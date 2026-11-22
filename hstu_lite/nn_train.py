"""Neural ranker.
Stage 1 (debiased pretrain on train):  python nn_train.py s1 --seed 0
Stage 2 (fine-tune on randomized val): python nn_train.py s2 --seed 0 --mode holdout   (pick epochs)
                                       python nn_train.py s2 --seed 0 --mode cv --epochs N
"""
import os, math, time, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .common import average_ranks, feed_rank_pct, Split, load_item_emb, gather_emb, ndcg_full, feed_rows, WORK, H_MAX, D_EMB, val_folds, val_holdout_mask

NNUM = 19      # numeric features built in the dataset
NEXTRA = 8     # features computed inside the model


# ----------------------------------------------------------------------------- data
class FeedData(torch.utils.data.Dataset):
    def __init__(self, split, feeds, cmax=0, train=False, emb_in_ram=False):
        self.split, self.feeds, self.cmax, self.train, self.ram = split, np.asarray(feeds), cmax, train, emb_in_ram
        self.S = self.E = None
        self.crop_seed = 0

    def _open(self):                     # lazy: every DataLoader worker opens its own memmaps
        if self.S is None:
            self.S = Split(self.split); self.E = load_item_emb(self.split, self.ram)

    def __len__(self):
        return len(self.feeds)

    def __getitem__(self, k):
        self._open(); S = self.S; f = int(self.feeds[k])
        c0, c1 = int(S.c_off[f]), int(S.c_off[f + 1])
        ci = np.arange(c0, c1)
        y = np.asarray(S.target[c0:c1], np.float32) if S.has_target else np.zeros(c1 - c0, np.float32)
        if self.train and self.cmax and len(ci) > self.cmax:      # crop long feeds, keep all positives
            p, n = np.flatnonzero(y > 0), np.flatnonzero(y <= 0)
            rng = np.random.default_rng(np.random.SeedSequence([self.crop_seed, f]))
            negatives = rng.choice(n, max(0, self.cmax - len(p)), replace=False)
            sel = np.sort(np.concatenate([p, negatives]))
            ci, y = ci[sel], y[sel]
        C = len(ci)
        cf = np.asarray(S.c_f[ci], np.float32)
        x = np.sign(cf) * np.log1p(np.abs(cf))
        z = (x - x.mean(0)) / (x.std(0) + 1e-6)
        rk = np.stack([average_ranks(column) for column in cf.T], axis=1) / max(C - 1, 1)
        ce, cp = gather_emb(self.E, np.asarray(S.c_eidx[ci]))
        h0 = int(S.h_off[f]); hl = min(int(S.h_off[f + 1]) - h0, H_MAX); hs = slice(h0, h0 + hl)
        he, hp = gather_emb(self.E, np.asarray(S.h_eidx[hs]))
        u = np.asarray(S.uemb[f])
        num = np.concatenate([x, z, rk, np.full((C, 1), np.log1p(C)), (~cp)[:, None],
                              np.full((C, 1), float((u == 0).all())), np.full((C, 1), float(hl == 0))], 1).astype(np.float32)
        es = np.asarray(S.h_es[hs]).astype(np.int64)
        es[(es < 0) | (es >= 26)] = 26          # rare / unseen sources -> one bucket
        return dict(f=f, ce=ce, cp=cp, ccat=np.asarray(S.c_cat[ci]), citem=np.asarray(S.c_item[ci]), num=num, y=y,
                    pos=np.asarray(S.pos[ci]) if S.has_pos else np.zeros(C, np.int16),
                    he=he, hp=hp, hcat=np.asarray(S.h_cat[hs]), het=np.asarray(S.h_et[hs]), hes=es,
                    hitem=np.asarray(S.h_item[hs]), u=u)


def collate(items):
    B = len(items); C = max(len(i["y"]) for i in items); H = max(1, max(len(i["hitem"]) for i in items))
    T = lambda shape, dt, fill=0: np.full(shape, fill, dt)
    o = dict(ce=T((B, C, D_EMB), np.int8), cp=T((B, C), bool), ccat=T((B, C), np.int64), citem=T((B, C), np.int64, -1),
             num=T((B, C, NNUM), np.float32), y=T((B, C), np.float32), pos=T((B, C), np.int64), cmask=T((B, C), bool),
             he=T((B, H, D_EMB), np.int8), hp=T((B, H), bool), hcat=T((B, H), np.int64), het=T((B, H), np.int64),
             hes=T((B, H), np.int64), hitem=T((B, H), np.int64, -2), hmask=T((B, H), bool), u=T((B, D_EMB), np.int8))
    for b, it in enumerate(items):
        c, h = len(it["y"]), len(it["hitem"])
        for k in ("ce", "cp", "ccat", "citem", "num", "y", "pos"): o[k][b, :c] = it[k]
        o["cmask"][b, :c] = True
        for k in ("he", "hp", "hcat", "het", "hes", "hitem"): o[k][b, :h] = it[k]
        o["hmask"][b, :h] = True; o["u"][b] = it["u"]
    out = {k: torch.from_numpy(v) for k, v in o.items()}
    out["f"] = torch.tensor([i["f"] for i in items]); out["n"] = torch.tensor([len(i["y"]) for i in items])
    return out


def size_batches(sizes, budget=16384, max_b=256):
    """inference batches of similar-size feeds (sorted) under a candidate budget"""
    order = np.argsort(sizes); batches, cur, mx = [], [], 0
    for i in order:
        m = max(mx, sizes[i])
        if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_b):
            batches.append(cur); cur, m = [], sizes[i]
        cur.append(i); mx = m
    if cur: batches.append(cur)
    return batches


# ----------------------------------------------------------------------------- model
class Cross(nn.Module):
    """candidate -> history target attention with additive bias from exact-item and same-category matches"""
    def __init__(self, d, nh, drop):
        super().__init__()
        self.nh, self.dh, self.p = nh, d // nh, drop
        self.lq, self.lkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.wm = nn.Parameter(torch.full((nh,), 1.0)); self.wc = nn.Parameter(torch.full((nh,), 0.3))
        self.drop = nn.Dropout(drop)

    def forward(self, x, kv, match, samecat, kmask):
        B, C, d = x.shape; L = kv.shape[1]
        q = self.q(self.lq(x)).view(B, C, self.nh, self.dh).transpose(1, 2)
        k, v = self.kv(self.lkv(kv)).view(B, L, 2, self.nh, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        bias = self.wm.view(1, -1, 1, 1) * match[:, None] + self.wc.view(1, -1, 1, 1) * samecat[:, None]
        bias = bias.masked_fill(~kmask[:, None, None, :], -1e4).to(q.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, C, d)))
        return x + self.drop(self.ffn(x))


def enc_layer(d, nh, drop):
    return nn.TransformerEncoderLayer(d, nh, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)


class Ranker(nn.Module):
    def __init__(self, d=128, nh=4, n_hist=2, n_cross=2, n_list=2, drop=0.1):
        super().__init__()
        self.cfg = dict(d=d, nh=nh, n_hist=n_hist, n_cross=n_cross, n_list=n_list, drop=drop)
        self.item_in = nn.Sequential(nn.Linear(D_EMB, d), nn.LayerNorm(d))
        self.user_in = nn.Sequential(nn.Linear(D_EMB, d), nn.LayerNorm(d))
        self.miss_c, self.miss_h = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.ccat, self.hcat = nn.Embedding(64, d), nn.Embedding(64, d)
        self.et, self.es, self.rec = nn.Embedding(16, d), nn.Embedding(32, d), nn.Embedding(H_MAX + 1, d)
        self.cnum = nn.Sequential(nn.Linear(NNUM + NEXTRA, d), nn.GELU(), nn.Linear(d, d))
        self.hist = nn.TransformerEncoder(enc_layer(d, nh, drop), n_hist, enable_nested_tensor=False)
        self.cross = nn.ModuleList([Cross(d, nh, drop) for _ in range(n_cross)])
        self.lst = nn.TransformerEncoder(enc_layer(d, nh, drop), n_list, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.pos_bias = nn.Embedding(128, 1); nn.init.zeros_(self.pos_bias.weight)

    @staticmethod
    def pos_bucket(p):
        p = p.clamp(min=0).float()
        b = torch.where(p < 64, p, 64 + torch.floor(torch.log2(p.clamp(min=64) / 64) * 8))
        return b.clamp(max=127).long()

    def forward(self, b, use_pos=False):
        cmask, hmask = b["cmask"], b["hmask"]
        B, C = cmask.shape; H = hmask.shape[1]; dev = cmask.device
        cef, hef, uf = b["ce"].float(), b["he"].float(), b["u"].float()
        with torch.autocast(device_type="cuda", enabled=False):          # exact similarity features in fp32
            cn, hn, un = F.normalize(cef, dim=-1), F.normalize(hef, dim=-1), F.normalize(uf, dim=-1)
            vh = hmask & b["hp"]
            sim = torch.bmm(cn, hn.transpose(1, 2))
            smax = sim.masked_fill(~vh[:, None, :], -2.0).amax(-1)
            smax = torch.where(vh.any(-1, keepdim=True), smax, 0.0)
            nvh = vh.sum(-1, keepdim=True).clamp(min=1)
            smean = (sim * vh[:, None, :]).sum(-1) / nvh
            rw = torch.exp(-torch.arange(H, device=dev, dtype=torch.float32) / 32.0)
            sdec = (sim * vh[:, None, :] * rw).sum(-1) / (vh * rw).sum(-1, keepdim=True).clamp(min=1e-6)
            cos_uc = (cn * un[:, None, :]).sum(-1)
            match = ((b["citem"][:, :, None] == b["hitem"][:, None, :]) & hmask[:, None, :]).float()
            samecat = ((b["ccat"][:, :, None] == b["hcat"][:, None, :]) & hmask[:, None, :]).float()
            hl = hmask.sum(-1, keepdim=True).clamp(min=1).float()
            cp = b["cp"].float()
            extra = torch.stack([smax * cp, smean * cp, sdec * cp, cos_uc, torch.log1p(match.sum(-1)), (match * rw).sum(-1),
                                 samecat.sum(-1) / hl, (samecat * rw).sum(-1) / rw[:H].sum()], -1)
        c = torch.where(b["cp"][..., None], self.item_in(cef / 64), self.miss_c)
        c = c + self.ccat(b["ccat"].clamp(0, 63)) + self.cnum(torch.cat([b["num"], extra], -1))
        h = torch.where(b["hp"][..., None], self.item_in(hef / 64), self.miss_h)
        h = h + self.hcat(b["hcat"].clamp(0, 63)) + self.et(b["het"].clamp(0, 15)) + self.es(b["hes"].clamp(0, 31))
        h = h + self.rec(torch.arange(H, device=dev))[None]
        kv = torch.cat([self.user_in(uf / 64)[:, None], h], 1)                    # user token always valid
        kmask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), hmask], 1)
        kv = self.hist(kv, src_key_padding_mask=~kmask)
        z1 = torch.zeros(B, C, 1, device=dev)
        m_, s_ = torch.cat([z1, match], -1), torch.cat([z1, samecat], -1)
        for blk in self.cross:
            c = blk(c, kv, m_, s_, kmask)
        c = self.lst(c, src_key_padding_mask=~cmask)                               # listwise context, no positions
        s = self.head(c).squeeze(-1).float()
        if use_pos and self.training:
            s = s + self.pos_bias(self.pos_bucket(b["pos"])).squeeze(-1).float()
        return s


# ----------------------------------------------------------------------------- losses
def softmax_loss(s, y, mask):
    lp = s.masked_fill(~mask, -1e4).log_softmax(-1)
    npos = y.sum(-1); ok = npos > 0
    return (-(lp * y).sum(-1) / npos.clamp_min(1) * ok).sum() / ok.sum().clamp_min(1)


def lambda_loss(s, y, mask):
    """LambdaRank: pairwise logistic weighted by |delta NDCG| (full-feed NDCG)"""
    s = s.masked_fill(~mask, -1e4); B, C = s.shape
    with torch.no_grad():
        negatives_first = y.argsort(dim=-1, stable=True)
        order = negatives_first.gather(1, s.gather(1, negatives_first).argsort(dim=-1, descending=True, stable=True))
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(C, device=s.device).expand(B, C))
        disc = 1.0 / torch.log2(rank.float() + 2.0)
        npos = y.sum(-1)
        ideal = torch.cumsum(1.0 / torch.log2(torch.arange(C, device=s.device).float() + 2.0), 0)
        idcg = ideal[(npos.long() - 1).clamp(min=0)]
        pair = (y[:, :, None] > y[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        w = (disc[:, :, None] - disc[:, None, :]).abs() / idcg[:, None, None] * pair
    l = F.softplus(-(s[:, :, None] - s[:, None, :])) * w
    return l.sum() / (npos > 0).sum().clamp(min=1)


# ----------------------------------------------------------------------------- utils
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else: self.shadow[k].copy_(v)


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.no_grad()
def predict(model, split, feeds, dev, workers=4, emb_in_ram=False):
    S = Split(split); model.eval()
    out = np.full(S.N, np.nan, np.float32)
    sizes = (S.c_off[feeds + 1] - S.c_off[feeds]).astype(np.int64)
    ds = FeedData(split, feeds, emb_in_ram=emb_in_ram)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=size_batches(sizes), collate_fn=collate, num_workers=workers,
                                     pin_memory=torch.device(dev).type == "cuda", persistent_workers=False)
    for b in dl:
        g = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda" and torch.cuda.is_bf16_supported()):
            s = model(g).float().cpu().numpy()
        for j, (f, n) in enumerate(zip(b["f"].tolist(), b["n"].tolist())):
            out[S.c_off[f]:S.c_off[f] + n] = s[j, :n]
    return out


def evaluate(model, dev, split="val", feeds=None, workers=4):
    S = Split(split); feeds = np.arange(S.F) if feeds is None else feeds
    p = predict(model, split, feeds, dev, workers)
    rows, sizes = feed_rows(S.c_off, feeds)
    sub_off = np.concatenate([[0], np.cumsum(sizes)])
    return ndcg_full(sub_off, p[rows], np.asarray(S.target)[rows])[0], p


def cosine_lr(step, total, warm, base, floor=0.05):
    if step < warm: return base * (step + 1) / warm
    t = (step - warm) / max(total - warm, 1)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def train_steps(model, ema, opt, loader, dev, steps_total, step0, a, use_pos, loss_w, mix_iter=None, on_step=None):
    model.train(); step = step0
    for b in loader:
        for g_ in opt.param_groups: g_["lr"] = cosine_lr(step, steps_total, a.warmup, a.lr)
        g = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda" and torch.cuda.is_bf16_supported()):
            s = model(g, use_pos=use_pos)
        loss = loss_w[0] * softmax_loss(s, g["y"], g["cmask"]) + loss_w[1] * lambda_loss(s, g["y"], g["cmask"])
        if mix_iter is not None:                    # regulariser: keep seeing (debiased) train feeds while fine-tuning
            gt = to_dev(next(mix_iter), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(dev).type == "cuda" and torch.cuda.is_bf16_supported()):
                st = model(gt, use_pos=True)
            loss = loss + a.mix_train * softmax_loss(st, gt["y"], gt["cmask"])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update(model)
        step += 1
        if step % 200 == 0: print(f"step {step}/{steps_total} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        if on_step: on_step(step)
        if step >= steps_total: break
    return step


def make_opt(model, lr, wd):
    emb_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    dec = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in emb_ids]
    nodec = [p for p in model.parameters() if not (p.ndim >= 2 and id(p) not in emb_ids)]
    return torch.optim.AdamW([{"params": dec, "weight_decay": wd}, {"params": nodec, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.98), fused=next(model.parameters()).is_cuda)


def infinite(loader):
    while True:
        for b in loader: yield b


def train_loader(split, feeds, a, cmax, bs):
    ds = FeedData(split, feeds, cmax=cmax, train=True, emb_in_ram=a.emb_in_ram)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=a.workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=a.workers > 0,
                                       prefetch_factor=4 if a.workers > 0 else None)


def feeds_with_pos(S, feeds):
    counts = np.add.reduceat(S.target, S.c_off[:-1], dtype=np.int32)
    return feeds[counts[feeds] > 0]


# ----------------------------------------------------------------------------- stages
def stage1(a, dev):
    St, Sv, Ste = Split("train"), Split("val"), Split("test")
    feeds = np.arange(St.F)
    if a.train_feeds: feeds = np.sort(np.argsort(np.asarray(St.ts), kind="stable")[-a.train_feeds:])
    feeds = feeds_with_pos(St, feeds)
    model = Ranker(a.d, a.heads, a.n_hist, a.n_cross, a.n_list, a.drop).to(dev)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M; train feeds", len(feeds))
    ema = EMA(model, a.ema); opt = make_opt(model, a.lr, a.wd)
    total = a.epochs * (len(feeds) // a.bs)
    best = [-1.0]; ck = os.path.join(WORK, f"nn1_s{a.seed}.pt")
    ev = Ranker(**model.cfg).to(dev)
    sel_feeds = np.arange(Sv.F)[~val_holdout_mask(Sv)]          # select checkpoint on 22 Sep only; 24 Sep stays clean

    def on_step(step):
        if step % a.eval_every == 0 or step == total:
            ev.load_state_dict(ema.shadow); m, _ = evaluate(ev, dev, "val", sel_feeds, a.workers)
            print(f"== step {step}: val(22 Sep) NDCG (EMA) {m:.5f}", flush=True)
            if m > best[0]:
                best[0] = m; torch.save({"cfg": model.cfg, "sd": ema.shadow, "val": m, "step": step}, ck)
            model.train()

    dl = train_loader("train", feeds, a, a.cmax, a.bs); step = 0
    while step < total:
        step = train_steps(model, ema, opt, dl, dev, total, step, a, use_pos=True, loss_w=(1.0, a.lambda_w), on_step=on_step)
    st = torch.load(ck, map_location=dev); ev.load_state_dict(st["sd"])
    print("best val(22 Sep) NDCG", st["val"], "at step", st["step"])
    print("full val NDCG:", evaluate(ev, dev, "val", None, a.workers)[0])
    np.save(os.path.join(WORK, f"nn1_s{a.seed}_val.npy"), predict(ev, "val", np.arange(Sv.F), dev, a.workers))
    np.save(os.path.join(WORK, f"nn1_s{a.seed}_test.npy"), predict(ev, "test", np.arange(Ste.F), dev, a.workers))


def finetune(a, dev, tr_feeds, ev_feeds_list, epochs, tag):
    st = torch.load(a.init or os.path.join(WORK, f"nn1_s{a.seed}.pt"), map_location=dev)
    model = Ranker(**st["cfg"]).to(dev); model.load_state_dict(st["sd"])
    ema = EMA(model, a.ema); opt = make_opt(model, a.lr, a.wd)
    Sv = Split("val"); tr_feeds = feeds_with_pos(Sv, tr_feeds)
    dl = train_loader("val", tr_feeds, a, a.cmax, a.bs)
    mix = None
    if a.mix_train > 0:
        St = Split("train"); tf = np.sort(np.argsort(np.asarray(St.ts), kind="stable")[-300000:])
        mix = infinite(train_loader("train", feeds_with_pos(St, tf), a, 128, a.bs))
    spe = math.ceil(len(tr_feeds) / a.bs); total = epochs * spe; step = 0; ev = Ranker(**st["cfg"]).to(dev); hist = []
    for ep in range(epochs):
        step = train_steps(model, ema, opt, dl, dev, total, step, a, use_pos=False, loss_w=(1.0, a.lambda_w), mix_iter=mix)
        ev.load_state_dict(ema.shadow)
        if ev_feeds_list is not None:
            m, _ = evaluate(ev, dev, "val", ev_feeds_list, a.workers); hist.append(m)
            print(f"[{tag}] epoch {ep + 1}: early tuning NDCG {m:.5f}", flush=True)
    return ev, hist


def stage2(a, dev):
    Sv, Ste = Split("val"), Split("test")
    allf = np.arange(Sv.F)
    if a.mode == "holdout":
        early = ~val_holdout_mask(Sv)
        groups = val_folds(Sv)
        tuning = early & (groups == 0)
        fitting = early & (groups != 0)
        ev0 = Ranker(**torch.load(os.path.join(WORK, f"nn1_s{a.seed}.pt"), map_location=dev)["cfg"]).to(dev)
        ev0.load_state_dict(torch.load(os.path.join(WORK, f"nn1_s{a.seed}.pt"), map_location=dev)["sd"])
        print("stage-1 NDCG on early tuning users:", evaluate(ev0, dev, "val", allf[tuning], a.workers)[0])
        _, hist = finetune(a, dev, allf[fitting], allf[tuning], a.epochs, "holdout")
        print("best epochs =", int(np.argmax(hist)) + 1, hist)
    else:
        folds = val_folds(Sv); oof = np.full(Sv.N, np.nan, np.float32); te = np.zeros(Ste.N, np.float64)
        for k in range(folds.max() + 1):
            ev, _ = finetune(a, dev, allf[folds != k], None, a.epochs, f"fold{k}")
            p = predict(ev, "val", allf[folds == k], dev, a.workers)
            rows, sizes = feed_rows(Sv.c_off, allf[folds == k])
            oof[rows] = feed_rank_pct(np.r_[0, sizes.cumsum()], p[rows])
            pt = predict(ev, "test", np.arange(Ste.F), dev, a.workers)
            te += feed_rank_pct(Ste.c_off, pt)
        print("OOF NDCG:", ndcg_full(Sv.c_off, oof, np.asarray(Sv.target))[0])
        np.save(os.path.join(WORK, f"nn2_s{a.seed}_oof.npy"), oof)
        np.save(os.path.join(WORK, f"nn2_s{a.seed}_test.npy"), (te / (folds.max() + 1)).astype(np.float32))


def check_perm(a, dev):
    """scores must not depend on candidate order (dropout off)"""
    st = torch.load(os.path.join(WORK, f"nn1_s{a.seed}.pt"), map_location=dev)
    m = Ranker(**st["cfg"]).to(dev); m.load_state_dict(st["sd"]); m.eval()
    ds = FeedData("val", np.arange(64)); items = [ds[i] for i in range(64)]
    with torch.inference_mode():
        s1 = m(to_dev(collate(items), dev)).detach()
    rng = np.random.default_rng(0); perms = []
    for it in items:
        p = rng.permutation(len(it["y"])); perms.append(p)
        for k in ("ce", "cp", "ccat", "citem", "num", "y", "pos"): it[k] = it[k][p]
    with torch.inference_mode():
        s2 = m(to_dev(collate(items), dev)).detach()
    err = max((s1[j, p] - s2[j, :len(p)]).abs().max().item() for j, p in enumerate(perms))
    print("max |delta| after permutation:", err); assert err < 1e-3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["s1", "s2", "perm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--mode", default="cv", choices=["holdout", "cv"])
    ap.add_argument("--init", default="")
    ap.add_argument("--train-feeds", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--cmax", type=int, default=None)
    ap.add_argument("--ema", type=float, default=None)
    ap.add_argument("--lambda-w", type=float, default=None)
    ap.add_argument("--mix-train", type=float, default=0.25)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-hist", type=int, default=2); ap.add_argument("--n-cross", type=int, default=2)
    ap.add_argument("--n-list", type=int, default=2); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--emb-in-ram", action="store_true")
    a = ap.parse_args()
    s1 = a.stage == "s1"
    for k, v1, v2 in [("epochs", 2, 4), ("bs", 128, 32), ("lr", 1e-3, 1e-4), ("warmup", 1000, 50), ("cmax", 128, 256),
                      ("ema", 0.999, 0.99), ("lambda_w", 0.3, 1.0)]:
        if getattr(a, k) is None: setattr(a, k, v1 if s1 else v2)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    os.makedirs(WORK, exist_ok=True); dev = torch.device(a.device)
    t0 = time.time()
    {"s1": stage1, "s2": stage2, "perm": check_perm}[a.stage](a, dev)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
