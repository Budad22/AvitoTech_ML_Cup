"""Compare frozen checkpoints on the same validation feeds."""
import ast
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from .data import Feeds, load_batch
from .model import HistoryRanker
from .state import Lock, atomic_write, code_identity, read_json, sha256, write_json

OFFICIAL_SHA = "87b731df2a5507957c075e93a7f219c1fa4f59b9b7bb9752c59c43087bb2cc06"
BASELINE_SHA = "e64954d8704843d7d490c26bd1f0c7d8437e76fed48405b0589f1742e4fdfe11"


def load_frozen(run, stage, data_hash):
    run = Path(run)
    config = read_json(run / "config.json")
    frozen = read_json(run / "frozen.json")
    path = run / "last.pt"
    digest = sha256(path)
    if (config["stage"] != stage or config["cache_sha256"] != data_hash
            or config != frozen["config"] or frozen["checkpoint_sha256"] != digest):
        raise ValueError(f"Invalid frozen {stage} checkpoint identity")
    step = config["epochs"] * math.ceil(config["feeds"] / config["batch"])
    from .train import STAGES, EFFECTIVE_BATCH
    if (config["epochs"] != STAGES[stage][0] or config["batch"] != EFFECTIVE_BATCH
            or frozen["step"] != step):
        raise ValueError(f"Incomplete stage {stage}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["steps"] != step or state["epoch"] != config["epochs"] or state["cursor"] != 0 or state["config"] != config:
        raise ValueError(f"Checkpoint cursor differs for {stage}")
    return state, digest


def baseline_model(source, checkpoint, compact, device):
    if sha256(source) != OFFICIAL_SHA or sha256(checkpoint) != BASELINE_SHA:
        raise ValueError("Official baseline source or checkpoint digest differs")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    ranker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Ranker")
    module = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=[ranker], type_ignores=[]), str(source), "exec"), module)
    mapping = read_json(Path(compact) / "metadata.json")["categories"]
    model = module["Ranker"](len(mapping)).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    codes = torch.zeros(54, dtype=torch.long, device=device)
    for original, encoded in mapping.items():
        codes[int(original)] = encoded

    def predict(data):
        item = F.normalize(data["item"], dim=-1)
        user = F.normalize(data["user"].float(), dim=-1)
        feats = torch.cat((data["floats"], item, data["missing"].unsqueeze(-1)), dim=-1)
        return model(feats, codes[data["category"]], user, ~data["valid"]).sigmoid().float()

    return predict


def history_model(state, device):
    model = HistoryRanker(state["config"]["query_chunk"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def ndcg(scores, labels, offsets):
    values = np.empty(len(offsets) - 1)
    discounts = 1.0 / np.log2(np.arange(2, np.diff(offsets).max() + 2, dtype=np.float64))
    for i, (left, right) in enumerate(zip(offsets, offsets[1:])):
        target = labels[left:right]
        positives = int(target.sum())
        order = np.lexsort((target, -scores[left:right]))
        values[i] = target[order] @ discounts[:right-left] / discounts[:positives].sum() if positives else 1.0
    return values


def paired_bootstrap(users, differences, repetitions=10000):
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    user_sums = np.stack([np.bincount(inverse, weights=d) for d in differences], axis=1)
    rng = np.random.default_rng(42)
    samples = []
    for offset in range(0, repetitions, 64):
        sample = rng.integers(len(counts), size=(min(64, repetitions-offset), len(counts)))
        samples.append(user_sums[sample].sum(axis=1) / counts[sample].sum(axis=1, keepdims=True))
    intervals = np.quantile(np.concatenate(samples), [0.025, 0.975], axis=0).T
    return intervals, len(counts)


@torch.no_grad()
def scores_for(dataset, predictors, microbatch, device):
    offsets = dataset.sidecar["offsets"]
    scores = {name: np.empty(offsets[-1], dtype=np.float32) for name in predictors}
    for left in range(0, len(dataset), microbatch):
        right = min(left + microbatch, len(dataset))
        features, _ = load_batch(dataset, range(left, right), device)
        destination = slice(offsets[left], offsets[right])
        for name, model in predictors.items():
            scores[name][destination] = model(features)[features["valid"]].cpu().numpy()
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise ValueError("Nonfinite evaluation scores")
    return scores


def evaluate(args):
    from catboost.utils import eval_metric

    cache_hash = sha256(args.cache / "metadata.json")
    pilot = args.command == "evaluate-pilot"
    # Require both frozen checkpoints before opening the held-out day.
    digests = {}
    if pilot:
        state, digests["pilot"] = load_frozen(args.pilot, "pilot", cache_hash)
        predictors = {"pilot": history_model(state, args.device)}
        split = "early"
    else:
        state_a, digests["a"] = load_frozen(args.stage_a, "a", cache_hash)
        state_b, digests["b"] = load_frozen(args.stage_b, "b", cache_hash)
        if state_b["config"]["parent_sha256"] != digests["a"]:
            raise ValueError("Stage B was initialized from a different A")
        manifest = read_json(args.cache / "metadata.json")
        if manifest["identity"]["scope"] != "full":
            raise ValueError("Full evaluation requires full cache")
        predictors = {"a": history_model(state_a, args.device), "b": history_model(state_b, args.device)}
        predictors["baseline"] = baseline_model(args.official, args.baseline,
                                                manifest["identity"]["compact"], args.device)
        digests["baseline"] = BASELINE_SHA
        split = "late"
    data = Feeds(args.cache, split)
    offsets = data.sidecar["offsets"]
    labels = data.sidecar["target"]
    users = data.sidecar["user_id"]
    feed = np.repeat(data.sidecar["feed_id"], np.diff(offsets))
    item = data.sidecar["item_id"]
    keys = (feed.astype(np.uint64) << np.uint64(32)) | item.astype(np.uint64)
    if len(keys) != data.info["candidates"] or np.unique(keys).size != len(keys):
        raise ValueError("Unexpected candidate coverage")
    args.output.mkdir(parents=True, exist_ok=True)
    with Lock(args.output / ".lock"):
        config = dict(scope=split, cache_sha256=cache_hash, checkpoints=digests,
                      code=code_identity(), device=args.device, microbatch=args.microbatch)
        path = args.output / "config.json"
        if path.exists() and read_json(path) != config:
            raise ValueError("Evaluation output belongs to a different run")
        write_json(path, config)
        score = scores_for(data, predictors, args.microbatch, args.device)
        per_feed = {name: ndcg(value, labels, offsets) for name, value in score.items()}
        metrics = {name: float(value.mean()) for name, value in per_feed.items()}
        verification = {}
        groups = feed.tolist()
        target = labels.tolist()
        for name, value in score.items():
            reference = eval_metric(target, value.tolist(), "NDCG:type=Base", group_id=groups, thread_count=4)[0]
            verification[name] = float(abs(metrics[name] - reference))
            if verification[name] > 1e-10:
                raise ValueError(f"CatBoost metric disagrees for {name}")
        output = args.output / "scores.parquet"
        table = pa.table(dict(feed_id=feed, item_id=item, **score))
        atomic_write(output, lambda tmp: pq.write_table(table, tmp))
        report = dict(feeds=len(data), candidates=len(keys), users=int(np.unique(users).size),
                      ndcg=metrics, catboost_abs_diff=verification, **config,
                      scores_sha256=sha256(output), feed_ids_sha256=data.info["files"]["feed_id.npy"])
        if not pilot:
            names = ("b-baseline", "a-baseline", "b-a")
            differences = [per_feed["b"] - per_feed["baseline"],
                           per_feed["a"] - per_feed["baseline"], per_feed["b"] - per_feed["a"]]
            intervals, users_count = paired_bootstrap(users, differences)
            report["paired"] = {key: dict(delta=float(values.mean()), ci95=ci.tolist(), users=users_count)
                                for key, values, ci in zip(names, differences, intervals)}
            report["historical_baseline"] = True
        write_json(args.output / "report.json", report)
        print(report, flush=True)
