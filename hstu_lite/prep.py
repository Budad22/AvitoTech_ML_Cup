"""Parquet -> packed flat binary arrays.  python prep.py --split val ; --split test ; --split train [--train-shards 512]"""
import os, glob, json, argparse, numpy as np
import pyarrow.parquet as pq, pyarrow.compute as pc
from pathlib import Path
from contextlib import ExitStack
from datetime import datetime, timezone
from .common import RAW, PACK, ALL_COLS, D_EMB

LISTC = {"c_item": "item_id", "c_eidx": "item_id__embedding_idx", "c_cat": "category_id",
         "target": "target", "pos": "pos",
         "h_item": "history__item_id", "h_eidx": "history__item_id__embedding_idx", "h_cat": "history__category_id",
         "h_et": "history__event_type", "h_es": "history__event_source"}


def flat(col):
    a = col.combine_chunks()
    return pc.list_flatten(a).to_numpy(zero_copy_only=False), pc.list_value_length(a).to_numpy(zero_copy_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--train-shards", type=int, default=0, help="Evenly spaced train shards, 0 = all")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RAW, a.split, "impressions", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet in {RAW}/{a.split}/impressions")
    if a.split == "train" and a.train_shards:
        if not 1 <= a.train_shards <= len(files):
            ap.error("Invalid shard count")
        files = [files[i] for i in np.linspace(0, len(files) - 1, a.train_shards, dtype=int)]
    destination = Path(PACK) / a.split
    if destination.exists():
        raise FileExistsError(f"Packed split already exists: {destination}")
    out = destination.with_name(destination.name + ".building")
    out.mkdir(parents=True)
    clen_all, hlen_all = [], []
    schema = pq.read_schema(files[0]).names
    with ExitStack() as stack:
        fh = {k: stack.enter_context((out / (k + ".bin")).open("wb")) for k in ALL_COLS}
        for index, f in enumerate(files, 1):
            t = pq.read_table(f)
            for key, column in (("feed_id", "feed_id"), ("user_id", "user_id"), ("ts", "timestamp")):
                values = t[column].to_numpy()
                packed = values.astype(np.int32)
                if not np.array_equal(values, packed):
                    raise ValueError(f"{column} overflows int32")
                fh[key].write(packed.tobytes())
            ue, ul = flat(t["user_embedding"])
            if not (ul == D_EMB).all():
                raise ValueError("Unexpected user vector width")
            fh["uemb"].write(ue.astype(np.int8).tobytes())
            ci, cl = flat(t["item_id"])
            if np.any(cl == 0):
                raise ValueError("Empty candidate feed")
            clen_all.append(cl)
            fh["c_item"].write(ci.astype(np.int32).tobytes())
            hi, hl = flat(t["history__item_id"])
            hlen_all.append(hl)
            fh["h_item"].write(hi.astype(np.int32).tobytes())
            for key, column in LISTC.items():
                if key in ("c_item", "h_item") or column not in schema:
                    continue
                values, lengths = flat(t[column])
                if not np.array_equal(lengths, hl if key.startswith("h_") else cl):
                    raise ValueError(f"Misaligned {column}")
                fh[key].write(values.astype(ALL_COLS[key][0]).tobytes())
            fs = [flat(t[f"f{i}"]) for i in range(1, 6)]
            if any(not np.array_equal(lengths, cl) for _, lengths in fs):
                raise ValueError("Misaligned numeric columns")
            numeric = np.stack([values for values, _ in fs], axis=1).astype(np.float32)
            if not np.isfinite(numeric).all():
                raise ValueError("Nonfinite numeric input")
            fh["c_f"].write(numeric.tobytes())
            if index % 16 == 0 or index == len(files):
                print(f"{a.split}: {index}/{len(files)} shards", flush=True)
    cl = np.concatenate(clen_all); hl = np.concatenate(hlen_all)
    np.save(os.path.join(out, "c_off.npy"), np.concatenate([[0], np.cumsum(cl)]).astype(np.int64))
    np.save(os.path.join(out, "h_off.npy"), np.concatenate([[0], np.cumsum(hl)]).astype(np.int64))
    meta = dict(files=[dict(name=Path(f).name, bytes=Path(f).stat().st_size) for f in files],
                source=str(Path(RAW).resolve()), packed_at=datetime.now(timezone.utc).isoformat(),
                feeds=int(len(cl)), cands=int(cl.sum()), hist=int(hl.sum()),
                max_hist=int(hl.max()), max_cands=int(cl.max()))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out.rename(destination)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
