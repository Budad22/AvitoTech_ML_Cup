"""Profile the downloaded audit subset; never treats a train sample as full train.

Run with .venv/bin/python scripts/audit_data.py after inspect_remote_zip.py --audit-sample.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def summary(values):
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {"count": int(a.size), "nonfinite": int(a.size)}
    return {"count": int(a.size), "nonfinite": int(a.size - finite.size),
            "min": float(finite.min()), "p01": float(np.quantile(finite, .01)),
            "p50": float(np.quantile(finite, .5)), "p95": float(np.quantile(finite, .95)),
            "p99": float(np.quantile(finite, .99)), "max": float(finite.max()),
            "mean": float(finite.mean()), "std": float(finite.std()),
            "zero_fraction": float(np.mean(finite == 0))}


def numpy(array):
    return array.to_numpy(zero_copy_only=False)


def ndcg(labels, scores):
    count = int(labels.sum())
    if not count:
        return 1.0
    discounts = 1 / np.log2(np.arange(len(labels)) + 2)
    order = np.lexsort((labels, -scores))  # Official pessimistic ties.
    return float(np.dot(labels[order], discounts) / discounts[:count].sum())


def profile(root, split):
    folder = root / split
    files = sorted((folder / "impressions").glob("*.parquet"))
    meta = json.loads((folder / "item_embeddings/data.meta").read_text())
    vector_path = folder / "item_embeddings/data.memmap"
    embeddings = np.memmap(vector_path, dtype=meta["dtype"], mode="r",
                           shape=tuple(meta["shape"]), order=meta["order"]) if vector_path.exists() else None
    result = {"files": len(files), "file_names": [p.name for p in files],
              "embedding_meta": meta, "has_local_item_vectors": embeddings is not None,
              "nulls": Counter(), "invalid": Counter(), "category_counts": Counter(),
              "history_event_type_counts": Counter(), "history_event_source_counts": Counter(),
              "history_category_counts": Counter(), "position_bins": {}, "score_controls": {}}
    arrays = {name: [] for name in ["feed_id", "user_id", "timestamp", "candidates", "history",
                                   "positives", "user_vector_norm", "user_vector_zero",
                                   "f1", "f2", "f3", "f4", "f5"]}
    item_ids, history_ids, packed_keys = set(), set(), []
    counters = Counter()
    controls = {}
    pos_edges = np.array([0, 1, 5, 10, 30, 100, 301])
    pos_n, pos_y = np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)
    for path in files:
        table = pq.read_table(path)
        data = {name: table[name].combine_chunks() for name in table.column_names}
        result["schema"] = {field.name: str(field.type) for field in table.schema}
        for name, column in data.items():
            result["nulls"][name] += column.null_count
            if pa.types.is_list(column.type):
                result["nulls"][name + ".elements"] += column.values.null_count
        flat = {name: numpy(column.values) for name, column in data.items()
                if pa.types.is_list(column.type)}
        n = numpy(pc.list_value_length(data["item_id"]))
        h = numpy(pc.list_value_length(data["history__item_id"]))
        offsets = numpy(data["item_id"].offsets)
        hist_offsets = numpy(data["history__item_id"].offsets)
        counters["feeds"] += len(n)
        counters["candidate_rows"] += int(n.sum())
        counters["history_events"] += int(h.sum())
        counters["empty_histories"] += int(np.sum(h == 0))
        counters["histories_at_limit"] += int(np.sum(h == 256))
        for name in ("feed_id", "user_id", "timestamp"):
            arrays[name].append(numpy(data[name]))
        arrays["candidates"].append(n)
        arrays["history"].append(h)
        for name in flat:
            expected = h if name.startswith("history__") else np.full(len(n), 128) if name == "user_embedding" else n
            lengths = numpy(pc.list_value_length(data[name]))
            result["invalid"][name + ".length"] += int(np.sum(lengths != expected))
        users = flat["user_embedding"].astype(np.float32).reshape(-1, 128)
        arrays["user_vector_norm"].append(np.linalg.norm(users, axis=1))
        arrays["user_vector_zero"].append(np.all(users == 0, axis=1))
        for name in ("category_id", "history__category_id", "history__event_type", "history__event_source"):
            key = {"category_id": "category_counts", "history__category_id": "history_category_counts",
                   "history__event_type": "history_event_type_counts", "history__event_source": "history_event_source_counts"}[name]
            codes, counts = np.unique(flat[name], return_counts=True)
            result[key].update({int(c): int(k) for c, k in zip(codes, counts)})
        for name in ("f1", "f2", "f3", "f4", "f5"):
            arrays[name].append(flat[name])
        for prefix, name in (("candidate", "item_id__embedding_idx"), ("history", "history__item_id__embedding_idx")):
            idx = flat[name]
            counters[prefix + "_missing_vectors"] += int(np.sum(idx == -1))
            result["invalid"][prefix + "_vector_indices"] += int(np.sum((idx < -1) | (idx >= meta["shape"][0])))
        item_ids.update(map(int, np.unique(flat["item_id"])))
        history_ids.update(map(int, np.unique(flat["history__item_id"])))
        feed = numpy(data["feed_id"]).astype(np.int64)
        packed_keys.append((np.repeat(feed, n) << 32) | flat["item_id"].astype(np.int64))
        labels = flat.get("target")
        if labels is not None:
            result["invalid"]["target_not_binary"] += int(np.sum((labels != 0) & (labels != 1)))
            prefix = np.r_[0, np.cumsum(labels, dtype=np.int64)]
            arrays["positives"].append(prefix[offsets[1:]] - prefix[offsets[:-1]])
            if "pos" in flat:
                bins = np.searchsorted(pos_edges, flat["pos"], side="right") - 1
                valid = (bins >= 0) & (bins < 6)
                pos_n += np.bincount(bins[valid], minlength=6)
                pos_y += np.bincount(bins[valid], weights=labels[valid], minlength=6).astype(np.int64)
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            candidates = flat["item_id"][start:end]
            history = flat["history__item_id"][hist_offsets[i]:hist_offsets[i + 1]]
            counters["duplicate_candidate_pairs"] += len(candidates) - len(np.unique(candidates))
            counters["repeated_history_events_by_item"] += len(history) - len(np.unique(history))
            counters["candidates_seen_in_history"] += int(np.isin(candidates, history).sum())
            if split == "val" and labels is not None:
                y = labels[start:end]
                scores = {name: flat[name][start:end].astype(np.float64) for name in ("f1", "f2", "f3", "f4", "f5")}
                scores["constant_pessimistic_ties"] = np.zeros(len(y))
                if embeddings is not None:
                    idx = flat["item_id__embedding_idx"][start:end]
                    present = idx >= 0
                    vectors = np.zeros((len(y), 128), dtype=np.float32)
                    vectors[present] = embeddings[idx[present]].astype(np.float32)
                    denominator = np.linalg.norm(vectors, axis=1) * np.linalg.norm(users[i])
                    dot = np.einsum("ij,j->i", vectors, users[i])
                    scores["user_item_cosine"] = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
                for name, score in scores.items():
                    if not np.isfinite(score).all():
                        raise ValueError(f"Nonfinite control score: {name} in {path.name}")
                    controls.setdefault(name, []).append(ndcg(y, score))
                discounts = 1 / np.log2(np.arange(len(y)) + 2)
                p = int(y.sum())
                controls.setdefault("expected_uniform_random", []).append(
                    float(p / len(y) * discounts.sum() / discounts[:p].sum()) if p else 1.)
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items() if parts}
    result["counts"] = dict(counters)
    for name in ("candidates", "history", "positives", "user_vector_norm", "f1", "f2", "f3", "f4", "f5"):
        if name in arrays:
            result[name] = summary(arrays[name])
    result["time_utc"] = [datetime.fromtimestamp(int(f(arrays["timestamp"])), timezone.utc).isoformat() for f in (np.min, np.max)]
    moscow = timezone(timedelta(hours=3))
    result["time_moscow"] = [datetime.fromtimestamp(int(f(arrays["timestamp"])), moscow).isoformat() for f in (np.min, np.max)]
    days = (arrays["timestamp"].astype(np.int64) + 3 * 3600) // 86400
    result["days_moscow"] = {}
    for day in np.unique(days):
        selected = days == day
        key = datetime.fromtimestamp(int(day) * 86400, timezone.utc).date().isoformat()
        entry = {"feeds": int(selected.sum()), "candidates": int(arrays["candidates"][selected].sum())}
        if "positives" in arrays:
            entry["positive_fraction"] = float(arrays["positives"][selected].sum() / entry["candidates"])
        result["days_moscow"][key] = entry
    result["unique_users"] = len(np.unique(arrays["user_id"]))
    result["unique_candidate_items"] = len(item_ids)
    result["unique_history_items"] = len(history_ids)
    result["invalid"]["duplicate_feed_ids"] = int(len(arrays["feed_id"]) - len(np.unique(arrays["feed_id"])))
    result["invalid"]["feeds_below_5_candidates"] = int(np.sum(arrays["candidates"] < 5))
    result["invalid"]["histories_over_256"] = int(np.sum(arrays["history"] > 256))
    result["zero_user_vectors"] = int(arrays["user_vector_zero"].sum())
    if "positives" in arrays:
        result["positive_fraction"] = float(arrays["positives"].sum() / arrays["candidates"].sum())
        result["feeds_without_positives"] = int(np.sum(arrays["positives"] == 0))
    for i in range(6):
        if pos_n[i]:
            result["position_bins"][f"{pos_edges[i]}..{pos_edges[i+1]-1}"] = {
                "rows": int(pos_n[i]), "positive_fraction": float(pos_y[i] / pos_n[i])}
    result["score_controls"] = {name: float(np.mean(values)) for name, values in controls.items()}
    if embeddings is not None:
        norms, zero_count, low_count, high_count = [], 0, 0, 0
        for start in range(0, len(embeddings), 65536):
            block = embeddings[start:start + 65536]
            norms.append(np.linalg.norm(block.astype(np.float32), axis=1))
            zero_count += int(np.all(block == 0, axis=1).sum())
            low_count += int(np.sum(block == -128))
            high_count += int(np.sum(block == 127))
        result["item_vector_norm"] = summary(np.concatenate(norms))
        result["item_vector_zero_rows"] = zero_count
        result["item_vector_extreme_components"] = {"minus_128": low_count, "plus_127": high_count}
    # Keep arrays only in memory for exact overlap and submission checks.
    state = {"feeds": set(map(int, arrays["feed_id"])), "users": set(map(int, arrays["user_id"])),
             "items": item_ids, "keys": np.concatenate(packed_keys)}
    return result, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/initial_audit/competition"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data.parent / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output = {"scope": "Local impressions files; coverage is explicit per split; local item vectors only",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "splits": {}, "overlap": {}, "versions": {"numpy": np.__version__, "pyarrow": pa.__version__}}
    states = {}
    for split in ("train", "val", "test"):
        output["splits"][split], states[split] = profile(args.data, split)
        expected_files = {m["name"].rsplit("/", 1)[-1] for m in manifest["members"]
                          if f"/{split}/impressions/" in m["name"] and m["name"].endswith(".parquet")}
        actual_files = set(output["splits"][split]["file_names"])
        output["splits"][split]["coverage"] = {"archive_files": len(expected_files),
            "local_files": len(actual_files), "complete_impressions": actual_files == expected_files}
        print(json.dumps({"split": split, "counts": output["splits"][split]["counts"]}), flush=True)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        output["overlap"][f"{a}_{b}"] = {name: len(states[a][name] & states[b][name]) for name in ("feeds", "users", "items")}
    template = pq.read_table(args.data / "sample_submission.parquet")
    template_keys = (numpy(template["feed_id"].combine_chunks()).astype(np.int64) << 32) | numpy(template["item_id"].combine_chunks()).astype(np.int64)
    output["submission"] = {"rows": len(template), "schema": str(template.schema),
                            "duplicate_pairs": int(len(template_keys) - len(np.unique(template_keys))),
                            "exact_test_key_coverage": bool(np.array_equal(np.sort(template_keys), np.sort(states["test"]["keys"])))}
    output["archive"] = manifest
    output["local_file_sha256"] = {}
    for path in sorted(args.data.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            output["local_file_sha256"][str(path.relative_to(args.data))] = digest.hexdigest()
    output["script_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(Path("scripts").glob("*.py"))}
    output["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
