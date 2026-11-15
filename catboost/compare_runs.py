"""Paired comparison on identical validation keys; diagnostic slices, no tuning."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from baseline import ranking_metrics, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys_data import PreparedSplit


def compare(data, control, treatment):
    old = json.loads((control / "result.json").read_text())
    new = json.loads((treatment / "result.json").read_text())
    if old["status"] != "complete" or new["status"] != "complete":
        raise ValueError("Both runs must be complete")
    if not (old["prepared_manifest_sha256"] == new["prepared_manifest_sha256"] == sha256(data / "manifest.json")):
        raise ValueError("Runs use different data")
    if old["config"] != new["config"]:
        raise ValueError("Model parameters differ; this comparison isolates features")
    batches = list(PreparedSplit(data, "val").iter_batches(include_labels=True))
    truth = pa.concat_tables([b.candidate_table(include_target=True).select(["feed_id", "item_id", "target"])
                             for b in batches])
    metadata = pa.concat_tables([b.features.select(["feed_id", "user_id", "timestamp", "history_length",
                                                   "candidate_count", "user_embedding_is_zero",
                                                   "candidate_embedding_present"]) for b in batches])
    labels = truth["target"].to_numpy()
    counts = metadata["candidate_count"].to_numpy()
    offsets = np.r_[0, np.cumsum(counts)]
    values = []
    for directory in (control, treatment):
        predictions = pq.read_table(directory / "val_predictions.parquet")
        if not predictions.select(["feed_id", "item_id"]).equals(truth.select(["feed_id", "item_id"])):
            raise ValueError("Predictions do not match validation keys and order")
        scores = predictions["score"].to_numpy()
        measured = ranking_metrics(truth["feed_id"].to_numpy(), labels, scores)
        if abs(measured["ndcg"] - json.loads((directory / "result.json").read_text())["validation"]["ndcg"]) > 1e-10:
            raise ValueError("Saved metric does not match predictions")
        per_feed = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            y = labels[start:end]
            ranked = y[np.lexsort((y, -scores[start:end]))]
            discounts = 1 / np.log2(np.arange(len(y)) + 2)
            denominator = discounts[:int(y.sum())].sum()
            per_feed.append(float(np.dot(ranked, discounts) / denominator) if denominator else 1.)
        values.append(np.asarray(per_feed))
    base, candidate = values
    delta = candidate - base

    def summarize(mask):
        n = int(mask.sum())
        return {"feeds": n, "control_ndcg": float(base[mask].mean()) if n else None,
                "treatment_ndcg": float(candidate[mask].mean()) if n else None,
                "delta_ndcg": float(delta[mask].mean()) if n else None}

    users = metadata["user_id"].to_numpy()
    _, inverse = np.unique(users, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(42)
    sampled_deltas = []
    for _ in range(1000):
        selected = rng.integers(0, len(sums), size=len(sums))
        sampled_deltas.append(float(sums[selected].sum() / sizes[selected].sum()))
    timestamps = metadata["timestamp"].to_numpy()
    days = (timestamps.astype(np.int64) + 3 * 3600) // 86400
    lengths = metadata["history_length"].to_numpy()
    unique_times = np.unique(timestamps)
    boundary = int(unique_times[len(unique_times) // 2])
    slices = {"day_moscow": {}, "history_length": {}, "candidate_count": {}, "user_vector": {},
              "candidate_vectors": {}, "time_partition": {}}
    for day in np.unique(days):
        name = datetime.fromtimestamp(int(day) * 86400, timezone.utc).date().isoformat()
        slices["day_moscow"][name] = summarize(days == day)
    for name, lower, upper in (("empty", 0, 1), ("1-15", 1, 16), ("16-63", 16, 64),
                               ("64-255", 64, 256), ("256", 256, 257)):
        slices["history_length"][name] = summarize((lengths >= lower) & (lengths < upper))
    for name, lower, upper in (("5-30", 5, 31), ("31-100", 31, 101), ("101+", 101, 1000000)):
        slices["candidate_count"][name] = summarize((counts >= lower) & (counts < upper))
    zero_user = metadata["user_embedding_is_zero"].to_numpy()
    missing_item = np.array([not all(mask) for mask in metadata["candidate_embedding_present"].to_pylist()])
    for name, mask in (("zero", zero_user), ("nonzero", ~zero_user)):
        slices["user_vector"][name] = summarize(mask)
    for name, mask in (("some_missing", missing_item), ("all_present", ~missing_item)):
        slices["candidate_vectors"][name] = summarize(mask)
    for name, mask in (("early", timestamps < boundary), ("late", timestamps >= boundary)):
        slices["time_partition"][name] = summarize(mask)
    result = {"control_run": str(control), "treatment_run": str(treatment),
              "same_data_and_model_config": True, "exact_validation_keys": True,
              "overall": summarize(np.ones(len(delta), dtype=bool)),
              "improved_feeds": int((delta > 0).sum()), "degraded_feeds": int((delta < 0).sum()),
              "unchanged_feeds": int((delta == 0).sum()),
              "cluster_bootstrap": {"unit": "user_id", "repetitions": 1000, "seed": 42,
                                    "delta_95_percentile_interval": np.quantile(sampled_deltas, [.025, .975]).tolist(),
                                    "limitation": "Conditional on selected checkpoints; excludes training-seed and selection uncertainty"},
              "time_boundary_utc": datetime.fromtimestamp(boundary, timezone.utc).isoformat(),
              "time_boundary_moscow": datetime.fromtimestamp(boundary, timezone(timedelta(hours=3))).isoformat(),
              "slices": slices,
              "limitation": "Both checkpoints selected on full val. All slices are diagnostic, not independent holdout.",
              "script_sha256": sha256(__file__),
              "predictions_sha256": {str(p): sha256(p / "val_predictions.parquet") for p in (control, treatment)}}
    output = treatment / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/prepared-v1"))
    parser.add_argument("--control", type=Path, default=Path("runs/catboost/EXP-015-r01"))
    parser.add_argument("--treatment", type=Path, default=Path("runs/catboost/EXP-016-r01"))
    args = parser.parse_args()
    compare(args.data, args.control, args.treatment)
