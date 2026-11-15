"""Minimal CPU CatBoostRanker: f1-f5 and category, complete feed groups."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import resource

import catboost
from catboost import CatBoostRanker, Pool
from catboost.utils import eval_metric
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys_data import PreparedSplit
from history_categories import add_history_categories, feature_names

FEATURES = ["f1", "f2", "f3", "f4", "f5", "category_id"]
KEY_SCHEMA = pa.schema([("feed_id", pa.int32()), ("item_id", pa.int32()), ("score", pa.float32())])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame(table, features=FEATURES):
    data = table.select(features).to_pandas()
    data["category_id"] = data["category_id"].astype(np.int32)
    return data


def labeled_data(root, split, windows=()):
    dataset = PreparedSplit(root, split)
    features = FEATURES + feature_names(windows)
    tables = [add_history_categories(batch, batch.candidate_table(include_target=True), windows)
              .select(["feed_id", "item_id", *features, "target"])
              for batch in dataset.iter_batches(include_labels=True)]
    table = pa.concat_tables(tables)
    pool = Pool(frame(table, features), label=table["target"].to_numpy(),
                cat_features=["category_id"], group_id=table["feed_id"].to_numpy())
    return table, pool


def ranking_metrics(feed_ids, targets, scores, *, verify=True):
    """Full, @10 and @30 NDCG, equal feed weights and pessimistic ties."""
    groups, targets, scores = np.asarray(feed_ids), np.asarray(targets), np.asarray(scores, dtype=np.float64)
    if not (len(groups) == len(targets) == len(scores)) or not len(groups):
        raise ValueError("Metric inputs must be nonempty and equally sized")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Metric requires finite scores and binary targets")
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    group_starts = groups[boundaries[:-1]]
    if len(np.unique(group_starts)) != len(group_starts):
        raise ValueError("Metric feed groups must be contiguous")
    values = np.empty((len(boundaries) - 1, 3))
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels = targets[start:end]
        ranked = labels[np.lexsort((labels, -scores[start:end]))]
        discount = 1 / np.log2(np.arange(end - start) + 2)
        positives = int(labels.sum())
        for j, cutoff in enumerate((None, 10, 30)):
            limit = len(labels) if cutoff is None else min(cutoff, len(labels))
            denominator = discount[:min(positives, limit)].sum()
            values[i, j] = float(np.dot(ranked[:limit], discount[:limit]) / denominator) if denominator else 1.
    result = dict(zip(("ndcg", "ndcg_at_10", "ndcg_at_30"), map(float, values.mean(axis=0))))
    if verify:
        native = eval_metric(targets.tolist(), scores.tolist(), "NDCG:type=Base",
                             group_id=groups.tolist(), thread_count=8)[0]
        if abs(native - result["ndcg"]) > 1e-10:
            raise RuntimeError(f"NDCG disagrees with CatBoost: {native} vs {result['ndcg']}")
        result["independent_catboost_check"] = True
    result["feeds"] = len(values)
    result["rows"] = len(groups)
    return result


def prediction_table(keys, scores):
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Predictions must be finite after float32 conversion")
    return pa.Table.from_arrays([keys["feed_id"].combine_chunks(), keys["item_id"].combine_chunks(), pa.array(scores)], schema=KEY_SCHEMA)


def validate_submission(path, template):
    predicted, expected = pq.read_table(path), pq.read_table(template)
    if predicted.schema != KEY_SCHEMA or len(predicted) != len(expected):
        raise ValueError("Submission schema or row count does not match template")
    if any(predicted[name].null_count for name in predicted.column_names):
        raise ValueError("Submission contains nulls")
    if not np.isfinite(predicted["score"].to_numpy()).all():
        raise ValueError("Submission scores must be finite")
    def keys(table):
        return (table["feed_id"].to_numpy().astype(np.int64) << 32) | table["item_id"].to_numpy().astype(np.int64)
    actual, required = keys(predicted), keys(expected)
    if len(np.unique(actual)) != len(actual) or not np.array_equal(np.sort(actual), np.sort(required)):
        raise ValueError("Submission keys do not exactly match the template")
    return {"rows": len(predicted), "exact_template_coverage": True, "unique_keys": True,
            "finite_float32_scores": True, "sha256": sha256(path), "bytes": Path(path).stat().st_size}


def run(data, output, config_path, feature_config_path=None, *, make_submission=False):
    started = time.perf_counter()
    feature_config = json.loads(feature_config_path.read_text()) if feature_config_path else {}
    if set(feature_config) - {"history_category_windows"}:
        raise ValueError("Unknown feature configuration key")
    windows = feature_config.get("history_category_windows", [])
    if (not isinstance(windows, list) or any(type(w) is not int or w < 1 or w > 256 for w in windows)
            or len(set(windows)) != len(windows)):
        raise ValueError("History windows must be distinct integers in [1, 256]")
    features = FEATURES + feature_names(windows)
    if output.exists():
        raise FileExistsError(f"Choose a new run directory: {output}")
    output.mkdir(parents=True)
    config = json.loads(config_path.read_text())
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "feature_config.json").write_text(json.dumps(feature_config, indent=2))
    provenance = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                  "features": features, "feature_config": feature_config,
                  "feature_script_sha256": sha256(Path(__file__).with_name("history_categories.py")),
                  "categorical_features": ["category_id"], "config": config,
                  "script_sha256": sha256(__file__), "prepared_manifest_sha256": sha256(data / "manifest.json"),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "versions": {"catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
                  "platform": platform.platform()}
    (output / "result.json").write_text(json.dumps(provenance, indent=2))
    print("Loading train and complete val", flush=True)
    train_table, train_pool = labeled_data(data, "train", windows)
    val_table, val_pool = labeled_data(data, "val", windows)
    provenance["load_seconds"] = time.perf_counter() - started
    provenance["train"] = {"feeds": len(np.unique(train_table["feed_id"].to_numpy())), "rows": len(train_table),
                           "complete": PreparedSplit(data, "train").info["complete_impressions"]}
    print(json.dumps(provenance["train"]), flush=True)
    model = CatBoostRanker(**config)
    before_fit = time.perf_counter()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=20)
    provenance["train_seconds"] = time.perf_counter() - before_fit
    model.save_model(str(output / "model.cbm"))
    provenance["best_iteration"] = model.get_best_iteration()
    provenance["trees"] = model.tree_count_
    provenance["training_eval_history"] = model.get_evals_result()
    val_predictions = prediction_table(val_table, model.predict(val_pool, thread_count=8))
    pq.write_table(val_predictions, output / "val_predictions.parquet", compression="zstd")
    provenance["validation"] = ranking_metrics(val_table["feed_id"].to_numpy(), val_table["target"].to_numpy(), val_predictions["score"].to_numpy())
    provenance["f1_control"] = ranking_metrics(val_table["feed_id"].to_numpy(), val_table["target"].to_numpy(), val_table["f1"].to_numpy())
    print("Validation: " + json.dumps(provenance["validation"]), flush=True)
    provenance["delta_ndcg_vs_f1"] = provenance["validation"]["ndcg"] - provenance["f1_control"]["ndcg"]
    del train_pool, val_pool, train_table, val_table
    provenance["submission_requested"] = make_submission
    artifact_names = ["model.cbm", "val_predictions.parquet"]
    if make_submission:
        before_predict = time.perf_counter()
        with pq.ParquetWriter(output / "submission.parquet", KEY_SCHEMA, compression="zstd") as writer:
            for batch in PreparedSplit(data, "test").iter_batches():
                table = add_history_categories(batch, batch.candidate_table(), windows)
                pool = Pool(frame(table, features), cat_features=["category_id"])
                writer.write_table(prediction_table(table, model.predict(pool, thread_count=8)))
        provenance["test_predict_seconds"] = time.perf_counter() - before_predict
        manifest = json.loads((data / "manifest.json").read_text())
        source = (data / manifest["source_relative_path"]).resolve()
        provenance["submission"] = validate_submission(output / "submission.parquet", source / "sample_submission.parquet")
        artifact_names.append("submission.parquet")
    provenance["status"] = "complete"
    provenance["total_seconds"] = time.perf_counter() - started
    # macOS reports bytes, Linux reports KiB; includes loading, fitting and submission checks.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    provenance["peak_rss_bytes"] = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["artifacts"] = {name: sha256(output / name) for name in artifact_names}
    (output / "result.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"validation": provenance["validation"], "submission": str(output / "submission.parquet") if make_submission else None, "seconds": provenance["total_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/prepared-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/catboost/EXP-015-r01"))
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/baseline.json")
    parser.add_argument("--feature-config", type=Path, help="Optional feature configuration; absent keeps the baseline inputs")
    parser.add_argument("--make-submission", action="store_true", help="Generate test predictions only when explicitly requested by the owner")
    args = parser.parse_args()
    run(args.data.resolve(), args.output.resolve(), args.config.resolve(),
        args.feature_config.resolve() if args.feature_config else None, make_submission=args.make_submission)


if __name__ == "__main__":
    main()
