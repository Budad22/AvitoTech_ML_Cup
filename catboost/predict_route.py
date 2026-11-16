"""Inference-only submission for saved history-aware checkpoints, on owner request."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

from catboost import CatBoostRanker, Pool
import numpy as np
import pyarrow.parquet as pq

from baseline import (KEY_SCHEMA, candidate_features, configured_features, frame,
                      prediction_table, sha256, validate_submission)
from history_signals import (CENTROID_FEATURES, EVENT_FEATURES,
                             add_history_centroid, add_history_event_types)
from history_neighbors import NEIGHBOR_FEATURES, add_history_neighbors
from recsys_data import PreparedSplit

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SOURCES = ["catboost/baseline.py", "catboost/history_categories.py", "catboost/cosine_features.py",
                   "catboost/feed_context.py", "catboost/history_signals.py",
                   "catboost/history_neighbors.py", "recsys_data/core.py", "recsys_data/__init__.py"]


def compatible_manifests(training, inference):
    """Training may use more train shards; val/test must be exactly the same prepared splits."""
    if training["format_version"] != inference["format_version"]:
        raise ValueError("Prepared format version changed")
    if not training.get("archive_etag") or training["archive_etag"] != inference.get("archive_etag"):
        raise ValueError("Data archive identity changed")
    for split in ("val", "test"):
        if training["splits"][split] != inference["splits"][split]:
            raise ValueError(f"Prepared {split} differs from the training dataset")


def feature_names(config):
    allowed = {"history_category_windows", "user_item_cosine", "feed_percentiles", "centroid", "event_types", "neighbors"}
    if set(config) - allowed:
        raise ValueError("Unsupported feature configuration")
    for name in allowed - {"history_category_windows"}:
        if type(config.get(name, False)) is not bool:
            raise ValueError(f"{name} must be boolean")
    windows = config.get("history_category_windows", [])
    if (not isinstance(windows, list) or len(set(windows)) != len(windows)
            or any(type(w) is not int or not 1 <= w <= 256 for w in windows)):
        raise ValueError("Invalid category-history windows")
    return (configured_features(windows, config.get("user_item_cosine", False), config.get("feed_percentiles", False))
            + (CENTROID_FEATURES if config.get("centroid") else [])
            + (EVENT_FEATURES if config.get("event_types") else [])
            + (NEIGHBOR_FEATURES if config.get("neighbors") else []))


def transform(batch, config):
    table = candidate_features(batch, config.get("history_category_windows", []),
                               config.get("user_item_cosine", False), config.get("feed_percentiles", False))
    if config.get("centroid"):
        table = add_history_centroid(batch, table)
    if config.get("event_types"):
        table = add_history_event_types(batch, table)
    if config.get("neighbors"):
        table = add_history_neighbors(batch, table)
    return table


def predict(data, training_data, run, output=None):
    data, training_data, run = [Path(p).resolve() for p in (data, training_data, run)]
    output = Path(output).resolve() if output else run / "submission.parquet"
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Choose a new filename; submissions and sidecars are never overwritten")
    training = json.loads((run / "result.json").read_text())
    if training["status"] != "complete":
        raise ValueError("Training run is incomplete")
    if sha256(training_data / "manifest.json") != training["prepared_manifest_sha256"]:
        raise ValueError("Training manifest checksum differs from the report")
    training_manifest = json.loads((training_data / "manifest.json").read_text())
    inference_manifest = json.loads((data / "manifest.json").read_text())
    compatible_manifests(training_manifest, inference_manifest)
    config = json.loads((run / "feature_config.json").read_text())
    if config != training["feature_config"]:
        raise ValueError("Saved feature config differs from the training report")
    features = feature_names(config)
    if features != training["features"]:
        raise ValueError("Feature order differs from training")
    snapshot = ROOT / training["source_snapshot"]
    if sha256(snapshot) != training["source_snapshot_sha256"]:
        raise ValueError("Training code snapshot checksum differs")
    with zipfile.ZipFile(snapshot) as archive:
        for source in FEATURE_SOURCES:
            expected = hashlib.sha256(archive.read(source)).hexdigest()
            if sha256(ROOT / source) != expected:
                raise ValueError(f"Feature source differs from training snapshot: {source}")
    model_path = run / "model.cbm"
    model_hash = sha256(model_path)
    if model_hash != training["artifacts"]["model.cbm"]:
        raise ValueError("Model checkpoint checksum differs")
    model = CatBoostRanker().load_model(str(model_path))
    if model.feature_names_ != features:
        raise ValueError("Checkpoint feature order differs")
    if sha256(run / "val_predictions.parquet") != training["artifacts"]["val_predictions.parquet"]:
        raise ValueError("Saved val prediction checksum differs")
    # Verify this exact inference path against saved scores before touching test.
    batch = next(PreparedSplit(data, "val").iter_batches(batch_size=137))
    table = transform(batch, config)
    actual = np.asarray(model.predict(Pool(frame(table, features), cat_features=["category_id"]), thread_count=8), dtype=np.float32)
    expected = pq.read_table(run / "val_predictions.parquet").slice(0, len(table))
    if not table.select(["feed_id", "item_id"]).equals(expected.select(["feed_id", "item_id"])):
        raise ValueError("Val probe keys differ from training")
    if not np.array_equal(actual, expected["score"].to_numpy()):
        raise ValueError("Inference path does not exactly reproduce saved val scores")
    print(f"Val probe passed: {len(batch.features)} feeds / {len(table):,} candidates, exact float32 scores", flush=True)
    probe = {"feeds": len(batch.features), "candidates": len(table), "exact_saved_float32_scores": True}
    dataset = PreparedSplit(data, "test")
    if not dataset.embeddings.available:
        raise FileNotFoundError(dataset.embeddings.path)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=output.parent) as temporary:
        staged = Path(temporary) / "submission.parquet"
        with pq.ParquetWriter(staged, KEY_SCHEMA, compression="zstd") as writer:
            for i, batch in enumerate(dataset.iter_batches(), 1):
                table = transform(batch, config)
                pool = Pool(frame(table, features), cat_features=["category_id"])
                writer.write_table(prediction_table(table, model.predict(pool, thread_count=8)))
                rows += len(table)
                if i % 10 == 0:
                    print(f"Predicted {rows:,}/{dataset.info['candidates']:,} test candidates", flush=True)
        source = (data / inference_manifest["source_relative_path"]).resolve()
        validation = validate_submission(staged, source / "sample_submission.parquet")
        if sha256(model_path) != model_hash:
            raise RuntimeError("Checkpoint changed during inference")
        report = {"status": "complete", "training_run": str(run), "run_id": training["run_id"],
                  "model_sha256": model_hash, "training_manifest_sha256": sha256(training_data / "manifest.json"),
                  "inference_manifest_sha256": sha256(data / "manifest.json"),
                  "same_val_test_manifest_entries": True, "source_snapshot_sha256": sha256(snapshot),
                  "feature_source_sha256": {name: sha256(ROOT / name) for name in FEATURE_SOURCES},
                  "features": features, "feature_config": config, "val_probe": probe,
                  "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "predict_and_validate_seconds": time.perf_counter()-started, "submission": validation,
                  "output": str(output), "script_sha256": sha256(__file__),
                  "retrained": False, "test_ndcg": None, "uploaded": False,
                  "test_embeddings": {"path": str(dataset.embeddings.path), "metadata": dataset.embeddings.meta,
                                      "bytes": dataset.embeddings.path.stat().st_size}}
        if output.exists() or sidecar.exists():
            raise FileExistsError("Output appeared during inference; refusing to overwrite")
        staged.rename(output)
        sidecar.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/prepared-v1")
    parser.add_argument("--training-data", type=Path, default=ROOT / "artifacts/prepared-train16-v1")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/catboost/route-023-024/EXP-CB-024-r01")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predict(args.data, args.training_data, args.run, args.output)
