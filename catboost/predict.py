"""Create a submission from a saved checkpoint, without training or changing training reports.

Run this only after an explicit request to create a submission.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

from catboost import CatBoostRanker, Pool
import pyarrow.parquet as pq

from baseline import (KEY_SCHEMA, candidate_features, configured_features, frame,
                      prediction_table, sha256, validate_submission)
from recsys_data import PreparedSplit


def predict(data, run, output=None):
    data, run = Path(data).resolve(), Path(run).resolve()
    output = Path(output).resolve() if output else run / "submission.parquet"
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Choose a new submission filename; existing files are never overwritten")
    training = json.loads((run / "result.json").read_text())
    if training["status"] != "complete":
        raise ValueError("Training run is not complete")
    manifest_hash = sha256(data / "manifest.json")
    if manifest_hash != training["prepared_manifest_sha256"]:
        raise ValueError("Prepared dataset differs from the training run")
    model_path = run / "model.cbm"
    model_hash = sha256(model_path)
    if model_hash != training["artifacts"]["model.cbm"]:
        raise ValueError("Checkpoint hash differs from the training report")
    feature_path = run / "feature_config.json"
    config = json.loads(feature_path.read_text()) if feature_path.exists() else {}
    windows = config.get("history_category_windows", [])
    cosine = config.get("user_item_cosine", False)
    context = config.get("feed_percentiles", False)
    features = configured_features(windows, cosine, context)
    if features != training["features"]:
        raise ValueError("Feature configuration differs from training")
    sources = {"baseline.py": "script_sha256", "history_categories.py": "feature_script_sha256",
               "cosine_features.py": "cosine_script_sha256", "feed_context.py": "context_script_sha256"}
    for name, field in sources.items():
        if field in training and sha256(Path(__file__).with_name(name)) != training[field]:
            raise ValueError(f"Feature code changed since training: {name}")
    dataset = PreparedSplit(data, "test")
    if cosine and not dataset.embeddings.available:
        raise FileNotFoundError(f"Test cosine requires {dataset.embeddings.path}")
    model = CatBoostRanker().load_model(str(model_path))
    if model.feature_names_ != features:
        raise ValueError("Checkpoint feature order differs from the saved configuration")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=output.parent) as temporary:
        staged = Path(temporary) / "submission.parquet"
        with pq.ParquetWriter(staged, KEY_SCHEMA, compression="zstd") as writer:
            for i, batch in enumerate(dataset.iter_batches(), 1):
                table = candidate_features(batch, windows, cosine, context)
                pool = Pool(frame(table, features), cat_features=["category_id"])
                writer.write_table(prediction_table(table, model.predict(pool, thread_count=8)))
                rows += len(table)
                if i % 10 == 0:
                    print(f"Predicted {rows:,}/{dataset.info['candidates']:,} test candidates", flush=True)
        manifest = json.loads((data / "manifest.json").read_text())
        source = (data / manifest["source_relative_path"]).resolve()
        validation = validate_submission(staged, source / "sample_submission.parquet")
        if sha256(model_path) != model_hash:
            raise RuntimeError("Checkpoint changed during prediction")
        report = {"status": "complete", "training_run": str(run), "model_sha256": model_hash,
                  "prepared_manifest_sha256": manifest_hash, "features": features, "feature_config": config,
                  "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                  "predict_and_validate_seconds": time.perf_counter() - started,
                  "submission": validation, "output": str(output),
                  "script_sha256": sha256(__file__), "retrained": False, "test_ndcg": None,
                  "uploaded": False}
        if cosine:
            report["test_embeddings"] = {"path": str(dataset.embeddings.path),
                                         "bytes": dataset.embeddings.path.stat().st_size,
                                         "metadata": dataset.embeddings.meta}
        if output.exists() or report_path.exists():
            raise FileExistsError("Submission output appeared during prediction; refusing to overwrite")
        staged.rename(output)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/prepared-v1"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predict(args.data, args.run, args.output)
