"""Bounded CatBoost route: nearest-history features, then conditional train scaling.

Only train and val are accessed. Rejected trials never replace the active configuration.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import catboost
import pyarrow as pa
import pyarrow.parquet as pq

from baseline import candidate_features, ranking_metrics, sha256
from history_signals import add_history_centroid
from history_neighbors import NEIGHBOR_FEATURES, add_history_neighbors
from snowball import (ROOT, MIN_DELTA, build_cache, dump, features_for, fit_run,
                      journal, paired, per_feed_values, temporal_partition)
from recsys_data import PreparedSplit

SERIES = ROOT / "runs/catboost/route-023-024"
DATA = ROOT / "artifacts/prepared-v1"


def snapshot(directory):
    path = directory / "source.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("catboost", "recsys_data"):
            for source in sorted((ROOT / folder).glob("*.py")):
                archive.write(source, source.relative_to(ROOT))
        for source in sorted((ROOT / "catboost/configs").glob("*.json")):
            archive.write(source, source.relative_to(ROOT))
    return {"source_snapshot": str(path.relative_to(ROOT)), "source_snapshot_sha256": sha256(path)}


def cached_neighbors(base_cache, directory):
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    started = time.perf_counter()
    for split in ("train", "val"):
        tables = []
        for i, batch in enumerate(PreparedSplit(DATA, split).iter_batches(), 1):
            keys = batch.candidate_table()
            tables.append(add_history_neighbors(batch, keys).select(["feed_id", "item_id", *NEIGHBOR_FEATURES]))
            if i % 10 == 0:
                print(f"Neighbors {split}: {i} batches", flush=True)
        pq.write_table(pa.concat_tables(tables), directory / f"{split}.parquet", compression="zstd")
    manifest = {"prepared_manifest_sha256": sha256(DATA / "manifest.json"),
                "base_cache_manifest_sha256": sha256(base_cache / "manifest.json"),
                "feature_source_sha256": sha256(ROOT / "catboost/history_neighbors.py"),
                "seconds": time.perf_counter() - started,
                "files": {f"{s}.parquet": sha256(directory / f"{s}.parquet") for s in ("train", "val")}}
    dump(directory / "manifest.json", manifest)
    return manifest


def attach(base, extra):
    if not base.select(["feed_id", "item_id"]).equals(extra.select(["feed_id", "item_id"])):
        raise ValueError("Nearest-history cache keys differ from base cache")
    for name in NEIGHBOR_FEATURES:
        base = base.append_column(name, extra[name])
    return base


def finalize(report, directory, val, scores, row_mask):
    report["validation"] = ranking_metrics(val["feed_id"].to_numpy(), val["target"].to_numpy(), scores)
    late = val.filter(pa.array(~row_mask))
    report["control"] = ranking_metrics(late["feed_id"].to_numpy(), late["target"].to_numpy(), scores[~row_mask])
    report["evaluation_note"] = "Acceptance frozen on selection before late/full evaluation"
    dump(directory / "result.json", report)
    dump(ROOT / "catboost/results" / f"{report['run_id']}.json", report)


def decide(report, scores, val, metadata, feed_mask, row_mask, control_report, control_scores, title):
    counts = metadata["candidate_count"].to_numpy()
    selection = val.filter(pa.array(row_mask))
    new_values = per_feed_values(selection, scores[row_mask], counts[feed_mask])
    old_values = per_feed_values(selection, control_scores[row_mask], counts[feed_mask])
    comparison = paired(new_values-old_values, metadata["user_id"].to_numpy()[feed_mask])
    accepted = comparison["delta_ndcg"] >= MIN_DELTA
    report["decision"] = {"accepted": accepted, "minimum_delta": MIN_DELTA, "hypothesis": title,
                          "control_run": control_report["run_id"], "selection_comparison": comparison}
    directory = SERIES / report["run_id"]
    # Persist the decision before opening the later-part metric.
    dump(directory / "comparison.json", report["decision"])
    dump(directory / "result.json", report)
    finalize(report, directory, val, scores, row_mask)
    full_delta = report["validation"]["ndcg"]-control_report["validation"]["ndcg"]
    late_delta = report["control"]["ndcg"]-control_report["control"]["ndcg"]
    journal(f"### {report['run_id']} — результат\n\n"
        f"- Контроль: {control_report['run_id']}; selection {report['selection']['ndcg']:.10f}; "
        f"Δ {comparison['delta_ndcg']:+.10f}; диагностический 95% bootstrap {comparison['user_cluster_bootstrap_95']}.\n"
        f"- Решение до оценки поздней/полной val: **{'оставить' if accepted else 'отклонить'}**, порог +{MIN_DELTA}.\n"
        f"- Полная val {report['validation']['ndcg']:.10f}, Δ {full_delta:+.10f}; "
        f"поздний control {report['control']['ndcg']:.10f}, Δ {late_delta:+.10f}.\n"
        f"- Fit {report['train_seconds']:.2f} с; {report['trees']} деревьев; "
        f"пиковый RSS процесса {report['process_peak_rss_bytes']/2**30:.2f} GiB (накопительный).")
    print(json.dumps({"run_id": report["run_id"], "decision": report["decision"],
                      "validation": report["validation"], "control": report["control"]}, ensure_ascii=False), flush=True)
    return accepted


def scaled_train(data, feature_config, output):
    """Use the verified original train matrix without modifying the train16 raw copy."""
    dataset = PreparedSplit(data, "train")
    source = PreparedSplit(DATA, "train")
    if dataset.manifest["archive_etag"] != source.manifest["archive_etag"]:
        raise ValueError("Scaled train comes from another archive")
    if dataset.embeddings.meta != source.embeddings.meta:
        raise ValueError("Scaled train embedding metadata differs")
    dataset.embeddings = source.embeddings
    if dataset.info["feeds"] != 98428 or len(dataset.info["files"]) != 16:
        raise ValueError("This bounded experiment requires the existing 16-shard dataset")
    if dataset.manifest["splits"]["val"] != source.manifest["splits"]["val"]:
        raise ValueError("Scaled dataset val differs from the common val")
    output.mkdir(parents=True)
    started = time.perf_counter()
    writer = None
    rows = 0
    try:
        for i, batch in enumerate(dataset.iter_batches(include_labels=True), 1):
            table = candidate_features(batch, [16, 64, 256], True, True, include_target=True)
            table = add_history_centroid(batch, table)
            if feature_config.get("neighbors"):
                table = add_history_neighbors(batch, table)
            table = table.select(["feed_id", "item_id", *features_for(feature_config), "target"])
            if writer is None:
                writer = pq.ParquetWriter(output / "train.parquet", table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(table)
            if i % 10 == 0:
                print(f"Scaled train: {rows:,} candidates", flush=True)
    finally:
        if writer is not None:
            writer.close()
    result = {"data": str(data.relative_to(ROOT)), "manifest_sha256": sha256(data / "manifest.json"),
              "embedding_override": {"path": str(source.embeddings.path), "metadata": source.embeddings.meta,
                  "bytes": source.embeddings.path.stat().st_size, "archive_etag": source.manifest["archive_etag"]},
              "rows": rows, "feeds": dataset.info["feeds"], "seconds": time.perf_counter()-started,
              "train_cache_sha256": sha256(output / "train.parquet")}
    dump(output / "manifest.json", result)
    return result


def main():
    occupied = [SERIES, ROOT / "catboost/results/EXP-CB-023-r01.json",
                ROOT / "catboost/results/EXP-CB-024-r01.json",
                ROOT / "catboost/results/route-023-024.json", ROOT / "catboost/configs/route_best.json"]
    if any(path.exists() for path in occupied):
        raise FileExistsError("Route already exists; reserve new IDs")
    SERIES.mkdir(parents=True)
    code = snapshot(SERIES)
    base_cache = ROOT / "artifacts/catboost-snowball-v1"
    build_cache(DATA, base_cache)  # Validate identity and checksums of the existing cache.
    extra_cache = ROOT / "artifacts/catboost-neighbors64-v1"
    preparation = cached_neighbors(base_cache, extra_cache)
    journal(f"- EXP-CB-023: подготовка nearest-history кэша {preparation['seconds']:.2f} с; "
            f"`{extra_cache.relative_to(ROOT)}`. Снимок кода `{code['source_snapshot']}`, SHA-256 `{code['source_snapshot_sha256']}`.")
    train, val = [attach(pq.read_table(base_cache / f"{split}.parquet"),
                         pq.read_table(extra_cache / f"{split}.parquet")) for split in ("train", "val")]
    metadata = pq.read_table(base_cache / "val_metadata.parquet")
    feed_mask, partition = temporal_partition(metadata)
    original_partition = json.loads((ROOT / "runs/catboost/snowball-020-022/partition.json").read_text())
    if partition != original_partition:
        raise ValueError("Validation partition changed")
    row_mask = np.repeat(feed_mask, metadata["candidate_count"].to_numpy())
    config = json.loads((ROOT / "catboost/configs/snowball_best.json").read_text())
    control_dir = ROOT / "runs/catboost/snowball-020-022" / config["run_id"]
    control_report = json.loads((control_dir / "result.json").read_text())
    control_predictions = pq.read_table(control_dir / "val_predictions.parquet")
    if not control_predictions.select(["feed_id", "item_id"]).equals(val.select(["feed_id", "item_id"])):
        raise ValueError("Control predictions have different validation keys")
    if sha256(control_dir / "val_predictions.parquet") != control_report["artifacts"]["val_predictions.parquet"]:
        raise ValueError("Control prediction checksum changed")
    control_scores = control_predictions["score"].to_numpy()
    feature_config = {"centroid": True, "event_types": False, "neighbors": True}
    provenance = {**code, "partition": partition, "prepared_manifest_sha256": sha256(DATA / "manifest.json"),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "platform": platform.platform(),
                  "versions": {"catboost": catboost.__version__, "numpy": np.__version__, "pyarrow": pa.__version__},
                  "base_cache_manifest_sha256": sha256(base_cache / "manifest.json"),
                  "neighbor_cache_manifest_sha256": sha256(extra_cache / "manifest.json"),
                  "train": {"feeds": 24592, "rows": len(train), "files": 4}}
    report, scores = fit_run("EXP-CB-023-r01", config["model_config"], feature_config, train, val,
                            row_mask, SERIES / "EXP-CB-023-r01", provenance)
    accepted = decide(report, scores, val, metadata, feed_mask, row_mask, control_report, control_scores,
                      "Максимум и среднее top-3 cosine к последним 64 событиям")
    reports = [report]
    best = {"run_id": control_report["run_id"], "run_directory": str(control_dir.relative_to(ROOT)),
            "model_config": config["model_config"], "feature_config": config["feature_config"]}
    if accepted:
        best.update(run_id=report["run_id"], run_directory=str((SERIES / report["run_id"]).relative_to(ROOT)),
                    feature_config=report["feature_config"])
        dump(SERIES / "best_before_scaling.json", best)
        del train
        large_data = ROOT / "artifacts/prepared-train16-v1"
        large_cache = ROOT / "artifacts/catboost-neighbors64-train16-v1"
        preparation = scaled_train(large_data, feature_config, large_cache)
        journal(f"- EXP-CB-024: подготовка train16 {preparation['seconds']:.2f} с, "
                f"{preparation['feeds']} лент / {preparation['rows']} кандидатов. "
                "Таблицы неизменны; исходная train-матрица переиспользована через проверенный EmbeddingStore без копирования/замены файлов.")
        large_train = pq.read_table(large_cache / "train.parquet")
        provenance.update(prepared_manifest_sha256=sha256(large_data / "manifest.json"),
                          train={"feeds": 98428, "rows": len(large_train), "files": 16},
                          scaled_train_cache_manifest_sha256=sha256(large_cache / "manifest.json"))
        large_report, large_scores = fit_run("EXP-CB-024-r01", config["model_config"], feature_config,
            large_train, val, row_mask, SERIES / "EXP-CB-024-r01", provenance)
        scaled_accepted = decide(large_report, large_scores, val, metadata, feed_mask, row_mask,
                                  report, scores, "Увеличение train с 4 до 16 файлов на принятых признаках")
        reports.append(large_report)
        if scaled_accepted:
            best.update(run_id=large_report["run_id"],
                        run_directory=str((SERIES / large_report["run_id"]).relative_to(ROOT)))
    summary = {"status": "complete", "next": "catboost_scaled" if accepted else "learned_history",
               "best": best, "runs": reports, "test_read": False, "submission_created": False}
    dump(SERIES / "summary.json", summary)
    dump(ROOT / "catboost/results/route-023-024.json", summary)
    dump(ROOT / "catboost/configs/route_best.json", best)
    journal(f"### Итог CatBoost-маршрута\n\n- Принятый вариант: **{best['run_id']}**. "
            f"Следующий этап: {summary['next']}. Конфиг `catboost/configs/route_best.json`. "
            "Test/submission/push не выполнялись.")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    existed = SERIES.exists()
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        if not existed and SERIES.exists():
            (SERIES / "error.txt").write_text(error)
        journal("### Ошибка CatBoost-маршрута\n\n```text\n" + error + "```\nИзмеренные предыдущие результаты сохранены.")
        raise
