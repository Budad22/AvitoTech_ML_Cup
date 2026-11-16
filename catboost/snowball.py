"""Three sequential train/val experiments; test is never opened.

Run from repository root with .venv/bin/python -u catboost/snowball.py.
Rejected changes remain reproducible in their run folders but do not enter best.json.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
import zipfile

import catboost
from catboost import CatBoostRanker, Pool
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from baseline import (candidate_features, configured_features, frame, prediction_table,
                      ranking_metrics, sha256)
from history_signals import (CENTROID_FEATURES, EVENT_FEATURES,
                             add_history_centroid, add_history_event_types)
from recsys_data import PreparedSplit
from history_neighbors import NEIGHBOR_FEATURES
from extended_features import LONG_FEATURES, PROJECTED_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MIN_DELTA = 0.001
BASE_FEATURE_CONFIG = {"history_category_windows": [16, 64, 256],
                       "user_item_cosine": True, "feed_percentiles": True}
STEPS = [("EXP-CB-020", "centroid", "Центроид последних 64 событий"),
         ("EXP-CB-021", "ndcg_loss", "YetiRank mode=NDCG с log-дисконтом"),
         ("EXP-CB-022", "event_types", "Категория кандидата по типам событий последних 64 событий")]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def journal(message):
    with (ROOT / "catboost_exp.md").open("a") as stream:
        stream.write("\n" + message + "\n")


def features_for(config):
    return configured_features([16, 64, 256], True, True) + (
        CENTROID_FEATURES if config["centroid"] else []) + (
        EVENT_FEATURES if config["event_types"] else []) + (
        NEIGHBOR_FEATURES if config.get("neighbors", False) else []) + (
        LONG_FEATURES if config.get("long_history", False) else []) + (
        PROJECTED_FEATURES if config.get("projected_products", False) else [])


def temporal_partition(metadata):
    timestamps = metadata["timestamp"].to_numpy()
    unique, counts = np.unique(timestamps, return_counts=True)
    cumulative = np.cumsum(counts)[:-1]
    boundary = int(unique[1:][np.argmin(abs(cumulative - .8 * len(timestamps)))])
    mask = timestamps < boundary
    return mask, {"boundary_timestamp": boundary,
                  "boundary_utc": datetime.fromtimestamp(boundary, timezone.utc).isoformat(),
                  "selection_feeds": int(mask.sum()), "control_feeds": int((~mask).sum()),
                  "rule": "nearest 80% of feeds by timestamp; equal timestamps stay together",
                  "limitation": "Full val used by historical experiments; control is not virgin holdout"}


def build_cache(data, directory):
    """One assembly pass per split; columns of rejected hypotheses are never put in Pool."""
    sources = [ROOT / "catboost" / name for name in
               ["baseline.py", "history_signals.py", "history_categories.py", "cosine_features.py", "feed_context.py"]]
    sources.append(ROOT / "recsys_data/core.py")
    identity = {"prepared_manifest_sha256": sha256(data / "manifest.json"),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "base_feature_config": BASE_FEATURE_CONFIG, "embedding_files": {}}
    for split in ("train", "val"):
        store = PreparedSplit(data, split).embeddings
        if not store.available:
            raise FileNotFoundError(store.path)
        stat = store.path.stat()
        identity["embedding_files"][split] = {"path": str(store.path), "bytes": stat.st_size,
                                               "mtime_ns": stat.st_mtime_ns, "metadata": store.meta}
    if directory.exists():
        saved = json.loads((directory / "manifest.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Feature cache identity changed; use a new cache directory")
        for name, expected in saved["files"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Feature cache checksum differs: {name}")
        return saved
    directory.mkdir(parents=True)
    started = time.perf_counter()
    for split in ("train", "val"):
        rows = 0
        metadata = []
        path = directory / f"{split}.parquet"
        writer = None
        try:
            for i, batch in enumerate(PreparedSplit(data, split).iter_batches(include_labels=True), 1):
                table = candidate_features(batch, [16, 64, 256], True, True, include_target=True)
                table = add_history_centroid(batch, table)
                table = add_history_event_types(batch, table)
                all_features = features_for({"centroid": True, "event_types": True})
                table = table.select(["feed_id", "item_id", *all_features, "target"])
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                rows += len(table)
                if split == "val":
                    metadata.append(batch.features.select(["feed_id", "user_id", "timestamp",
                        "history_length", "candidate_count", "user_embedding_is_zero"]))
                if i % 10 == 0:
                    print(f"Cache {split}: {rows:,} candidates", flush=True)
        finally:
            if writer is not None:
                writer.close()
        if split == "val":
            pq.write_table(pa.concat_tables(metadata), directory / "val_metadata.parquet")
    names = ["train.parquet", "val.parquet", "val_metadata.parquet"]
    result = {"identity": identity, "seconds": time.perf_counter() - started,
              "files": {name: sha256(directory / name) for name in names}}
    dump(directory / "manifest.json", result)
    return result


def per_feed_values(table, scores, counts):
    offsets = np.r_[0, np.cumsum(counts)]
    labels = table["target"].to_numpy()
    values = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        y = labels[start:end]
        ranked = y[np.lexsort((y, -scores[start:end]))]
        discounts = 1 / np.log2(np.arange(len(y)) + 2)
        ideal = discounts[:int(y.sum())].sum()
        values.append(float(np.dot(ranked, discounts) / ideal) if ideal else 1.)
    return np.asarray(values)


def paired(delta, users):
    _, inverse = np.unique(users, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(1000):
        selected = rng.integers(0, len(sums), len(sums))
        draws.append(sums[selected].sum() / sizes[selected].sum())
    return {"delta_ndcg": float(delta.mean()), "feeds": len(delta),
            "user_cluster_bootstrap_95": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_repetitions": 1000,
            "limitation": "Diagnostic; checkpoint selection and training-seed uncertainty excluded"}


def fit_run(run_id, config, feature_config, train, val, selection_rows, output, provenance):
    started = time.perf_counter()
    output.mkdir()
    features = features_for(feature_config)
    report = {**provenance, "run_id": run_id, "status": "running", "config": config,
              "feature_config": {**BASE_FEATURE_CONFIG, **feature_config}, "features": features,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "submission_requested": False, "test_read": False}
    dump(output / "config.json", config)
    dump(output / "feature_config.json", report["feature_config"])
    dump(output / "result.json", report)
    journal(f"### {run_id} — запуск\n\n- Статус: в работе; старт {report['started_at_utc']}.\n"
            f"- Конфиг: `{output.relative_to(ROOT)}/config.json`; признаки: `{output.relative_to(ROOT)}/feature_config.json`.\n"
            f"- Снимок кода SHA-256: `{provenance['source_snapshot_sha256']}`. Test не читается.")
    train_pool = Pool(frame(train, features), label=train["target"].to_numpy(),
                      cat_features=["category_id"], group_id=train["feed_id"].to_numpy())
    selection = val.filter(pa.array(selection_rows))
    selection_pool = Pool(frame(selection, features), label=selection["target"].to_numpy(),
                          cat_features=["category_id"], group_id=selection["feed_id"].to_numpy())
    report["pool_seconds"] = time.perf_counter() - started
    model = CatBoostRanker(**config)
    before_fit = time.perf_counter()
    model.fit(train_pool, eval_set=selection_pool, use_best_model=True, verbose=20)
    report["train_seconds"] = time.perf_counter() - before_fit
    model.save_model(str(output / "model.cbm"))
    scores = np.asarray(model.predict(frame(val, features), thread_count=8), dtype=np.float32)
    pq.write_table(prediction_table(val, scores), output / "val_predictions.parquet", compression="zstd")
    report["selection"] = ranking_metrics(selection["feed_id"].to_numpy(),
        selection["target"].to_numpy(), scores[selection_rows])
    report["best_iteration"] = model.get_best_iteration()
    report["trees"] = model.tree_count_
    report["training_eval_history"] = model.get_evals_result()
    report["status"] = "complete"
    report["total_seconds"] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["artifacts"] = {name: sha256(output / name) for name in ["model.cbm", "val_predictions.parquet"]}
    dump(output / "result.json", report)
    print(f"Completed {run_id}: selection={report['selection']['ndcg']:.10f}", flush=True)
    return report, scores


def run(data, series, cache):
    if series.exists():
        raise FileExistsError(series)
    report_paths = [ROOT / "catboost/results" / f"{exp_id}-r01.json" for exp_id, _, _ in STEPS]
    report_paths.extend([ROOT / "catboost/results/EXP-CB-020-control-r01.json",
                         ROOT / "catboost/results/snowball-020-022.json",
                         ROOT / "catboost/configs/snowball_best.json"])
    if any(path.exists() for path in report_paths):
        raise FileExistsError("These experiment IDs already have results; reserve new IDs before another series")
    series.mkdir(parents=True)
    with zipfile.ZipFile(series / "source.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for path in sorted((ROOT / "catboost").glob("*.py")):
            snapshot.write(path, path.relative_to(ROOT))
        for path in sorted((ROOT / "recsys_data").glob("*.py")):
            snapshot.write(path, path.relative_to(ROOT))
        for path in sorted((ROOT / "catboost/configs").glob("*.json")):
            snapshot.write(path, path.relative_to(ROOT))
    cache_manifest = build_cache(data, cache)
    train, val = [pq.read_table(cache / f"{split}.parquet") for split in ("train", "val")]
    metadata = pq.read_table(cache / "val_metadata.parquet")
    counts = metadata["candidate_count"].to_numpy()
    selection_feeds, partition = temporal_partition(metadata)
    selection_rows = np.repeat(selection_feeds, counts)
    selection = val.filter(pa.array(selection_rows))
    dump(series / "partition.json", partition)
    journal("### Зафиксированное временное разбиение серии\n\n"
            f"- Граница UTC: {partition['boundary_utc']}; selection {partition['selection_feeds']} лент, "
            f"control {partition['control_feeds']} лент. Равные timestamp не разделяются.\n"
            f"- Кэш признаков: `{cache.relative_to(ROOT)}`, подготовка {cache_manifest['seconds']:.2f} с "
            "(при повторном использовании — время исходной сборки). Все дополнительные признаки вычислены "
            "без меток; в Pool попадают только признаки соответствующего опыта.")
    provenance = {"prepared_manifest_sha256": sha256(data / "manifest.json"), "partition": partition,
                  "cache_manifest_sha256": sha256(cache / "manifest.json"),
                  "source_snapshot": str((series / "source.zip").relative_to(ROOT)),
                  "source_snapshot_sha256": sha256(series / "source.zip"),
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "versions": {"catboost": catboost.__version__, "numpy": np.__version__, "pyarrow": pa.__version__},
                  "platform": platform.platform(), "train": {"feeds": 24592, "rows": len(train), "files": 4}}
    model_config = json.loads((ROOT / "catboost/configs/baseline.json").read_text())
    feature_config = {"centroid": False, "event_types": False}
    base_id = "EXP-CB-020-control-r01"
    base_report, base_scores = fit_run(base_id, model_config, feature_config, train, val,
                                      selection_rows, series / base_id, provenance)
    base_values = per_feed_values(selection, base_scores[selection_rows], counts[selection_feeds])
    current_id, current_values = base_id, base_values
    reports = [base_report]
    scores_by_run = {base_id: base_scores}
    decisions = []
    dump(series / "decisions.json", decisions)
    journal(f"- Контроль {base_id}: selection NDCG={base_report['selection']['ndcg']:.10f}; "
            f"fit {base_report['train_seconds']:.2f} с, {base_report['trees']} деревьев.")
    for exp_id, hypothesis, title in STEPS:
        trial_config, trial_features = dict(model_config), dict(feature_config)
        if hypothesis == "ndcg_loss":
            trial_config["loss_function"] = "YetiRank:mode=NDCG;dcg_type=Base;dcg_denominator=LogPosition"
        else:
            trial_features[hypothesis] = True
        run_id = exp_id + "-r01"
        report, scores = fit_run(run_id, trial_config, trial_features, train, val,
                                selection_rows, series / run_id, provenance)
        values = per_feed_values(selection, scores[selection_rows], counts[selection_feeds])
        comparison = paired(values - current_values, metadata["user_id"].to_numpy()[selection_feeds])
        accepted = comparison["delta_ndcg"] >= MIN_DELTA
        decision = {"run_id": run_id, "hypothesis": title, "control_run": current_id,
                    "accepted": accepted, "minimum_delta": MIN_DELTA, "selection_comparison": comparison,
                    "reason": "positive practical gain" if accepted else "negative or near-zero gain"}
        report["decision"] = decision
        dump(series / run_id / "result.json", report)
        dump(series / run_id / "comparison.json", decision)
        decisions.append(decision)
        dump(series / "decisions.json", decisions)
        reports.append(report)
        scores_by_run[run_id] = scores
        journal(f"### {run_id} — результат и решение\n\n"
                f"- Контроль: {current_id}; selection NDCG={report['selection']['ndcg']:.10f}; "
                f"Δ={comparison['delta_ndcg']:+.10f}; диагностический 95% bootstrap "
                f"{comparison['user_cluster_bootstrap_95']}.\n"
                f"- Fit {report['train_seconds']:.2f} с; {report['trees']} деревьев; "
                f"пиковый RSS процесса {report['process_peak_rss_bytes']/2**30:.2f} GiB "
                "(накопительный максимум всей серии).\n"
                f"- Решение: **{'оставить' if accepted else 'отклонить'}**; порог +{MIN_DELTA}. "
                f"Следующий контроль: {run_id if accepted else current_id}. "
                "Поздний control пока не оценивался.")
        if accepted:
            model_config, feature_config = trial_config, trial_features
            current_id, current_values = run_id, values
    # Freeze decisions before evaluating the later control or full val for any run.
    best = {"run_id": current_id, "model_config": model_config,
            "feature_config": {**BASE_FEATURE_CONFIG, **feature_config},
            "decision_metric": "selection full-length NDCG", "minimum_delta": MIN_DELTA,
            "decisions_sha256": sha256(series / "decisions.json")}
    dump(series / "best.json", best)
    all_values = {}
    for report in reports:
        run_id = report["run_id"]
        scores = scores_by_run[run_id]
        report["validation"] = ranking_metrics(val["feed_id"].to_numpy(), val["target"].to_numpy(), scores)
        late = val.filter(pa.array(~selection_rows))
        report["control"] = ranking_metrics(late["feed_id"].to_numpy(), late["target"].to_numpy(), scores[~selection_rows])
        report["evaluation_note"] = "Late/full metrics calculated only after all acceptance decisions frozen"
        all_values[run_id] = per_feed_values(val, scores, counts)
        dump(series / run_id / "result.json", report)
        dump(ROOT / "catboost/results" / f"{run_id}.json", report)
    final_comparison = {}
    days = (metadata["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
    lengths = metadata["history_length"].to_numpy()
    zero_user = metadata["user_embedding_is_zero"].to_numpy()
    sizes = metadata["candidate_count"].to_numpy()
    for report in reports[1:]:
        run_id = report["run_id"]
        control_id = report["decision"]["control_run"]
        delta = all_values[run_id] - all_values[control_id]
        masks = {"selection": selection_feeds, "late_control": ~selection_feeds,
                 "all": np.ones(len(counts), dtype=bool), "empty_history": lengths == 0,
                 "history_1_15": (lengths > 0) & (lengths < 16),
                 "history_16_63": (lengths >= 16) & (lengths < 64),
                 "history_64_255": (lengths >= 64) & (lengths < 256), "history_256": lengths == 256,
                 "zero_user": zero_user, "nonzero_user": ~zero_user,
                 "candidates_5_30": sizes <= 30, "candidates_31_100": (sizes > 30) & (sizes <= 100),
                 "candidates_101_plus": sizes > 100}
        masks.update({f"day_moscow_{int(day)}": days == day for day in np.unique(days)})
        final_comparison[run_id] = {"control_run": control_id, "slices": {}}
        for name, mask in masks.items():
            final_comparison[run_id]["slices"][name] = {
                "feeds": int(mask.sum()), "control_ndcg": float(all_values[control_id][mask].mean()) if mask.any() else None,
                "treatment_ndcg": float(all_values[run_id][mask].mean()) if mask.any() else None,
                "delta_ndcg": float(delta[mask].mean()) if mask.any() else None}
        final_comparison[run_id]["full_val_bootstrap"] = paired(delta, metadata["user_id"].to_numpy())
    dump(series / "final_comparison.json", final_comparison)
    summary = {"status": "complete", "best": best, "partition": partition,
               "decisions": decisions, "runs": reports, "comparison": final_comparison,
               "test_read": False, "submission_created": False}
    dump(series / "summary.json", summary)
    dump(ROOT / "catboost/results/snowball-020-022.json", summary)
    dump(ROOT / "catboost/configs/snowball_best.json", best)
    lines = ["## Итог последовательной серии EXP-CB-020..022", "",
             "Решения зафиксированы до расчёта поздней/полной val. После этих метрик состав модели не менялся.", "",
             "| Run-ID | Selection NDCG | Поздний control NDCG | Полная val NDCG | Решение |",
             "| --- | ---: | ---: | ---: | --- |"]
    for report in reports:
        decision = report.get("decision")
        label = "контроль" if decision is None else ("оставить" if decision["accepted"] else "отклонить")
        lines.append(f"| {report['run_id']} | {report['selection']['ndcg']:.10f} | "
                     f"{report['control']['ndcg']:.10f} | {report['validation']['ndcg']:.10f} | {label} |")
    lines.extend(["", f"- Итоговая модель: **{current_id}**; конфигурация `catboost/configs/snowball_best.json`.",
                  f"- Артефакты серии: `{series.relative_to(ROOT)}`; отчёт `catboost/results/snowball-020-022.json`.",
                  "- Полная val ранее использовалась; один seed=42, bootstrap не учитывает выбор checkpoint и разброс обучения.",
                  "- Отклонённые гипотезы исключены из итоговой конфигурации; код/модели/прогнозы оставлены для воспроизводимости.",
                  "- Test-инференс, submission и публикация не выполнялись."])
    journal("\n".join(lines))
    print(json.dumps({"best": best, "metrics": [{"run": r["run_id"], "selection": r["selection"]["ndcg"],
        "control": r["control"]["ndcg"], "full": r["validation"]["ndcg"]} for r in reports]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/prepared-v1")
    parser.add_argument("--series", type=Path, default=ROOT / "runs/catboost/snowball-020-022")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/catboost-snowball-v1")
    args = parser.parse_args()
    series_existed = args.series.exists()
    try:
        run(args.data.resolve(), args.series.resolve(), args.cache.resolve())
    except Exception:
        error = traceback.format_exc()
        if not series_existed and args.series.exists():
            (args.series / "error.txt").write_text(error)
        journal("### Сбой последовательной серии\n\n```text\n" + error + "```\n"
                "Метрики незавершённых запусков не заявляются; успешные предыдущие записи сохранены.")
        raise
