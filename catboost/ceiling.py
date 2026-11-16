"""Bounded CatBoost-only search: small screens, large confirmations, paired seed check.

No test access, submissions, ensemble, branch change or publication.
"""
import gc
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
import zipfile

import catboost
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from baseline import ranking_metrics, sha256
from compact_embeddings import prepare_train_cache
from extended_features import (LONG_FEATURES, PROJECTED_FEATURES, PROJECTION,
                               PROJECTION_SHA256, add_extended_features)
from history_neighbors import NEIGHBOR_FEATURES
from recsys_data import PreparedSplit
from snowball import (ROOT, MIN_DELTA, build_cache, dump, features_for, fit_run,
                      journal, paired, per_feed_values, temporal_partition)

SERIES = ROOT / "runs/catboost/ceiling-025-038"
SMALL = ROOT / "artifacts/prepared-v1"
LARGE = ROOT / "artifacts/prepared-train16-v1"
BASE_CACHE = ROOT / "artifacts/catboost-snowball-v1"
EXTRA_SMALL = ROOT / "artifacts/catboost-extended-train4-v1"
EXTRA_LARGE = ROOT / "artifacts/catboost-extended-train16-v1"
GROUPS = [
    (25,26,"long_history", "История256 сверх сохранённой истории64"),
    (27,28,"projected_products", "32 произведения проекций user/item"),
    (29,30,"depth8", "Глубина 8"),
    (31,32,"l2", "L2=10"),
    (33,34,"depth4", "Глубина 4"),
    (35,36,"lr", "Learning rate 0,05, до 400 деревьев"),
]
BASE_FLAGS = {"centroid": True, "event_types": False, "neighbors": True,
              "long_history": False, "projected_products": False}


def snapshot():
    with zipfile.ZipFile(SERIES / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("catboost", "recsys_data"):
            for path in sorted((ROOT / folder).glob("*.py")):
                archive.write(path, path.relative_to(ROOT))
        for path in sorted((ROOT / "catboost/configs").glob("*.json")):
            archive.write(path, path.relative_to(ROOT))
    np.save(SERIES / "projection.npy", PROJECTION)
    return {"source_snapshot": str((SERIES / "source.zip").relative_to(ROOT)),
            "source_snapshot_sha256": sha256(SERIES / "source.zip"),
            "projection_sha256": PROJECTION_SHA256,
            "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "platform": platform.platform(),
            "versions": {"catboost":catboost.__version__,"numpy":np.__version__,"pyarrow":pa.__version__}}


def verify_cache(directory, manifest, files_key="files"):
    for name, expected in manifest[files_key].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Cache checksum differs: {directory / name}")


def extra_cache(data, output, compact, splits):
    identity = {"data_manifest_sha256": sha256(data / "manifest.json"),
                "compact_manifest_sha256": sha256(compact.directory / "manifest.json"),
                "feature_source_sha256": sha256(ROOT / "catboost/extended_features.py"),
                "projection_sha256": PROJECTION_SHA256, "splits": list(splits)}
    if output.exists():
        saved=json.loads((output / "manifest.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Extended cache identity changed")
        verify_cache(output,saved)
        return saved
    output.mkdir(parents=True)
    started=time.perf_counter()
    counts={}
    for split in splits:
        dataset=PreparedSplit(data,split)
        if split=="train":
            if (dataset.embeddings.meta != compact.meta or
                dataset.manifest["archive_etag"] != compact.manifest["identity"]["source_archive_etag"]):
                raise ValueError("Compact cache belongs to another train matrix")
            dataset.embeddings=compact
        writer=None
        rows=0
        try:
            for i,batch in enumerate(dataset.iter_batches(batch_size=256),1):
                table=add_extended_features(batch,batch.candidate_table()).select(
                    ["feed_id","item_id",*LONG_FEATURES,*PROJECTED_FEATURES])
                if writer is None:
                    writer=pq.ParquetWriter(output / f"{split}.parquet",table.schema,compression="zstd")
                writer.write_table(table)
                rows+=len(table)
                if i%20==0:
                    print(f"Extended {data.name}/{split}: {rows:,} candidates",flush=True)
        finally:
            if writer is not None:writer.close()
        counts[split]=rows
    result={"identity":identity,"seconds":time.perf_counter()-started,"rows":counts,
            "files":{f"{s}.parquet":sha256(output / f"{s}.parquet") for s in splits}}
    dump(output / "manifest.json",result)
    journal(f"- Extended-кэш `{output.relative_to(ROOT)}`: {result['seconds']:.2f} с, строки {counts}. "
            "Обе группы вычислены без меток; в Pool входят только явно выбранные столбцы.")
    return result


def attach(table, extra, columns):
    if not table.select(["feed_id","item_id"]).equals(extra.select(["feed_id","item_id"])):
        raise ValueError("Feature cache keys differ")
    for name in columns:table=table.append_column(name,extra[name])
    return table


def read_small(split):
    names=features_for(BASE_FLAGS)
    base_names=[name for name in names if name not in NEIGHBOR_FEATURES]
    table=pq.read_table(BASE_CACHE / f"{split}.parquet",columns=["feed_id","item_id",*base_names,"target"])
    table=attach(table,pq.read_table(ROOT / f"artifacts/catboost-neighbors64-v1/{split}.parquet"),NEIGHBOR_FEATURES)
    return attach(table,pq.read_table(EXTRA_SMALL / f"{split}.parquet"),LONG_FEATURES+PROJECTED_FEATURES)


def read_large(flags,compact):
    directory=ROOT / "artifacts/catboost-neighbors64-train16-v1"
    saved=json.loads((directory / "manifest.json").read_text())
    if saved["manifest_sha256"] != sha256(LARGE / "manifest.json"):
        raise ValueError("Large base cache uses another prepared dataset")
    table=pq.read_table(directory / "train.parquet")
    extra_names=(LONG_FEATURES if flags["long_history"] else [])+(
        PROJECTED_FEATURES if flags["projected_products"] else [])
    if extra_names:
        extra_cache(LARGE,EXTRA_LARGE,compact,("train",))
        table=attach(table,pq.read_table(EXTRA_LARGE / "train.parquet",columns=["feed_id","item_id",*extra_names]),extra_names)
    return table


def saved_run(path,truth):
    report=json.loads((path / "result.json").read_text())
    if report["status"] != "complete" or sha256(path / "model.cbm") != report["artifacts"]["model.cbm"]:
        raise ValueError("Invalid control checkpoint")
    predictions=pq.read_table(path / "val_predictions.parquet")
    if (not predictions.select(["feed_id","item_id"]).equals(truth.select(["feed_id","item_id"]))
            or sha256(path / "val_predictions.parquet") != report["artifacts"]["val_predictions.parquet"]):
        raise ValueError("Invalid control predictions")
    return report,predictions["score"].to_numpy()


def changes(config,flags,kind):
    config,flags=dict(config),dict(flags)
    if kind in ("long_history","projected_products"):flags[kind]=True
    elif kind=="depth8":config["depth"]=8
    elif kind=="depth4":config["depth"]=4
    elif kind=="l2":config["l2_leaf_reg"]=10
    elif kind=="lr":config.update(learning_rate=.05,iterations=400)
    else:raise ValueError(kind)
    return config,flags


def main():
    if SERIES.exists() or (ROOT / "catboost/configs/ceiling_best.json").exists():
        raise FileExistsError("Reserve fresh IDs/output for another search")
    for number in range(25,39):
        if (ROOT / f"catboost/results/EXP-CB-{number:03d}-r01.json").exists():
            raise FileExistsError(f"EXP-CB-{number:03d} already has results")
    SERIES.mkdir(parents=True)
    provenance=snapshot()
    build_cache(SMALL,BASE_CACHE)
    neighbor_dir=ROOT / "artifacts/catboost-neighbors64-v1"
    verify_cache(neighbor_dir,json.loads((neighbor_dir / "manifest.json").read_text()))
    large_base=ROOT / "artifacts/catboost-neighbors64-train16-v1"
    saved=json.loads((large_base / "manifest.json").read_text())
    if sha256(large_base / "train.parquet") != saved["train_cache_sha256"]:
        raise ValueError("Large base cache checksum changed")
    compact=prepare_train_cache(LARGE,SMALL,ROOT / "artifacts/compact-embeddings-train16-v1")
    provenance["compact_manifest_sha256"]=sha256(compact.directory / "manifest.json")
    extra_cache(SMALL,EXTRA_SMALL,compact,("train","val"))
    train4=read_small("train")
    val=read_small("val")
    metadata=pq.read_table(BASE_CACHE / "val_metadata.parquet")
    counts=metadata["candidate_count"].to_numpy()
    feed_mask,partition=temporal_partition(metadata)
    if partition != json.loads((ROOT / "runs/catboost/snowball-020-022/partition.json").read_text()):
        raise ValueError("Validation partition changed")
    row_mask=np.repeat(feed_mask,counts)
    selection=val.filter(pa.array(row_mask))
    original4=ROOT / "runs/catboost/route-023-024/EXP-CB-023-r01"
    original16=ROOT / "runs/catboost/route-023-024/EXP-CB-024-r01"
    small_report,small_scores=saved_run(original4,val)
    large_report,large_scores=saved_run(original16,val)
    original_report,original_scores=large_report,large_scores
    model_config=dict(large_report["config"])
    flags=dict(BASE_FLAGS)
    provenance.update(partition=partition,extended_small_manifest_sha256=sha256(EXTRA_SMALL / "manifest.json"))
    all_reports=[]
    all_scores={small_report["run_id"]:small_scores,large_report["run_id"]:large_scores}
    decisions=[]

    def fit(number,config,feature_flags,large=False):
        data=LARGE if large else SMALL
        table=read_large(feature_flags,compact) if large else train4
        prov={**provenance,"prepared_manifest_sha256":sha256(data / "manifest.json"),
              "train":{"feeds":98428 if large else 24592,"rows":len(table),"files":16 if large else 4}}
        if large and EXTRA_LARGE.exists():prov["extended_large_manifest_sha256"]=sha256(EXTRA_LARGE / "manifest.json")
        run_id=f"EXP-CB-{number:03d}-r01"
        report,scores=fit_run(run_id,config,feature_flags,table,val,row_mask,SERIES / run_id,prov)
        all_reports.append(report)
        all_scores[run_id]=scores
        if large:
            del table
            gc.collect()
            pa.default_memory_pool().release_unused()
        return report,scores

    def compare(report,scores,control,control_scores,phase,title):
        new=per_feed_values(selection,scores[row_mask],counts[feed_mask])
        old=per_feed_values(selection,control_scores[row_mask],counts[feed_mask])
        result=paired(new-old,metadata["user_id"].to_numpy()[feed_mask])
        passed=result["delta_ndcg"]>=MIN_DELTA
        report["decision"]={"phase":phase,"hypothesis":title,"control_run":control["run_id"],
                            "passed":passed,"minimum_delta":MIN_DELTA,"selection_comparison":result}
        dump(SERIES / report["run_id"] / "result.json",report)
        dump(SERIES / report["run_id"] / "comparison.json",report["decision"])
        journal(f"### {report['run_id']} — {phase}: {'прошёл порог' if passed else 'отклонить'}\n\n"
                f"- {title}; контроль {control['run_id']}; selection {report['selection']['ndcg']:.10f}, "
                f"Δ {result['delta_ndcg']:+.10f}; bootstrap {result['user_cluster_bootstrap_95']}.\n"
                f"- Fit {report['train_seconds']:.2f} с, {report['trees']} деревьев; "
                "поздняя/полная val пока не оценивались.")
        return passed

    for screen_id,confirm_id,kind,title in GROUPS:
        trial_config,trial_flags=changes(model_config,flags,kind)
        report,scores=fit(screen_id,trial_config,trial_flags)
        passed=compare(report,scores,small_report,small_scores,"4-file screen",title)
        decision={"screen_run":report["run_id"],"screen_passed":passed,"accepted":False,"hypothesis":title,
                  "confirm_run":None,"control_4":small_report["run_id"],"control_16":large_report["run_id"]}
        if passed:
            confirmed,confirmed_scores=fit(confirm_id,trial_config,trial_flags,large=True)
            decision["confirm_run"]=confirmed["run_id"]
            accepted=compare(confirmed,confirmed_scores,large_report,large_scores,"16-file confirmation",title)
            decision["accepted"]=accepted
            if accepted:
                model_config,flags=trial_config,trial_flags
                small_report,small_scores=report,scores
                large_report,large_scores=confirmed,confirmed_scores
        else:
            journal(f"- EXP-CB-{confirm_id:03d}: не запускался — 4-file проба не прошла порог.")
        decisions.append(decision)
        dump(SERIES / "decisions.json",decisions)
        journal(f"- Итог группы «{title}»: {'оставить' if decision['accepted'] else 'не включать в основной вариант'}; "
                f"следующий основной контроль {large_report['run_id']}.")
        print(json.dumps(decision,ensure_ascii=False),flush=True)

    # A predeclared paired second seed, never selecting the best seed checkpoint.
    seed_config=dict(original_report["config"],random_seed=17)
    repeat_control,repeat_control_scores=fit(37,seed_config,BASE_FLAGS,large=True)
    candidate_changed=large_report["run_id"]!=original_report["run_id"]
    seed_confirmed=True
    if candidate_changed:
        repeat_config=dict(model_config,random_seed=17)
        repeat_leader,repeat_leader_scores=fit(38,repeat_config,flags,large=True)
        seed_confirmed=compare(repeat_leader,repeat_leader_scores,repeat_control,repeat_control_scores,
                               "seed17 paired confirmation","Проверка суммарного улучшения на другом seed")
    else:
        journal("- EXP-CB-038: не запускался, конфигурация лидера совпадает с исходным контролем. Seed17 не используется для выбора checkpoint.")
    chosen=large_report if seed_confirmed else original_report
    chosen_dir=SERIES / chosen["run_id"] if chosen is not original_report else original16
    chosen_flags=flags if seed_confirmed else BASE_FLAGS
    best={"run_id":chosen["run_id"],"run_directory":str(chosen_dir.relative_to(ROOT)),
          "model_config":chosen["config"],"feature_config":chosen["feature_config"],
          "seed_confirmation_passed":seed_confirmed if candidate_changed else None,"candidate_changed":candidate_changed,
          "selection_threshold":MIN_DELTA,"projection_sha256":PROJECTION_SHA256,
          "provisional_candidate":large_report["run_id"] if not seed_confirmed else None}
    # Freeze all decisions and the selected seed42 model before late/full evaluation.
    dump(SERIES / "best.json",best)
    dump(SERIES / "frozen_decisions.json",{"groups":decisions,"seed_confirmed":seed_confirmed,"best":best})
    for report in all_reports:
        scores=all_scores[report["run_id"]]
        report["validation"]=ranking_metrics(val["feed_id"].to_numpy(),val["target"].to_numpy(),scores)
        late=val.filter(pa.array(~row_mask))
        report["control"]=ranking_metrics(late["feed_id"].to_numpy(),late["target"].to_numpy(),scores[~row_mask])
        report["evaluation_note"]="Full/late measured only after all search and seed decisions were frozen"
        dump(SERIES / report["run_id"] / "result.json",report)
        dump(ROOT / f"catboost/results/{report['run_id']}.json",report)
    summary={"status":"complete","best":best,"groups":decisions,"runs":all_reports,
             "test_read":False,"submission_created":False,"ensemble_created":False,
             "limit_note":"Bounded search on 4/16 shards; not a ceiling of CatBoost on full train"}
    dump(SERIES / "summary.json",summary)
    dump(ROOT / "catboost/results/ceiling-025-038.json",summary)
    dump(ROOT / "catboost/configs/ceiling_best.json",best)
    lines=["## Итог CatBoost-only серии EXP-CB-025..038","",
           "| Run-ID | Train-файлы | Selection | Поздний control | Полная val | Роль |",
           "| --- | ---: | ---: | ---: | ---: | --- |"]
    for report in all_reports:
        role=report.get("decision",{}).get("phase","seed17 control")
        lines.append(f"| {report['run_id']} | {report['train']['files']} | {report['selection']['ndcg']:.10f} | "
                     f"{report['control']['ndcg']:.10f} | {report['validation']['ndcg']:.10f} | {role} |")
    lines.extend(["",f"- Итоговый основной вариант: **{best['run_id']}**, `catboost/configs/ceiling_best.json`.",
                  f"- Подтверждение нового кандидата другим seed: {best['seed_confirmation_passed']}; новый кандидат отличается от исходного: {candidate_changed}. При отсутствии нового кандидата выполнен только повтор контроля.",
                  "- Прежний онлайн-score относится только к EXP-CB-024; новых online-метрик нет.",
                  "- Исследован ограниченный набор сигналов/параметров на 4/16 файлах. Предел класса моделей на полном train не установлен.",
                  "- Test, submission, ансамбли, другие архитектуры, новые ветки и публикации не выполнялись."])
    journal("\n".join(lines))
    print(json.dumps({"best":best,"metrics":[{"run":r["run_id"],"selection":r["selection"]["ndcg"],
        "late":r["control"]["ndcg"],"full":r["validation"]["ndcg"]} for r in all_reports]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    existed=SERIES.exists()
    try:main()
    except Exception:
        error=traceback.format_exc()
        if not existed and SERIES.exists():(SERIES / "error.txt").write_text(error)
        journal("### Ошибка CatBoost-only серии\n\n```text\n"+error+"```\nПредыдущие результаты сохранены.")
        raise
