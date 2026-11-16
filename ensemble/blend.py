"""Validate frozen parent predictions and select a blend using selection only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'catboost'))
from baseline import ranking_metrics, prediction_table, sha256

WEIGHTS=[0.,.25,.5,.75,1.]
METHODS=['rank','zscore']
MIN_DELTA=.001


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def packed(table):
    return (table['feed_id'].to_numpy().astype(np.int64)<<32)|table['item_id'].to_numpy().astype(np.int64)


def align(truth,predictions):
    expected_schema=pa.schema([('feed_id',pa.int32()),('item_id',pa.int32()),('score',pa.float32())])
    if predictions.schema!=expected_schema:
        raise ValueError('Prediction schema must be int32/int32/float32')
    if len(predictions)!=len(truth) or any(predictions[c].null_count for c in predictions.column_names):
        raise ValueError('Prediction row count or nulls invalid')
    actual,required=packed(predictions),packed(truth)
    order=np.argsort(actual)
    sorted_actual=actual[order]
    if (len(np.unique(actual))!=len(actual) or len(np.unique(required))!=len(required)
            or not np.array_equal(sorted_actual,np.sort(required))):
        raise ValueError('Predictions do not cover the exact unique val pairs')
    scores=predictions['score'].to_numpy()
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    return scores[order[np.searchsorted(sorted_actual,required)]]


def normalize(scores,counts,method):
    scores=np.asarray(scores,dtype=np.float64)
    if len(scores)!=int(np.sum(counts)) or not np.isfinite(scores).all():
        raise ValueError('Invalid normalization input')
    result=np.empty_like(scores)
    offsets=np.r_[0,np.cumsum(counts)]
    for start,end in zip(offsets[:-1],offsets[1:]):
        part=scores[start:end]
        if method=='rank':
            _,inverse,n=np.unique(part,return_inverse=True,return_counts=True)
            result[start:end]=((np.cumsum(n)-.5*n)/len(part))[inverse]
        elif method=='zscore':
            mean=math.fsum(part.tolist())/len(part)
            std=math.sqrt(math.fsum(((part-mean)**2).tolist())/len(part))
            result[start:end]=(part-mean)/std if std else 0.
        else:raise ValueError(method)
    return result


def mixture(cat,mlp,cat_normalized,mlp_normalized,weight):
    if not 0<=weight<=1:raise ValueError('Invalid weight')
    if weight==0:return np.asarray(mlp,dtype=np.float32)
    if weight==1:return np.asarray(cat,dtype=np.float32)
    result=np.asarray(weight*cat_normalized+(1-weight)*mlp_normalized,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite blend')
    return result


def per_feed(truth,scores,counts):
    y=truth['target'].to_numpy()
    result=[]
    for start,end in zip(np.r_[0,np.cumsum(counts)[:-1]],np.cumsum(counts)):
        labels=y[start:end]
        ranked=labels[np.lexsort((labels,-scores[start:end]))]
        discount=1/np.log2(np.arange(end-start)+2)
        ideal=discount[:int(labels.sum())].sum()
        result.append(float(np.dot(ranked,discount)/ideal) if ideal else 1.)
    return np.asarray(result)


def bootstrap(delta,users):
    _,inverse=np.unique(users,return_inverse=True)
    sums=np.bincount(inverse,weights=delta);counts=np.bincount(inverse)
    rng=np.random.default_rng(42);draws=[]
    for _ in range(1000):
        indices=rng.integers(0,len(sums),len(sums))
        draws.append(sums[indices].sum()/counts[indices].sum())
    return {'unit':'user_id','repetitions':1000,'seed':42,
            'delta_ndcg':float(delta.mean()),'interval_95':np.quantile(draws,[.025,.975]).tolist(),
            'limitation':'Conditional on fixed checkpoints; excludes model/weight selection uncertainty'}


def run(source,inputs,output):
    if output.exists() or (ROOT / 'ensemble/results/EXP-ENS-001-002.json').exists():
        raise FileExistsError('Reserve fresh output/IDs before repeating the experiment')
    output.mkdir(parents=True)
    started=time.perf_counter()
    cb_report=json.loads((inputs/'catboost_result.json').read_text())
    mlp_report=json.loads((inputs/'mlp_val_predictions.json').read_text())
    provenance=json.loads((inputs/'provenance.json').read_text())
    for name,digest in provenance['files'].items():
        if sha256(inputs/name)!=digest:raise ValueError(f'Input file changed: {name}')
    if sha256(inputs/'catboost_val_predictions.parquet')!=cb_report['artifacts']['val_predictions.parquet']:
        raise ValueError('CatBoost score hash differs')
    if sha256(inputs/'mlp_val_predictions.parquet')!=mlp_report['prediction_sha256']:
        raise ValueError('MLP score hash differs')
    cache=source/'artifacts/catboost-snowball-v1'
    cache_manifest=json.loads((cache/'manifest.json').read_text())
    if sha256(source/'artifacts/prepared-v1/manifest.json')!=cache_manifest['identity']['prepared_manifest_sha256']:
        raise ValueError('Canonical prepared data changed')
    for name in ['val.parquet','val_metadata.parquet']:
        if sha256(cache/name)!=cache_manifest['files'][name]:raise ValueError(f'Canonical val cache changed: {name}')
    truth=pq.read_table(cache/'val.parquet',columns=['feed_id','item_id','target'])
    meta=pq.read_table(cache/'val_metadata.parquet')
    counts=meta['candidate_count'].to_numpy()
    if not np.array_equal(truth['feed_id'].to_numpy(),np.repeat(meta['feed_id'].to_numpy(),counts)):
        raise ValueError('Metadata/group order differs from truth')
    cat=align(truth,pq.read_table(inputs/'catboost_val_predictions.parquet'))
    mlp=align(truth,pq.read_table(inputs/'mlp_val_predictions.parquet'))
    boundary=mlp_report['selection_control_boundary']
    if boundary!=cb_report['partition']['boundary_timestamp']:raise ValueError('Selection boundaries differ')
    timestamps=meta['timestamp'].to_numpy();selection= timestamps<boundary
    row_selection=np.repeat(selection,counts)
    if int(selection.sum())!=11534 or int((~selection).sum())!=2889:
        raise ValueError('Validation partition sizes changed')
    st=truth.filter(pa.array(row_selection))
    parent_selection={name:ranking_metrics(st['feed_id'].to_numpy(),st['target'].to_numpy(),scores[row_selection])
                      for name,scores in [('catboost',cat),('mlp',mlp)]}
    if abs(parent_selection['catboost']['ndcg']-cb_report['selection']['ndcg'])>1e-10:
        raise ValueError('CatBoost selection score differs')
    if abs(parent_selection['mlp']['ndcg']-mlp_report['selection_ndcg_at_checkpoint'])>1e-10:
        raise ValueError('MLP selection score differs')
    stronger=max(parent_selection,key=lambda name:parent_selection[name]['ndcg'])
    baseline_score=parent_selection[stronger]['ndcg']
    normalized={method:(normalize(cat,counts,method),normalize(mlp,counts,method)) for method in METHODS}
    grid=[]
    for index,method in enumerate(METHODS,1):
        for weight in WEIGHTS:
            scores=mixture(cat,mlp,*normalized[method],weight)
            metric=ranking_metrics(st['feed_id'].to_numpy(),st['target'].to_numpy(),scores[row_selection],verify=False)
            row={'experiment':f'EXP-ENS-{index:03d}','method':method,'catboost_weight':weight,
                 'mlp_weight':1-weight,'selection':metric,'delta_vs_stronger_parent':metric['ndcg']-baseline_score}
            grid.append(row)
            print(json.dumps(row),flush=True)
    # Stable ordering resolves exact ties: rank first, then lower CatBoost weight.
    candidate=max(grid,key=lambda row:row['selection']['ndcg'])
    accepted=(0<candidate['catboost_weight']<1 and candidate['delta_vs_stronger_parent']>=MIN_DELTA)
    decision={'candidate':candidate,'stronger_parent':stronger,'parent_selection':parent_selection,
              'accepted':accepted,'minimum_delta':MIN_DELTA,'selection_boundary':int(boundary),
              'chosen':'blend' if accepted else stronger,'mlp_source_commit':provenance['mlp_commit'],
              'parent_prediction_hashes':{'catboost':cb_report['artifacts']['val_predictions.parquet'],
                                          'mlp':mlp_report['prediction_sha256']},
              'script_sha256':sha256(__file__),'cache_manifest_sha256':sha256(cache/'manifest.json'),
              'test_read':False,'submission_created':False}
    effective_weight=candidate['catboost_weight'] if accepted else (1. if stronger=='catboost' else 0.)
    decision.update(effective_method=candidate['method'] if accepted else 'identity',
                    effective_catboost_weight=effective_weight,effective_mlp_weight=1-effective_weight,
                    parent_checkpoint_hashes={'catboost':cb_report['artifacts']['model.cbm'],
                                              'mlp':mlp_report['checkpoint_sha256']})
    dump(output/'selection_grid.json',grid)
    dump(output/'frozen_decision.json',decision)
    candidate_scores=mixture(cat,mlp,*normalized[candidate['method']],candidate['catboost_weight'])
    baseline=cat if stronger=='catboost' else mlp
    chosen=candidate_scores if accepted else baseline
    late=truth.filter(pa.array(~row_selection))
    metrics={}
    for name,scores in [('catboost',cat),('mlp',mlp),('candidate',candidate_scores),('chosen',chosen)]:
        metrics[name]={'selection':ranking_metrics(st['feed_id'].to_numpy(),st['target'].to_numpy(),scores[row_selection]),
                       'control':ranking_metrics(late['feed_id'].to_numpy(),late['target'].to_numpy(),scores[~row_selection]),
                       'full':ranking_metrics(truth['feed_id'].to_numpy(),truth['target'].to_numpy(),scores)}
    if abs(metrics['catboost']['full']['ndcg']-cb_report['validation']['ndcg'])>1e-10:
        raise ValueError('CatBoost full val score differs')
    if abs(metrics['mlp']['full']['ndcg']-mlp_report['full_val_ndcg'])>1e-10:
        raise ValueError('MLP full val score differs')
    base_values=per_feed(truth,baseline,counts);new_values=per_feed(truth,chosen,counts)
    delta=new_values-base_values;users=meta['user_id'].to_numpy()
    masks={'selection':selection,'late_control':~selection,'all':np.ones(len(counts),dtype=bool)}
    lengths=meta['history_length'].to_numpy();zero=meta['user_embedding_is_zero'].to_numpy()
    masks.update(empty_history=lengths==0,history_1_15=(lengths>0)&(lengths<16),
                 history_16_63=(lengths>=16)&(lengths<64),history_64_255=(lengths>=64)&(lengths<256),
                 history_256=lengths==256,zero_user=zero,nonzero_user=~zero,
                 candidates_5_30=counts<=30,candidates_31_100=(counts>30)&(counts<=100),candidates_101plus=counts>100)
    days=(timestamps.astype(np.int64)+10800)//86400
    for day in np.unique(days):
        label=datetime.fromtimestamp(int(day)*86400,timezone.utc).date().isoformat()
        masks['day_moscow_'+label]=days==day
    slices={name:{'feeds':int(mask.sum()),'parent_ndcg':float(base_values[mask].mean()) if mask.any() else None,
                  'chosen_ndcg':float(new_values[mask].mean()) if mask.any() else None,
                  'delta_ndcg':float(delta[mask].mean()) if mask.any() else None} for name,mask in masks.items()}
    intervals={name:bootstrap(delta[mask],users[mask]) for name,mask in list(masks.items())[:3]}
    pq.write_table(prediction_table(truth,chosen),output/'val_predictions.parquet',compression='zstd')
    pq.write_table(prediction_table(truth,candidate_scores),output/'candidate_val_predictions.parquet',compression='zstd')
    report={'status':'complete','decision':decision,'grid':grid,'metrics':metrics,'slices':slices,
            'bootstrap':intervals,'rows':len(truth),'feeds':len(meta),'seconds':time.perf_counter()-started,
            'artifacts':{name:sha256(output/name) for name in ['val_predictions.parquet','candidate_val_predictions.parquet','frozen_decision.json']},
            'limitation':'Historical val reused by both model families; intervals exclude model/weight selection uncertainty',
            'test_read':False,'submission_created':False}
    dump(output/'result.json',report)
    (ROOT/'ensemble/results').mkdir(exist_ok=True)
    dump(ROOT/'ensemble/results/EXP-ENS-001-002.json',report)
    (ROOT/'ensemble/configs').mkdir(exist_ok=True)
    dump(ROOT/'ensemble/configs/best.json',decision)
    with (ROOT/'ensemble_exp.md').open('a') as stream:
        stream.write('\n## Результаты EXP-ENS-001-r01 / EXP-ENS-002-r01\n\n')
        stream.write('| Метод | Доля CatBoost | Доля MLP | Selection NDCG | Δ к сильному родителю |\n| --- | ---: | ---: | ---: | ---: |\n')
        for row in grid:
            stream.write(f"| {row['method']} | {row['catboost_weight']} | {row['mlp_weight']} | {row['selection']['ndcg']:.10f} | {row['delta_vs_stronger_parent']:+.10f} |\n")
        stream.write(f"\n- Сильный родитель: {stronger}. Решение до оценки поздней/full val: {'оставить ансамбль' if accepted else 'оставить одиночного родителя'}.\n")
        stream.write(f"- Выбранный кандидат: {candidate['method']}, CatBoost {candidate['catboost_weight']}, MLP {candidate['mlp_weight']}.\n")
        for name,value in metrics.items():
            stream.write(f"- {name}: selection {value['selection']['ndcg']:.10f}; поздний control {value['control']['ndcg']:.10f}; full val {value['full']['ndcg']:.10f}.\n")
        stream.write(f"- Full-val Δ выбранного результата: {delta.mean():+.10f}; bootstrap {intervals['all']['interval_95']}. Поздний control Δ {delta[~selection].mean():+.10f}, bootstrap {intervals['late_control']['interval_95']}.\n")
        stream.write(f"- Время {report['seconds']:.2f} с. Метрики родителей независимо совпали с их отчётами, все пары/граница подтверждены. Артефакты: `{output.relative_to(ROOT)}`; отчёт `ensemble/results/EXP-ENS-001-002.json`.\n")
        stream.write('- Новых обучений, test-инференса, submission и публикаций не выполнялось.\n')
    print(json.dumps({'decision':decision,'metrics':metrics,'seconds':report['seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT.parent/'AvitoTech_ML_Cup')
    parser.add_argument('--inputs',type=Path,default=ROOT/'runs/ensemble/inputs')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/ensemble/EXP-ENS-001-002')
    args=parser.parse_args()
    try:run(args.source_root.resolve(),args.inputs.resolve(),args.output.resolve())
    except Exception:
        import traceback
        with (ROOT/'ensemble_exp.md').open('a') as stream:
            stream.write('\n## Сбой опыта\n\n```text\n'+traceback.format_exc()+'```\n')
        raise
