"""Export a selected checkpoint to official val and test, without training or uploading."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from recsys_data import PreparedSplit
from recsys_data.core import _flat
from ftt.data import digest
from ftt.atomic import write_json
from ftt.inference import model_inputs
from ftt.model import Ranker
from ftt.training import ndcg

SCHEMA = pa.schema([('feed_id',pa.int32()),('item_id',pa.int32()),('score',pa.float32())])


def pair_keys(feed, item):
    return (np.asarray(feed).astype('uint32').astype('uint64') << 32) | np.asarray(item).astype('uint32').astype('uint64')


def validate(path, expected):
    table = pq.read_table(path)
    if table.column_names != SCHEMA.names or any(table.schema.field(n).type != SCHEMA.field(n).type for n in SCHEMA.names):
        raise ValueError('Prediction schema must be int32/int32/float32')
    if any(table[n].null_count for n in table.column_names) or not np.isfinite(table['score'].to_numpy()).all():
        raise ValueError('Prediction contains null/NaN/Inf')
    keys = pair_keys(table['feed_id'].to_numpy(),table['item_id'].to_numpy())
    if len(np.unique(keys)) != len(keys):
        raise ValueError('Duplicate prediction pairs')
    if not np.array_equal(np.sort(keys),np.sort(expected)):
        raise ValueError('Prediction pairs do not cover the official split')
    return dict(rows=len(keys),feeds=len(np.unique(table['feed_id'].to_numpy())),sha256=digest(path))


@torch.inference_mode()
def export(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    checkpoint=Path(args.checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    weights=state['model']
    model=Ranker(state['identity']['config'],weights['mean'],weights['scale'],weights['edges']).to(args.device)
    model.load_state_dict(weights,strict=True)
    model.eval()
    provenance=dict(run_id=args.run_id,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                    checkpoint_step=state['step'],seed=state['identity']['seed'],
                    config=state['identity']['config'],training_commit=args.training_commit,
                    inference_commit=args.inference_commit,training_source_sha256=args.training_source_sha256,
                    manifest_sha256=digest(Path(args.data)/'manifest.json'),scores='raw float32 logits',splits={})
    for split in ('val','test'):
        dataset=PreparedSplit(args.data,split)
        temporary=output/f'{split}.tmp.parquet'
        metrics, timestamps=[],[]
        with pq.ParquetWriter(temporary,SCHEMA,compression='zstd') as writer:
            # Match the original per-feed validation kernels when checking checkpoint parity.
            for batch in dataset.iter_batches(1 if split=='val' else args.batch_size,include_labels=split=='val'):
                inputs=model_inputs(batch,args.device)
                scores=model(inputs).cpu().numpy().astype('float32')
                candidates=batch.candidate_table()
                writer.write_table(pa.table(dict(feed_id=candidates['feed_id'],item_id=candidates['item_id'],
                                                  score=pa.array(scores)),schema=SCHEMA))
                if split=='val':
                    target=_flat(batch.labels,'target').to_numpy()
                    offsets=batch.candidate_offsets
                    metrics.extend(ndcg(target[a:b],scores[a:b]) for a,b in zip(offsets[:-1],offsets[1:]))
                    timestamps.extend(batch.features['timestamp'].to_numpy().tolist())
        # Independently read the official pair columns, not the arrays used by the forward pass.
        expected=[]
        for record in dataset.info['files']:
            features=pq.read_table(Path(args.data)/split/'features'/record['name'],columns=['feed_id','item_id'])
            counts=pc.list_value_length(features['item_id']).to_numpy()
            expected.append(pair_keys(np.repeat(features['feed_id'].to_numpy(),counts),_flat(features,'item_id').to_numpy()))
        validation=validate(temporary,np.concatenate(expected))
        if split=='val':
            if validation['rows']!=1389971 or validation['feeds']!=14423:
                raise ValueError('Incomplete official val export')
            values=np.asarray(metrics)
            times=np.asarray(timestamps)
            boundary=int(np.sort(times)[int(len(times)*.8)])
            selection=np.flatnonzero(times<boundary)
            rng=np.random.default_rng(2026)
            fast=np.sort(rng.permutation(selection)[:2000])
            fast_score=float(values[fast].mean())
            if abs(fast_score-state['best'])>1e-7:
                raise ValueError(f'Frozen checkpoint parity failed: {fast_score} != {state["best"]}')
            validation.update(full_ndcg=float(values.mean()),selection_ndcg=float(values[selection].mean()),
                              control_ndcg=float(values[times>=boundary].mean()),fast_ndcg=fast_score,boundary=boundary)
        else:
            template=pq.read_table(args.sample,columns=['feed_id','item_id'])
            validate(temporary,pair_keys(template['feed_id'].to_numpy(),template['item_id'].to_numpy()))
        final=output/('ftt_val_predictions.parquet' if split=='val' else 'ftt_submission.parquet')
        temporary.replace(final)
        provenance['splits'][split]=dict(file=str(final),**validation)
        write_json(output/'metadata.json',provenance)
        print(json.dumps(dict(split=split,**validation)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ('data','checkpoint','output','sample','run-id','training-commit','inference-commit','training-source-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--cpu-threads',type=int,default=4)
    export(parser.parse_args())


if __name__=='__main__':
    main()
