"""Export raw test logits from an existing research checkpoint for manual upload."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mlp_next.extended_models import ExtendedRanker
from mlp_trials.data import batches, context_features, pack
from mlp_trials.models import history_stats
from mlp_trials.state import write_json
from scripts.validate_submission import validate


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs_for_batch(batch, categories, config, device):
    packed = pack(batch, categories, kind='history_stats',
                  history_window=256 if config.get('sequence') else 128)
    tensors = packed['tensors']
    short = {k: v[:, :128] if k.startswith('h_') and v.ndim >= 2 else v
             for k, v in tensors.items()}
    values = {k: tensors[k] for k in ('dense', 'category', 'counts', 'weights')}
    values['history'] = history_stats(short)
    if config['context']:
        values['context'] = torch.from_numpy(context_features(tensors['dense'].numpy(),
                              tensors['category'].numpy(), tensors['counts'].numpy()))
    if config.get('sequence'):
        values.update({k: tensors[k] for k in ('h_vectors', 'h_mask', 'h_present',
                      'h_category', 'h_type', 'h_source')})
        values['h_vectors'] = values['h_vectors'].half()
    return packed['table'], {k: v if k == 'counts' else v.to(device) for k, v in values.items()}


@torch.inference_mode()
def export(args):
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cache_hash, data_hash = sha256(args.cache / 'metadata.json'), sha256(args.data / 'manifest.json')
    info = json.loads((args.cache / 'metadata.json').read_text(encoding='utf-8'))
    full_hash = hashlib.sha256((cache_hash + data_hash).encode()).hexdigest()
    if state['identity']['cache'] != full_hash and not (
            state['identity']['cache'] == cache_hash and info['signature']['manifest'] == data_hash):
        raise ValueError('Checkpoint, cache and prepared data identities differ')
    config = state['identity']['config']
    categories = {int(k): v for k, v in info['categories'].items()}
    with np.load(args.cache / 'statistics.npz') as archive:
        statistics = {k: archive[k].copy() for k in archive.files}
    model = ExtendedRanker(len(categories), config, statistics).to(args.device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    args.output.mkdir(parents=True)
    prediction, temporary = args.output / 'submission.parquet', args.output / 'submission.tmp.parquet'
    manifest = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    total = manifest['splits']['test']['feeds']
    started = time.monotonic()
    feeds = rows = 0
    writer = None
    source = batches(args.data, 'test', args.batch_size, history=True)
    try:
        for batch in source:
            table, inputs = inputs_for_batch(batch, categories, config, args.device)
            scores = model(inputs).cpu().numpy().astype('float32')
            if len(scores) != len(table) or not np.isfinite(scores).all():
                raise ValueError('Incomplete or nonfinite test logits')
            result = pa.table({'feed_id': table['feed_id'].cast(pa.int32()),
                              'item_id': table['item_id'].cast(pa.int32()),
                              'score': pa.array(scores, type=pa.float32())})
            if writer is None:
                writer = pq.ParquetWriter(temporary, result.schema)
            writer.write_table(result)
            feeds += len(batch.features)
            rows += len(result)
            write_json(args.output / 'status.json', dict(status='running', phase='predict',
                       step=feeds, steps=total, rows=rows, elapsed_seconds=time.monotonic()-started))
        if writer:
            writer.close()
            writer = None
        if feeds != total:
            raise ValueError('Incomplete test feed coverage')
        audit = validate(temporary, args.template)
        temporary.replace(prediction)
        audit['file'] = str(prediction.resolve())
        metadata = dict(run_id=args.run_id, checkpoint=str(args.checkpoint),
            checkpoint_sha256=sha256(args.checkpoint), selected_step=state['step'],
            model=config, identity=state['identity'], cache_sha256=cache_hash,
            data_manifest_sha256=data_hash, boundary=info['boundary'],
            inference_code_commit=args.code_commit, score_transform='none; raw float32 logits',
            split='test', audit=audit, elapsed_seconds=time.monotonic()-started,
            created_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output / 'submission.json', metadata)
        write_json(args.output / 'status.json', dict(status='completed', phase='completed',
                   step=feeds, steps=total, rows=rows, sha256=audit['sha256']))
        return metadata
    except Exception as error:
        if writer:
            writer.close()
        temporary.unlink(missing_ok=True)
        write_json(args.output / 'status.json', dict(status='error', error=str(error), rows=rows, step=feeds))
        raise
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'cache', 'checkpoint', 'template', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    pa.set_cpu_count(2)
    print(json.dumps(export(args), ensure_ascii=False))


if __name__ == '__main__':
    main()
