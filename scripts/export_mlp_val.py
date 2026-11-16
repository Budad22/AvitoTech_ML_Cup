"""Export all official val logits and provenance for a keyed model ensemble."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mlp.metrics import ndcg
from mlp_trials.state import write_json
from recsys_data import PreparedSplit
from recsys_data.core import _flat


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def packed_keys(feeds, items):
    return (feeds.astype(np.uint32).astype(np.uint64) << 32) | items.astype(np.uint32).astype(np.uint64)


def validate_val(prediction, data, *, expected_rows=1389971, expected_feeds=14423):
    """Align by keys against official labels; no dependence on prediction row order."""
    table = pq.read_table(prediction)
    expected = pa.schema([('feed_id', pa.int32()), ('item_id', pa.int32()), ('score', pa.float32())])
    if not table.schema.equals(expected, check_metadata=False):
        raise ValueError('Required schema: feed_id:int32, item_id:int32, score:float32')
    if len(table) != expected_rows or any(column.null_count for column in table.columns):
        raise ValueError('Wrong row count or null predictions')
    scores = table['score'].to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    prediction_keys = packed_keys(table['feed_id'].to_numpy(), table['item_id'].to_numpy())
    order = np.argsort(prediction_keys)
    prediction_keys = prediction_keys[order]
    if np.any(prediction_keys[1:] == prediction_keys[:-1]):
        raise ValueError('Duplicate prediction pairs')
    reference_keys, labels = [], []
    for batch in PreparedSplit(data, 'val').iter_batches(256, include_labels=True,
            feature_columns=['feed_id', 'item_id', 'candidate_count']):
        feeds = np.repeat(batch.features['feed_id'].to_numpy(), batch.candidate_counts)
        reference_keys.append(packed_keys(feeds, _flat(batch.features, 'item_id').to_numpy()))
        labels.append(_flat(batch.labels, 'target').to_numpy())
    reference_keys, labels = np.concatenate(reference_keys), np.concatenate(labels)
    reference_order = np.argsort(reference_keys)
    if not np.array_equal(prediction_keys, reference_keys[reference_order]):
        raise ValueError('Prediction pairs do not exactly cover official val')
    feed_keys = prediction_keys >> 32
    offsets = np.r_[0, np.flatnonzero(feed_keys[1:] != feed_keys[:-1]) + 1, len(feed_keys)]
    if len(offsets) - 1 != expected_feeds:
        raise ValueError('Wrong feed count')
    labels, scores = labels[reference_order], scores[order]
    value = sum(ndcg(labels[start:end], scores[start:end]) for start, end in zip(offsets[:-1], offsets[1:])) / expected_feeds
    return dict(rows=len(table), feeds=expected_feeds, full_val_ndcg=value, prediction_sha256=sha256(prediction),
                checks=['schema', 'row_count', 'feed_count', 'nulls', 'finite_logits', 'unique_pairs',
                        'exact_official_val_key_coverage', 'ndcg_recomputed_from_saved_logits'])


def export(args):
    args.output.mkdir(parents=True, exist_ok=True)
    prediction = args.output / 'mlp_val_predictions.parquet'
    metadata = args.output / 'mlp_val_predictions.json'
    if prediction.exists() or metadata.exists():
        raise FileExistsError('Use a fresh output directory; exports are immutable')
    torch.set_num_threads(args.cpu_threads)
    pa.set_cpu_count(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    reference = state.get('reference', state)
    if sha256(args.data / 'manifest.json') != reference['fingerprint']:
        raise ValueError('Checkpoint and prepared manifest differ')
    if 'reference' in state:
        from mlp_trials.models import create_model
        from mlp_trials.training import evaluate
        kind, window = state['config']['kind'], state['config']['history_window']
        model = create_model(len(reference['categories']), kind, window).to(args.device)
        model.load_state_dict(state['model'])
        metrics = evaluate(model, args.data, reference, args.device, kind, window, args.batch_size,
                           part='all', output=prediction, split='val')
    else:
        from mlp.model import Ranker
        from mlp.training import evaluate
        kind = 'baseline'
        model = Ranker(len(reference['categories'])).to(args.device)
        model.load_state_dict(state['model'])
        metrics = evaluate(model, args.data, reference['categories'], args.device, reference['boundary'],
                           part='all', batch_size=args.batch_size, output=prediction, split='val')
    audit = validate_val(prediction, args.data)
    if abs(audit['full_val_ndcg'] - metrics['ndcg']) > 1e-12:
        raise ValueError('Recomputed NDCG differs from inference metric')
    sources = {}
    for folder in ('mlp', 'mlp_trials', 'recsys_data'):
        sources.update({path.as_posix(): sha256(path) for path in Path(folder).glob('*.py')})
    result = dict(run_id=args.run_id, architecture=kind, selected_epoch=state['epoch'],
                  selection_ndcg_at_checkpoint=state['best'], checkpoint_sha256=sha256(args.checkpoint),
                  code_commit=args.code_commit, inference_source_sha256=sources,
                  export_script_sha256=sha256(__file__), split='official_val', partition='all',
                  score_type='raw_logits', score_dtype='float32', sigmoid_applied=False, rounded=False,
                  selection_control_boundary=reference['boundary'],
                  selection_rule='timestamp < boundary', control_rule='timestamp >= boundary',
                  partitions=reference['sizes'], prepared_manifest_sha256=reference['fingerprint'],
                  device=args.device, metrics=metrics, created_at=datetime.now(timezone.utc).isoformat(),
                  **audit)
    write_json(metadata, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--code-commit', required=True, help='Training model source revision, not export date')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=2)
    export(parser.parse_args())
