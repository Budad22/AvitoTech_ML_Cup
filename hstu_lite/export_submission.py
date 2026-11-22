"""Package existing NN-1 test logits as a verified manual submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mlp_trials.state import write_json
from scripts.validate_submission import validate
from .common import Split


def export(run, seed, template, output):
    work = run / 'work'
    name = f'nn1_s{seed}'
    result = json.loads((work / name / 'final.json').read_text(encoding='utf-8'))
    if result['seed'] != seed:
        raise ValueError('Seed differs from the completed evaluation')
    checkpoint = work / f'{name}.pt'
    test = Split('test')
    scores = np.load(work / f'{name}_test.npy', allow_pickle=False, mmap_mode='r')
    if scores.shape != (test.N,) or scores.dtype != np.float32:
        raise ValueError('Test logits must be a float32 vector covering every candidate')
    table = pa.table({'feed_id': pa.array(np.repeat(test.feed_id, np.diff(test.c_off)), type=pa.int32()),
                      'item_id': pa.array(test.c_item, type=pa.int32()),
                      'score': pa.array(scores, type=pa.float32())})
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'submission.tmp.parquet'
    submission = output / 'submission.parquet'
    try:
        pq.write_table(table, temporary)
        audit = validate(temporary, template)
        with checkpoint.open('rb') as stream:
            checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
        metadata = dict(run_id=run.name, checkpoint_sha256=checkpoint_sha256,
                        prediction_sha256=audit['sha256'], metrics=result, rows=audit['rows'],
                        feeds=audit['feeds'], split='test', checks=audit['checks'])
        write_json(output / 'submission.json', metadata)
        os.replace(temporary, submission)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.seed, args.template, args.output)))


if __name__ == '__main__':
    main()
