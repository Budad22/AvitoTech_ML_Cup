"""Validate a local Parquet for manual submission; never contacts the competition."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def validate(prediction, template, expected_sha256=None):
    prediction, template = Path(prediction), Path(template)
    table = pq.read_table(prediction)
    expected = {'feed_id': pa.int32(), 'item_id': pa.int32(), 'score': pa.float32()}
    if table.column_names != list(expected) or any(table.schema.field(k).type != v for k, v in expected.items()):
        raise ValueError('Required schema: feed_id:int32, item_id:int32, score:float32')
    if not len(table) or any(table[name].null_count for name in expected):
        raise ValueError('Empty predictions or null values')
    if not np.isfinite(table['score'].to_numpy()).all():
        raise ValueError('Nonfinite scores')
    reference = pq.read_table(template, columns=['feed_id', 'item_id'])
    if len(reference) != len(table):
        raise ValueError('Row count differs from the template')

    def keys(value):
        feeds = value['feed_id'].to_numpy().astype(np.uint32).astype(np.uint64)
        items = value['item_id'].to_numpy().astype(np.uint32).astype(np.uint64)
        result = np.sort((feeds << 32) | items)
        if np.any(result[1:] == result[:-1]):
            raise ValueError('Duplicate (feed_id, item_id) pairs')
        return result
    if not np.array_equal(keys(table), keys(reference)):
        raise ValueError('Candidate pairs differ from the template')
    with prediction.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError('File SHA-256 differs from the expected source checksum')
    return dict(status='ready_for_manual_submission', file=str(prediction.resolve()),
                template=str(template.resolve()), bytes=prediction.stat().st_size, rows=len(table),
                feeds=len(np.unique(table['feed_id'].to_numpy())), sha256=digest,
                checks=['schema', 'nonempty', 'nulls', 'finite_scores', 'unique_pairs', 'exact_template_coverage'],
                checked_at=datetime.now(timezone.utc).isoformat(), agent_upload_attempts=0)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    audit = validate(args.prediction, args.template, args.expected_sha256)
    if args.output:
        args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
