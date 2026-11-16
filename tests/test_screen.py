"""Nested sample views must preserve full inputs and keep promotion free of control scores."""
import hashlib
import json
import unittest
from argparse import Namespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from recsys_data import PreparedSplit
from scripts.mlp_screen import promotion, sample_indices, sample_view, run
from scripts.gpu_suite import VARIANTS
import test_mlp_trials


class ScreenContracts(unittest.TestCase):
    def test_two_screens_then_only_larger_sample_winner_on_full_data(self):
        fixture = test_mlp_trials.TrialContracts()
        fixture.setUp()
        try:
            commands = []
            class Child:
                pid = 999999
                returncode = 0
                def __init__(self, command, **kwargs):
                    commands.append(command)
                    output = fixture.root / command[command.index('--directory') + 1]
                    output.mkdir(exist_ok=True)
                    stage = output.name
                    kinds = [command[-1]] if stage == 'full' else VARIANTS
                    cases = [dict(kind=kind, run_id=kind, status='completed', selection_ndcg=
                                  .9 if kind == ('softmax' if stage == '1p56' else 'baseline') else .2) for kind in kinds]
                    (output / 'report.json').write_text(json.dumps(dict(cases=cases)))
                def poll(self):
                    return 0
            args = Namespace(directory=fixture.root / 'suite', data=fixture.data, screen_hours=.2, active_root=None)
            with patch('scripts.mlp_screen.sample_indices', return_value=([0], [0])), \
                    patch('scripts.mlp_screen.sample_view'), patch('scripts.mlp_screen.subprocess.Popen', Child):
                run(args)
                run(args)
            self.assertEqual(len(commands), 3)
            self.assertTrue(all('--selection-only' in c for c in commands[:2]))
            self.assertNotIn('--selection-only', commands[2])
            self.assertEqual(commands[2][-2:], ['--variants', 'softmax'])
            self.assertEqual(json.loads((args.directory / 'status.json').read_text())['completed'], 23)
            args.directory = fixture.root / 'gpu-only'
            args.stages = ['1p56', 'full']
            commands.clear()
            with patch('scripts.mlp_screen.sample_indices', return_value=([0], [0])), \
                    patch('scripts.mlp_screen.sample_view'), patch('scripts.mlp_screen.subprocess.Popen', Child):
                run(args)
            self.assertEqual(len(commands), 2)
            self.assertEqual(json.loads((args.directory / 'status.json').read_text())['completed'], 12)
        finally:
            fixture.tearDown()

    def test_nested_indices_and_promotion_gate(self):
        small, large = sample_indices(512)
        self.assertEqual(small, [0, 170, 341, 511])
        self.assertEqual(len(large), 8)
        self.assertTrue(set(small) < set(large))
        cases = [dict(kind=kind, stage='1p56', status='completed', selection_ndcg=.2 + index / 100)
                 for index, kind in enumerate(VARIANTS)]
        cases += [dict(kind='baseline', stage='0p78', status='completed', selection_ndcg=.99)]
        self.assertEqual(promotion(cases)['kind'], 'low_lr')
        with self.assertRaises(ValueError):
            promotion(cases[1:])

    def test_sample_preserves_source_labels_history_and_validation(self):
        fixture = test_mlp_trials.TrialContracts()
        fixture.setUp()
        try:
            root = fixture.data
            manifest_path = root / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            original = manifest['splits']['train']['files'][0]
            files = []
            for index in range(8):
                name = f'screen-{index}.parquet'
                for kind in ('features', 'labels', 'diagnostics'):
                    source = root / 'train' / kind / original['name']
                    if source.exists():
                        table = pq.read_table(source)
                        table = table.set_column(table.schema.get_field_index('feed_id'), 'feed_id',
                                                 pa.array([10 + index * 3 + j for j in range(3)], type=pa.int32()))
                        pq.write_table(table, source.with_name(name))
                files.append(dict(original, name=name))
            manifest['splits']['train'].update(files=files, feeds=24, candidates=120, history_events=48)
            manifest_path.write_text(json.dumps(manifest))
            before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            small_indices, large_indices = sample_indices(8)
            small = fixture.root / 'small'
            large = fixture.root / 'large'
            first = sample_view(root, small, small_indices)
            second = sample_view(root, large, large_indices)
            self.assertEqual(first['splits']['train']['feeds'], 12)
            self.assertEqual(second['splits']['train']['feeds'], 24)
            self.assertEqual(first['splits']['val'], manifest['splits']['val'])
            sampled = list(PreparedSplit(small, 'train').iter_batches(3, include_labels=True))
            self.assertEqual(sum(len(batch.features) for batch in sampled), 12)
            self.assertTrue(all(batch.labels is not None for batch in sampled))
            self.assertIn('history__event_type', sampled[0].features.column_names)
            self.assertEqual(before, hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertEqual(sample_view(root, small, small_indices), first)
            with self.assertRaises(ValueError):
                sample_view(root, small, large_indices)
            with self.assertRaises(ValueError):
                sample_view(root, fixture.root / 'invalid', [0, 0])
            del sampled
        finally:
            fixture.tearDown()
