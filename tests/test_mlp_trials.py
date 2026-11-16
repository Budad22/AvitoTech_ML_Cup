"""Architecture, ranking, alignment, and exact-resume regression contracts."""
from argparse import Namespace
from functools import partial
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import test_data
from recsys_data import prepare_dataset, PreparedSplit
from mlp.training import select
from mlp_trials.data import batches, context_features, pack
from mlp_trials.models import KINDS, create_model, score
from mlp_trials.losses import objective
from mlp_trials.training import schema, train
from scripts.gpu_suite import winner
from scripts.gpu_suite import run as run_suite, VARIANTS
from training_logs import RunLogger


class TrialContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = test_data.DataContracts()
        self.fixture.setUp()
        self.root = self.fixture.root
        for split in ('val', 'test'):
            shutil.copytree(self.fixture.source / 'train/item_embeddings', self.fixture.source / split / 'item_embeddings')
            folder = self.fixture.source / split / 'impressions'
            folder.mkdir()
            table = test_data.feeds(split)
            table = table.set_column(0, 'feed_id', pa.array([100, 101, 102] if split == 'val' else [200, 201, 202], type=pa.int32()))
            pq.write_table(table, folder / 'part-00000.parquet')
        self.data = self.root / 'prepared'
        prepare_dataset(self.fixture.source, self.data, ('train', 'val', 'test'))
        self.categories = {2: 1, 3: 2, 4: 3, 5: 4}

    def tearDown(self):
        self.fixture.tearDown()

    def test_all_architectures_gradients_permutation_and_label_independence(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                batch = next(batches(self.data, 'train', 3, history=True))
                packed = pack(batch, self.categories, kind=kind, history_window=128)
                inputs = packed['tensors']
                model = create_model(len(self.categories), kind, 128).eval()
                scores = score(model, inputs, kind)
                self.assertEqual(tuple(scores.shape), (15,))
                self.assertTrue(torch.isfinite(scores).all())
                loss = objective(scores, inputs, kind)
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                order = torch.randperm(len(scores))
                reordered = {k: v[order] if k in ('dense', 'category', 'target', 'weights', 'feed', 'row', 'context', 'norms') else v
                             for k, v in inputs.items()}
                reordered['row'] = torch.empty_like(reordered['row'])
                for feed in range(3):
                    mask = reordered['feed'] == feed
                    reordered['row'][mask] = torch.arange(mask.sum())
                torch.testing.assert_close(score(model, reordered, kind), scores[order], atol=2e-6, rtol=2e-5)
                changed = dict(inputs, target=1 - inputs['target'])
                torch.testing.assert_close(score(model, changed, kind), scores, atol=0, rtol=0)
                empty = select(batch, np.array([1]))
                empty_inputs = pack(empty, self.categories, kind=kind, history_window=128)['tensors']
                self.assertTrue(torch.isfinite(score(model, empty_inputs, kind)).all())
                del batch, empty, packed, inputs, empty_inputs

    def test_history_missing_masks_and_signed_codes(self):
        batch = next(batches(self.data, 'train', 3, history=True))
        values = pack(batch, self.categories, kind='history_stats', history_window=8)['tensors']
        self.assertEqual(values['h_mask'].sum().item(), 6)
        self.assertEqual(values['h_present'].sum().item(), 4)
        self.assertEqual(values['h_source'][0, 0].item(), 129)  # Real source zero is not padding.
        self.assertEqual(values['h_source'][1, 0].item(), 0)
        self.assertEqual(values['h_vectors'][0, 1].abs().sum().item(), 0)
        self.assertEqual(values['h_vectors'][:, 3:].abs().sum().item(), 0)
        del batch

    def test_ranking_losses_padding_equal_feed_and_no_positive(self):
        inputs = dict(counts=torch.tensor([2, 3]), feed=torch.tensor([0, 0, 1, 1, 1]),
                      row=torch.tensor([0, 1, 0, 1, 2]), target=torch.tensor([1., 0., 1., 0., 0.]))
        for kind in ('softmax', 'lambda'):
            with self.subTest(kind=kind):
                zero = torch.zeros(5, requires_grad=True)
                initial = objective(zero, inputs, kind)
                initial.backward()
                self.assertTrue((zero.grad[inputs['target'] == 1] < 0).all())
                self.assertTrue((zero.grad[inputs['target'] == 0] > 0).all())
                improved = objective(torch.tensor([2., -2., 2., -2., -2.]), inputs, kind)
                self.assertLess(improved.item(), initial.item())
                empty = dict(inputs, target=torch.zeros(5))
                loss = objective(torch.zeros(5, requires_grad=True), empty, kind)
                self.assertEqual(loss.item(), 0)
                loss.backward()
        expected = (np.log(2) + np.log(3)) / 2
        self.assertAlmostEqual(objective(torch.zeros(5), inputs, 'softmax').item(), expected, places=6)

    def test_context_ties_and_permutation(self):
        dense = np.zeros((4, 519), dtype=np.float32)
        dense[:, :5] = np.array([1., 1., 3., 2.])[:, None]
        categories = np.array([1, 1, 2, 1])
        expected = context_features(dense, categories, [4])
        np.testing.assert_allclose(expected[:, 0], [1/6, 1/6, 1, 2/3])
        order = np.array([3, 0, 2, 1])
        np.testing.assert_array_equal(context_features(dense[order], categories[order], [4]), expected[order])

    def test_resume_matches_uninterrupted_training_and_skips_shards(self):
        reference = self.root / 'schema.json'
        schema(self.data, reference)
        def arguments(output, resume=None):
            return Namespace(output=output, data=self.data, schema=reference, resume=resume,
                             seed=42, kind='history_attention', history_window=8, device='cpu',
                             batch_size=1, epochs=1, hours=1., learning_rate=.001)
        uninterrupted, interrupted = self.root / 'full', self.root / 'resume'
        with RunLogger(uninterrupted, 'full', 'test', echo=False) as logger:
            train(arguments(uninterrupted), logger)
        interrupted.mkdir()
        (interrupted / 'STOP_AFTER_CHECKPOINT').touch()
        with RunLogger(interrupted, 'resume', 'test', echo=False) as logger:
            train(arguments(interrupted), logger)
        state = torch.load(interrupted / 'last.pt', weights_only=False)
        self.assertEqual(state['cursor'], 1)
        (interrupted / 'STOP_AFTER_CHECKPOINT').unlink()
        with RunLogger(interrupted, 'resume', 'test', echo=False) as logger:
            train(arguments(interrupted, interrupted / 'last.pt'), logger)
        full = torch.load(uninterrupted / 'last.pt', weights_only=False)
        resumed = torch.load(interrupted / 'last.pt', weights_only=False)
        for key in full['model']:
            torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
        all_batches = list(batches(self.data, 'train', 1, seed=43, history=True))
        remaining = list(batches(self.data, 'train', 1, seed=43, start_batch=2, history=True))
        self.assertEqual(remaining[0].features.to_pydict(), all_batches[2].features.to_pydict())
        del all_batches, remaining

    def test_winner_uses_only_successful_selection_scores(self):
        selected = winner([dict(status='completed', selection_ndcg=.4, control_ndcg=.1, kind='a'),
                           dict(status='completed', selection_ndcg=.3, control_ndcg=.9, kind='b'),
                           dict(status='failed', selection_ndcg=1, kind='c')])
        self.assertEqual(selected['kind'], 'a')
        self.assertIsNone(winner([dict(status='failed')]))

    def test_queue_continues_after_case_failure_and_resumes_without_retraining(self):
        directory = self.root / 'queue'
        directory.mkdir()
        (directory / 'schema.json').write_text('{"fingerprint":"fixture"}')
        template = pa.table({'feed_id': pa.array([200], type=pa.int32()), 'item_id': pa.array([101], type=pa.int32())})
        pq.write_table(template, self.fixture.source / 'sample_submission.parquet')
        commands = []
        class Child:
            pid = 999999
            def __init__(child, arguments, **kwargs):
                commands.append(arguments)
                target = Path(arguments[arguments.index('--output') + 1])
                child.code = 0
                if 'train' in arguments:
                    kind = arguments[arguments.index('--kind') + 1]
                    if kind == 'history_pool':
                        child.code = 1
                    else:
                        (target / 'best.pt').write_bytes(b'fixture')
                        (target / 'result.json').write_text(json.dumps(dict(status='completed', kind=kind,
                                       selection_ndcg=.9 if kind == 'baseline' else .2)))
                elif 'evaluate' in arguments:
                    target.write_text('{"ndcg":0.9}')
                elif 'predict' in arguments:
                    pq.write_table(template.append_column('score', pa.array([.1], type=pa.float32())), target)
            def wait(child):
                return child.code
        args = Namespace(directory=directory, data=self.data, epochs=5, hours=2, batch_size=64, active_root=None)
        with patch('scripts.gpu_suite.subprocess.Popen', Child):
            run_suite(args)
            count = len(commands)
            run_suite(args)
        state = json.loads((directory / 'status.json').read_text())
        self.assertEqual(state['status'], 'completed_with_errors')
        self.assertEqual(state['completed'], 10)
        self.assertEqual(state['failed'], 1)
        self.assertEqual(len(commands), count)
        self.assertEqual(sum('train' in c for c in commands), len(VARIANTS))
        self.assertEqual(sum('control' in c for c in commands), 1)
        self.assertEqual(json.loads((directory / 'report.json').read_text())['winner']['kind'], 'baseline')
        screening = self.root / 'screening'
        screening.mkdir()
        (screening / 'schema.json').write_text('{"fingerprint":"fixture"}')
        args.directory = screening
        args.selection_only = True
        args.run_prefix = 'EXP-026-small'
        args.device = 'cpu'
        commands.clear()
        with patch('scripts.gpu_suite.subprocess.Popen', Child):
            run_suite(args)
        self.assertFalse(any('evaluate' in command or 'predict' in command for command in commands))
        self.assertTrue(all(command[command.index('--device') + 1] == 'cpu' for command in commands))
        self.assertTrue((screening / 'report.json').exists())
        self.assertFalse((screening / 'submission-audit.json').exists())
