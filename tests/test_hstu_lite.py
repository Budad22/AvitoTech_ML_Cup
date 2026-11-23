import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hstu_lite.common import average_ranks, ndcg_full
from hstu_lite.nn_train import Ranker, collate, lambda_loss, softmax_loss
from hstu_lite.monitor import Progress
from mlp.metrics import ndcg


class HstuLiteContracts(unittest.TestCase):
    def test_monitor_recovers_from_windows_reader_lock_without_rewriting_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            monitor = Progress(root)
            replace = os.replace
            stage_replacements = []

            def locked_once(source, destination):
                if destination.name == 'stage.txt':
                    stage_replacements.append(destination)
                    if len(stage_replacements) == 1:
                        raise PermissionError(13, 'WinError 5 simulated')
                replace(source, destination)

            with patch('mlp_trials.state.os.replace', side_effect=locked_once):
                monitor.publish('train', step=1)
                monitor.publish('train', step=2)
            self.assertEqual((root / 'stage.txt').read_text(), 'train')
            self.assertEqual(len(stage_replacements), 2)

    def test_prolonged_status_lock_leaves_previous_snapshot_and_training_continues(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            monitor = Progress(root)
            monitor.publish('train', step=1)
            with patch('mlp_trials.state.os.replace', side_effect=PermissionError(13, 'WinError 5 simulated')), \
                    patch('mlp_trials.state.time.sleep'), self.assertLogs('hstu_lite.monitor', 'WARNING'):
                monitor.publish('train', step=2)
            status = root / 'status.json'
            self.assertEqual(json.loads(status.read_text())['step'], 1)
            monitor.publish('train', step=3)
            self.assertEqual(json.loads(status.read_text())['step'], 3)

    def test_equal_ranks_and_full_ndcg_match_existing_evaluator(self):
        np.testing.assert_array_equal(average_ranks(np.array([3, 1, 3, 2])), [2.5, 0, 2.5, 1])
        scores = np.array([2., 2., 1., 0., 0.])
        labels = np.array([1, 0, 1, 0, 0])
        value, _ = ndcg_full(np.array([0, 3, 5]), scores, labels)
        self.assertAlmostEqual(value, (ndcg(labels[:3], scores[:3]) + 1.) / 2)

    def test_masked_ranking_trains_and_inference_ignores_position_and_candidate_order(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(99)
        def feed(count, history):
            return dict(f=0, ce=rng.integers(-16, 16, (count, 128), dtype=np.int8),
                        cp=np.ones(count, dtype=bool), ccat=np.arange(count) + 2,
                        citem=np.arange(count) + 10, num=rng.normal(size=(count, 19)).astype(np.float32),
                        y=np.arange(count) == 0, pos=np.arange(count),
                        he=np.zeros((history, 128), np.int8), hp=np.zeros(history, bool),
                        hcat=np.arange(history) + 2, het=np.ones(history, int),
                        hes=np.zeros(history, int), hitem=np.arange(history) + 10,
                        u=np.ones(128, np.int8))
        inputs = collate([feed(4, 2), feed(2, 0)])
        model = Ranker(d=16, nh=4, n_hist=1, n_cross=1, n_list=1, drop=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        score = model(inputs, use_pos=True)
        loss = softmax_loss(score, inputs['y'], inputs['cmask']) + lambda_loss(score, inputs['y'], inputs['cmask'])
        self.assertTrue(torch.isfinite(loss))
        before = model.head[-1].weight.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head[-1].weight))
        self.assertGreater(model.pos_bias.weight.abs().sum().item(), 0)
        model.eval()
        order = torch.tensor([2, 0, 3, 1])
        permuted = dict(inputs)
        for key in ('ce', 'cp', 'ccat', 'citem', 'num', 'cmask'):
            permuted[key] = inputs[key].clone()
            permuted[key][0] = inputs[key][0, order]
        permuted.pop('pos')
        with torch.inference_mode():
            original = model(inputs)
            torch.testing.assert_close(model(inputs, use_pos=True), original)
            torch.testing.assert_close(model(permuted)[0], original[0, order], atol=1e-5, rtol=1e-5)
