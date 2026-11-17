"""Regression contracts for shared scientific policy and worker ownership."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from ftt.policy import paired_summary, select_candidates, validation_partition
from ftt.runtime import read_json, record_failure, worker_lock


class PolicyContracts(unittest.TestCase):
    def test_control_top_candidates_and_slow_learner_are_retained(self):
        def result(n, best, early):
            return dict(number=n, best=best, seconds=1,
                        history=[dict(step=200, ndcg=early), dict(step=800, ndcg=best)])
        results = [result(1, 1., .9), result(2, .7, .1), result(3, .9, .8),
                   result(4, .8, .7), result(5, .5, .3)]
        self.assertEqual(select_candidates(results, 1, 2), [1, 3, 4, 2])
        self.assertEqual(select_candidates(results, 1, 3), [1, 3, 4, 2, 5])
        self.assertEqual(select_candidates(results[:1], 1, 2), [1])

    def test_partition_and_paired_bootstrap_regression(self):
        rng = np.random.default_rng(2026)
        selection, fast, confirmation = validation_partition(np.array([8,1,2,10,9,9,3,4]), 9, rng)
        np.testing.assert_array_equal(selection, [0,1,2,6,7])
        np.testing.assert_array_equal(fast, [1,6])
        np.testing.assert_array_equal(confirmation, [0,2,7])
        control = np.array([[.2,.3,.4,.5],[.3,.2,.5,.4],[.4,.3,.2,.5]])
        delta = np.array([[.01,-.02,.03,.01],[.02,.01,-.01,.04],[-.01,.02,.01,.02]])
        summary = paired_summary(control+delta, control, np.array([7,7,8,9]), rng, repetitions=512)
        np.testing.assert_allclose(summary['interval'], [.0013333333333333346,.024083333333333238], atol=1e-15)
        np.testing.assert_allclose(summary['per_seed'], [.0075,.015,.01], atol=1e-15)
        self.assertTrue(summary['qualifies'])

    def test_worker_lock_prevents_duplicate_and_releases_on_exit(self):
        import os
        if os.name != 'nt':
            self.skipTest('Windows worker ownership contract')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'worker.lock'
            with worker_lock(path):
                with self.assertRaises(OSError):
                    with worker_lock(path):
                        self.fail('Second controller acquired the same lock')
            with worker_lock(path):
                pass

    def test_failure_records_budget_exhaustion_and_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = SimpleNamespace(root=Path(folder), state=dict(status='running'))
            try:
                raise TimeoutError('synthetic budget')
            except TimeoutError as error:
                record_failure(queue, error)
            state = read_json(queue.root/'status.json', {})
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertIn('TimeoutError: synthetic budget', (queue.root/'failure.txt').read_text())
