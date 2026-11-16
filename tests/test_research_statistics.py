"""Architecture decisions require paired confirmation evidence, not a high training score."""
import unittest

import numpy as np

from mlp_research.statistics import conclusion, paired_interval


class ResearchStatisticalContracts(unittest.TestCase):
    def test_seed_user_bootstrap_zero_and_multiple_comparisons(self):
        users = np.repeat(np.arange(20), 2)
        control = np.full((3, 40), .3)
        null = paired_interval(control, control, users, repetitions=300)
        self.assertEqual(null['delta'], 0)
        self.assertFalse(null['evidence'])
        candidate = control + np.linspace(.01, .03, 40)[None, :]
        raw = paired_interval(candidate, control, users, repetitions=500)
        corrected = paired_interval(candidate, control, users, comparisons=20, repetitions=500)
        self.assertTrue(corrected['evidence'])
        self.assertLessEqual(corrected['lower'], raw['lower'])
        self.assertGreaterEqual(corrected['upper'], raw['upper'])
        with self.assertRaises(ValueError):
            paired_interval(candidate, control, users[:-1])

    def test_decisions_keep_inconclusive_outcome_and_incomplete_runs(self):
        control = np.full((3, 30), .3)
        families = dict(history_control='control', modified='plr', transformer='alternative')
        users = np.arange(30)
        fast = dict(history_control=.3, modified=.32, transformer=.31)
        metrics = dict(history_control=control, modified=control + .02, transformer=control + .01)
        self.assertEqual(conclusion(metrics, users, families, fast)['decision'], 'continue_mlp')
        metrics['transformer'] = control + .05
        self.assertEqual(conclusion(metrics, users, families, fast)['decision'], 'prioritize_alternative')
        self.assertEqual(conclusion(metrics, users, families, fast, incomplete=True)['decision'], 'inconclusive_incomplete')
        metrics['modified'] = control.copy()
        metrics['transformer'] = control.copy()
        self.assertEqual(conclusion(metrics, users, families, fast)['decision'], 'inconclusive_no_confirmed_gain')
