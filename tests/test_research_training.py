"""Step continuation must preserve optimizer, dropout RNG, data order, and budget."""
import unittest

import torch

import test_mlp_trials
from mlp_research.cache import build
from mlp_research.catalog import hypotheses
from mlp_research.training import Trainer, feed_order


class ResearchTrainingContracts(unittest.TestCase):
    def test_cached_resume_is_identical_to_uninterrupted_steps(self):
        fixture = test_mlp_trials.TrialContracts()
        fixture.setUp()
        first = resumed = None
        try:
            cache = fixture.root / 'cache'
            build(fixture.data, cache, lambda *args, **kwargs: None)
            config = next(c for c in hypotheses() if c['name'] == 'plr_raw_skip')
            first = Trainer(cache, fixture.root / 'first', lambda *args, **kwargs: None, batch_size=1)
            resumed = Trainer(cache, fixture.root / 'resumed', lambda *args, **kwargs: None, batch_size=1)
            first.run(config, 42, 4)
            resumed.run(config, 42, 2)
            resumed.run(config, 42, 4)
            before = torch.load(fixture.root / 'first/runs/plr_raw_skip-s42/last.pt', weights_only=False)
            after = torch.load(fixture.root / 'resumed/runs/plr_raw_skip-s42/last.pt', weights_only=False)
            for key in before['model']:
                torch.testing.assert_close(before['model'][key], after['model'][key], atol=0, rtol=0)
            self.assertEqual(after['step'], 4)
            self.assertEqual(after['optimizer']['state'][0]['step'], 4)
            final = resumed.final(config, 42)
            self.assertGreaterEqual(final['confirmation_ndcg'], 0)
            self.assertGreater(final['best_step'], 0)
        finally:
            if first:
                first.close()
            if resumed:
                resumed.close()
            fixture.tearDown()

    def test_step_order_covers_whole_feeds_and_repeats_deterministically(self):
        self.assertEqual(feed_order(7, 42, 0, 3).tolist(), feed_order(7, 42, 0, 3).tolist())
        order = [i for step in range(3) for i in feed_order(7, 42, step, 3)]
        self.assertEqual(sorted(order), list(range(7)))
