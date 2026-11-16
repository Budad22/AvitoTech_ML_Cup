"""Cached training preserves features, whole feeds, and sealed late validation."""
import json
import unittest

import numpy as np
import torch

import test_mlp_trials
from mlp_research.cache import build, CachedSplit, fast_partition
from mlp_trials.data import batches, pack
from mlp_trials.models import history_stats


class ResearchCacheContracts(unittest.TestCase):
    def test_cache_matches_stream_and_never_contains_control(self):
        fixture = test_mlp_trials.TrialContracts()
        fixture.setUp()
        try:
            cache = fixture.root / 'cache'
            info = build(fixture.data, cache, lambda *args, **kwargs: None)
            train = CachedSplit(cache, 'train')
            batch = next(batches(fixture.data, 'train', 3, history=True))
            expected = pack(batch, fixture.categories, kind='history_stats', history_window=128)['tensors']
            actual = train.batch([0, 1, 2])
            for key in ('dense', 'category', 'target', 'weights', 'counts'):
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            torch.testing.assert_close(actual['history'], history_stats(expected), atol=0, rtol=0)
            shuffled = train.batch([2, 0])
            torch.testing.assert_close(shuffled['dense'], torch.cat((actual['dense'][10:15], actual['dense'][:5])))
            validation = CachedSplit(cache, 'val')
            self.assertTrue((validation.arrays['timestamp'] < info['boundary']).all())
            fast = np.load(cache / 'fast.npy')
            confirmation = np.load(cache / 'confirmation.npy')
            self.assertFalse(set(fast) & set(confirmation))
            self.assertEqual(sorted([*fast, *confirmation]), list(range(validation.feeds)))
            self.assertEqual(build(fixture.data, cache, lambda *args, **kwargs: None)['signature'], info['signature'])
            manifest = fixture.data / 'manifest.json'
            changed = json.loads(manifest.read_text())
            changed['extra'] = 'changed'
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'signature'):
                build(fixture.data, cache, lambda *args, **kwargs: None)
            del batch
            train.close()
            validation.close()
        finally:
            fixture.tearDown()

    def test_frozen_partition_is_deterministic_and_metadata_only(self):
        metadata = dict(feed_id=np.arange(100), timestamp=np.arange(100) * 3600,
                        history_length=np.arange(100) % 20, candidate_count=np.arange(100) % 15 + 5)
        first = fast_partition(metadata, 30)
        second = fast_partition(metadata, 30)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(len(first[0]), 30)
        self.assertEqual(len(first[1]), 70)
