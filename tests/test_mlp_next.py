"""New history signals preserve labels, masks, candidate symmetry and controls."""
import unittest
import numpy as np
import torch

from mlp_next.models import configurations, HistoryRanker
from mlp_next.runner import cached_order
from mlp_next.streaming import sequential_order
from mlp_research.models import ResearchRanker, loss
from test_research_models import statistics


class NextMLPContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_history_optimizer_update(self):
        x = dict(dense=torch.rand(2, 519, device='cuda'), category=torch.tensor([1, 2], device='cuda'),
                 history=torch.rand(2, 529, device='cuda'), context=torch.rand(2, 17, device='cuda'),
                 counts=torch.tensor([2]), weights=torch.tensor([.5, .5], device='cuda'),
                 target=torch.tensor([1., 0.], device='cuda'),
                 h_vectors=torch.rand(1, 256, 128, device='cuda').half(),
                 h_mask=torch.ones(1, 256, dtype=torch.bool, device='cuda'),
                 h_present=torch.ones(1, 256, dtype=torch.bool, device='cuda'),
                 h_category=torch.ones(1, 256, dtype=torch.long, device='cuda'),
                 h_type=torch.ones(1, 256, dtype=torch.long, device='cuda'),
                 h_source=torch.ones(1, 256, dtype=torch.long, device='cuda'))
        for name, config in configurations().items():
            with self.subTest(name=name):
                model = HistoryRanker(5, config, statistics()).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                before = model.network[-1].weight.detach().clone()
                value = loss(model(x), x, config['loss'])
                value.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(value))
                self.assertFalse(torch.equal(before, model.network[-1].weight))

    def test_sequence_cache_streaming_and_blend_alignment(self):
        import test_mlp_trials
        from mlp_research.cache import build, CachedSplit
        from mlp_next.data import build_sequences, SequenceSplit
        from mlp_next.streaming import StreamingSplit
        from mlp_next.diagnostics import exposure
        from mlp_next.blend import aligned_scores, keys
        import pyarrow as pa
        import pyarrow.parquet as pq
        fixture = test_mlp_trials.TrialContracts()
        fixture.setUp()
        try:
            cache = fixture.root / 'cache'
            build(fixture.data, cache, lambda *a, **k: None)
            build_sequences(fixture.data, cache, fixture.root / 'sequences', lambda *a, **k: None)
            dataset = SequenceSplit(cache, 'train')
            expected = dataset.batch([0, 1, 2])
            stream = StreamingSplit(fixture.data, fixture.categories, configurations()['stats_din128'], 'cpu')
            actual = stream.batch([0, 1, 2])
            for name in ('dense', 'history', 'category', 'target', 'counts', 'weights',
                         'h_vectors', 'h_mask', 'h_present', 'h_category', 'h_type', 'h_source'):
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            self.assertEqual(sum(exposure(fixture.data)['counts']), 15)
            path = fixture.root / 'predictions.parquet'
            pq.write_table(pa.table(dict(feed_id=[2, 1], item_id=[20, 10], score=[.2, .1])), path)
            np.testing.assert_allclose(aligned_scores(path, keys([1, 2], [10, 20])), [.1, .2])
            with self.assertRaises(ValueError):
                aligned_scores(path, keys([1, 2], [11, 20]))
            dataset.close()
            stream.close()
            from mlp_research.training import Trainer
            config = configurations()['stats_din128']
            first = Trainer(cache, fixture.root / 'resumed', lambda *a, **k: None,
                            model_class=HistoryRanker, split_class=SequenceSplit, order_fn=cached_order)
            first.run(config, 42, 20)
            first.run(config, 42, 30)
            first.close()
            direct = Trainer(cache, fixture.root / 'direct', lambda *a, **k: None,
                             model_class=HistoryRanker, split_class=SequenceSplit, order_fn=cached_order)
            direct.run(config, 42, 30)
            direct.close()
            left = torch.load(fixture.root / 'resumed/runs/stats_din128-s42/last.pt', weights_only=False)
            right = torch.load(fixture.root / 'direct/runs/stats_din128-s42/last.pt', weights_only=False)
            for key in left['model']:
                torch.testing.assert_close(left['model'][key], right['model'][key], atol=0, rtol=0)
        finally:
            fixture.tearDown()

    def test_controls_preserve_initialization_and_new_models_are_finite(self):
        torch.set_num_threads(2)
        x = dict(dense=torch.rand(5, 519), category=torch.tensor([0, 1, 2, 3, 4]),
                 history=torch.rand(5, 529), context=torch.rand(5, 17),
                 counts=torch.tensor([2, 3]), weights=torch.tensor([.5, .5, 1/3, 1/3, 1/3]),
                 target=torch.tensor([1., 0., 0., 1., 0.]),
                 h_vectors=torch.rand(2, 256, 128).half(),
                 h_mask=torch.cat((torch.zeros(1, 256, dtype=torch.bool), torch.ones(1, 256, dtype=torch.bool))),
                 h_present=torch.ones(2, 256, dtype=torch.bool),
                 h_category=torch.ones(2, 256, dtype=torch.long),
                 h_type=torch.ones(2, 256, dtype=torch.long), h_source=torch.ones(2, 256, dtype=torch.long))
        order = torch.tensor([1, 0, 4, 2, 3])
        for name, config in configurations().items():
            with self.subTest(name=name):
                torch.manual_seed(42)
                model = HistoryRanker(5, config, statistics()).eval()
                torch.manual_seed(42)
                baseline = ResearchRanker(5, config, statistics()).eval()
                # Zero-initialized correction exactly preserves the strong base.
                torch.testing.assert_close(model(x), baseline(x), atol=0, rtol=0)
                if config['sequence']:
                    with torch.no_grad():
                        model.correction[-1].weight.fill_(.01)
                score = model(x)
                self.assertTrue(torch.isfinite(score).all())
                objective = loss(score, x, config['loss'])
                objective.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                permuted = {k: v if k.startswith('h_') or k == 'counts' else v[order] for k, v in x.items()}
                torch.testing.assert_close(model(permuted), score[order], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(model(dict(x, target=1-x['target'])), score, atol=0, rtol=0)
                if config['sequence']:
                    changed = dict(x, h_vectors=x['h_vectors'].clone())
                    changed['h_vectors'][0] = 99
                    torch.testing.assert_close(model(changed)[:2], score[:2], atol=0, rtol=0)

    def test_cached_order_matches_old_shuffle_and_sequential_covers_full_epoch(self):
        from mlp_research.training import feed_order
        for step in range(9):
            np.testing.assert_array_equal(cached_order(37, 42, step, 16), feed_order(37, 42, step, 16))
        order = np.concatenate([sequential_order(37, 42, step, 16) for step in range(3)])
        np.testing.assert_array_equal(order, np.arange(37))
