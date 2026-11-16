"""The complete research catalog must be finite, label-independent, and equivariant."""
import unittest

import numpy as np
import torch

from mlp_research.catalog import hypotheses, promote, CONTROLS
from mlp_research.models import ResearchRanker, loss
from mlp_research.runner import finalists
from mlp_trials.models import create_model


def statistics():
    return dict(mean=np.zeros(22, 'float32'), scale=np.ones(22, 'float32'),
                quantiles=np.tile(np.linspace(0, 1, 257, dtype='float32'), (5, 1)))


class ResearchModelContracts(unittest.TestCase):
    def test_all_models_gradients_and_candidate_permutation(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        inputs = dict(dense=torch.rand(12, 519), category=torch.arange(12) % 5,
                      history=torch.rand(12, 529), context=torch.rand(12, 17),
                      target=torch.tensor([1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
                      counts=torch.tensor([5, 7]), weights=torch.tensor([1 / 5] * 5 + [1 / 7] * 7))
        order = torch.tensor([3, 0, 4, 1, 2, 10, 6, 11, 8, 7, 9, 5])
        for config in hypotheses():
            with self.subTest(name=config['name']):
                model = ResearchRanker(5, config, statistics()).eval()
                logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (12,))
                objective = loss(logits, inputs, config['loss'])
                self.assertTrue(torch.isfinite(objective))
                objective.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                permuted = {name: value if name == 'counts' else value[order] for name, value in inputs.items()}
                torch.testing.assert_close(model(permuted), logits[order], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(model(dict(inputs, target=1 - inputs['target'])), logits, atol=0, rtol=0)

    def test_controls_reproduce_existing_network_initialization(self):
        configs = {c['name']: c for c in hypotheses()}
        inputs = dict(dense=torch.rand(4, 519), category=torch.tensor([0, 1, 2, 3]), history=torch.rand(4, 529))
        for name, kind in [('baseline', 'baseline'), ('history_control', 'history_stats')]:
            torch.manual_seed(42)
            old = create_model(5, kind).eval()
            torch.manual_seed(42)
            new = ResearchRanker(5, configs[name], statistics()).eval()
            for before, after in zip(old.parameters(), new.parameters()):
                torch.testing.assert_close(before, after, atol=0, rtol=0)
            columns = [inputs['dense'], old.category(inputs['category'])]
            if name == 'history_control':
                columns.append(inputs['history'])
            expected = old.network(torch.cat(columns, dim=1)).squeeze(-1)
            torch.testing.assert_close(new(inputs), expected, atol=0, rtol=0)

    def test_promotions_keep_families_controls_and_slow_learners(self):
        catalog = hypotheses()
        results = [dict(name=c['name'], status='completed', fast_ndcg=.3 - i / 1000)
                   for i, c in enumerate(catalog)]
        selected = promote(results, catalog)
        self.assertTrue(set(CONTROLS).issubset(selected))
        self.assertIn('feature_transformer2', selected)
        self.assertIn('plr_low_lr', selected)
        by_name = {c['name']: c for c in catalog}
        self.assertTrue({'plr', 'encoding', 'mlp', 'alternative'}.issubset({by_name[n]['family'] for n in selected}))
        final = finalists(results, catalog)
        self.assertTrue(set(CONTROLS).issubset(final))
        self.assertEqual(len(final), 7)
