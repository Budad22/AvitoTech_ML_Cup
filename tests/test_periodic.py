"""Numerical PLR contracts beyond the common ranking-model checks."""
import io
import math
import unittest

import torch

from mlp_trials.periodic import PeriodicEmbeddings


class PeriodicContracts(unittest.TestCase):
    def test_formula_feature_independence_and_trainable_frequencies(self):
        torch.manual_seed(7)
        layer = PeriodicEmbeddings(features=2, frequencies=3, embedding_size=4)
        # Positive biases keep this fixture away from a zero-gradient ReLU region.
        with torch.no_grad():
            layer.bias.fill_(2)
        values = torch.tensor([[.13, .28], [.64, .91]], requires_grad=True)
        result = layer(values)
        expected = []
        for index in range(2):
            angles = 2 * math.pi * values[:, index, None] * layer.frequencies[index]
            expected.append(torch.relu(torch.cat((angles.cos(), angles.sin()), -1) @
                                       layer.weight[index] + layer.bias[index]))
        torch.testing.assert_close(result, torch.stack(expected, dim=1))
        changed = values.detach().clone()
        changed[:, 0] += .17
        torch.testing.assert_close(layer(changed)[:, 1], result[:, 1])
        result.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(layer.frequencies.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(layer.frequencies.grad).all())

    def test_checkpoint_round_trip_and_input_validation(self):
        layer = PeriodicEmbeddings()
        values = torch.rand(6, 5)
        buffer = io.BytesIO()
        torch.save(layer.state_dict(), buffer)
        buffer.seek(0)
        restored = PeriodicEmbeddings()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(values), layer(values), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            layer(torch.zeros(6, 4))
        with self.assertRaises(ValueError):
            PeriodicEmbeddings(frequency_scale=0)
