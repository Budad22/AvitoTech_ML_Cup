import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from history_ranker.data import collate, percentiles
from history_ranker.evaluate import ndcg, paired_bootstrap
from history_ranker.losses import PositionBCE, lambda_rank, swap_weights
from history_ranker.model import HistoryRanker
from history_ranker.train import build_training, restore_checkpoint, save_checkpoint, train_step


def example(candidates, history, seed):
    g = torch.Generator().manual_seed(seed)
    item = torch.randn(candidates, 128, generator=g)
    floats = torch.rand(candidates, 5, generator=g)
    return dict(
        user=torch.zeros(128), item=item, missing=torch.zeros(candidates, dtype=torch.bool),
        category=torch.arange(candidates) % 3, item_id=torch.arange(1, candidates + 1),
        floats=floats, percentile=torch.from_numpy(percentiles(floats.numpy())),
        history_item=torch.zeros(history, 128), history_missing=torch.ones(history, dtype=torch.bool),
        history_category=torch.arange(history) % 3, history_item_id=torch.arange(1, history + 1),
        history_type=torch.zeros(history, dtype=torch.long),
        history_source=torch.zeros(history, dtype=torch.long),
        labels=(torch.arange(candidates) == 1).float(), pos=torch.arange(candidates) * 2,
    )


class RankerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_midrank_ties(self):
        values = np.array([[5, 0], [3, 0], [5, 1], [4, 1]], dtype=np.float32)
        np.testing.assert_allclose(percentiles(values), [[.75, .25], [.125, .25], [.75, .75], [.375, .75]])

    def test_padding_invariance_and_candidate_permutation(self):
        model = HistoryRanker().eval()
        empty, long = example(3, 0, 1), example(7, 5, 2)
        order = torch.tensor([2, 0, 1])
        candidates = ("item", "missing", "category", "item_id", "floats", "percentile", "labels", "pos")
        permuted = {key: value[order] if key in candidates else value for key, value in empty.items()}
        with torch.no_grad():
            solo, target = collate([empty])
            self.assertNotIn("labels", solo)
            self.assertNotIn("pos", solo)
            self.assertIn("pos", target)
            score = model(solo)
            padded, _ = collate([empty, long])
            torch.testing.assert_close(model(padded)[0, :3], score[0], atol=3e-6, rtol=1e-5)
            inputs, _ = collate([permuted])
            torch.testing.assert_close(model(inputs)[0], score[0, order], atol=3e-6, rtol=1e-5)

    def test_item_match_uses_ids_even_without_vectors(self):
        model = HistoryRanker(query_chunk=1).eval()
        with torch.no_grad():
            model.query.weight.zero_()
            model.query.bias.zero_()
            model.key_value.weight.zero_()
            model.key_value.bias.zero_()
            model.key_value.weight[128:, :] = torch.eye(128)
            model.cross_output.weight.copy_(torch.eye(128))
            model.cross_output.bias.zero_()
            model.match_bias.zero_()
            model.match_bias[0].fill_(math.log(2))
            row = example(3, 2, 3)
            # Known match, unknown, no match. Both history vectors are explicitly missing.
            row["item_id"] = torch.tensor([1, 0, 5])
            row["history_item_id"] = torch.tensor([1, 0])
            data, _ = collate([row])
            memory = torch.zeros(1, 3, 128)
            memory[0, 1, :] = 1
            output = model.cross_attention(
                torch.zeros(1, 3, 128), memory, data, torch.ones(1, 3, dtype=torch.bool),
            )
            torch.testing.assert_close(output[0, :, 0], torch.tensor([0.5, 1/3, 1/3]))

    def test_position_loss_uses_gaps_and_excludes_padding(self):
        model = HistoryRanker().eval()
        features, target = collate([example(2, 0, 1), example(3, 2, 2)])
        loss = PositionBCE(4)
        with torch.no_grad():
            scores = model(features)
            before = loss(scores, target["labels"], features["valid"], target["pos"])
            loss.bias[2] = 1
            after = loss(scores, target["labels"], features["valid"], target["pos"])
            self.assertTrue(torch.all(after < before))
            torch.testing.assert_close(model(features), scores, atol=0, rtol=0)

    def test_lambda_weights_and_gradient(self):
        scores = torch.tensor([0.2, 0.4, 0.7, -0.1], requires_grad=True)
        target = torch.tensor([1., 0., 0., 1.])
        ids = torch.arange(4)
        predicted = swap_weights(scores, target, ids).numpy()
        original = scores.detach().numpy()
        base = ndcg(original, target.numpy(), np.array([0, 4]))[0]
        for a, i in enumerate((0, 3)):
            for b, j in enumerate((1, 2)):
                swapped = original.copy()
                swapped[[i, j]] = swapped[[j, i]]
                self.assertAlmostEqual(predicted[a, b], abs(ndcg(swapped, target.numpy(), [0, 4])[0] - base))
        valid = torch.ones(1, 4, dtype=torch.bool)
        loss = lambda_rank(scores[None], target[None], valid, ids[None]).sum()
        loss.backward()
        self.assertTrue((scores.grad[target.bool()] < 0).all())
        self.assertTrue((scores.grad[~target.bool()] > 0).all())

    def test_pessimistic_ties_and_no_pairs(self):
        torch.testing.assert_close(
            swap_weights(torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([3, 2, 1])),
            torch.tensor([[1 / math.log2(3) - 0.5, 0.5]]),
        )
        score = torch.ones(1, 3, requires_grad=True)
        loss = lambda_rank(score, torch.ones_like(score), torch.ones_like(score, dtype=torch.bool),
                           torch.arange(3)[None])
        self.assertEqual(0, float(loss.detach()))
        loss.sum().backward()
        torch.testing.assert_close(score.grad, torch.zeros_like(score))

    def test_exact_cpu_resume_with_dropout(self):
        rows = [example(3, 0, 1), example(4, 5, 2), example(5, 2, 3)]
        torch.manual_seed(42)
        model, optimizer, pos = build_training("a", {"max_pos": 8}, "cpu", None)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            config = {"seed": 42}
            train_step(model, optimizer, pos, rows, [0, 1], 1, "cpu")
            save_checkpoint(path, model, optimizer, pos, config, "cpu", 0, 2, 1, 12.5)
            expected = train_step(model, optimizer, pos, rows, [1, 2], 1, "cpu")
            second, opt2, pos2 = build_training("a", {"max_pos": 8}, "cpu", None)
            self.assertEqual((0, 2, 1, 12.5), restore_checkpoint(path, second, opt2, pos2, config, "cpu"))
            actual = train_step(second, opt2, pos2, rows, [1, 2], 1, "cpu")
            self.assertEqual(expected, actual)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(second.state_dict()[key], value, atol=0, rtol=0)
            torch.testing.assert_close(pos2.bias, pos.bias, atol=0, rtol=0)

    def test_bootstrap_resamples_users_and_weights_feeds(self):
        interval, users = paired_bootstrap(np.array([9, 9, 2]), [np.array([1., 1., -1.])], 1000)
        self.assertEqual(2, users)
        np.testing.assert_allclose(interval, [[-1., 1.]])


if __name__ == "__main__":
    unittest.main()
