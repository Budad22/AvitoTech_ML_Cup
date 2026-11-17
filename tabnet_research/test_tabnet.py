"""Synthetic contracts for shared inputs, ranking, device buffers and exact resume."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import numpy as np
import torch

from tabnet_research.run import configure_source, run_case, sha256

ROOT = Path(__file__).resolve().parents[1]
configure_source(ROOT / 'artifacts/mlp-source-d4f9edb', ROOT / 'artifacts/tabnet-deps')
from tabnet_research.model import make_model, training_loss

STATISTICS = dict(mean=np.zeros(22, dtype='float32'), scale=np.ones(22, dtype='float32'),
                  quantiles=np.tile(np.linspace(0, 1, 257, dtype='float32'), (5, 1)))


class SyntheticFeeds:
    feeds = 4

    def batch(self, feeds, **options):
        counts = np.array([5 + int(i) for i in feeds])
        blocks = []
        for i, count in zip(feeds, counts):
            rng = np.random.default_rng(100 + int(i))
            blocks.append(dict(dense=torch.from_numpy(rng.normal(size=(count, 519)).astype('float32')),
                               category=torch.from_numpy(rng.integers(1, 5, size=count)),
                               history=torch.from_numpy(rng.normal(size=(count, 529)).astype('float32')),
                               target=torch.tensor([1.] + [0.] * (count - 1))))
        return {**{name: torch.cat([block[name] for block in blocks])
                   for name in ('dense', 'category', 'history', 'target')},
                'counts': torch.from_numpy(counts),
                'weights': torch.from_numpy(np.repeat((1 / counts).astype('float32'), counts))}


class TabNetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_encoder_and_eval_permutation_label_independence(self):
        torch.manual_seed(42)
        control = make_model('mlp', 4, STATISTICS)
        torch.manual_seed(42)
        candidate = make_model('tabnet', 4, STATISTICS).eval()
        for name, value in control.state_dict().items():
            if name.startswith(('category.', 'numerical.')):
                self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        inputs = SyntheticFeeds().batch([0, 1])
        order = torch.tensor([4, 3, 2, 1, 0, 10, 9, 8, 7, 6, 5])
        permuted = {name: values[order] if name != 'counts' else values for name, values in inputs.items()}
        permuted['target'] = 1 - permuted['target']
        with torch.no_grad():
            expected = candidate(inputs)[order]
            actual = candidate(permuted)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertIn('network.network.encoder.group_attention_matrix', candidate.state_dict())

    def test_ranking_gradients_sparsity_and_checkpoint_device_buffer(self):
        torch.manual_seed(42)
        model = make_model('tabnet', 4, STATISTICS)
        inputs = SyntheticFeeds().batch([0, 1])
        value = training_loss(model, inputs, 'tabnet')
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(model.numerical.frequencies.grad.abs().sum()), 0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertLessEqual(float(model.network.mask_entropy.detach()), 1e-6)
        if torch.cuda.is_available():
            model.to('cuda')
            self.assertEqual(model.network.network.encoder.group_attention_matrix.device.type, 'cuda')
            cuda_inputs = {name: x.to('cuda') if name != 'counts' else x for name, x in inputs.items()}
            self.assertTrue(torch.isfinite(training_loss(model, cuda_inputs, 'tabnet')))

    def test_interrupted_resume_matches_uninterrupted_optimizer_and_scores(self):
        data = SyntheticFeeds()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            common = dict(categories=4, device='cpu', steps=3, batch_size=2)
            for kind in ('mlp', 'tabnet'):
                full, resumed = base / (kind + '-full'), base / (kind + '-resumed')
                full.mkdir()
                resumed.mkdir()
                def invoke(folder, emit):
                    return run_case(SimpleNamespace(output=folder, **common), kind, 42, data, data,
                                    STATISTICS, np.arange(data.feeds), {}, emit, time.monotonic() + 60)
                invoke(full, lambda *a, **k: None)
                def interrupt(event, **fields):
                    if event == 'batch_completed' and fields['step'] == 1:
                        (resumed / 'STOP_AFTER_CHECKPOINT').touch()
                with self.assertRaises(InterruptedError):
                    invoke(resumed, interrupt)
                (resumed / 'STOP_AFTER_CHECKPOINT').unlink()
                invoke(resumed, lambda *a, **k: None)
                left = torch.load(full / f'{kind}-s42/last.pt', weights_only=False)
                right = torch.load(resumed / f'{kind}-s42/last.pt', weights_only=False)
                self.assertEqual(left['step'], right['step'])
                self.assertTrue(torch.equal(left['rng'], right['rng']))
                for name, tensor in left['model'].items():
                    self.assertTrue(torch.equal(tensor, right['model'][name]), name)
                for index, values in left['optimizer']['state'].items():
                    for name, tensor in values.items():
                        self.assertTrue(torch.equal(tensor, right['optimizer']['state'][index][name]), name)

    def test_longer_run_forks_optimizer_rng_without_changing_parent(self):
        data = SyntheticFeeds()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            parent_identity = dict(experiment='parent', steps=3)
            child_identity = dict(experiment='child', steps=6)
            for kind in ('mlp', 'tabnet'):
                parent, child, full = (base / (kind + suffix) for suffix in ('-parent', '-child', '-full'))
                for folder in (parent, child, full):
                    folder.mkdir()
                common = dict(categories=4, device='cpu', batch_size=2, evaluation_interval=2)
                def invoke(folder, steps, identity, **options):
                    args = SimpleNamespace(output=folder, steps=steps, **common, **options)
                    return run_case(args, kind, 42, data, data, STATISTICS, np.arange(data.feeds),
                                    identity, lambda *a, **k: None, time.monotonic() + 60)
                invoke(parent, 3, parent_identity)
                source = parent / f'{kind}-s42/last.pt'
                digest = sha256(source)
                result = invoke(child, 6, child_identity, resume_from=parent, parent_identity=parent_identity)
                invoke(full, 6, child_identity)
                self.assertEqual(sha256(source), digest)
                self.assertEqual(result['steps_this_attempt'], 3)
                resumed = torch.load(child / f'{kind}-s42/last.pt', weights_only=False)
                expected = torch.load(full / f'{kind}-s42/last.pt', weights_only=False)
                self.assertEqual(resumed['identity'], dict(child_identity, kind=kind, seed=42))
                self.assertTrue(torch.equal(resumed['rng'], expected['rng']))
                for name, tensor in expected['model'].items():
                    self.assertTrue(torch.equal(tensor, resumed['model'][name]), name)
                for index, values in expected['optimizer']['state'].items():
                    for name, tensor in values.items():
                        self.assertTrue(torch.equal(tensor, resumed['optimizer']['state'][index][name]), name)


if __name__ == '__main__':
    unittest.main()
