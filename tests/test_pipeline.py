"""Performance paths preserve data, gradients, ordering and restart position."""
import hashlib
import shutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mlp.pipeline import feed_loss, prepared_batches
from mlp.training import batches, train
from training_logs import RunLogger
from recsys_data import EmbeddingStore, prepare_dataset
import test_data


class PipelineTests(unittest.TestCase):
    def test_checkpoint_resume_preserves_training_with_prefetch(self):
        fixture = test_data.DataContracts()
        fixture.setUp()
        try:
            data = fixture.root / 'prepared'
            shutil.copytree(fixture.source / 'train', fixture.source / 'val')
            validation = test_data.feeds('val')
            validation = validation.set_column(0, 'feed_id', pa.array([100, 101, 102], type=pa.int32()))
            pq.write_table(validation, fixture.source / 'val/impressions/part-00000.parquet')
            prepare_dataset(fixture.source, data, ('train', 'val'))
            def run(output, stop=None, resume=None, init_from=None, learning_rate=.001):
                args = SimpleNamespace(data=data, output=output, resume=resume, device='cpu',
                                       batch_size=1, seed=42, epochs=2, hours=.1, log_interval=10,
                                       embedding_cache='ram', cache_budget_gib=1, prefetch=2,
                                       cpu_threads=1, arrow_threads=1, stop_file=stop,
                                       init_from=init_from, learning_rate=learning_rate)
                with RunLogger(output, 'synthetic', 'MLP', echo=False) as log:
                    train(args, log)
                return torch.load(output / 'last.pt', weights_only=False)
            expected = run(fixture.root / 'full')
            stopped = fixture.root / 'paused'
            flag = fixture.root / 'stop'
            flag.touch()
            checkpoint = run(stopped, flag)
            self.assertEqual((checkpoint['epoch'], checkpoint['cursor']), (0, 1))
            flag.unlink()
            actual = run(stopped, flag, stopped / 'last.pt')
            self.assertEqual((actual['epoch'], actual['cursor']), (2, 0))
            for name, value in expected['model'].items():
                torch.testing.assert_close(actual['model'][name], value, rtol=0, atol=0)
            self.assertTrue(torch.equal(actual['torch_rng'], expected['torch_rng']))
            flag.touch()
            warm = fixture.root / 'warm'
            initialized = run(warm, flag, init_from=stopped / 'last.pt', learning_rate=.0003)
            self.assertEqual((initialized['epoch'], initialized['cursor']), (0, 1))
            self.assertEqual(initialized['history'], [])
            self.assertEqual(initialized['optimizer']['param_groups'][0]['lr'], .0003)
            self.assertTrue(all(value['step'] == 1 for value in initialized['optimizer']['state'].values()))
            flag.unlink()
            continued = run(warm, flag, resume=warm / 'last.pt', learning_rate=.001)
            self.assertEqual(continued['optimizer']['param_groups'][0]['lr'], .0003)
        finally:
            fixture.tearDown()

    def test_equal_feed_loss_and_gradients_match_reference(self):
        torch.manual_seed(11)
        counts = [5, 19, 7]
        a = torch.randn(sum(counts), requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        labels = torch.randint(2, (sum(counts),)).float()
        reference = torch.stack([part.mean() for part in torch.split(
            torch.nn.functional.binary_cross_entropy_with_logits(a, labels, reduction='none'), counts)]).mean()
        weights = torch.repeat_interleave(1 / torch.tensor(counts).float(), torch.tensor(counts))
        actual = feed_loss(b, labels, weights, len(counts))
        reference.backward()
        actual.backward()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(a.grad, b.grad)

    def test_prefetch_order_and_exception_propagation(self):
        with patch('mlp.pipeline.pack', side_effect=lambda item, *args: item):
            with prepared_batches(range(10), {}, prefetch=2) as values:
                self.assertEqual(list(values), list(range(10)))
            def failed():
                yield 1
                raise RuntimeError('producer failure')
            with self.assertRaisesRegex(RuntimeError, 'producer failure'):
                with prepared_batches(failed(), {}, prefetch=2) as values:
                    list(values)

    def test_ram_cache_matches_memmap_and_raw_files_stay_unchanged(self):
        fixture = test_data.DataContracts()
        fixture.setUp()
        path = fixture.source / 'train/item_embeddings'
        try:
            before = hashlib.sha256((path / 'data.memmap').read_bytes()).hexdigest()
            mapped = EmbeddingStore(path)
            ram = EmbeddingStore(path, cache='ram')
            ids = np.array([0, -1, 1, 2], dtype=np.int32)
            expected, expected_mask = mapped.lookup(ids)
            actual, actual_mask = ram.lookup(ids)
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(expected_mask, actual_mask)
            self.assertFalse(ram._vectors.flags.writeable)
            self.assertEqual(before, hashlib.sha256((path / 'data.memmap').read_bytes()).hexdigest())
            del mapped, ram
            output = fixture.root / 'prepared'
            prepare_dataset(fixture.source, output, ('train',))
            full = [batch.features['feed_id'].to_pylist() for batch in batches(output, 'train', 1, 42)]
            resumed = [batch.features['feed_id'].to_pylist() for batch in batches(output, 'train', 1, 42, start_batch=1)]
            self.assertEqual(full[1:], resumed)
        finally:
            fixture.tearDown()
