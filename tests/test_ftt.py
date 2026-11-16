"""Synthetic contracts for feature attention and whole-feed ranking."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ftt.model import Ranker, configuration, objective
from ftt.runner import ndcg
from ftt.data import Cache, prepare
from ftt.runner import Queue
from recsys_data import prepare_dataset
import test_data as fixtures
import pyarrow as pa
import pyarrow.parquet as pq
import shutil


class TransformerTests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(42)
        return dict(numeric=torch.randn(5, 5), category=torch.tensor([1,2,3,4,5]),
                    item=torch.randn(5,128), user=torch.randn(5,128), history_mean=torch.randn(5,128),
                    flags=torch.randn(5,4), counts=[3,2], target=torch.tensor([1.,0.,1.,0.,1.]),
                    history_vectors=torch.randn(2,50,128), history_category=torch.zeros(2,50,dtype=torch.long),
                    history_event=torch.zeros(2,50,dtype=torch.long), history_source=torch.zeros(2,50,dtype=torch.long),
                    history_mask=torch.zeros(2,50,dtype=torch.bool))

    def model(self, n):
        return Ranker(configuration(n), np.zeros(5), np.ones(5), np.tile(np.linspace(-2,2,17),(5,1)))

    def test_all_variants_finite_gradients_and_empty_history(self):
        torch.set_num_threads(2)
        for number in range(1,9):
            with self.subTest(number=number):
                batch = self.batch()
                model = self.model(number)
                scores = model(batch)
                self.assertEqual(tuple(scores.shape),(5,))
                value = objective(scores,batch['target'],batch['counts'],configuration(number)['loss'])
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_candidate_permutation_and_history_padding(self):
        permutation = torch.tensor([2,0,1,4,3])
        for number in (1,7,8):
            batch = self.batch()
            model = self.model(number).eval()
            with torch.no_grad():
                expected = model(batch)
                permuted = {k:(v[permutation] if isinstance(v,torch.Tensor) and v.shape[0]==5 else v)
                            for k,v in batch.items()}
                actual = model(permuted)
                torch.testing.assert_close(actual,expected[permutation],atol=1e-5,rtol=1e-5)
                batch['history_vectors'].fill_(100)
                torch.testing.assert_close(model(batch),expected,atol=1e-5,rtol=1e-5)

    def test_metric_ties_and_no_positive(self):
        self.assertEqual(ndcg(np.array([0.,0.]),np.array([1.,2.])),1.)
        self.assertLess(ndcg(np.array([1.,0.]),np.array([0.,0.])),1.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_updates_all_variants(self):
        for n in range(1,9):
            model = self.model(n).cuda()
            batch = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in self.batch().items()}
            before = model.output.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
            objective(model(batch),batch['target'],batch['counts'],configuration(n)['loss']).backward()
            optimizer.step()
            self.assertFalse(torch.equal(before,model.output.weight))

    def test_cache_and_exact_optimizer_resume(self):
        fixture = fixtures.DataContracts()
        fixture.setUp()
        try:
            shutil.copytree(fixture.source/'train/item_embeddings',fixture.source/'val/item_embeddings')
            folder = fixture.source/'val/impressions'
            folder.mkdir()
            val = fixtures.feeds('val')
            val = val.set_column(val.schema.get_field_index('feed_id'),'feed_id',pa.array([110,111,112],type=pa.int32()))
            pq.write_table(val,folder/'part.parquet')
            data = fixture.root/'prepared'
            prepare_dataset(fixture.source,data,splits=('train','val'))
            root = fixture.root/'queue'
            prepare(data,root/'cache',lambda *a,**k:None)
            cache = Cache(root/'cache','train')
            batch = cache.batch([2,0],'cpu')
            self.assertEqual(batch['feed_id'].tolist(),[12,10])
            self.assertEqual(batch['counts'],[5,5])
            self.assertEqual(batch['history_vectors'].shape,(2,50,128))
            args = SimpleNamespace(output=str(root),device='cpu',hours=8,data=str(data))
            q = Queue(args)
            q.validation = Cache(root/'cache','val')
            q.fast = np.array([0])
            q.trial(2,42,1,'small')
            q.trial(2,42,2,'small')
            resumed = torch.load(root/'small/FTT-002-s42/last.pt',weights_only=False)
            other = fixture.root/'other'
            shutil.copytree(root/'cache',other/'cache')
            args.output = str(other)
            q = Queue(args)
            q.validation = Cache(other/'cache','val')
            q.fast = np.array([0])
            q.trial(2,42,2,'small')
            straight = torch.load(other/'small/FTT-002-s42/last.pt',weights_only=False)
            for key in resumed['model']:
                torch.testing.assert_close(resumed['model'][key],straight['model'][key],rtol=0,atol=0)
        finally:
            if 'cache' in locals():
                cache.close()
            if 'q' in locals():
                q.validation.close()
            fixture.tearDown()
