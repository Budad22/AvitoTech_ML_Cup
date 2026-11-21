"""New families remain finite, label-independent and safe with empty histories."""
import unittest
import numpy as np
import torch

from mlp_next.extended_models import ExtendedRanker, extended_configurations, extended_loss
from mlp_next.overnight import family_leaders
from test_research_models import statistics


def inputs(device='cpu'):
    return dict(dense=torch.rand(5,519,device=device),category=torch.tensor([0,1,2,3,4],device=device),
                history=torch.rand(5,529,device=device),context=torch.rand(5,17,device=device),
                counts=torch.tensor([2,3]),weights=torch.tensor([.5,.5,1/3,1/3,1/3],device=device),
                target=torch.tensor([1.,0.,0.,1.,0.],device=device),
                position_bucket=torch.tensor([0,5,2,1,3],device=device),
                h_vectors=torch.rand(2,256,128,device=device).half(),
                h_mask=torch.cat((torch.zeros(1,256,dtype=torch.bool,device=device),
                                  torch.ones(1,256,dtype=torch.bool,device=device))),
                h_present=torch.ones(2,256,dtype=torch.bool,device=device),
                h_category=torch.ones(2,256,dtype=torch.long,device=device),
                h_type=torch.ones(2,256,dtype=torch.long,device=device),
                h_source=torch.ones(2,256,dtype=torch.long,device=device))


class OvernightContracts(unittest.TestCase):
    def test_pending_checkpoint_resumes_after_minimum_window_and_is_not_duplicated(self):
        from pathlib import Path
        from unittest.mock import Mock
        import tempfile
        from mlp_next.overnight import Overnight
        with tempfile.TemporaryDirectory() as folder:
            queue = object.__new__(Overnight)
            queue.root = Path(folder)
            queue.state = dict(elapsed_seconds=19 * 3600)
            queue.resume_case = dict(phase='full_reserve', current=dict(
                name='all_scalars_softmax', seed=43, target=590856))
            checkpoint = queue.root / 'full_reserve/runs/all_scalars_softmax-s43/last.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            queue.ledger = {}
            queue.case = Mock()
            queue.resume_pending_case()
            queue.case.assert_called_once_with('all_scalars_softmax', 43, 590856, 'full_reserve', full=True)
            queue.case.reset_mock()
            queue.ledger['full_reserve/all_scalars_softmax-s43/590856'] = {'status': 'completed'}
            queue.resume_pending_case()
            queue.case.assert_not_called()

    def test_queue_explicitly_constructs_cuda_trainer(self):
        from unittest.mock import patch
        from pathlib import Path
        from mlp_next.overnight import Overnight
        queue=object.__new__(Overnight)
        queue.medium_cache=Path('unused');queue.root=Path('unused');queue.event=lambda *a,**k:None
        with patch('mlp_next.overnight.Trainer') as factory:
            queue.trainer(Path('unused'))
        self.assertEqual(factory.call_args.kwargs['device'],'cuda')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_full_queue_trainer_prefetch_and_model_update_on_cuda(self):
        import test_mlp_trials
        from types import SimpleNamespace
        from mlp_research.cache import build
        from mlp_next.data import build_sequences
        from mlp_next.extended_data import prepare_positions
        from mlp_next.overnight import Overnight
        from mlp_next.models import configurations
        fixture=test_mlp_trials.TrialContracts();fixture.setUp();trainer=None
        try:
            queue=object.__new__(Overnight)
            queue.root=fixture.root;queue.args=SimpleNamespace(data=fixture.data)
            queue.configs=configurations();queue.event=lambda *a,**k:None
            queue.medium_cache=fixture.root/'cache'
            build(fixture.data,queue.medium_cache,queue.event)
            build_sequences(fixture.data,queue.medium_cache,fixture.root/'sequences',queue.event)
            prepare_positions(fixture.data,queue.medium_cache,fixture.root/'positions')
            trainer=queue.trainer(fixture.root/'full',full=True,name='plr_softmax')
            trainer.validation.need_sequence=False
            result=trainer.run(queue.configs['plr_softmax'],42,2)
            self.assertEqual(result['device'],'cuda')
            self.assertEqual(result['steps'],2)
        finally:
            if trainer:trainer.close()
            fixture.tearDown()

    def test_predecessor_exit_and_pid_reuse_are_normal_transitions(self):
        from unittest.mock import patch
        from mlp_next.overnight import predecessor_alive
        for actual,expected in ((None,False),(123,True),(124,False)):
            with self.subTest(actual=actual), \
                    patch('mlp_next.overnight.read',return_value=dict(controller_pid=7,controller_identity=123)), \
                    patch('mlp_next.overnight.process_identity',return_value=actual):
                self.assertEqual(predecessor_alive('unused'),expected)

    def test_wait_continues_after_predecessor_exit_without_restarting_it(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from pathlib import Path
        import tempfile
        from mlp_next.overnight import Overnight
        with tempfile.TemporaryDirectory() as directory:
            queue=object.__new__(Overnight)
            queue.args=SimpleNamespace(previous=Path(directory)/'prior',minimum_hours=12,maximum_hours=18)
            queue.root=Path(directory)
            queue.new_names=[]
            queue.state={};queue.event=lambda *a,**k:None
            with patch('mlp_next.overnight.predecessor_alive',side_effect=[True,False]), \
                    patch('mlp_next.overnight.read',return_value={'status':'completed'}), \
                    patch('mlp_next.overnight.time.sleep'), \
                    patch('mlp_next.overnight.write_json'), \
                    patch('mlp_next.overnight.subprocess.check_output',return_value='0, 304'), \
                    patch('mlp_next.overnight.subprocess.run',return_value=SimpleNamespace(returncode=0)) as tests:
                queue.wait()
            self.assertEqual(queue.state['status'],'running')
            self.assertEqual(tests.call_count,1)

    def test_worker_records_named_progress_and_never_retrains_same_full_seed(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import tempfile,json
        from mlp_next.overnight import Overnight
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'data').mkdir()
            (root/'data/manifest.json').write_text(json.dumps({'splits':{'train':{'feeds':37}}}))
            queue=Overnight(SimpleNamespace(output=root/'run',data=root/'data',minimum_hours=12,maximum_hours=18))
            queue.state['status']='running'
            fake=SimpleNamespace(train=SimpleNamespace(feeds=37),validation=SimpleNamespace(),batch_size=64,
                                 close=lambda:None,
                                 run=lambda config,seed,target,**kw:dict(name=config['name'],seed=seed,
                                      status='completed',steps=target,fast_ndcg=.35))
            with patch.object(queue,'trainer',return_value=fake) as factory:
                first=queue.full('plr_softmax',42,'first')
                second=queue.full('plr_softmax',42,'reserve')
            self.assertEqual(factory.call_count,1)
            self.assertEqual(first,second)
            self.assertEqual(queue.state['completed'],1)
            self.assertEqual(json.loads((root/'run/status.json').read_text())['progress']['name'],'plr_softmax')
            queue.logger.handler.close()

    def exercise(self, device):
        torch.set_num_threads(2)
        x=inputs(device)
        order=torch.tensor([1,0,4,2,3],device=device)
        for name, config in extended_configurations().items():
            with self.subTest(name=name,device=device):
                model=ExtendedRanker(5,config,statistics()).to(device).eval()
                score=model(x)
                self.assertTrue(torch.isfinite(score).all())
                changed={k:v if k.startswith('h_') or k=='counts' else v[order] for k,v in x.items()}
                torch.testing.assert_close(model(changed),score[order],atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(model(dict(x,target=1-x['target'])),score,atol=0,rtol=0)
                # Original positions are never needed or consumed during inference.
                torch.testing.assert_close(model({k:v for k,v in x.items() if k!='position_bucket'}),score,atol=0,rtol=0)
                model.train()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
                value=extended_loss(model(x),x,config['loss'])
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                before=model.network[-1].weight.detach().clone()
                optimizer.step()
                self.assertFalse(torch.equal(before,model.network[-1].weight))
                if config.get('memory'):
                    model.eval()
                    original=model(x)
                    padded=dict(x,h_vectors=x['h_vectors'].clone())
                    padded['h_vectors'][0]=99
                    torch.testing.assert_close(model(padded)[:2],original[:2],atol=0,rtol=0)

    def test_all_new_families_cpu(self):
        self.exercise('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_all_new_families_cuda(self):
        self.exercise('cuda')

    def test_family_promotion_excludes_failures_and_retains_diversity(self):
        configs=extended_configurations()
        records=[dict(name=n,status='completed',fast_ndcg=i/1000) for i,n in enumerate(configs)]
        records[0]['status']='failed'
        selected=family_leaders(records,configs)
        self.assertEqual({configs[n]['family'] for n in selected}, {c['family'] for c in configs.values()})
        self.assertNotIn(records[0]['name'],selected)

    def test_positions_and_prefetch_match_source_without_touching_raw_data(self):
        import test_mlp_trials
        from mlp_research.cache import build
        from mlp_next.data import build_sequences
        from mlp_next.extended_data import prepare_positions, ExtendedSplit, PrefetchedStream
        from mlp_next.streaming import StreamingSplit
        fixture=test_mlp_trials.TrialContracts();fixture.setUp()
        try:
            cache=fixture.root/'cache';position=fixture.root/'positions'
            build(fixture.data,cache,lambda *a,**k:None)
            build_sequences(fixture.data,cache,fixture.root/'sequences',lambda *a,**k:None)
            prepare_positions(fixture.data,cache,position)
            cached=ExtendedSplit(cache,'train',positions=position)
            config=extended_configurations()['position_nuisance']
            direct=StreamingSplit(fixture.data,fixture.categories,config,'cpu')
            prefetched=PrefetchedStream(fixture.data,fixture.categories,config,'cpu',batch_size=3)
            reference=direct.batch(np.arange(3));actual=prefetched.batch(np.arange(3))
            from_cache=cached.batch(np.arange(3))
            for name in reference:
                torch.testing.assert_close(actual[name],reference[name],atol=0,rtol=0)
                torch.testing.assert_close(from_cache[name],reference[name],atol=0,rtol=0)
            cached.close();direct.close();prefetched.close()
        finally:
            fixture.tearDown()

    def test_plateau_stop_waits_for_minimum_coverage(self):
        import test_mlp_trials
        from unittest.mock import patch
        from mlp_research.cache import build
        from mlp_research.training import Trainer
        from mlp_next.models import configurations
        fixture=test_mlp_trials.TrialContracts();fixture.setUp()
        trainer=None
        try:
            torch.set_num_threads(2)
            cache=fixture.root/'cache'
            build(fixture.data,cache,lambda *a,**k:None)
            trainer=Trainer(cache,fixture.root/'run',lambda *a,**k:None)
            config=dict(configurations()['history_control'],early_stop_patience=1,minimum_steps=801)
            with patch('mlp_research.training.evaluate',return_value={'ndcg':.3}):
                result=trainer.run(config,42,3200)
            self.assertEqual(result['steps'],1600)
            self.assertTrue(result['stopped_early'])
        finally:
            if trainer:trainer.close()
            fixture.tearDown()
