import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import test_mlp_trials
from mlp_next.extended_models import ExtendedRanker, extended_configurations
from mlp_research.cache import build
from mlp_trials.data import batches
from scripts.export_research_submission import export, sha256


class ResearchSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_mlp_trials.TrialContracts()
        self.fixture.setUp()
        self.root, self.data = self.fixture.root, self.fixture.data
        self.cache = self.root / 'export-cache'
        self.info = build(self.data, self.cache, lambda *a, **k: None)
        with np.load(self.cache / 'statistics.npz') as archive:
            statistics = {k: archive[k].copy() for k in archive.files}
        config = extended_configurations()['all_scalars_softmax']
        model = ExtendedRanker(len(self.info['categories']), config, statistics)
        with torch.no_grad():
            model.network[-1].weight.zero_()
            model.network[-1].bias.fill_(5.)
        self.checkpoint = self.root / 'best.pt'
        fingerprint = hashlib.sha256((sha256(self.cache / 'metadata.json') + sha256(self.data / 'manifest.json')).encode()).hexdigest()
        torch.save(dict(identity=dict(config=config, seed=43, cache=fingerprint, batch_size=64),
                        model=model.state_dict(), step=78000), self.checkpoint)
        batch = next(batches(self.data, 'test', 3, history=True))
        reference = batch.candidate_table(include_target=False).select(['feed_id', 'item_id'])
        self.template = self.root / 'sample.parquet'
        pq.write_table(reference.append_column('score', pa.array(np.zeros(len(reference)), type=pa.float32())), self.template)

    def tearDown(self):
        self.fixture.tearDown()

    def args(self):
        return SimpleNamespace(data=self.data, cache=self.cache, checkpoint=self.checkpoint,
            template=self.template, output=self.root / 'export', device='cpu', batch_size=2,
            run_id='test-export', code_commit='test')

    def test_whole_test_exact_keys_and_raw_logits_without_target(self):
        args = self.args()
        result = export(args)
        table = pq.read_table(args.output / 'submission.parquet')
        np.testing.assert_array_equal(table['score'].to_numpy(), np.full(len(table), 5, dtype='float32'))
        self.assertEqual(result['selected_step'], 78000)
        self.assertEqual(result['audit']['rows'], 15)
        self.assertEqual(result['audit']['feeds'], 3)
        self.assertEqual(result['audit']['sha256'], sha256(args.output / 'submission.parquet'))
        with self.assertRaises(FileExistsError):
            export(args)

    def test_wrong_template_never_publishes_a_prediction(self):
        args = self.args()
        pq.write_table(pq.read_table(self.template).slice(0, 14), self.template)
        with self.assertRaisesRegex(ValueError, 'Row count'):
            export(args)
        self.assertFalse((args.output / 'submission.parquet').exists())
        self.assertFalse((args.output / 'submission.tmp.parquet').exists())

    def test_wrong_data_identity_is_rejected_before_export(self):
        args = self.args()
        state = torch.load(self.checkpoint, weights_only=False)
        state['identity']['cache'] = 'wrong'
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            export(args)
        self.assertFalse(args.output.exists())
