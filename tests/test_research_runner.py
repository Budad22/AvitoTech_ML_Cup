"""The whole campaign must freeze finalists before opening confirmation labels."""
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from mlp_research.runner import run


class ResearchRunnerContracts(unittest.TestCase):
    def test_stages_resume_and_confirmation_is_last(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'cache'
            cache.mkdir()
            (cache / 'metadata.json').write_text('{}')
            events = []

            class FakeTrainer:
                fingerprint = 'fixture'
                info = {}

                def __init__(self, cache, directory, emit):
                    self.directory = directory

                def run(self, config, seed, target, **kwargs):
                    events.append(('train', config['name'], seed, target))
                    folder = self.directory / 'runs' / f"{config['name']}-s{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    return dict(name=config['name'], seed=seed, status='completed', steps=target,
                                fast_ndcg=.3, seconds=1., history=[dict(step=target // 2, fast_ndcg=.3),
                                                                  dict(step=target, fast_ndcg=.3)])

                def final(self, config, seed):
                    events.append(('evaluate_confirmation', config['name'], seed))
                    path = self.directory / 'runs' / f"{config['name']}-s{seed}" / 'final.json'
                    path.write_text('{}')

                def close(self):
                    pass

            args = Namespace(directory=root / 'run', cache=cache, maximum_hours=1.)
            with patch('mlp_research.runner.Trainer', FakeTrainer), \
                    patch('mlp_research.runner.report', return_value=dict(decision='fixture')):
                run(args)
                count = len(events)
                run(args)
            self.assertEqual(len(events), count)
            first_confirmation = next(i for i, event in enumerate(events) if event[0] == 'evaluate_confirmation')
            self.assertTrue(all(event[0] == 'train' for event in events[:first_confirmation]))
            self.assertTrue(all(event[0] == 'evaluate_confirmation' for event in events[first_confirmation:]))
            self.assertEqual(len(events[first_confirmation:]), 21)  # Seven finalists, three seeds each.
            self.assertEqual({event[3] for event in events[:first_confirmation]}, {200, 800, 3200})
            self.assertEqual(json.loads((args.directory / 'status.json').read_text())['status'], 'completed')
