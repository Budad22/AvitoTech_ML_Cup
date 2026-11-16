"""A PLR queue must not promote failures or rerun an already stronger control."""
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.mlp_screen import compare_control, promotion, run


class PLRScreenContracts(unittest.TestCase):
    def test_custom_variants_gate_and_control_compatibility(self):
        variants = ('plr', 'history_stats_plr')
        cases = [dict(kind=kind, stage='1p56', status='completed', selection_ndcg=.3 + i / 10)
                 for i, kind in enumerate(variants)]
        self.assertEqual(promotion(cases, variants)['kind'], 'history_stats_plr')
        with self.assertRaises(ValueError):
            promotion(cases[:1], variants)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('control', 'candidate'):
                folder = root / name
                folder.mkdir()
                (folder / 'schema.json').write_text('{"fingerprint":"same","boundary":10}')
                (folder / 'suite-config.json').write_text(json.dumps(dict(
                    seed=42, batch_size=64, epochs=5, hours=.2, device='cuda')))
            report = root / 'control/report.json'
            report.write_text(json.dumps(dict(winner=dict(status='completed', selection_ndcg=.5))))
            self.assertEqual(compare_control(report, root / 'candidate')['selection_ndcg'], .5)
            (root / 'candidate/schema.json').write_text('{"fingerprint":"changed","boundary":10}')
            with self.assertRaisesRegex(ValueError, 'data/split'):
                compare_control(report, root / 'candidate')

    def test_full_training_is_skipped_or_promoted_using_existing_control(self):
        for control_score, expected_commands in ((.9, 2), (.1, 3)):
            with self.subTest(control_score=control_score), TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / 'data'
                data.mkdir()
                (data / 'manifest.json').write_text('{"splits":{"train":{"files":[{}]}}}')
                control = root / 'control'
                control.mkdir()
                config = dict(seed=42, batch_size=64, epochs=5, hours=.2, device='cuda')
                (control / 'schema.json').write_text('{"fingerprint":"same","boundary":10}')
                (control / 'suite-config.json').write_text(json.dumps(config))
                (control / 'report.json').write_text(json.dumps(dict(winner=dict(
                    status='completed', run_id='existing', selection_ndcg=control_score))))
                commands = []

                class Child:
                    pid = 999999
                    returncode = 0

                    def __init__(self, command, **kwargs):
                        commands.append(command)
                        target = Path(command[command.index('--directory') + 1])
                        target.mkdir(exist_ok=True)
                        kinds = command[command.index('--variants') + 1:]
                        cases = [dict(kind=kind, run_id=kind, status='completed', selection_ndcg=.4)
                                 for kind in kinds]
                        (target / 'report.json').write_text(json.dumps(dict(cases=cases)))
                        (target / 'schema.json').write_bytes((control / 'schema.json').read_bytes())
                        (target / 'suite-config.json').write_text(json.dumps(config))

                    def poll(self):
                        return 0

                args = Namespace(directory=root / 'queue', data=data, screen_hours=.2, active_root=None,
                                 variants=['plr', 'history_stats_plr'], promotion_control=control / 'report.json')
                with patch('scripts.mlp_screen.sample_indices', return_value=([0], [0])), \
                        patch('scripts.mlp_screen.sample_view'), \
                        patch('scripts.mlp_screen.subprocess.Popen', Child):
                    run(args)
                    run(args)  # Already completed stages must not launch twice.
                self.assertEqual(len(commands), expected_commands)
                state = json.loads((args.directory / 'status.json').read_text())
                self.assertEqual(state['status'], 'completed')
                self.assertEqual(state['completed'], state['total'])
                decision = json.loads((args.directory / 'promotion.json').read_text())
                self.assertEqual(decision['full_training'], control_score < .4)
