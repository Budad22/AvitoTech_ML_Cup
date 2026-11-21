import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.research_monitor import monitor, publish, TRAIN_STAGE


class ResearchMonitorTests(unittest.TestCase):
    def test_dashboard_excludes_small_screen_and_limits_summary_to_leaders_and_baseline(self):
        from scripts.research_monitor import completed_results
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stage, name, seed, value in [('screen', 'memory_cosine', 42, .99),
                    ('full_new', 'all_scalars_softmax', 42, .372),
                    ('full_reserve', 'all_scalars_softmax', 43, .374),
                    ('full_new', 'memory_cosine', 42, .373),
                    ('full_reserve', 'experts_4', 42, .370),
                    ('full_priority', 'plr_softmax', 42, .369)]:
                path = root / stage / 'runs' / f'{name}-s{seed}' / 'final.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(name=name, seed=seed, selection_ndcg=value)))
            result = completed_results(root)
            self.assertEqual(len(result['ndcg']), 3)
            self.assertEqual(result['ndcg']['Все числовые · selection'], .374)
            self.assertEqual(result['ndcg']['Внимание к истории · selection'], .373)
            self.assertEqual(result['ndcg']['Бейзлайн · selection'], .369)
            self.assertEqual(len(result['details']), 5)

    def test_training_progress_preserves_real_steps_without_false_eta(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'status.json').write_text(json.dumps(dict(status='running', phase='full_reserve',
                current=dict(name='all_scalars_softmax', seed=43, target=590856),
                progress=dict(step=78295), elapsed_seconds=68360)))
            self.assertEqual(publish(root, running=True), TRAIN_STAGE)
            progress = json.loads((root / 'api/progress.json').read_text())
            self.assertEqual(progress['step'], 78295)
            self.assertEqual(progress['steps'], 590856)
            self.assertEqual(progress['model'], 'all_scalars_softmax')
            self.assertIsNone(progress['elapsed_seconds'])

    def test_completed_metrics_are_explicitly_selection_and_failures_are_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'full_reserve/runs/test-s43'
            output.mkdir(parents=True)
            (output / 'final.json').write_text(json.dumps(dict(name='test', seed=43, selection_ndcg=.37)))
            (root / 'status.json').write_text('{"status":"completed"}')
            self.assertEqual(publish(root, running=False), 'completed')
            result = json.loads((root / 'api/comparison.json').read_text())
            self.assertEqual(next(iter(result['ndcg'].values())), .37)
            self.assertIn('selection', next(iter(result['ndcg'])))
            (root / 'status.json').write_text('{"status":"error","error":"test failure"}')
            self.assertEqual(publish(root, running=False), 'failed')
            self.assertIn('test failure', (root / 'api/failure.log').read_text())

    def test_wrapper_tracks_a_real_child_until_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            command = [sys.executable, '-c', "from pathlib import Path;Path('status.json').write_text('{\"status\":\"completed\"}')"]
            self.assertEqual(monitor(root, command, poll=.02), 0)
            self.assertEqual((root / 'api/stage.txt').read_text(), 'completed')
