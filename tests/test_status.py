"""Progress reports must distinguish an active training process from old logs."""
import unittest

from scripts.mlp_status import render, stage


def snapshot(**changes):
    value = {'task': {'state': 'Running', 'result': 267009}, 'processes': [],
             'worker': [], 'history': [], 'gpu': [], 'total_bytes': 20_000_000_000}
    value.update(changes)
    return value


class StatusTests(unittest.TestCase):
    def test_download_uses_compressed_archive_bytes(self):
        value = snapshot(partial={'bytes': 5_000_000_000})
        self.assertEqual(stage(value)[0], 'download')
        text = render(value, 'server')
        self.assertIn('25.0%', text)
        self.assertNotIn('267009', text)

    def test_stale_training_log_does_not_mean_running(self):
        value = snapshot(task={'state': 'Ready', 'result': 1},
                         worker=['{"event":"training_started"}', '{"epoch":1,"batch":100,"loss":0.2}'])
        self.assertEqual(stage(value)[0], 'stopped')
        self.assertNotIn('Эпоха: 1/', render(value, 'server'))

    def test_live_training_and_opt_in_logs(self):
        value = snapshot(processes=[{'ProcessId': 123, 'CommandLine': 'python -m mlp train --data E:/AVITOCUP/data'}],
                         worker=['{"event":"training_started"}', '{"epoch":2,"batch":100,"loss":0.12345,"seconds":3600}'])
        self.assertEqual(stage(value)[0], 'training')
        text = render(value, 'server')
        self.assertIn('Эпоха: 2/5', text)
        self.assertNotIn('training_started', text)
        self.assertIn('training_started', render(value, 'server', logs=10))
        self.assertIn('PID 123', render(value, 'server', verbose=True))

    def test_trial_process_is_training_and_queue_progress_is_visible(self):
        from training_logs.view import build_view
        from scripts.suite_status import render_suite
        value = snapshot(processes=[{'CommandLine': 'python -m mlp_trials train --kind baseline'}],
                         worker=['{"event":"training_started"}'],
                         suite=dict(completed=2, total=11, failed=0, current=dict(kind='history_pool')))
        phase, label = stage(value)
        self.assertEqual(phase, 'training')
        view = build_view(value, host='server', phase=phase, label=label)
        self.assertIn('2/11', view['warnings'][0])
        result = render_suite(dict(suite=dict(suite_id='test', directory='Q:/test', status='completed', total=11,
                                              completed=11, failed=0, cases=[], current=None), task='Ready', gpu=[]))
        self.assertIn('11/11', result)

    def test_structured_trial_logs_support_verbose_legacy_rendering(self):
        value = snapshot(processes=[{'ProcessId': 123, 'CommandLine': 'python -m mlp_trials train'}],
                         worker=['{"event":"training_started"}',
                                 '{"epoch":1,"batch":5,"elapsed_seconds":20,"metrics":{"train_loss":0.125}}'])
        self.assertIn('loss: 0.12500', render(value, 'server', verbose=True))

    def test_preparation_and_initialization_are_not_training(self):
        for command, expected in (('python -m recsys_data', 'preparation'), ('python -m mlp train', 'initializing')):
            self.assertEqual(stage(snapshot(processes=[{'CommandLine': command}]))[0], expected)

    def test_completion_and_unknown_total(self):
        self.assertEqual(stage(snapshot(complete=True))[0], 'complete')
        text = render(snapshot(partial={'bytes': 123}, total_bytes=None), 'server')
        self.assertNotIn('%', text)

    def test_download_rate_needs_two_samples(self):
        previous = snapshot(checked_at='2026-11-15T19:00:00+00:00', partial={'bytes': 0})
        current = snapshot(checked_at='2026-11-15T19:00:10+00:00', partial={'bytes': 100_000_000})
        self.assertIn('10.0 МБ/с', render(current, 'server', previous=previous))

    def test_extraction_is_separate_from_archive_download(self):
        value = snapshot(archive={'bytes': 20_000_000_000}, extracted_bytes=10_000_000_000,
                         extracted_total_bytes=40_000_000_000)
        self.assertEqual(stage(value)[0], 'extraction')
        self.assertIn('Распаковано не менее: 10.00 ГБ / 40.00 ГБ (25.0%)', render(value, 'server'))
        value['extracted_bytes'] = 0
        self.assertEqual(stage(value)[0], 'verification')
