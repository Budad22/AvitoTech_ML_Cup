"""Failure messages must be visible immediately, outside cropped live panels."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.diagnostics import readable_line, render_diagnostics
from training_logs.terminal import refresh_interval
from training_logs.view import build_view


class DiagnosticTests(unittest.TestCase):
    def test_logs_default_to_static_but_explicit_watch_is_respected(self):
        args = SimpleNamespace(once=False, watch=None, json=False, format='rich', logs=30, verbose=False)
        with patch('sys.stdout.isatty', return_value=True):
            self.assertEqual(refresh_interval(args), 0)
            args.watch = 5
            self.assertEqual(refresh_interval(args), 5)

    def test_error_is_first_and_traceback_and_source_remain_available(self):
        output = render_diagnostics(dict(status='error', error='Child failed', logs=[dict(path='worker.log',
                                    text='Traceback (most recent call last):\n  line 42\nValueError: broken data')]))
        self.assertLess(output.index('ОШИБКА'), output.index('Файл:'))
        self.assertIn('ValueError: broken data', output)
        self.assertIn('worker.log', output)
        self.assertIn('line 42', output)

    def test_structured_errors_and_metrics_are_readable_without_raw_json(self):
        line = json.dumps(dict(event='run_failed', level='ERROR', error='CUDA out of memory'), separators=(',', ':'))
        output = render_diagnostics(dict(status='running', logs=[dict(path='events.jsonl', text=line)]))
        self.assertIn('ОШИБКА', output)
        self.assertIn('CUDA out of memory', output)
        formatted = readable_line(json.dumps(dict(event='epoch_completed', epoch=2, metrics=dict(val_ndcg=.36))))
        self.assertIn('selection NDCG=0.360000', formatted)
        self.assertNotIn('schema_version', formatted)

    def test_suite_error_is_visible_in_normal_dashboard(self):
        snapshot = dict(suite=dict(error='Stage controller crashed', status='error'), task=dict(state='Ready'))
        view = build_view(snapshot, host='gpu', phase='stopped', label='Stopped')
        self.assertIn('Stage controller crashed', view['warnings'][0])
