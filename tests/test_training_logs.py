"""Log durability, terminal safety and compatibility with an active legacy run."""
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console

from training_logs import RunLogger
from training_logs.dashboard import dashboard
from training_logs.__main__ import demo_view
from training_logs.view import build_view
from training_logs.animation import AnimatedDashboard
from training_logs.terminal import refresh_interval
from types import SimpleNamespace


class TrainingLogTests(unittest.TestCase):
    def test_completed_pipeline_displays_selected_checkpoint_metrics(self):
        snapshot = dict(history=[dict(ndcg=.3), dict(ndcg=.2)], metrics={
            'selection': dict(ndcg=.3, logloss=.15), 'control': dict(ndcg=.29), 'all': dict(ndcg=.298)})
        view = build_view(snapshot, host='server', phase='complete', label='Завершено')
        self.assertEqual(view['metrics']['val_ndcg'], .3)
        self.assertEqual(view['metrics']['full_val_ndcg'], .298)
        self.assertEqual(view['metrics']['control_ndcg'], .29)

    def test_animation_preserves_observed_numbers_and_resets_between_epochs(self):
        animated = AnimatedDashboard(duration=1)
        old = demo_view()
        old['progress']['current'] = 10
        old['gpu']['utilization'] = 10
        animated.update(old, now=0)
        new = demo_view()
        new['progress']['current'] = 20
        new['gpu']['utilization'] = 90
        animated.update(new, now=1)
        frame = animated.frame(1.5)
        self.assertEqual(frame['progress']['current'], 20)
        self.assertEqual(frame['gpu']['utilization'], 90)
        self.assertGreater(frame['progress']['display_current'], 10)
        self.assertLess(frame['progress']['display_current'], 20)
        self.assertGreater(frame['gpu']['display_utilization'], 10)
        self.assertLess(frame['gpu']['display_utilization'], 90)
        self.assertNotIn('display_current', new['progress'])
        self.assertEqual(animated.frame(2)['gpu']['display_utilization'], 90)
        new['progress'].update(epoch=3, current=0)
        animated.update(new, now=3)
        self.assertEqual(animated.frame(3)['progress']['display_current'], 0)

    def test_live_default_requires_a_terminal_and_supports_one_shot(self):
        args = SimpleNamespace(once=False, watch=None, json=False, format='auto')
        with patch('sys.stdout.isatty', return_value=True):
            self.assertEqual(refresh_interval(args), 1)
            args.once = True
            self.assertEqual(refresh_interval(args), 0)
            args.once, args.json = False, True
            self.assertEqual(refresh_interval(args), 0)
        with patch('sys.stdout.isatty', return_value=False):
            args.json = False
            self.assertEqual(refresh_interval(args), 0)
            args.watch = 3
            self.assertEqual(refresh_interval(args), 3)

    def test_atomic_state_retries_a_transient_windows_reader_lock(self):
        import os
        original = os.replace
        calls = []
        def replace(source, target):
            calls.append(target)
            if len(calls) == 1:
                raise PermissionError('reader has a brief shared handle')
            return original(source, target)
        with tempfile.TemporaryDirectory() as temporary:
            with RunLogger(temporary, 'exp', 'MLP', echo=False) as log:
                with patch('training_logs.events.os.replace', side_effect=replace):
                    log.emit('step')
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads((Path(temporary) / 'progress.json').read_text())['event'], 'step')

    def test_events_are_utf8_redacted_bounded_and_metrics_survive(self):
        rng = random.getstate()
        with tempfile.TemporaryDirectory() as temporary:
            with RunLogger(temporary, 'exp', 'Transformer', echo=False, max_bytes=600, backups=2) as log:
                log.emit('step', metrics={'loss': .1}, progress={'current': 1, 'total': 10},
                         config={'password': 'do-not-save', 'nested': {'api_key': 'also-private'}})
                log.emit('checkpoint_saved', metrics={'accuracy': .9})
                state = json.loads((Path(temporary) / 'progress.json').read_text(encoding='utf-8'))
                self.assertEqual(state['metrics'], {'loss': .1, 'accuracy': .9})
                self.assertEqual(state['config']['password'], '[REDACTED]')
                self.assertEqual(state['config']['nested']['api_key'], '[REDACTED]')
                self.assertEqual(state['architecture'], 'Transformer')
                for i in range(15):
                    log.emit('step', progress={'current': i, 'total': 15})
            files = list(Path(temporary).glob('events.jsonl*'))
            self.assertLessEqual(len(files), 3)
            for path in files:
                for line in path.read_text(encoding='utf-8').splitlines():
                    record = json.loads(line)
                    self.assertEqual(record['schema_version'], 1)
                    self.assertNotIn('\x1b', line)
                    self.assertNotIn('do-not-save', line)
            self.assertFalse((Path(temporary) / 'progress.json.tmp').exists())
        self.assertEqual(random.getstate(), rng)

    def test_exception_is_recorded_without_swallowing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'failure'):
                with RunLogger(temporary, 'exp', 'MLP', echo=False) as log:
                    log.emit('started')
                    raise ValueError('failure')
            state = json.loads((Path(temporary) / 'progress.json').read_text())
            self.assertEqual(state['level'], 'ERROR')
            self.assertEqual(state['error_type'], 'ValueError')

    def test_schema_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            with RunLogger(temporary, 'exp', 'CNN', echo=False) as log:
                with self.assertRaises(ValueError):
                    log.emit('step', metrics={'loss': float('nan')})

    def test_checkpoint_is_not_presented_as_live(self):
        state = dict(run_id='exp', architecture='CNN', source='checkpoint', elapsed_seconds=100,
                     checkpoint_at='2026-11-15T12:00:00+00:00',
                     progress=dict(current=3, total=100, epoch=1, epochs=5), metrics={})
        view = build_view({'training_state': state}, host='server', phase='training', label='Обучение')
        self.assertEqual(view['source'], 'checkpoint')
        self.assertEqual(view['progress']['current'], 3)
        self.assertNotIn('train_loss', view['metrics'])
        newer = build_view({'training_state': state, 'batches_per_epoch': 100}, host='server',
                           phase='training', label='Обучение',
                           legacy_events=[dict(epoch=1, batch=10, seconds=120, loss=.2)])
        self.assertEqual(newer['source'], 'worker.log')
        self.assertEqual(newer['metrics']['train_loss'], .2)

    def test_dashboard_width_and_literal_text(self):
        for width in (40, 80, 120):
            output = io.StringIO()
            console = Console(file=output, width=width, color_system=None, legacy_windows=False)
            view = demo_view()
            view['run_id'] = '[red]literal[/red]'
            console.print(dashboard(view, width=width))
            text = output.getvalue()
            self.assertIn('[red]literal[/red]', text)
            if width >= 80:
                self.assertIn('18 420 / 49 475', text)
            self.assertNotIn('\x1b', text)
            self.assertTrue(all(len(line) <= width for line in text.splitlines()))

    def test_compact_dashboard_fits_standard_terminal(self):
        output = io.StringIO()
        console = Console(file=output, width=80, color_system=None, legacy_windows=False)
        console.print(dashboard(demo_view(), width=80, height=24))
        self.assertLessEqual(len(output.getvalue().splitlines()), 24)
