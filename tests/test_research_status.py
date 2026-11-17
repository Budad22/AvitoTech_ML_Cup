"""GPU monitoring must report real process state and usable error context."""
import unittest

from scripts.research_status import render_research
from scripts.diagnostics import readable_line
import json
import io
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console
from scripts.research_status import build_research_view, display_research


class ResearchStatusContracts(unittest.TestCase):
    def test_watch_follows_live_successor_after_predecessor_completes(self):
        snapshot=self.completed()
        snapshot['queued']=dict(experiment='EXP-032',hypotheses=26,minimum_hours=12,
                                alive=True,state=dict(status='waiting'))
        successor=self.completed()
        successor['research']['experiment']='EXP-032'
        args=SimpleNamespace(format='auto',json=True,no_animation=True,logs=0,host='gpu')
        output=io.StringIO()
        with patch('sys.stdout',output),patch('time.sleep'), \
                patch('scripts.research_status.read_research',return_value=successor) as poll:
            display_research(None,snapshot,args,1)
        self.assertEqual(poll.call_count,1)
        self.assertEqual([json.loads(line)['research']['experiment'] for line in output.getvalue().splitlines()],
                         ['EXP-030','EXP-032'])

    def completed(self):
        return dict(research=dict(experiment='EXP-030', device='cuda', status='completed', completed=42,
                    failed=0, directory='Q:/research', phase='completed', current=None, progress={},
                    hypotheses=7, elapsed_seconds=1088, updated_at='2026-11-16T18:50:42Z'),
                    task='Ready', alive=False, gpu=['RTX 3060 Ti, 0, 299, 8192'], logs=[],
                    budget_seconds=7200, stages=dict(confirm_3200=dict(completed=21, failed=0),
                    confirm_6400=dict(completed=21, failed=0)), summary=dict(name='plr_softmax',
                    confirmation_ndcg=.359852245, reference_ndcg=.355407814, delta=.004444431))

    def test_completed_dashboard_uses_queue_progress_and_selected_model_metrics(self):
        view = build_research_view(self.completed(), 'gpu')
        self.assertEqual(view['phase'], 'complete')
        self.assertEqual(view['progress']['current'], 42)
        self.assertEqual(view['progress']['total'], 42)
        self.assertEqual(view['metrics']['confirmation_ndcg'], .359852245)
        self.assertEqual(view['budget_seconds'], 7200)
        self.assertEqual(view['gpu']['used'], 299)

    def test_rich_panel_is_used_and_logs_remain_outside_the_panel(self):
        snapshot = self.completed()
        snapshot['logs'] = [dict(path='worker.err.log', text='ValueError: example error')]
        args = SimpleNamespace(format='rich', json=False, no_animation=True, logs=30, host='gpu')
        output = io.StringIO()
        console = Console(file=output, width=120, color_system=None, legacy_windows=False)
        with patch('training_logs.terminal.create_console', return_value=console):
            display_research(None, snapshot, args, 0)
        text = output.getvalue()
        self.assertIn('ПРОГРЕСС', text)
        self.assertIn('ВРЕМЯ И РЕСУРСЫ', text)
        self.assertIn('Результаты всей очереди', text)
        self.assertIn('worker.err.log', text)
        self.assertIn('ValueError: example error', text)

    def test_json_output_stays_machine_readable(self):
        args = SimpleNamespace(format='auto', json=True, no_animation=True, logs=0, host='gpu')
        output = io.StringIO()
        with patch('sys.stdout', output):
            display_research(None, self.completed(), args, 0)
        self.assertEqual(json.loads(output.getvalue())['research']['status'], 'completed')

    def test_stale_controller_errors_and_gpu_numbers_are_visible(self):
        snapshot = dict(research=dict(experiment='EXP-030', device='cuda', status='running', completed=1,
                        failed=0, directory='Q:/research', phase='confirm_3200',
                        current=dict(name='plr_history', seed=42, target=3200),
                        progress=dict(step=12, loss=.2, steps_per_second=80., loader_wait_pct=1.)),
                        task='Running', alive=False, gpu=['RTX 3060 Ti, 70, 4500, 8192'], logs=[])
        value = render_research(snapshot)
        self.assertIn('контроллера остановлен', value)
        self.assertIn('12/3200', value)
        self.assertIn('4500/8192', value)
        self.assertIn('80.00', value)

    def test_direct_research_events_have_readable_losses_and_steps(self):
        value = readable_line(json.dumps(dict(event='batch_completed', name='plr_history',
                              step=120, total=3200, loss=.2)))
        self.assertIn('plr_history', value)
        self.assertIn('120/3200', value)
        self.assertIn('loss=0.200000', value)
