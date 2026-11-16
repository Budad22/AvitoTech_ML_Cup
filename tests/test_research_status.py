"""GPU monitoring must report real process state and usable error context."""
import unittest

from scripts.research_status import render_research
from scripts.diagnostics import readable_line
import json


class ResearchStatusContracts(unittest.TestCase):
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
