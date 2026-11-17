"""The follow-up queue must wait for active work and preserve its control."""
import unittest
from unittest.mock import patch

from ftt.followup import gpu_busy
from ftt.model import configuration


class FollowupContracts(unittest.TestCase):
    def test_active_predecessor_prevents_gpu_query_or_start(self):
        with patch('ftt.followup.subprocess.check_output',return_value='2\n') as command:
            self.assertTrue(gpu_busy())
            self.assertEqual(command.call_count,1)

    def test_idle_gpu_and_busy_gpu(self):
        for gpu,expected in [('0, 299\n',False),('70, 299\n',True),('0, 1900\n',True)]:
            with self.subTest(gpu=gpu),patch('ftt.followup.subprocess.check_output',side_effect=['0\n',gpu]):
                self.assertEqual(gpu_busy(),expected)

    def test_each_hypothesis_changes_one_factor(self):
        base=configuration(9)
        for n in range(13,21):
            new=configuration(n)
            changes=[k for k,v in new.items() if k!='name' and base.get(k)!=v]
            self.assertEqual(len(changes),1)
            self.assertTrue(new['interactions'])
