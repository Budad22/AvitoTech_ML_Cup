"""The follow-up queue must wait for active work and preserve its control."""
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path

from ftt.followup import gpu_busy,import_control
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
        base=configuration(17)
        for n in range(21,27):
            new=configuration(n)
            self.assertEqual(len([k for k,v in new.items() if k!='name' and base.get(k)!=v]),1)

    def test_frozen_control_does_not_import_future_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'prior';target=root/'new'
            folder=source/'medium/FTT-017-s42';folder.mkdir(parents=True)
            (folder/'best.pt').write_bytes(b'future checkpoint')
            (folder/'last.pt').write_bytes(b'future state')
            (folder/'confirmed-best.pt').write_bytes(b'confirmed 12800 budget')
            ledger={f'medium/FTT-017-s42/{budget}':dict(number=17,seed=42,target=budget,directory=str(folder))
                    for budget in (12800,51200)}
            (source/'ledger.json').write_text(json.dumps(ledger))
            import_control(target,source,control=17,frozen_budget=12800)
            result=json.loads((target/'ledger.json').read_text())
            self.assertNotIn('medium/FTT-017-s42/51200',result)
            self.assertEqual((target/'medium/FTT-017-s42/best.pt').read_bytes(),b'confirmed 12800 budget')

    def test_import_copies_control_and_leaves_resumed_files_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'prior'; target=root/'new'
            folder=source/'medium/FTT-009-s42'
            folder.mkdir(parents=True)
            (folder/'last.pt').write_bytes(b'checkpoint')
            ledger={'medium/FTT-009-s42/6400':dict(number=9,seed=42,directory=str(folder))}
            (source/'ledger.json').write_text(json.dumps(ledger))
            import_control(target,source)
            copied=target/'medium/FTT-009-s42/last.pt'
            self.assertEqual(copied.read_bytes(),b'checkpoint')
            copied.write_bytes(b'continued checkpoint')
            import_control(target,source)
            self.assertEqual(copied.read_bytes(),b'continued checkpoint')
            self.assertEqual((folder/'last.pt').read_bytes(),b'checkpoint')
