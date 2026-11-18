"""Exercise real child failure/restart without touching a real training process."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from mlp_next.supervisor import supervise


class SupervisorContracts(unittest.TestCase):
    def test_crashed_child_resumes_saved_progress_then_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            child=root/'child.py'
            child.write_text("""import json,os
from pathlib import Path
r=Path.cwd();p=r/'checkpoint.json'
if not p.exists():
 p.write_text(json.dumps({'step':7}))
 (r/'status.json').write_text(json.dumps({'status':'error','phase':'test','current':{'name':'test','seed':42}}))
 os._exit(17)
s=json.loads(p.read_text());assert s['step']==7
p.write_text(json.dumps({'step':8}))
(r/'status.json').write_text(json.dumps({'status':'completed'}))
""",encoding='utf-8')
            result=supervise(root,[sys.executable,str(child)],poll=.02,retry_delay=.01)
            self.assertEqual(result,'completed')
            self.assertEqual(json.loads((root/'checkpoint.json').read_text())['step'],8)
            self.assertEqual(json.loads((root/'supervisor.json').read_text())['restarts'],1)

    def test_repeated_failure_is_bounded_and_owner_pause_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            result=supervise(root,[sys.executable,'-c',"import sys;sys.exit(19)"],
                             poll=.02,retry_delay=.01,max_retries=1)
            self.assertEqual(result,'blocked')
            self.assertEqual(json.loads((root/'supervisor.json').read_text())['restarts'],2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'STOP_AFTER_CASE').touch()
            self.assertEqual(supervise(root,['missing-command'],poll=.02),'paused')

    def test_already_completed_run_is_never_restarted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'status.json').write_text('{"status":"completed"}')
            self.assertEqual(supervise(root,['missing-command'],poll=.02),'completed')
