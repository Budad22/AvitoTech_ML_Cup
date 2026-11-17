"""Readers must not corrupt status or terminate training on sharing conflicts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ftt.atomic import replace, write_json


class AtomicOutputs(unittest.TestCase):
    def test_transient_reader_lock_preserves_old_json_until_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            path.write_text('{"step": 1}', encoding='utf-8')
            from os import replace as real_replace
            calls = []

            def locked(source, destination):
                calls.append(1)
                self.assertEqual(json.loads(path.read_text()), {'step': 1})
                if len(calls) < 3:
                    raise PermissionError('reader sharing conflict')
                real_replace(source, destination)

            with patch('ftt.atomic.os.replace', side_effect=locked), patch('ftt.atomic.time.sleep'):
                write_json(path, {'step': 2})
            self.assertEqual(json.loads(path.read_text()), {'step': 2})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_persistent_lock_is_bounded_and_old_output_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            path.write_text('{"step": 1}', encoding='utf-8')
            with patch('ftt.atomic.os.replace', side_effect=PermissionError('locked')) as operation, \
                    patch('ftt.atomic.time.sleep'):
                with self.assertRaises(PermissionError):
                    write_json(path, {'step': 2})
            self.assertEqual(operation.call_count, 21)
            self.assertEqual(json.loads(path.read_text()), {'step': 1})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_unrelated_error_is_not_retried(self):
        with patch('ftt.atomic.os.replace', side_effect=FileNotFoundError) as operation:
            with self.assertRaises(FileNotFoundError):
                replace('missing', 'destination')
            self.assertEqual(operation.call_count, 1)
