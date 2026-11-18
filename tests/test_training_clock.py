import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from history_ranker import train as trainer
from history_ranker.state import read_json, write_json


class SmallDataset:
    metadata = {"identity": {"scope": "smoke"}}
    info = {"max_pos": 2}

    def __len__(self):
        return 33


class TrainingClockTest(unittest.TestCase):
    def test_interruption_resume_and_completed_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            write_json(root / "metadata.json", {})
            with patch.object(trainer, "Feeds", return_value=SmallDataset()), patch.object(
                trainer, "train_step", side_effect=[0.5, RuntimeError("interrupt")]
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    trainer.train(root, run, "pilot", 8, "cpu")
            status_path = run / "status.json"
            interrupted = read_json(status_path)
            self.assertGreater(interrupted["elapsed_seconds"], 0)
            # Simulate a previous long attempt without a blocking timer.
            write_json(status_path, dict(interrupted, elapsed_seconds=600))
            with patch.object(trainer, "Feeds", return_value=SmallDataset()), patch.object(
                trainer, "train_step", return_value=0.5
            ) as step:
                trainer.train(root, run, "pilot", 8, "cpu")
                final = read_json(status_path)
                self.assertGreater(final["elapsed_seconds"], 600)
                trainer.train(root, run, "pilot", 8, "cpu")
                self.assertEqual(final, read_json(status_path))
                self.assertEqual(2, step.call_count)


class FullCacheProfileTest(unittest.TestCase):
    def test_measured_steps_sample_full_cache_after_warmup(self):
        import numpy as np

        class ProfileDataset:
            metadata = {"identity": {"scope": "full"}}
            info = {}
            sidecar = {name: np.arange(1001) for name in ("offsets", "history_offsets")}

            def __len__(self):
                return 1000

        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "metadata.json", {})
            with patch.object(trainer, "Feeds", return_value=ProfileDataset()), patch.object(
                trainer, "build_training", return_value=(None, None, None)
            ), patch.object(trainer, "train_step") as step:
                profile = trainer.smoke(tmp, tmp, "cpu", optimizer_steps=3)
                self.assertEqual(3, profile["optimizer_steps"])
                self.assertEqual(1000, profile["dataset_feeds"])
                self.assertEqual("full", profile["scope"])
                sampled = np.concatenate([call.args[4] for call in step.call_args_list[4:]])
                np.testing.assert_array_equal(sampled, trainer.order_for_epoch(1000, 0)[64:160])
                with self.assertRaises(ValueError):
                    trainer.smoke(tmp, tmp, "cpu", optimizer_steps=30)
