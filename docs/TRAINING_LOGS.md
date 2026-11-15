# Training logs

The logging system is shared by architectures. It does not import a training
framework, inspect or change the dataset, or control the optimizer. Presentation
uses Rich Live, panels, progress bars and responsive tables; persistence uses
Python logging with bounded JSONL rotation and an atomic latest-state file.

## Monitor the current GPU job

Install local monitor dependencies once:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-monitor.txt
.\.venv\Scripts\python.exe scripts/remote.py status --host 25.9.106.39 --user bl1tz2200 --watch 10
```

Rich is automatic in a terminal. `--format rich` forces it; `--format plain`
is suitable for redirected text. `--verbose` adds process/task diagnostics,
`--logs 30` adds recent logs, and `--json` emits one UTF-8 JSON snapshot per
refresh without terminal controls. Ctrl+C stops only the viewer.

The default screen focuses on epoch/batch progress, train/validation metrics,
measured elapsed time, the remaining training budget, GPU/VRAM and freshness.
Unmeasured metrics are displayed as pending. Budget remaining is not an ETA
for completing all epochs. Low measured throughput is flagged without changing
the job. Detailed historical setup logs are available only on request.

## Reuse from any trainer

```python
from training_logs import RunLogger

with RunLogger("runs/transformer-01", "transformer-01", "Transformer") as log:
    log.emit("training_started", budget_seconds=43200)
    # After a completed optimizer step; throttle updates, e.g. every 10 seconds.
    log.emit("batch_completed", elapsed_seconds=123.4, global_step=120,
             progress={"epoch": 1, "epochs": 5, "current": 120,
                       "total": 2000, "unit": "batch"},
             metrics={"train_loss": 0.12, "learning_rate": 0.001})
    log.emit("validation_completed", phase="validate",
             metrics={"val_ndcg": 0.73, "best_ndcg": 0.73})
    log.emit("training_completed", phase="completed")
```

Phases: `initialize`, `train`, `validate`, `predict`, `completed`, `error`.
Metric names are open; standard names receive readable labels. Optional
`resources={"gpu": {"name": "GPU", "used": 1024, "total": 8192,
"utilization": 50}}` supplies local GPU values in MiB and percent. Emit measured
values, not estimates of unobserved steps. MLP logs its first completed batch
and subsequently every `--log-interval 10` seconds, plus validation/checkpoints.

```powershell
python -m training_logs --run-dir runs/transformer-01 --watch 10
python -m training_logs --demo --format rich
```

The demo is explicitly fictional. The local viewer needs no SSH or GPU.

## Files and compatibility

- `events.jsonl`: UTF-8 events with schema_version=1, UTC timestamp, run_id,
  architecture, producer, phase, level and event. Default rotation: 20 MiB,
  three backups, at most four history files.
- `progress.json`: latest state, replaced atomically after each event. Metrics
  remain visible across checkpoints; progress is cleared when phases change.
- `worker.log` / `worker.err.log`: original stdout/stderr remain available.

There must be one event writer per run. Logger context managers record failures
and re-raise them. Secret-like dictionary keys are redacted recursively; callers
must keep credentials out of free-form messages as well. No local variables are
dumped on exceptions. Logs contain plain JSON, never ANSI escape sequences.

The existing MLP process predates these events. The independent Windows task
`AvitoTrainingLogs` uses `training_logs.bridge` to observe trusted `last.pt` on
CPU when its timestamp changes. It never modifies weights or restarts training.
The dashboard labels its source **checkpoint**: this is saved progress, typically
updated every ten minutes, and the live job may be ahead. Newer legacy batch
events take precedence. Missing loss is not reconstructed or invented.
The observer exits when the training task stops or a native trainer takes over
events. New runs use RunLogger directly and do not need this adapter.
