# MLP experiment

Run commands from the repository root with an activated Python 3.11 environment.
Install `requirements-mlp.txt`; on the GPU server install the CUDA build of
PyTorch 2.10.0 from its official wheel index. `paramiko` is needed only on the
computer running `scripts/remote.py`.

```powershell
python -m recsys_data --data ../competition --output artifacts/prepared-full-v1
python -m mlp train --data artifacts/prepared-full-v1 --output runs/EXP-013-r01
python -m mlp train --data artifacts/prepared-full-v1 --output runs/EXP-013-r01 --resume runs/EXP-013-r01/last.pt
python -m mlp evaluate --data artifacts/prepared-full-v1 --model runs/EXP-013-r01/best.pt --output runs/EXP-013-r01/metrics.json
python -m mlp predict --data artifacts/prepared-full-v1 --model runs/EXP-013-r01/best.pt --output runs/EXP-013-r01/predictions-test.parquet
python -m unittest discover -s tests -v
```

The MLP combines normalized user/item vectors, their product and absolute
difference, f1–f5, missing-vector masks and a learned category embedding.
It does not use history, IDs, timestamps or original positions as features.
The loss gives every feed equal weight. Checkpoint selection uses full-length
NDCG on the earlier ~80% of validation timestamps; the later part is held back
until final evaluation. Ties follow the pessimistic official metric policy.

Defaults: seed 42, 64 feeds per batch, AdamW lr 0.001, weight decay 0.0001,
5 epochs, patience 2, and a 12-hour training budget. Data preparation is outside
this budget. Final evaluation/inference may continue after the budget expires.
Training is FP32; no negative sampling or mixed precision is used.

`last.pt` contains weights, optimizer, RNG, epoch, batch cursor and elapsed
budget. Resume requires the same prepared manifest and uses the original run
configuration. Checkpoints are atomic and written at least every ten minutes
between batches and at epoch boundaries. A crash may lose work since the last
checkpoint. Load checkpoints only from trusted sources.

## Windows GPU worker

```powershell
python scripts/remote.py deploy --host HOST --user USER
python scripts/remote.py status --host HOST --user USER
python scripts/remote.py status --host HOST --user USER --watch 10
python scripts/remote.py status --host HOST --user USER --verbose --logs 30
python scripts/remote.py status --host HOST --user USER --json
python scripts/remote.py fetch --host HOST --user USER
```

The password is prompted, never saved. The SSH host must already be trusted in
`~/.ssh/known_hosts`. Deployment uses `E:\AVITOCUP` and creates the `AvitoMLP`
scheduled task. It downloads the official dataset and verifies it against the
local archive SHA-256, installs the environment, prepares data, trains, evaluates
and predicts. No GitHub or competition submission calls are made.

Logs: `E:\AVITOCUP\bootstrap.log` and
`E:\AVITOCUP\AvitoTech_ML_Cup\runs\EXP-013-r01\worker.log`.
Task Scheduler owns the process, so disconnecting SSH does not stop it.
The worker temporarily prevents automatic sleep. It never shuts down or reboots
Windows. Deployment refuses a second instance while the task is running.
After a failure, inspect the log before restarting the scheduled task; existing
checkpoints are resumed automatically. `fetch` retrieves logs, metrics and
checkpoints, while large predictions remain on the server.

Status is read-only and does not restart the worker. Its default output shows
the current stage, archive/extraction progress, GPU usage, training batch/loss,
best validation NDCG and checkpoint age when available. Archive progress divides
the server file size by the local `competition.zip` size; extraction progress
uses the ZIP inventory's uncompressed size. Without a local archive, totals are
omitted. `--watch` adds download speed/ETA after two samples; Ctrl+C stops only
the viewer. `--verbose` shows task/process details, `--logs N` adds log tails,
and `--json` emits the full snapshot for scripts. A running scheduler task by
itself never means the optimizer is already training.
