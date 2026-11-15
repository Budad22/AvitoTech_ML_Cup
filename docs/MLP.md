# MLP experiment

For the full competition train, prepare a fresh `artifacts/prepared-full-v2`
with `--allow-train-duplicate-items` and use that path in the commands below.
This explicit policy preserves repeated train exposures, including their
separate labels and original order; val/test keys remain unique. See DATA.md.
The Windows worker enables this policy automatically. Category inspection
reads only category/timestamp columns, and MLP batches omit unused histories.
On resume, an unchanged archive with a matching successful extraction marker
reuses the prior verification rather than reading and extracting it again.

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
5 epochs, patience 2, and a 2-hour training budget for new runs. Data preparation is outside
this budget. Final evaluation/inference may continue after the budget expires.
Training is FP32; no negative sampling or mixed precision is used.

Use `--learning-rate 0.0003` to test a lower constant AdamW learning rate.
`--init-from runs/EXP-013-r02/best.pt` initializes only model weights in a new
output directory: optimizer, RNG, counters and budget start fresh. The prepared
manifest and category/validation contracts must match. In contrast, `--resume`
restores the complete run, including its saved learning rate and budget; old
12-hour checkpoints retain that original limit. Initialization provenance is
hashed into config.json. Neither command submits predictions online.

`last.pt` contains weights, optimizer, RNG, epoch, batch cursor and elapsed
budget. Resume requires the same prepared manifest and uses the original run
configuration. Checkpoints are atomic and written at least every ten minutes
between batches and at epoch boundaries. A crash may lose work since the last
checkpoint. Load checkpoints only from trusted sources.

## Windows GPU worker

The active runtime is migrated to `Q:\AVITOCUP\AvitoTech_ML_Cup` on the NVMe SSD.
Its Python installation, virtual environment, prepared tables and raw embeddings
all live on Q:. The E: workspace is retained as a recovery copy. `remote.py`
automatically selects Q: after `Q:\AVITOCUP\active-workspace.json` is published.
The owner cancelled EXP-013-r01 and requested a fresh EXP-013-r02 on the same
verified dataset. Its old E: run directory was deleted. The activation marker
also identifies the current run for status/fetch; old weights are not resumed.
Status reports measured recent throughput and time waiting for input batches;
GPU utilization is an instantaneous sample, not the run's average.

The frozen-data runner resumes `last.pt` without downloading or preparing data:

```powershell
python scripts/run_prepared.py
# Request a safe pause after the next completed batch:
New-Item runs/EXP-013-r02/STOP_AFTER_CHECKPOINT -ItemType File
# Before restarting AvitoMLP, remove that request:
Remove-Item runs/EXP-013-r02/STOP_AFTER_CHECKPOINT
```

Use `--embedding-cache ram --cache-budget-gib 16 --prefetch 2` for training on
this 64 GiB host. The int8 train matrix occupies about 9.72 GiB, is loaded once
per process, and remains read-only. A single background producer prepares at
most two additional batches and pins host buffers for CUDA transfer. The cache
budget rejects excess allocation. `--embedding-cache memmap --prefetch 0` is the
lower-memory fallback. CPU/Arrow thread counts are independently configurable.
Batch size, sample order, FP32 model and equal-feed loss semantics are preserved.
CPU feature assembly writes into one contiguous float32 output, reuses owned
item-vector buffers and computes user masks before candidate expansion. The
shared embedding lookup casts directly into its float32 destination.
`scripts/benchmark_features.py --help` describes a CPU-only comparison of saved
code trees on identical batches, with exact byte checks before alternating
timing runs. Warmup is excluded; concurrent-training measurements are labelled.

`scripts/benchmark_pipeline.py` compares SSD memmap, RAM, and RAM with prefetch
using cloned model/optimizer state on the same batches. Its CUDA-event timings
separate transfer and compute; production checkpoints are never updated.

The commands below describe the original first deployment on E:.

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
Windows can defer file-size metadata updates until a large extracted file is
closed. Extraction percentage is therefore a lower estimate; current process
I/O rates help distinguish active extraction from unchanged file-size metadata.
