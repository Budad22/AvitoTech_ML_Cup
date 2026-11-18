"""Checkpointed two-stage training and bounded device profiling."""
import math
from pathlib import Path
import time

import torch

from .data import Feeds, load_batch
from .losses import PositionBCE, lambda_rank
from .model import HistoryRanker
from .state import Lock, atomic_write, code_identity, read_json, sha256, write_json

SEED = 42
EFFECTIVE_BATCH = 32
STAGES = {"a": (5, 3e-4), "b": (1, 3e-5), "pilot": (1, 3e-4)}


def order_for_epoch(size, epoch):
    return torch.randperm(size, generator=torch.Generator().manual_seed(SEED + epoch)).numpy()


def train_step(model, optimizer, position_loss, dataset, ids, microbatch, device):
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for start in range(0, len(ids), microbatch):
        features, target = load_batch(dataset, ids[start:start + microbatch], device)
        scores = model(features)
        if position_loss is None:
            losses = lambda_rank(scores, target["labels"], features["valid"], features["item_id"])
        else:
            losses = position_loss(scores, target["labels"], features["valid"], target["pos"])
        loss = losses.sum() / len(ids)
        loss.backward()
        total += loss.detach()
    if not bool(total.isfinite()):
        raise ValueError("Nonfinite optimizer-step loss")
    optimizer.step()
    return float(total)


def build_training(stage, info, device, query_chunk, initializer=None):
    model = HistoryRanker(query_chunk).to(device)
    position_loss = None if stage == "b" else PositionBCE(info["max_pos"]).to(device)
    if initializer is not None:
        model.load_state_dict(initializer["model"])
    params = list(model.parameters())
    if position_loss is not None:
        params.extend(position_loss.parameters())
    optimizer = torch.optim.AdamW(params, lr=STAGES[stage][1], weight_decay=1e-4)
    return model, optimizer, position_loss


def restore_checkpoint(path, model, optimizer, position_loss, config, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != config:
        raise ValueError("Checkpoint configuration differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if position_loss is not None:
        position_loss.load_state_dict(state["position_loss"])
    torch.set_rng_state(state["cpu_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch"], state["cursor"], state["steps"], state.get("elapsed_seconds", 0.0)


def save_checkpoint(path, model, optimizer, position_loss, config, device, epoch, cursor, steps, elapsed_seconds=0.0):
    state = dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 position_loss=None if position_loss is None else position_loss.state_dict(),
                 cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                 epoch=epoch, cursor=cursor, steps=steps, elapsed_seconds=elapsed_seconds)
    atomic_write(path, lambda tmp: torch.save(state, tmp))


def train(cache, output, stage, microbatch, device, parent=None, query_chunk=None):
    dataset = Feeds(cache, "early" if stage == "b" else "train")
    scope = dataset.metadata["identity"]["scope"]
    if scope != ("smoke" if stage == "pilot" else "full"):
        raise ValueError("Full training requires full cache, pilot requires smoke cache")
    data_hash = sha256(Path(cache) / "metadata.json")
    epochs, lr = STAGES[stage]
    initializer = None
    parent_hash = None
    if stage == "b":
        from .evaluate import load_frozen
        initializer, parent_hash = load_frozen(parent, "a", data_hash)
    elif parent is not None:
        raise ValueError("Parent is only valid for stage B")
    config = dict(stage=stage, cache_sha256=data_hash, code=code_identity(),
                  seed=SEED, lr=lr, epochs=epochs, feeds=len(dataset),
                  batch=EFFECTIVE_BATCH, microbatch=microbatch, device=device,
                  parent_sha256=parent_hash, query_chunk=query_chunk, torch=str(torch.__version__))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Lock(output / ".lock"):
        config_path = output / "config.json"
        if config_path.exists() and read_json(config_path) != config:
            raise ValueError("Run configuration differs from existing run")
        status_path = output / "status.json"
        previous = read_json(status_path) if status_path.exists() else {}
        if previous.get("status") == "completed":
            from .evaluate import load_frozen
            load_frozen(output, stage, data_hash)
            return
        write_json(config_path, config)
        torch.manual_seed(SEED)
        model, optimizer, position_loss = build_training(stage, dataset.info, device, query_chunk, initializer)
        last = output / "last.pt"
        epoch, cursor, steps = 0, 0, 0
        carried = previous.get("elapsed_seconds", 0.0)
        if last.exists():
            epoch, cursor, steps, checkpoint_seconds = restore_checkpoint(last, model, optimizer, position_loss, config, device)
            carried = max(carried, checkpoint_seconds)
        started = time.perf_counter()

        def elapsed():
            return carried + (time.perf_counter() - started)

        total_steps = epochs * math.ceil(len(dataset) / EFFECTIVE_BATCH)
        start_step = steps
        try:
            for current in range(epoch, epochs):
                order = order_for_epoch(len(dataset), current)
                for start in range(cursor, len(dataset), EFFECTIVE_BATCH):
                    ids = order[start:start + EFFECTIVE_BATCH]
                    loss = train_step(model, optimizer, position_loss, dataset, ids, microbatch, device)
                    steps += 1
                    position = start + len(ids)
                    done = position == len(dataset)
                    if done or steps % 100 == 0:
                        save_checkpoint(last, model, optimizer, position_loss, config, device,
                                        current + int(done), 0 if done else position, steps, elapsed())
                    if done or steps % 25 == 0 or steps == start_step + 1:
                        status = dict(status="running", epoch=current + 1, step=steps, steps=total_steps,
                                      loss=loss, elapsed_seconds=elapsed())
                        write_json(output / "status.json", status)
                        print(status, flush=True)
                cursor = 0
            write_json(output / "frozen.json", dict(config=config, step=steps, checkpoint_sha256=sha256(last)))
            write_json(output / "status.json", dict(status="completed", step=steps, steps=total_steps,
                                                    elapsed_seconds=elapsed()))
        except BaseException as error:
            write_json(output / "status.json", dict(status="interrupted", step=steps, elapsed_seconds=elapsed(), error=str(error)))
            raise


def smoke(cache, output, device, optimizer_steps=None):
    dataset = Feeds(cache, "train")
    scope = dataset.metadata["identity"]["scope"]
    if scope not in ("smoke", "full"):
        raise ValueError(f"Invalid profiling scope: {scope}")
    if optimizer_steps is not None and (optimizer_steps <= 0 or 64 + optimizer_steps * EFFECTIVE_BATCH > len(dataset)):
        raise ValueError("Measured steps must fit in the dataset after two warmup batches")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    ordered = order_for_epoch(len(dataset), 0)
    lengths = dataset.sidecar["history_offsets"][1:] - dataset.sidecar["history_offsets"][:-1]
    candidates = dataset.sidecar["offsets"][1:] - dataset.sidecar["offsets"][:-1]
    stress = [int(index) for index in (lengths.argmax(), lengths.argmin(), candidates.argmax())]
    # Verify both phases, including the longest histories and the largest candidate set.
    attempts = []
    for microbatch, chunk in ((8, None), (4, None), (2, None), (1, None), (1, 64)):
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            model, optimizer, pos = build_training("a", dataset.info, device, chunk)
            train_step(model, optimizer, pos, dataset, stress, microbatch, device)
            train_step(model, optimizer, None, dataset, stress, microbatch, device)
            for start in range(0, 64, EFFECTIVE_BATCH):
                train_step(model, optimizer, pos, dataset, ordered[start:start + EFFECTIVE_BATCH], microbatch, device)
            steps = optimizer_steps or (200 if scope == "full" else math.ceil(100 / math.ceil(EFFECTIVE_BATCH / microbatch)))
            started = time.perf_counter()
            for i in range(steps):
                start = 64 + i * EFFECTIVE_BATCH
                train_step(model, optimizer, pos, dataset, ordered[start:start + EFFECTIVE_BATCH], microbatch, device)
            duration = time.perf_counter() - started
            speed = steps * EFFECTIVE_BATCH / duration
            profile = dict(scope=scope, dataset_feeds=len(dataset), device=device, microbatch=microbatch, query_chunk=chunk, duration_seconds=duration,
                           optimizer_steps=steps, microbatches=steps * math.ceil(EFFECTIVE_BATCH / microbatch),
                           stress_history_lengths=[int(lengths[i]) for i in stress],
                           stress_candidate_counts=[int(candidates[i]) for i in stress],
                           feeds_per_second=speed, full_epoch_hours=3151208 / speed / 3600,
                           five_epoch_hours=5 * 3151208 / speed / 3600,
                           peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                           cache_sha256=sha256(Path(cache) / "metadata.json"), code=code_identity(),
                           oom_attempts=attempts)
            write_json(output / "profile.json", profile)
            print(profile, flush=True)
            return profile
        except torch.cuda.OutOfMemoryError:
            attempts.append(dict(microbatch=microbatch, query_chunk=chunk))
        if device == "cuda":
            del model, optimizer, pos
            torch.cuda.empty_cache()
    raise RuntimeError(f"Could not profile model with {attempts}")
