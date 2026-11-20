"""Read-only JSON API for Avito training tasks on Windows."""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
from threading import Lock
import time
from urllib.parse import urlsplit


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")


def read_file(path, *, json_data=False):
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        return None
    return json.loads(text) if json_data and text else text


def scheduler_states():
    # Locale-independent COM states: 2 = queued, 3 = ready, 4 = running.
    script = """
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.Encoding]::UTF8
$s=New-Object -ComObject Schedule.Service
$s.Connect()
$states=@{}
foreach($t in $s.GetFolder('\\').GetTasks(0)){
    if($t.Name -like 'Avito*'){$states[$t.Name]=[int]$t.State}
}
ConvertTo-Json -InputObject $states -Compress
"""
    out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                         check=True, capture_output=True, timeout=8)
    return json.loads(out.stdout.decode("utf-8-sig"))


def describe_job(job, directory, task_state, now):
    root = directory / job["directory"]
    stage = read_file(root / "stage.txt")
    stages = job["stages"]
    result = {"id": job["id"], "task": job["task"], "stage": stage, "next_stages": []}
    if task_state == 4:
        status = "running"
    elif task_state == 2:
        status = "queued"
    elif stage == "completed":
        status = "completed"
    elif (root / "failure.log").exists():
        status = "failed"
    elif stage:
        status = "stopped"
    else:
        status = "idle"
    result["status"] = status
    if status == "running" and stage in stages:
        result["next_stages"] = stages[stages.index(stage) + 1:]
    path = job.get("progress", {}).get(stage)
    if path:
        file = root / path
        progress = read_file(file, json_data=True)
        if progress:
            step, total = progress["step"], progress.get("steps")
            elapsed = progress.get("elapsed_seconds")
            remaining = None
            updated = file.stat().st_mtime
            if status == "running" and step and total and elapsed and now - updated < 180:
                remaining = math.ceil(elapsed / step * max(0, total - step))
            result["progress"] = dict(step=step, steps=total, epoch=progress.get("epoch"),
                percent=round(100 * step / total, 2) if total else None, elapsed_seconds=elapsed,
                updated_at=timestamp(updated), remaining_seconds=remaining,
                estimated_stage_end=timestamp(updated+remaining) if remaining is not None else None)
    if stage == "completed" and job.get("report"):
        report = read_file(root / job["report"], json_data=True)
        if report:
            result["ndcg"] = report.get("ndcg")
    return result


def training_queue(configs, jobs):
    """Report declared successors in plan order, plus queued monitored tasks."""
    by_id = {job["id"]: job for job in jobs}
    queue = []
    for cfg in configs:
        job = by_id[cfg["id"]]
        after = cfg.get("after")
        if job["status"] == "queued":
            state = "queued"
        elif after and job["status"] == "idle":
            predecessor = by_id[after]["status"]
            if predecessor in {"failed", "stopped"}:
                state = "blocked"
            else:
                state = "awaiting_start" if predecessor == "completed" else "waiting"
        else:
            continue
        queue.append({"id": job["id"], "task": job["task"], "state": state, "after": after})
    return queue


class Monitor:
    def __init__(self, config):
        self.config = config
        self.lock = Lock()
        self.expires = 0

    def snapshot(self):
        with self.lock:
            if time.monotonic() >= self.expires:
                cfg = read_file(self.config, json_data=True)
                states = scheduler_states()
                now = time.time()
                jobs = [describe_job(j, Path(cfg["experiments"]), states.get(j["task"]), now)
                        for j in cfg["jobs"]]
                queue = training_queue(cfg["jobs"], jobs)
                self.data = dict(updated_at=timestamp(now), jobs=jobs,
                    current=[j for j in jobs if j["status"] == "running"],
                    queue=queue)
                self.expires = time.monotonic() + 5
            return self.data


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, monitor, **kwargs):
        self.monitor = monitor
        super().__init__(*args, **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        if path == "":
            from dashboard import render
            try:
                data = self.monitor.snapshot()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                data = None
            return self.send_body(200 if data else 503, "text/html; charset=utf-8", render(data).encode())
        if path == "/api":
            return self.reply(200, {"endpoints":["/api/status", "/api/current", "/api/queue"]})
        if path not in ("/api/status", "/api/current", "/api/queue"):
            return self.reply(404, {"error":"Unknown endpoint"})
        try:
            data = self.monitor.snapshot()
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return self.reply(503, {"error":"Training status unavailable"})
        key = path.split("/")[-1]
        self.reply(200, data if key == "status" else {"updated_at":data["updated_at"], key:data[key]})

    def reply(self, code, obj):
        body = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
        self.send_body(code, "application/json; charset=utf-8", body)

    def send_body(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bind", required=True)
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args()
    handler = partial(Handler, monitor=Monitor(args.config))
    ThreadingHTTPServer((args.bind, args.port), handler).serve_forever()
