"""Download a nested, evenly spaced train-shard sample without changing the old dataset."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from inspect_remote_zip import HTTPRangeReader


def expand(source, output, shards):
    archive_manifest = json.loads((source / "archive_manifest.json").read_text())
    train = sorted(m["name"] for m in archive_manifest["members"]
                   if "/train/impressions/" in m["name"] and m["name"].endswith(".parquet"))
    if not 2 <= shards <= len(train):
        raise ValueError("Shard count must be between 2 and the full train size")
    selected = [train[round(i * (len(train) - 1) / (shards - 1))] for i in range(shards)]
    existing = {f"competition/train/impressions/{p.name}" for p in (source / "competition/train/impressions").glob("*.parquet")}
    if not existing <= set(selected):
        raise ValueError("Selected sample must include every original train shard")
    if output.exists():
        raise FileExistsError(f"Choose a new raw dataset directory: {output}")
    # Verify source identity before writing anything. Matrix files are not downloaded.
    remote = HTTPRangeReader(archive_manifest["url"])
    if remote.etag != archive_manifest["etag"] or remote.size != archive_manifest["archive_bytes"]:
        raise ValueError("Remote archive changed since the initial audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".expand-train-", dir=output.parent) as temporary:
        staging = Path(temporary) / "raw"
        # Existing raw files are immutable; hard links avoid duplicating val vectors.
        shutil.copytree(source, staging, copy_function=os.link)
        with zipfile.ZipFile(remote) as archive:
            for name in selected:
                target = (staging / name).resolve()
                if staging.resolve() not in target.parents:
                    raise ValueError("Unsafe archive path")
                info = archive.getinfo(name)
                if name in existing:
                    if target.stat().st_size != info.file_size:
                        raise ValueError(f"Existing shard size mismatch: {name}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                print(json.dumps({"extracted": name, "bytes": info.file_size}), flush=True)
        selection = {"source": str(source), "shards": shards, "selection": "round(i*(N-1)/(shards-1))",
                     "archive_etag": remote.etag, "selected_members": selected,
                     "downloaded_bytes": remote.downloaded}
        (staging / "train_selection.json").write_text(json.dumps(selection, indent=2))
        staging.rename(output)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/initial_audit"))
    parser.add_argument("--output", type=Path, default=Path("data/train16"))
    parser.add_argument("--shards", type=int, default=16)
    args = parser.parse_args()
    expand(args.source, args.output, args.shards)
