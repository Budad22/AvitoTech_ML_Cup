"""List or extract selected official ZIP members using bounded HTTP Range reads.

Avoids downloading the entire competition archive for an initial audit.
Every extracted member is verified by zipfile's CRC check.
"""
import argparse
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

URL = "https://storage.yandexcloud.net/avitotech-mlcup-2026/competition.zip"
NETWORK_ERRORS = (URLError, OSError, socket.timeout, http.client.IncompleteRead)


def retry(operation, attempts=6):
    for attempt in range(attempts):
        try:
            return operation()
        except NETWORK_ERRORS as error:
            if attempt + 1 == attempts:
                raise
            delay = min(2 ** attempt, 16)
            print(f"Network error: {error}; retry {attempt + 1}/{attempts - 1} in {delay}s", file=sys.stderr, flush=True)
            time.sleep(delay)


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, *, cache_dir=None):
        super().__init__()
        self.url = url
        def metadata():
            with urlopen(Request(url, method="HEAD"), timeout=120) as response:
                return int(response.headers["Content-Length"]), response.headers.get("ETag")
        self.size, self.etag = retry(metadata)
        self.position = 0
        self.downloaded = 0
        self.cache_hits = 0
        self.cache = None
        if cache_dir is not None:
            if not self.etag:
                raise ValueError("Persistent cache requires an archive ETag")
            identity = hashlib.sha256(f"{url}\n{self.etag}\n{self.size}".encode()).hexdigest()
            self.cache = Path(cache_dir) / identity
            self.cache.mkdir(parents=True, exist_ok=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        end = self.position + size - 1
        headers = {"Range": f"bytes={self.position}-{end}"}
        if self.etag:
            headers["If-Match"] = self.etag
        cached = self.cache / f"{self.position}-{size}.bin" if self.cache else None
        if cached is not None and cached.is_file():
            data = cached.read_bytes()
            if len(data) != size:
                raise RuntimeError(f"Invalid cached Range size: {cached}")
            self.cache_hits += size
        else:
            def fetch():
                with urlopen(Request(self.url, headers=headers), timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("Server ignored Range; refusing a full archive download")
                    expected = f"bytes {self.position}-{end}/{self.size}"
                    if response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Unexpected Content-Range")
                    block = response.read(size + 1)
                if len(block) != size:
                    raise http.client.IncompleteRead(block, size)
                return block
            data = retry(fetch)
            self.downloaded += size
            if cached is not None:
                with tempfile.NamedTemporaryFile(dir=self.cache, delete=False) as temporary:
                    temporary.write(data)
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, cached)
        self.position += size
        return data


def extract_member(archive, info, destination, *, resume=False):
    """Replay/verify an existing uncompressed prefix, then append; ZIP verifies full CRC."""
    if destination.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {destination}; use --resume")
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > info.file_size:
        raise ValueError(f"Existing file exceeds archive member size: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"{info.filename}: {existing:,}/{info.file_size:,} bytes already on disk. "
          "Replaying and verifying the compressed stream before appending.", flush=True)
    # Reading with a separate handle keeps existing bytes intact if download or verification fails.
    with destination.open("ab") as target, destination.open("rb") as prefix, archive.open(info) as source:
        processed = 0
        last_progress = time.monotonic()
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            check = min(len(block), max(0, existing - processed))
            if check and prefix.read(check) != block[:check]:
                raise ValueError(f"Existing prefix differs from archive: {destination}; no old bytes overwritten")
            if check < len(block):
                target.write(block[check:])
                target.flush()
            processed += len(block)
            if time.monotonic() - last_progress >= 5 or processed == info.file_size:
                print(f"{info.filename}: stream {processed / info.file_size:.1%}; "
                      f"file {max(existing, processed):,}/{info.file_size:,} bytes", flush=True)
                last_progress = time.monotonic()
        if processed != info.file_size:
            raise RuntimeError("Incomplete decompressed archive member")
    print(f"CRC verified: {info.filename}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/initial_audit"))
    parser.add_argument("--extract", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true",
                        help="Cache compressed ranges; verify existing output prefix and append missing bytes")
    parser.add_argument("--audit-sample", action="store_true",
                        help="Four evenly spaced train shards, all val/test tables, val vectors and metadata")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remote = HTTPRangeReader(URL, cache_dir=args.output / ".http-cache" if args.resume else None)
    manifest_path = args.output / "archive_manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["etag"] != remote.etag or previous["archive_bytes"] != remote.size:
            raise ValueError("Archive changed since the original download; existing prefix is preserved")
    with zipfile.ZipFile(remote) as archive:
        members = [{"name": m.filename, "bytes": m.file_size,
                    "compressed_bytes": m.compress_size, "method": m.compress_type,
                    "crc32": f"{m.CRC:08x}"} for m in archive.infolist()]
        manifest = {"url": URL, "archive_bytes": remote.size, "etag": remote.etag,
                    "members": members}
        (args.output / "archive_manifest.json").write_text(json.dumps(manifest, indent=2))
        names = list(args.extract)
        if args.audit_sample:
            train = sorted(m["name"] for m in members if "/train/impressions/" in m["name"]
                           and m["name"].endswith(".parquet"))
            names += [train[i] for i in (0, len(train) // 3, 2 * len(train) // 3, len(train) - 1)]
            names += [m["name"] for m in members if (
                m["name"].endswith(("readme.md", "data.meta", "sample_submission.parquet"))
                or ("/val/" in m["name"] and not m["name"].endswith("/"))
                or ("/test/impressions/" in m["name"] and m["name"].endswith(".parquet")))]
        for name in dict.fromkeys(names):
            info = archive.getinfo(name)
            destination = (args.output / name).resolve()
            if args.output.resolve() not in destination.parents:
                raise ValueError("Unsafe member path")
            extract_member(archive, info, destination, resume=args.resume)
            print(json.dumps({"extracted": name, "bytes": info.file_size}), flush=True)
    print(json.dumps({"manifest": str(args.output / "archive_manifest.json"),
                      "member_count": len(members), "downloaded_bytes": remote.downloaded,
                      "cached_bytes_reused": remote.cache_hits}), flush=True)


if __name__ == "__main__":
    main()
