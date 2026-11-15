"""List or extract selected official ZIP members using bounded HTTP Range reads.

Avoids downloading the entire competition archive for an initial audit.
Every extracted member is verified by zipfile's CRC check.
"""
import argparse
import io
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

URL = "https://storage.yandexcloud.net/avitotech-mlcup-2026/competition.zip"


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        with urlopen(Request(url, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.position = 0
        self.downloaded = 0

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
        with urlopen(Request(self.url, headers=headers), timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Server ignored Range; refusing a full archive download")
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected:
                raise RuntimeError("Unexpected Content-Range")
            data = response.read(size + 1)
        if len(data) != size:
            raise RuntimeError("Incomplete or excessive Range response")
        self.position += size
        self.downloaded += size
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/initial_audit"))
    parser.add_argument("--extract", nargs="*", default=[])
    parser.add_argument("--audit-sample", action="store_true",
                        help="Four evenly spaced train shards, all val/test tables, val vectors and metadata")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remote = HTTPRangeReader(URL)
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
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            print(json.dumps({"extracted": name, "bytes": info.file_size}), flush=True)
    print(json.dumps({"manifest": str(args.output / "archive_manifest.json"),
                      "member_count": len(members), "downloaded_bytes": remote.downloaded}), flush=True)


if __name__ == "__main__":
    main()
