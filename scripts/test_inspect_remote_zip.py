import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from inspect_remote_zip import HTTPRangeReader, extract_member


class Response:
    status = 206

    def __init__(self, headers, data=b"", fail=False):
        self.headers, self.data, self.fail = headers, data, fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        if self.fail:
            raise TimeoutError("simulated socket timeout")
        return self.data[:size]


class DownloadRecovery(unittest.TestCase):
    def test_timeout_retries_same_range_then_reuses_disk_cache(self):
        requests = []

        def open_request(request, timeout):
            if request.get_method() == "HEAD":
                return Response({"Content-Length": "12", "ETag": '"fixed"'})
            requests.append(request.get_header("Range"))
            self.assertEqual(request.get_header("If-match"), '"fixed"')
            return Response({"Content-Range": "bytes 2-5/12"}, b"cdef", fail=len(requests) == 1)

        with tempfile.TemporaryDirectory() as root, patch("inspect_remote_zip.urlopen", side_effect=open_request), \
                patch("inspect_remote_zip.time.sleep"):
            reader = HTTPRangeReader("https://example.test/archive", cache_dir=root)
            reader.seek(2)
            self.assertEqual(reader.read(4), b"cdef")
            self.assertEqual(requests, ["bytes=2-5", "bytes=2-5"])
            self.assertEqual(reader.tell(), 6)
            restarted = HTTPRangeReader("https://example.test/archive", cache_dir=root)
            restarted.seek(2)
            self.assertEqual(restarted.read(4), b"cdef")
            self.assertEqual(len(requests), 2)
            self.assertEqual(restarted.downloaded, 0)
            self.assertEqual(restarted.cache_hits, 4)

    def test_existing_decompressed_prefix_is_verified_and_appended(self):
        content = b"embedding bytes" * 10000
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.memmap", content)
        with tempfile.TemporaryDirectory() as root, zipfile.ZipFile(buffer) as archive:
            path = Path(root) / "data.memmap"
            prefix = content[:7777]
            path.write_bytes(prefix)
            with self.assertRaises(FileExistsError):
                extract_member(archive, archive.getinfo("data.memmap"), path)
            self.assertEqual(path.read_bytes(), prefix)
            extract_member(archive, archive.getinfo("data.memmap"), path, resume=True)
            self.assertEqual(path.read_bytes(), content)
            # A finished file can be verified again, without appending duplicate bytes.
            extract_member(archive, archive.getinfo("data.memmap"), path, resume=True)
            self.assertEqual(path.read_bytes(), content)
            path.write_bytes(b"wrong prefix")
            with self.assertRaisesRegex(ValueError, "prefix differs"):
                extract_member(archive, archive.getinfo("data.memmap"), path, resume=True)
            self.assertEqual(path.read_bytes(), b"wrong prefix")

    def test_interruption_keeps_partial_output_and_restart_completes(self):
        content = b"a" * (9 * 1024 * 1024)
        info = zipfile.ZipInfo("data.memmap")
        info.file_size = len(content)

        class Interrupted(io.BytesIO):
            def read(self, size):
                if self.tell():
                    raise TimeoutError("interrupted after first block")
                return super().read(size)

        class Archive:
            def __init__(self, interrupted):
                self.interrupted = interrupted

            def open(self, info):
                return Interrupted(content) if self.interrupted else io.BytesIO(content)

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data.memmap"
            with self.assertRaises(TimeoutError):
                extract_member(Archive(True), info, path, resume=True)
            self.assertEqual(path.stat().st_size, 8 * 1024 * 1024)
            extract_member(Archive(False), info, path, resume=True)
            self.assertEqual(path.read_bytes(), content)

    def test_server_ignoring_range_is_rejected(self):
        response = Response({})
        response.status = 200
        with patch("inspect_remote_zip.urlopen", side_effect=[Response({"Content-Length": "12"}), response]):
            reader = HTTPRangeReader("https://example.test/archive")
            with self.assertRaisesRegex(RuntimeError, "ignored Range"):
                reader.read(4)


if __name__ == "__main__":
    unittest.main()
