"""Fetch only train impressions; reuse transferred raw train/val without test reads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import os
import threading
import time
import zipfile

from .archive_http import HTTPRangeReader, extract_member


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--transfer-manifest',type=Path,required=True);p.add_argument('--transfer-ready',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.raw.mkdir(parents=True,exist_ok=True)
    source=json.loads(a.source_config.read_text());transfer=json.loads(a.transfer_manifest.read_text())
    members=source['members'];assert len(members)==512
    preloaded=set(transfer['preloaded_train_shards'])
    needed=[m for m in members if Path(m['name']).name not in preloaded]
    local=threading.local();opened=[];lock=threading.Lock();results={}
    def archive():
        if not hasattr(local,'zip'):
            reader=HTTPRangeReader(source['url'])
            if reader.etag!=source['etag'] or reader.size!=source['archive_bytes']:raise ValueError('Archive identity changed')
            local.zip=zipfile.ZipFile(reader)
            with lock:opened.append((local.zip,reader))
        return local.zip
    def fetch(member):
        name=member['name']
        if not name.startswith('competition/train/impressions/') or not name.endswith('.parquet'):raise ValueError('Unexpected member')
        relative=Path('train/impressions')/Path(name).name;dest=a.raw/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        z=archive();info=z.getinfo(name)
        if info.file_size!=member['bytes'] or f'{info.CRC:08x}'!=member['crc32']:raise ValueError('ZIP member metadata differs')
        if dest.exists():
            import zlib
            crc=0
            with dest.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):crc=zlib.crc32(b,crc)
            if dest.stat().st_size==info.file_size and crc==info.CRC:
                return relative.as_posix(),dict(bytes=info.file_size,sha256=sha256(dest),crc32=member['crc32'])
            raise ValueError('Existing completed shard failed verification')
        partial=dest.with_suffix(dest.suffix+'.partial')
        for attempt in range(3):
            try:
                extract_member(z,info,partial,resume=True)
                os.replace(partial,dest)
                return relative.as_posix(),dict(bytes=info.file_size,sha256=sha256(dest),crc32=member['crc32'])
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
        raise RuntimeError('unreachable')
    started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(fetch,m):m for m in needed}
            for future in as_completed(futures):
                name,result=future.result();results[name]=result
                status=dict(stage='downloading',completed=len(results),total=len(needed),
                            preloaded=len(preloaded),completed_bytes=sum(x['bytes'] for x in results.values()),
                            elapsed_seconds=time.monotonic()-started,updated_at=time.time())
                write(a.raw/'download-progress.json',status)
                print(json.dumps(status),flush=True)
        until=time.monotonic()+3600
        while not a.transfer_ready.exists():
            if time.monotonic()>until:raise TimeoutError('Mac transfer not confirmed')
            time.sleep(3)
        for name,expected in transfer['files'].items():
            path=a.raw/name
            if not path.exists() or path.stat().st_size!=expected['bytes'] or sha256(path)!=expected['sha256']:
                raise ValueError('Transferred file failed SHA verification: '+name)
            results[name]=expected
        actual={p.name for p in (a.raw/'train/impressions').glob('*.parquet')}
        required={Path(m['name']).name for m in members}
        if actual!=required:raise ValueError('Full train coverage differs')
        manifest=dict(status='completed',files=results,archive_etag=source['etag'],
                      source_config_sha256=sha256(a.source_config),train_shards=len(actual),
                      network_bytes=sum(r.downloaded for _,r in opened),test_read=False)
        write(a.raw/'raw-ready.json',manifest)
        write(a.raw/'download-progress.json',dict(stage='completed',completed=len(needed),total=len(needed)))
        print(json.dumps(dict(status='completed',train_shards=len(actual),network_bytes=manifest['network_bytes'])),flush=True)
    finally:
        for z,r in opened:z.close();r.close()


if __name__=='__main__':main()
