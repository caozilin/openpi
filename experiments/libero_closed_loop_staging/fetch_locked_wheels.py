"""Fetch large official wheels by ranges, verifying exact hashes in uv.lock.

Network workaround only. It does not resolve dependencies or edit the lockfile.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import os
from pathlib import Path
import time
import tomllib
import urllib.parse
import urllib.request

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--wheels', type=Path)
parser.add_argument('--min-bytes', type=int, default=1024**2)
args = parser.parse_args()
ROOT = args.project.resolve()
TARGET = args.wheels.resolve() if args.wheels else ROOT/'locked_wheels'
TARGET.mkdir(exist_ok=True)
supported = set(cpython_tags(python_version=(3, 12))) | set(compatible_tags(python_version=(3, 12)))
packages = tomllib.loads((ROOT/'uv.lock').read_text())['package']
jobs = []
for package in packages:
    options = [w for w in package.get('wheels', []) if 'url' in w and parse_wheel_filename(urllib.parse.unquote(w['url'].split('/')[-1]))[3] & supported]
    if not options:
        continue
    wheel = min(options, key=lambda w:w.get('size', 10**12))
    if wheel.get('size', 0) < args.min_bytes:
        continue
    filename = urllib.parse.unquote(wheel['url'].split('/')[-1])
    path = TARGET/filename
    expected_hash = wheel['hash'].removeprefix('sha256:')
    if path.is_file() and hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == expected_hash:
        print('CACHED '+filename, flush=True)
        continue
    jobs.append((wheel, path, expected_hash))


def fetch_range(job, fd, start, end):
    wheel, _, _ = job
    for attempt in range(4):
        try:
            request = urllib.request.Request(wheel['url'], headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 206:
                    raise ValueError('Expected HTTP 206')
                data = response.read()
            if len(data) != end-start+1:
                raise ValueError('Incomplete range')
            os.pwrite(fd, data, start)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(.5*(attempt+1))


with ThreadPoolExecutor(max_workers=12) as pool:
    futures = {}
    descriptors = []
    for job in jobs:
        wheel, path, expected = job
        temporary = path.with_suffix(path.suffix+'.partial')
        fd = os.open(temporary, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        os.ftruncate(fd, wheel['size'])
        descriptors.append((fd, temporary, path, expected))
        for start in range(0, wheel['size'], 1024**2):
            futures[pool.submit(fetch_range, job, fd, start, min(start+1024**2-1, wheel['size']-1))] = path.name
    try:
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index%50 == 0:
                print(f'RANGES {index}/{len(futures)}', flush=True)
    finally:
        for fd, *_ in descriptors:
            os.close(fd)
for _, temporary, path, expected in descriptors:
    with temporary.open('rb') as file:
        actual = hashlib.file_digest(file, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('SHA256 mismatch: '+path.name)
    temporary.replace(path)
    print('VERIFIED '+path.name, flush=True)
