"""The CLI remains attached to its terminal and releases its port on exit."""
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('entry,extra,stop_signal', [
    ('serve_mask_annotation_viewer.py', [], signal.SIGINT),
    ('serve_mask_annotation_viewer.py', ['--background'], signal.SIGINT),
    ('serve_mask_annotation_viewer.py', ['--benchmark'], signal.SIGTERM),
    ('serve_benchmark_viewer.py', [], signal.SIGHUP),
])
def test_foreground_cli_exits_with_terminal_signal(tmp_path, entry, extra, stop_signal):
    (tmp_path / 'index.html').write_text('<html>test</html>')
    args = [sys.executable, '-u', str(ROOT / 'examples/libero' / entry), '--port', '0', *extra]
    if entry == 'serve_mask_annotation_viewer.py' and '--benchmark' not in extra:
        args.extend(['--directory', str(tmp_path)])
    process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(4):
            line = process.stdout.readline()
            match = re.search(r'http://127\.0\.0\.1:(\d+)', line)
            if match:
                break
        assert match, (line, process.poll())
        port = int(match.group(1))
        assert process.poll() is None
        with socket.create_connection(('127.0.0.1', port), timeout=2):
            pass
        process.send_signal(stop_signal)
        _, errors = process.communicate(timeout=5)
        assert process.returncode == 0, errors
        with socket.socket() as s:
            s.settimeout(.5)
            assert s.connect_ex(('127.0.0.1', port)) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
