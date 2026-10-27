"""Loopback-only, read-only static preview with MP4 byte-range seeking."""
from __future__ import annotations
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil
import json
import subprocess
import sys
import time
from urllib.request import urlopen

DEFAULT = Path(__file__).resolve().parents[2] / 'data/libero40_mask_annotation/20261019_viewer_10demos_v2'

def byte_range(header: str | None, size: int):
    if not header: return None
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
    if not match or size <= 0: raise ValueError('Unsupported range')
    first,last=match.groups()
    if not first:
        suffix=int(last) if last else 0
        if suffix<=0:raise ValueError('Invalid suffix')
        return max(0,size-suffix),size-1
    start=int(first);end=min(int(last),size-1) if last else size-1
    if start>=size or start>end:raise ValueError('Unsatisfiable range')
    return start,end

class ViewerHandler(SimpleHTTPRequestHandler):
    """Serve only files inside the viewer root, never source HDF5 paths."""
    def send_head(self):
        root=Path(self.directory).resolve();path=Path(self.translate_path(self.path)).resolve()
        self._response_range=None
        if not path.is_relative_to(root):self.send_error(403,'Outside viewer directory');return None
        if path.is_dir():path=path/'index.html'
        if not path.is_file():self.send_error(404,'File not found');return None
        size=path.stat().st_size
        try:requested=byte_range(self.headers.get('Range'),size)
        except ValueError:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
        file=path.open('rb')
        if requested:
            start,end=requested;file.seek(start);self._response_range=requested
            self.send_response(206);self.send_header('Content-Range',f'bytes {start}-{end}/{size}');length=end-start+1
        else:self.send_response(200);length=size
        self.send_header('Content-Type',self.guess_type(str(path)));self.send_header('Content-Length',str(length))
        self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-cache')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();return file

    def copyfile(self, source, outputfile):
        if self._response_range is None:shutil.copyfileobj(source,outputfile);return
        remaining=self._response_range[1]-self._response_range[0]+1
        while remaining:
            block=source.read(min(2**16,remaining))
            if not block:break
            outputfile.write(block);remaining-=len(block)

    def log_message(self, format, *args):
        if args and str(args[1] if len(args)>1 else '') not in ('200','206','304'):
            super().log_message(format,*args)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parent if Path(__file__).name=='serve.py' else DEFAULT
    ap.add_argument('--directory',type=Path,default=default);ap.add_argument('--port',type=int,default=8768)
    ap.add_argument('--background',action='store_true',help='Keep the loopback preview alive after the launching shell exits')
    args=ap.parse_args();directory=args.directory.resolve()
    if not (directory/'index.html').is_file():raise SystemExit(f'No viewer index.html: {directory}')
    if args.background:
        if args.port==0:raise SystemExit('--background requires an explicit port')
        qa=directory/'qa';qa.mkdir(exist_ok=True)
        with (qa/'preview_server.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--directory',str(directory),'--port',str(args.port)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        url=f'http://127.0.0.1:{args.port}/index.html'
        for _ in range(50):
            if child.poll() is not None:raise SystemExit(f'Preview exited; see {qa / "preview_server.log"}')
            try:
                with urlopen(url,timeout=.5) as response:
                    if response.status==200:
                        (qa/'preview_server.json').write_text(json.dumps(dict(pid=child.pid,url=url,directory=str(directory)),indent=2))
                        print(f'LIBERO Dataset Studio: {url} (background PID {child.pid})',flush=True)
                        return
            except OSError:pass
            time.sleep(.1)
        child.terminate();raise SystemExit('Preview did not become ready in5seconds')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(ViewerHandler,directory=str(directory)))
    print(f'LIBERO Dataset Studio: http://127.0.0.1:{server.server_port}/index.html',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
