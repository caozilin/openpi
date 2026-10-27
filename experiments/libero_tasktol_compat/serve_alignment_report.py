"""Loopback-only static report server restricted to experiment evidence."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path('/media/czl/sata/franka_my_code')
ALLOWED=(ROOT/'openpi/experiments/libero_tasktol_compat/results',ROOT/'franka_mujoco/experiments/libero_closed_loop/results')


class EvidenceHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path=Path(self.translate_path(self.path)).resolve()
        if not any(path.is_relative_to(root) for root in ALLOWED):
            self.send_error(403,'Only experiment evidence is served')
            return None
        return super().send_head()


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8770),partial(EvidenceHandler,directory=str(ROOT))).serve_forever()
