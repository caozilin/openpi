"""Launch the authorized finite reconstruction batch beyond a temporary shell.

No solver configuration or source selection changes. All outputs stay under
the existing OpenPI experiment/data directories.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_v1'
DATASET = ROOT/'data/libero_tasktol_state/20261024_task_pose_default_ipopt400_v1'


def main():
    policy = json.loads((RUN/'policy.json').read_text())
    for path, expected in policy['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    metadata_path = RUN/'batch_host_process.json'
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text())
        for key in ('batch_pid', 'finish_pid'):
            try:
                os.kill(prior[key], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Existing host process may still run: {key}={prior[key]}')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT)+':'+str(ROOT.parent/'franka_mujoco'),
        NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba', MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
    commands = [
        [sys.executable, '-u', str(Path(__file__).with_name('rerecord_task_pose_batch.py')),
         '--output', str(RUN), '--resume', '--workers', '6'],
        [sys.executable, '-u', str(Path(__file__).with_name('finish_task_pose_batch.py')),
         '--run', str(RUN), '--dataset', str(DATASET), '--wait'],
    ]
    pids = []
    for command, name in zip(commands, ('batch_host.log', 'finish_host.log'), strict=True):
        with (RUN/name).open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        pids.append(process.pid)
    metadata = dict(batch_pid=pids[0], finish_pid=pids[1], commands=commands,
        launched_at_utc=datetime.now(timezone.utc).isoformat(), finite_plan_episodes=400,
        saved_completed_episodes=len(list(RUN.glob('case*/selection.json'))), optimizer_unchanged=True)
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
