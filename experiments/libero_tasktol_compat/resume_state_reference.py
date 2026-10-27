"""Resume interrupted state-reference batches without rerunning completed jobs.

The original experiment implementation and parameters are checked unchanged.
Incomplete shards are archived, never accepted as successful recordings.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(results, modes):
    return {mode: {key: sum(bool(report[field]) for job in results for report in job.get('results', []) if report['mode'] == mode)
                   for key, field in [('final', 'success'), ('ever', 'ever_success'), ('rejected', 'execution_rejected')]}
            for mode in modes}


def resume(root, workers):
    root = root.resolve()
    provenance = json.loads((root / 'provenance.json').read_text())
    for path, expected in provenance['source_sha256'].items():
        assert sha(path) == expected, f'Experiment implementation changed: {path}'
    runner = Path(__file__).with_name('rerecord_state_reference.py').resolve()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    session = root / 'resume_sessions' / stamp
    session.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, session / Path(__file__).name)
    plan_path = session / 'plan.json'
    plan_path.write_text(json.dumps(dict(plan=provenance['plan']), indent=2) + '\n')
    completed, pending, archived = [], [], []
    for job in provenance['plan']:
        dest = root / job['suite'] / f"{job['task_id']:02d}" / job['demo']
        summary = dest / 'summary.json'
        if summary.exists():
            value = json.loads(summary.read_text())
            assert [r['mode'] for r in value['results']] == provenance['modes']
            for report in value['results']:
                folder = dest / report['mode']
                assert (folder / 'recording.hdf5').exists() and (folder / 'trajectory.json').exists()
                assert json.loads((folder / 'report.json').read_text()) == report
            completed.append(value)
        else:
            if dest.exists():
                archive = session / 'interrupted' / job['suite'] / f"{job['task_id']:02d}" / job['demo']
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(archive))
                archived.append(str(archive))
            pending.append(job)
    metadata = dict(resumer_sha256=sha(__file__), unchanged_experiment_sources=True,
                    reused_completed=len(completed), pending=pending, archived=archived)
    (session / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def save():
        value = dict(complete=len(completed) == len(provenance['plan']), aggregate=aggregate(completed, provenance['modes']), results=completed)
        temporary = root / 'summary.json.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(root / 'summary.json')
        print(json.dumps(dict(completed=len(completed), total=len(provenance['plan']), aggregate=value['aggregate'])), flush=True)

    def run(job):
        log = session / f"case_{job['case']:02d}_{job['demo']}.log"
        env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, str(runner), '--worker', '--plan', str(plan_path), '--case', str(job['case']),
                   '--demo', job['demo'], '--output', str(root), '--time-scale', str(provenance.get('time_scale', 1)),
                   '--lookahead', str(provenance['lookahead_steps']),
                   '--gripper-policy', provenance.get('gripper_policy', 'preserve'),
                   '--label-method', provenance.get('label_method', 'achieved_state_fk'),
                   '--pulse-phase', str(provenance.get('pulse_phase', 0)), '--modes', *provenance['modes']]
        with log.open('w') as stream:
            done = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        summary = root / job['suite'] / f"{job['task_id']:02d}" / job['demo'] / 'summary.json'
        return json.loads(summary.read_text()) if summary.exists() else dict(**job, error='worker_failed', returncode=done.returncode, log=str(log))

    save()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in pending]):
            completed.append(future.result())
            save()
    if any('error' in job for job in completed):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    with (args.root / 'resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resume(args.root, args.workers)


if __name__ == '__main__':
    main()
