"""Reevaluate every original seed0..5 once to establish the formal nominal baseline."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104'
PYTHON = ROOT.parent / 'franka_mujoco/experiments/libero_closed_loop/.venv/bin/python'
FROZEN = ROOT / 'data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def assignments():
    return {str(z): [dict(suite=s,task_id=t,seed=z) for s in SUITES for t in range(10)] for z in range(6)}


def validate(request):
    expected = {(s, t, z) for s in SUITES for t in range(10) for z in range(6)}
    selected = {(r['suite'], r['task_id'], r['seed']) for r in request['selected_episodes']}
    assert selected == expected and len(request['selected_episodes']) == 240
    assert request['worker_assignments'] == assignments()
    assert request['expected_episodes'] == 240 and request['checkpoint_step'] == 10000
    assert request['tolerance_enabled'] is False and request['semantic_mask'] == '000'
    assert request['max_consecutive_solver_failures'] == 20
    for path, expected_sha in request['source_sha256'].items():
        assert sha(Path(path)) == expected_sha, path
    for path, expected_sha in request['preserved_artifact_sha256'].items():
        assert sha(Path(path)) == expected_sha, path


def summarize(run, request):
    rows = [read(p) for p in sorted(run.glob('episodes/*/task_*/seed_*/result.json'))]
    keys = [(r['suite'], r['task_id'], r['seed']) for r in rows]
    expected = {(r['suite'], r['task_id'], r['seed']) for r in request['selected_episodes']}
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    assert all(r['complete'] and r['eval_identity'] == request['eval_identity'] for r in rows)
    errors = [str(p) for p in run.glob('episodes/*/task_*/seed_*/infrastructure_error.json')]
    summary = dict(complete=set(keys) == expected and not errors, expected_episodes=240,
                   completed_episodes=len(rows), successes=sum(r['success'] for r in rows),
                   infrastructure_errors=len(errors), infrastructure_error_files=errors,
                   per_suite={s: dict(episodes=sum(r['suite'] == s for r in rows),
                                      successes=sum(r['suite'] == s and r['success'] for r in rows)) for s in SUITES},
                   updated_at=datetime.datetime.now().astimezone().isoformat())
    dump(run / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--worker', type=int, choices=range(6))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    request = read(run / 'evaluation_request.json')
    sys.path.insert(0, str(ROOT))
    from examples.libero.success_continuation import validate_request as validate_success_request
    validate_success_request(request)
    validate(request)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(FROZEN), str(ROOT), str(ROOT / 'third_party/libero'), str(ROOT / 'packages/openpi-client/src')]
    from examples.libero.solver_failure_hold import validate_failure_policy_request
    validate_failure_policy_request(request, 20)
    if args.worker is not None:
        from examples.libero.eval_reconstructed_libero_local import worker
        jobs = request['worker_assignments'][str(args.worker)]
        episode_keys = {(x['suite'], x['task_id'], x['seed']) for x in jobs}
        for seed in sorted({job['seed'] for job in jobs}):
            worker(run, seed, 20, replica_index=args.worker, episode_keys=episode_keys)
        return
    lock = (run / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump(run / 'evaluation_process.json', dict(pid=os.getpid(), command=sys.argv,
         started_at=datetime.datetime.now().astimezone().isoformat()))
    import urllib.request
    deadline = time.monotonic() + 1800
    for port in request['ports']:
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=5) as response:
                    assert response.status == 200
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Inference readiness timed out') from None
                time.sleep(5)
    children = []
    def stop(signum,frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(signum,stop)
    try:
        for replica in range(6):
            with (run / f'client_{replica}.log').open('a') as log:
                p = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()), '--run-dir', str(run),
                                      '--worker', str(replica)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(p)
            dump(run / f'client_{replica}_process.json', dict(pid=p.pid, replica=replica,
                 command=str(Path(__file__).resolve()), assigned_episodes=len(request['worker_assignments'][str(replica)])))
        dump(run / 'evaluation_state.json', dict(stage='evaluating', pids=[p.pid for p in children], expected_episodes=240))
        while any(p.poll() is None for p in children):
            summarize(run, request)
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Client infrastructure error; inspect per-client logs')
            time.sleep(5)
        summary = summarize(run, request)
        assert summary['complete'] and all(p.returncode == 0 for p in children)
        dump(run / 'evaluation_completion.json', dict(summary, exit_codes=[p.returncode for p in children], eval_identity=request['eval_identity']))
        dump(run / 'evaluation_state.json', dict(stage='complete', episodes=240, successes=summary['successes']))
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == '__main__':
    main()
