"""Extend the audited formal nominal400 with seed10..49 exactly once."""

import argparse
import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from examples.libero.eval_formal_nominal import ROOT, PYTHON, FROZEN, SUITES, read, dump, sha

RUN = ROOT / 'data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105'
BASELINE = ROOT / 'data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_formal_seed0to5_20261104'
IDENTITY = 'libero_reconstructed_step10000_nominal_hold20_seed10to49_20261105'
REMOTE = '/mnt/SharedData/czl/runs/libero_eval_pi05_reconstructed_10000_nominal_hold20_seed10to49_20261105'


def assignments():
    jobs = [dict(suite=s, task_id=t, seed=z) for z in range(10, 50) for s in SUITES for t in range(10)]
    return {str(i): jobs[i::6] for i in range(6)}


def validate(request):
    expected = {(s, t, z) for s in SUITES for t in range(10) for z in range(10, 50)}
    selected = {(r['suite'], r['task_id'], r['seed']) for r in request['selected_episodes']}
    assert selected == expected and len(request['selected_episodes']) == 1600
    assert request['worker_assignments'] == assignments()
    assert request['expected_episodes'] == 1600 and request['target_combined_episodes'] == 2000
    assert request['checkpoint_step'] == 10000 and request['client_location'] == 'local'
    assert request['tolerance_enabled'] is False and request['semantic_mask'] == '000'
    assert request['max_consecutive_solver_failures'] == 20
    assert request['gpu_ids'] == [0, 1, 2, 3, 4, 6]
    assert request['ports'] == list(range(18760, 18766))
    from examples.libero.success_continuation import validate_request
    from examples.libero.solver_failure_hold import validate_failure_policy_request
    validate_request(request)
    validate_failure_policy_request(request, 20)
    for path, digest in request['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in request['preserved_manifest_sha256'].items():
        assert sha(Path(path)) == digest, path


def summarize(run, request):
    rows = [read(p) for p in sorted(run.glob('episodes/*/task_*/seed_*/result.json'))]
    keys = [(r['suite'], r['task_id'], r['seed']) for r in rows]
    expected = {(r['suite'], r['task_id'], r['seed']) for r in request['selected_episodes']}
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    assert all(r['complete'] and r['eval_identity'] == request['eval_identity'] for r in rows)
    errors = list(run.glob('episodes/*/task_*/seed_*/infrastructure_error.json'))
    summary = dict(complete=set(keys) == expected and not errors, expected_episodes=1600,
                   completed_episodes=len(rows), successes=sum(r['success'] for r in rows),
                   infrastructure_errors=len(errors), infrastructure_error_files=list(map(str, errors)),
                   per_suite={s: dict(episodes=sum(r['suite'] == s for r in rows),
                       successes=sum(r['suite'] == s and r['success'] for r in rows)) for s in SUITES},
                   updated_at=datetime.datetime.now().astimezone().isoformat())
    dump(run / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--worker', type=int, choices=range(6))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    request = read(run / 'evaluation_request.json')
    validate(request)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(FROZEN), str(ROOT), str(ROOT / 'third_party/libero'), str(ROOT / 'packages/openpi-client/src')]
    if args.worker is not None:
        from examples.libero.eval_reconstructed_libero_local import worker
        jobs = request['worker_assignments'][str(args.worker)]
        keys = {(j['suite'], j['task_id'], j['seed']) for j in jobs}
        for seed in sorted({j['seed'] for j in jobs}):
            worker(run, seed, 20, replica_index=args.worker, episode_keys=keys)
        return
    lock = (run / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'evaluation_completion.json').exists():
        assert read(run / 'evaluation_completion.json')['complete']
        print('Already complete; no rollouts restarted.', flush=True)
        return
    dump(run / 'evaluation_process.json', dict(pid=os.getpid(), command=sys.argv,
         started_at=datetime.datetime.now().astimezone().isoformat()))
    identities = read(run / 'inference_identity_audit.json')
    assert identities['complete'] and len(identities['replicas']) == 6
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        for replica in range(6):
            with (run / f'client_{replica}.log').open('a') as log:
                process = subprocess.Popen([str(PYTHON), '-u', '-m', 'examples.libero.eval_nominal50',
                    '--run-dir', str(run), '--worker', str(replica)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(process)
            dump(run / f'client_{replica}_process.json', dict(pid=process.pid, replica=replica,
                 command='examples.libero.eval_nominal50', assigned_episodes=len(request['worker_assignments'][str(replica)])))
        dump(run / 'evaluation_state.json', dict(stage='evaluating', pids=[p.pid for p in children], expected_episodes=1600))
        while any(p.poll() is None for p in children):
            summarize(run, request)
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Client infrastructure error; inspect client logs. Completed failures are never replaced.')
            time.sleep(10)
        summary = summarize(run, request)
        assert summary['complete'] and all(p.returncode == 0 for p in children)
        dump(run / 'evaluation_completion.json', dict(summary, exit_codes=[p.returncode for p in children],
             eval_identity=request['eval_identity']))
        dump(run / 'evaluation_state.json', dict(stage='complete', episodes=1600, successes=summary['successes']))
    except BaseException:
        import traceback
        dump(run / 'evaluation_state.json', dict(stage='failed_or_interrupted', error=traceback.format_exc()))
        raise
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
