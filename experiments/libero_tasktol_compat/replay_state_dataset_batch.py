"""Cold replay all packaged successful recordings with bounded parallelism."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--include-failures', action='store_true')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    jobs = [job for job in manifest['episodes'] if job['success'] or args.include_failures]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shared=Path(__file__).resolve().parents[3]/'franka_mujoco'
    sources=[Path(__file__).resolve(),Path(__file__).with_name('replay_state_dataset.py'),
        Path(__file__).with_name('observe_online_ipopt.py')]+[shared/s for s in (
            'tasktol/control/ipopt/controller.py','tasktol/control/ipopt/adapter.py',
            'tasktol/control/task_space/objective.py','tasktol/control/task_space/kinematics.py',
            'tasktol/simulation/libero.py','tasktol/simulation/libero_joint_controller.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources}
    (output/'source_snapshot').mkdir()
    for i,p in enumerate(sources):(output/'source_snapshot'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    (output / 'plan.json').write_text(json.dumps(dict(dataset=str(dataset), jobs=jobs,source_sha256=hashes), indent=2) + '\n')
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR='/tmp/tasktol_cold_replay_numba', MPLCONFIGDIR='/tmp/tasktol_cold_replay_mpl')

    def run(job):
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        folder = output / job['suite'] / f"{job['task_id']:02d}" / job['demo']
        folder.parent.mkdir(parents=True, exist_ok=True)
        log = output / f"case_{job['case']:02d}_{job['demo']}.log"
        command = [sys.executable, str(Path(__file__).with_name('replay_state_dataset.py')), '--dataset', str(dataset),
                   '--case', str(job['case']), '--demo', job['demo'], '--output', str(folder)]
        if args.include_failures:
            command += ['--include-failure']
        with log.open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report = folder / 'report.json'
        if report.exists():
            return dict(returncode=result.returncode, report=str(report), **json.loads(report.read_text()))
        return dict(case=job['case'], demo=job['demo'], error='replay_process_failed', returncode=result.returncode, log=str(log))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            results.append(future.result())
            summary = dict(complete=len(results) == len(jobs), planned=len(jobs), replayed=len(results),
                           final_success=sum(result.get('final_success', False) for result in results),
                           bitwise_state_match=sum(result.get('states_bitwise_equal', False) for result in results),
                           bitwise_joint_target_match=sum(result.get('joint_targets_bitwise_equal', False) for result in results),
                           reproduced_recording_bitwise=sum(result.get('reproduces_recording_bitwise',False) for result in results),
                           matched_recorded_outcome=sum(result.get('recorded_outcome_matches',False) for result in results),
                           failures_included=args.include_failures,
                           failures=sum(result['returncode'] != 0 for result in results), results=results)
            temporary = output / 'summary.json.tmp'
            temporary.write_text(json.dumps(summary, indent=2) + '\n')
            temporary.replace(output / 'summary.json')
            print(json.dumps({key: value for key, value in summary.items() if key != 'results'}), flush=True)
    assert all(result['returncode'] == 0 for result in results)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (output/'source_unchanged_at_end.json').write_text(json.dumps(dict(unchanged=True))+'\n')
    if args.include_failures:
        from validate_state_dataset import validate
        validate(dataset, output)


if __name__ == '__main__':
    main()
