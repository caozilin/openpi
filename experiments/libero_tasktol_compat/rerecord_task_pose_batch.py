"""Fixed 40 x 10 original-speed default-IPOPT reconstruction experiment.

Every selected source episode is counted, including infeasible runs. Native
physics, observations and videos are produced only by the maintained recorder.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def make_plan(seed=20261024):
    selection = ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'
    tasks = json.loads(selection.read_text())['tasks']
    assert len(tasks) == 40
    plan = []
    for case, task in enumerate(tasks):
        with h5py.File(task['source_hdf5'], 'r') as source:
            demos = sorted(source['data'], key=lambda d: int(d[5:]))
            assert len(demos) == 50
            rng = np.random.default_rng(seed+case)
            indices = sorted(rng.choice(len(demos), size=10, replace=False).tolist())
            with h5py.File(task['processed_hdf5'], 'r') as ledger:
                available = set(ledger['regeneration/episodes'])
            for index in indices:
                demo = demos[index]
                assert demo in available
                group = source['data'][demo]
                states, actions = group['states'][()], group['actions'][()]
                plan.append(dict(case=case, suite=task['suite'], task_id=task['task_id'], demo=demo,
                    source_hdf5=task['source_hdf5'], planned_frames=len(states)-1,
                    source_input_sha256=hashlib.sha256(states.tobytes()+actions.tobytes()+str(group.attrs['model_file']).encode()).hexdigest()))
    return dict(seed=seed, sampling='10 of all 50 source demos without replacement per task; RNG seed+case; no outcome filter',
                tasks=40, episodes=400, selection_manifest=str(selection), selection_manifest_sha256=sha(selection), plan=plan)


def accepts(previous, trial):
    return bool(trial['full_length'] and not trial['rejected']
                and (not previous['success'] or trial['success'])
                and trial['objective'] < previous['objective'])


def summarize(results, plan):
    valid = [r for r in results if 'error' not in r]
    baseline = sum(r['baseline']['success'] for r in valid)
    selected = sum(r['selected']['success'] for r in valid)
    summary = dict(complete=len(results) == len(plan['plan']), planned=len(plan['plan']), completed=len(results),
        process_errors=len(results)-len(valid), baseline_success=baseline, selected_success=selected,
        baseline_rejected=sum(r['baseline']['rejected'] for r in valid),
        selected_rejected=sum(r['selected']['rejected'] for r in valid),
        baseline_success_rate_all_planned=baseline/len(plan['plan']),
        selected_success_rate_all_planned=selected/len(plan['plan']),
        physical_recordings=sum(len(r['trials']) for r in valid), results=sorted(results, key=lambda r: (r['case'], int(r['demo'][5:]))))
    summary['suites'] = {}
    for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
        group = [r for r in valid if r['suite'] == suite]
        summary['suites'][suite] = dict(planned=sum(j['suite'] == suite for j in plan['plan']), completed=len(group),
            baseline_success=sum(r['baseline']['success'] for r in group), selected_success=sum(r['selected']['success'] for r in group),
            baseline_rejected=sum(r['baseline']['rejected'] for r in group), selected_rejected=sum(r['selected']['rejected'] for r in group))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.resume:
        plan = json.loads((output/'plan.json').read_text())
        policy = json.loads((output/'policy.json').read_text())
        assert policy['source_sha256'][str(Path(__file__).resolve())] == sha(Path(__file__))
    else:
        output.mkdir(parents=True, exist_ok=False)
        plan = make_plan()
        atomic_json(output/'plan.json', plan)
        shared = ROOT.parent/'franka_mujoco'
        sources = [Path(__file__), Path(__file__).with_name('rerecord_task_pose_reference.py'),
            Path(__file__).with_name('rerecord_state_reference.py'),
            Path(__file__).with_name('probe_original_speed_reference.py'),
            Path(__file__).with_name('task_pose_object_metrics.py'),
            Path(__file__).with_name('observe_online_ipopt.py'),
            shared/'tasktol/control/ipopt/controller.py', shared/'tasktol/control/ipopt/adapter.py',
            shared/'tasktol/control/task_space/objective.py', shared/'tasktol/control/task_space/kinematics.py',
            shared/'tasktol/simulation/libero.py', shared/'tasktol/simulation/libero_joint_controller.py']
        policy = dict(gains=[.5, .25], projection_posture_weight=0., robot_joint_tracking_penalty=False,
            finger_joint_tracking_penalty=False, online_optimizer='unchanged IpoptSettings() and default objective',
            optimizer_max_iterations=20, optimizer_max_cpu_time_s=.095, control_hz=20, physics_hz=500,
            gripper='original unchanged', settling=0, skip_noops=False, controller='maintained B10 full seven-joint reference',
            trials='baseline plus 2 uniform IRLC trials for every full-length selected reference, successful or not',
            rejected_prefix_policy='retain rejection; no correction when no full rollout exists; no OSC fallback',
            acceptance='full feasible rollout, retain success, strictly decrease fixed EE+object cost',
            cem_search=False, historical_candidates_reused=False,
            source_sha256={str(p.resolve()): sha(p) for p in sources})
        atomic_json(output/'policy.json', policy)
        (output/'source_snapshot').mkdir()
        for index, path in enumerate(sources):
            (output/'source_snapshot'/f'{index}_{path.name}').write_bytes(path.read_bytes())
    if args.prepare_only:
        print(json.dumps(dict(prepared=True, tasks=plan['tasks'], episodes=plan['episodes'], output=str(output))))
        return
    for path, digest in policy['source_sha256'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',
        MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
    script = Path(__file__).with_name('rerecord_task_pose_reference.py')
    analyzer = Path(__file__).with_name('analyze_state_reference.py')

    def trial(job, folder, parent=None, gain=None):
        command = [sys.executable, str(script), '--case', str(job['case']), '--demo', job['demo'], '--output', str(folder)]
        if parent is not None:
            command += ['--parent', str(parent), '--gain', str(gain)]
        if not (folder/'task_pose_object_metrics.json').is_file():
            if folder.exists():
                raise RuntimeError(f'Incomplete trial preserved for inspection: {folder}')
            with folder.with_suffix('.log').open('w') as stream:
                done = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if done.returncode:
                raise RuntimeError(f'Recorder failed ({done.returncode}): {folder}')
        if not (folder/'comparison.json').exists():
            with folder.with_suffix('.validation.log').open('w') as stream:
                done = subprocess.run([sys.executable, str(analyzer), '--root', str(folder)], env=env, stdout=stream, stderr=subprocess.STDOUT)
            if done.returncode:
                raise RuntimeError(f'Recording audit failed ({done.returncode}): {folder}')
        comparison = json.loads((folder/'comparison.json').read_text())
        assert all(comparison['source_checks'].values())
        report_path = folder/job['suite']/f"{job['task_id']:02d}"/job['demo']/'ipopt_state_B10/report.json'
        report = json.loads(report_path.read_text())
        assert report['source_input_sha256'] == job['source_input_sha256']
        assert report['solver_settings']['max_iterations'] == 20 and report['solver_settings']['max_cpu_time_s'] == .095
        return json.loads((folder/'task_pose_object_metrics.json').read_text())

    def run(job):
        root = output/f"case{job['case']:02d}_{job['demo']}"
        root.mkdir(exist_ok=True)
        marker = root/'selection.json'
        if marker.exists():
            return json.loads(marker.read_text())
        started = time.monotonic()
        try:
            initial = root/'baseline'
            baseline = trial(job, initial)
            parent, current = initial, baseline
            trials = [dict(root=str(initial), stage='baseline', accepted=True, **baseline)]
            skipped = []
            for index, gain in enumerate(policy['gains']):
                if not current['full_length'] or current['rejected']:
                    skipped.append(dict(iteration=index+1, gain=gain, reason='no full feasible parent rollout'))
                    continue
                folder = root/f'iteration_{index+1}'
                metric = trial(job, folder, parent, gain)
                accepted = accepts(current, metric)
                decision = dict(root=str(folder), parent=str(parent), iteration=index+1, gain=gain,
                                accepted=accepted, previous_objective=current['objective'], **metric)
                atomic_json(folder/'acceptance.json', decision)
                trials.append(decision)
                if accepted:
                    parent, current = folder, metric
            result = dict(**job, baseline_root=str(initial), selected_root=str(parent), baseline=baseline,
                          selected=current, trials=trials, skipped=skipped, wall_s=time.monotonic()-started)
            atomic_json(marker, result)
            return result
        except Exception as error:
            result = dict(**job, error=str(error), wall_s=time.monotonic()-started)
            atomic_json(root/'process_error.json', result)
            return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in plan['plan']]):
            results.append(future.result())
            summary = summarize(results, plan)
            atomic_json(output/'summary.json', summary)
            print(json.dumps({k: summary[k] for k in ('complete', 'planned', 'completed', 'process_errors',
                'baseline_success', 'selected_success', 'baseline_rejected', 'selected_rejected', 'physical_recordings')}), flush=True)
    rows = []
    for result in results:
        if 'error' in result:
            rows.append(dict(case=result['case'], suite=result['suite'], task_id=result['task_id'], demo=result['demo'],
                             process_error=result['error'], baseline_success=False, selected_success=False,
                             rejected=False, selected_root='', task_cost='', executed_frames=0))
        else:
            rows.append(dict(case=result['case'], suite=result['suite'], task_id=result['task_id'], demo=result['demo'],
                process_error='', baseline_success=result['baseline']['success'], selected_success=result['selected']['success'],
                rejected=result['selected']['rejected'], selected_root=result['selected_root'],
                task_cost=result['selected']['objective'], executed_frames=result['selected']['frames']-1))
    with (output/'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assert summary['process_errors'] == 0
    assert all(sha(path) == digest for path, digest in policy['source_sha256'].items())


if __name__ == '__main__':
    main()
