"""A fixed, auditable offline reconstruction search, not controller fallback.

Try state/pulse, state/original-grip, then sustained-open-contact labels. Each
trial is a fresh full IPOPT recording. Accept only complete final successes;
never execute OSC as a fallback or pass a future joint witness to the client.
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

import h5py

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SELECTION = ROOT / 'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'
CANDIDATES = ['state_s2_pulse', 'state_s2_original_grip', 'state_s2_open_contact_pulse']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_for(provenance):
    if (provenance.get('time_scale', 1), provenance['lookahead_steps'], provenance.get('task_kp', 150)) != (2, 1, 150):
        return None
    label = provenance.get('label_method', 'achieved_state_fk')
    grip = provenance.get('gripper_policy', 'preserve')
    if label == 'achieved_state_fk':
        return 'state_s2_pulse' if grip == 'native_integrator_pulse' else 'state_s2_original_grip'
    if label == 'state_free_source_bias_sustained_open_contact_bounded_FK' and grip == 'native_integrator_pulse':
        return 'state_s2_open_contact_pulse'
    return None


def inspect_trial(root, job, expected_source_sha):
    comparison = json.loads((root / 'comparison.json').read_text())
    assert all(comparison['source_checks'].values())
    dest = root / job['suite'] / f"{job['task_id']:02d}" / job['demo']
    folder = dest / 'ipopt_state_B10'
    report_path = folder / 'report.json'
    report = json.loads(report_path.read_text())
    assert report['source_input_sha256'] == expected_source_sha
    assert report['tolerance'] == 'off' and report['full_joint_reference']
    assert report['time_scale'] == 2 and report['lookahead_steps'] == 1
    assert report.get('task_kp', 150) == 150
    rows = json.loads((folder / 'trajectory.json').read_text())
    assert sum(not row['aborted'] for row in rows) == report['executed_frames']
    assert all(row['physics_steps'] == (0 if row['aborted'] else 25) for row in rows)
    if report['success']:
        assert not report['execution_rejected']
        assert report['executed_frames'] == report['expected_frames'] == len(rows)
        assert all(row['optimizer']['feasible'] and row['native_joint_goal_error_rad'] < 1e-10 for row in rows)
        assert rows[-1]['success']
    recording = folder / 'recording.hdf5'
    with h5py.File(recording, 'r') as file:
        assert (job['demo'] in file['data']) == report['success']
    return dict(root=str(root.resolve()), success=report['success'], rejected=report['execution_rejected'],
                executed_frames=report['executed_frames'], expected_frames=report['expected_frames'],
                report=str(report_path.resolve()), report_sha256=sha(report_path),
                recording=str(recording.resolve()), recording_sha256=sha(recording),
                comparison_sha256=sha(root / 'comparison.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-runs', type=Path, nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output / 'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())['plan']
    assert len({(j['case'], j['demo']) for j in plan}) == len(plan)
    if args.resume:
        assert json.loads((output / 'plan.json').read_text())['plan'] == plan
        saved_policy = json.loads((output / 'policy.json').read_text())
        assert saved_policy['candidates'] == CANDIDATES
        for path, expected in saved_policy['source_sha256'].items():
            if Path(path).resolve() != Path(__file__).resolve():
                assert sha(path) == expected, f'Candidate implementation changed: {path}'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    session = output / 'execution_sessions' / stamp
    session.mkdir(parents=True)
    shutil.copyfile(__file__, session / Path(__file__).name)
    selection = json.loads(SELECTION.read_text())['tasks']
    cache = {}
    for root in args.cache_runs:
        root = root.resolve()
        provenance = json.loads((root / 'provenance.json').read_text())
        candidate = candidate_for(provenance)
        if candidate is None:
            raise ValueError(f'Cache has no supported fixed candidate: {root}')
        assert (root / 'comparison.json').exists()
        for job in provenance['plan']:
            key = (job['case'], job['demo'], candidate)
            assert key not in cache, f'Duplicate candidate cache: {key}'
            cache[key] = root
    if not args.resume:
        (output / 'plan.json').write_text(json.dumps(dict(plan=plan), indent=2) + '\n')
    policy = dict(schema='libero_state_reconstruction_search_v1', candidates=CANDIDATES,
                  acceptance='all N-1 transitions at S2 completed, every IPOPT solve feasible, final native task predicate true',
                  runtime=dict(task_kp=150, null_kp=10, tolerance='off', control_hz=20, physics_hz=500),
                  controller_fallback=False, source_joint_witness_as_online_input=False,
                  metric='offline recovery after <=3 reannotation trials, distinct from first-trial or learned VLA success',
                  source_sha256={str(p): sha(p) for p in [Path(__file__), HERE / 'rerecord_state_reference.py',
                      HERE / 'probe_open_contact_hybrid.py', HERE / 'open_contact_labeler.py']})
    if not args.resume:
        (output / 'policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    (session / 'policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def run(job):
        folder = output / f"case_{job['case']:02d}_{job['demo']}"
        folder.mkdir(exist_ok=args.resume)
        task = selection[job['case']]
        with h5py.File(task['source_hdf5']) as file:
            group = file['data'][job['demo']]
            source_hash = hashlib.sha256(group['states'][()].tobytes() + group['actions'][()].tobytes() + str(group.attrs['model_file']).encode()).hexdigest()
        selection_path = folder / 'selection.json'
        if args.resume and selection_path.exists():
            saved = json.loads(selection_path.read_text())
            assert saved['source_input_sha256'] == source_hash
            for index, trial in enumerate(saved['trials']):
                assert trial['candidate'] == CANDIDATES[index]
                verified = inspect_trial(Path(trial['root']), job, source_hash)
                for key, value in verified.items():
                    assert trial[key] == value
            assert saved['recovered'] == saved['trials'][-1]['success']
            return saved
        trials = []
        for index, candidate in enumerate(CANDIDATES):
            key = (job['case'], job['demo'], candidate)
            cached = key in cache
            root = cache[key] if cached else folder / f'trial_{index:02d}_{candidate}'
            if not cached and args.resume and root.exists():
                if (root / 'comparison.json').exists():
                    cached = True
                else:
                    partial_summary = root / 'summary.json'
                    finished = partial_summary.exists() and json.loads(partial_summary.read_text()).get('complete', False)
                    if not finished:
                        archive = session / 'interrupted' / folder.name / root.name
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(root), str(archive))
            if not cached:
                if candidate == 'state_s2_open_contact_pulse':
                    command = [sys.executable, str(HERE / 'probe_open_contact_hybrid.py'), '--case', str(job['case']), '--demo', job['demo'], '--output', str(root)]
                else:
                    grip = 'native_integrator_pulse' if candidate == 'state_s2_pulse' else 'preserve'
                    command = [sys.executable, str(HERE / 'rerecord_state_reference.py'), '--plan', str(output / 'plan.json'),
                               '--case', str(job['case']), '--demo', job['demo'], '--output', str(root), '--time-scale', '2',
                               '--gripper-policy', grip, '--modes', 'ipopt_state_B10', '--workers', '1']
                if not root.exists():
                    with (session / f'{folder.name}_trial_{index:02d}.log').open('w') as stream:
                        done = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    assert done.returncode == 0, f'Trial process failed: {folder}'
                with (folder / f'verification_{index:02d}.log').open('w') as stream:
                    verified = subprocess.run([sys.executable, str(HERE / 'analyze_state_reference.py'), '--root', str(root)], env=env, stdout=stream, stderr=subprocess.STDOUT)
                assert verified.returncode == 0, f'Trial verification failed: {folder}'
            trial = dict(candidate=candidate, cached=cached, **inspect_trial(root, job, source_hash))
            trials.append(trial)
            if trial['success']:
                break
        result = dict(**job, source_input_sha256=source_hash, trials=trials,
                      recovered=trials[-1]['success'], selected_root=trials[-1]['root'] if trials[-1]['success'] else None,
                      selected_candidate=trials[-1]['candidate'] if trials[-1]['success'] else None)
        (folder / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in plan]):
            completed.append(future.result())
            totals = dict(original_episodes=len(plan), evaluated=len(completed), first_trial_success=sum(j['trials'][0]['success'] for j in completed),
                          recovered=sum(j['recovered'] for j in completed), reconstruction_trials=sum(len(j['trials']) for j in completed))
            result = dict(complete=len(completed) == len(plan), policy='policy.json', totals=totals, results=completed)
            temporary = output / 'summary.json.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(output / 'summary.json')
            print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
