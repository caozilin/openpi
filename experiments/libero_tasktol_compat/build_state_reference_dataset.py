"""Package verified IPOPT recollections into task HDF5 indexes.

Images and simulator states link to the retained rollout shards. Failures remain
in attempts, but never appear in the successful training data group.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / 'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'
CONTRACT = dict(
    shape=[7], position='base/world metres, adjacent independent target increments',
    rotation='world SO(3) rotation vector radians, left composition', gripper='-1 open / +1 close; 0 holds native integrator when pulse retiming is enabled; never binarize 0',
    execution='anchor at initial actual FK; accumulate target; strict IPOPT; full seven-joint reference via B10',
    frequency_hz=20, physics_hz=500, tolerance='off', extra_delta_transform=False,
    gripper_decode=dict(negative=-1, hold=0, positive=1, deadband=.5, after='policy action unnormalization'),
    image_orientation='raw LIBERO; rotate 180 degrees once at policy export',
    observation_timing='native pre-command obs; cached joint/EE kinematics lag final physics state by one 2ms tick, except forwarded initial frame; full pre_action_states stores actual state',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(runs, output, reconstruction=None, refinement=None, task_refinement=None):
    selected = json.loads(SELECTION.read_text())['tasks']
    selection_by_key = None
    refinement_by_key = None
    task_refinement_by_key = None
    assert sum(value is not None for value in (reconstruction, refinement, task_refinement)) <= 1
    if task_refinement is not None:
        task_search = json.loads((task_refinement / 'summary.json').read_text())
        assert task_search['complete'] and task_search['process_errors'] == 0
        task_policy = json.loads((task_refinement / 'policy.json').read_text())
        assert task_policy['robot_joint_tracking_penalty'] is False
        assert task_policy['finger_joint_tracking_penalty'] is False
        task_refinement_by_key = {}
        for job in task_search['results']:
            root = Path(job['selected_root']).resolve()
            assert job['selected']['objective'] <= job['baseline']['objective']
            assert not job['baseline']['success'] or job['selected']['success']
            task_refinement_by_key[(str(root), job['case'], job['demo'])] = job
        assert len(task_refinement_by_key) == task_search['planned']
        runs = list(dict.fromkeys(Path(key[0]) for key in task_refinement_by_key))
    if refinement is not None:
        refined = json.loads((refinement / 'summary.json').read_text())
        assert refined['complete']
        refinement_by_key = {}
        for job in refined['results']:
            assert job['objective'] <= job['baseline_objective']
            assert not job['baseline_success'] or job['success']
            root = Path(job['selected_root']).resolve()
            refinement_by_key[(str(root), job['case'], job['demo'])] = job
        assert len(refinement_by_key) == len(refined['results'])
        runs = list(dict.fromkeys(Path(key[0]) for key in refinement_by_key))
    if reconstruction is not None:
        search = json.loads((reconstruction / 'summary.json').read_text())
        assert search['complete'] and search['totals']['evaluated'] == search['totals']['original_episodes']
        policy = json.loads((reconstruction / 'policy.json').read_text())
        selection_by_key = {}
        for job in search['results']:
            assert [trial['candidate'] for trial in job['trials']] == policy['candidates'][:len(job['trials'])]
            assert not any(trial['success'] for trial in job['trials'][:-1])
            assert job['recovered'] == job['trials'][-1]['success']
            for trial in job['trials']:
                assert sha(trial['report']) == trial['report_sha256']
                assert sha(trial['recording']) == trial['recording_sha256']
            root = Path(job['trials'][-1]['root']).resolve()
            selection_by_key[(str(root), job['case'], job['demo'])] = job
        runs = list(dict.fromkeys(Path(key[0]) for key in selection_by_key))
    tasks, identities, entries, method_parameters = {}, set(), [], set()
    solver_parameters = set()
    for root in runs:
        root = root.resolve()
        comparison = json.loads((root / 'comparison.json').read_text())
        progress = json.loads((root / 'summary.json').read_text())
        assert progress['complete'] and all(comparison['source_checks'].values())
        assert len(progress['results']) == len(comparison['episodes'])
        for job in comparison['episodes']:
            selection_key = (str(root), job['case'], job['demo'])
            if selection_by_key is not None and selection_key not in selection_by_key:
                continue
            if refinement_by_key is not None and selection_key not in refinement_by_key:
                continue
            if task_refinement_by_key is not None and selection_key not in task_refinement_by_key:
                continue
            mode = next(x for x in job['modes'] if x['mode'] == 'ipopt_state_B10')
            identity = (job['case'], job['demo'])
            assert identity not in identities, f'Duplicate recording: {identity}'
            identities.add(identity)
            task = selected[job['case']]
            dest = root / job['suite'] / f"{job['task_id']:02d}" / job['demo']
            folder = dest / 'ipopt_state_B10'
            report = json.loads((folder / 'report.json').read_text())
            solver_parameters.add(json.dumps(report.get('solver_settings'), sort_keys=True))
            assert report['success'] == mode['final'] and report['full_joint_reference']
            provenance = json.loads((root / 'provenance.json').read_text())
            method_parameters.add((report.get('time_scale', 1), report['lookahead_steps'],
                                   provenance.get('gripper_policy', 'preserve'), provenance.get('pulse_phase', 0),
                                   provenance.get('label_method', 'achieved_state_fk'), provenance.get('task_kp', 150)))
            path, reference = folder / 'recording.hdf5', dest / 'state_reference.hdf5'
            rows = json.loads((folder / 'trajectory.json').read_text())
            count = len(rows)
            with h5py.File(path, 'r') as file:
                group = file['attempts'][job['demo']]
                assert len(group['actions']) == count
                assert (job['demo'] in file['data']) == report['success']
                if report['success']:
                    assert count == report['expected_frames'] == report['executed_frames']
                    assert all(r['optimizer']['feasible'] and not r['aborted'] and r['physics_steps'] == 25 for r in rows)
            entry = dict(case=job['case'], suite=job['suite'], task_id=job['task_id'], demo=job['demo'],
                         success=report['success'], ever_success=report['ever_success'], rejected=report['execution_rejected'],
                         attempted_frames=count, executed_frames=report['executed_frames'], planned_frames=report['expected_frames'],
                         recording=str(path), recording_sha256=sha(path), reference=str(reference), reference_sha256=sha(reference),
                         report=str(folder / 'report.json'), report_sha256=sha(folder / 'report.json'),
                         source_input_sha256=report['source_input_sha256'], source_hdf5=task['source_hdf5'])
            entries.append(entry)
            if task_refinement_by_key is not None:
                entry['task_refinement'] = task_refinement_by_key[selection_key]
                entry['task_refinement_artifacts'] = []
                for trial in entry['task_refinement']['trials']:
                    trial_root = Path(trial['root'])
                    trial_dest = trial_root / task['suite'] / f"{task['task_id']:02d}" / job['demo']
                    paths = [trial_dest/'ipopt_state_B10/recording.hdf5', trial_dest/'ipopt_state_B10/report.json',
                             trial_dest/'state_reference.hdf5', trial_root/'task_pose_object_metrics.json',
                             trial_root/'optimizer_default_audit.json']
                    entry['task_refinement_artifacts'].append(dict(root=str(trial_root),
                        files=[dict(path=str(p), sha256=sha(p)) for p in paths]))
            if refinement_by_key is not None:
                entry['refinement'] = refinement_by_key[selection_key]
                entry['refinement_artifacts'] = []
                refined_job = entry['refinement']
                for trial_root in [refined_job['baseline_root']] + [t['root'] for t in refined_job['trials']]:
                    trial_root = Path(trial_root)
                    trial_dest = trial_root / task['suite'] / f"{task['task_id']:02d}" / job['demo']
                    artifact = dict(root=str(trial_root), files=[])
                    for artifact_path in [trial_dest/'ipopt_state_B10/recording.hdf5', trial_dest/'ipopt_state_B10/report.json',
                                          trial_dest/'state_reference.hdf5', trial_root/'physical_state_metrics.json']:
                        artifact['files'].append(dict(path=str(artifact_path), sha256=sha(artifact_path)))
                    entry['refinement_artifacts'].append(artifact)
            if selection_by_key is not None:
                reconstructed = selection_by_key[selection_key]
                entry.update(reconstruction_trials=reconstructed['trials'],
                             selected_candidate=reconstructed['selected_candidate'],
                             candidate_parameters=dict(time_scale=report.get('time_scale', 1), lookahead=report['lookahead_steps'],
                                 gripper_policy=provenance.get('gripper_policy', 'preserve'),
                                 label_method=provenance.get('label_method', 'achieved_state_fk')))
            tasks.setdefault(job['case'], []).append(entry)
    assert len(method_parameters) == 1 or any(p is not None for p in (reconstruction, refinement, task_refinement)), 'Select one fixed method or an explicit reconstruction/refinement policy'
    task_kps = {params[-1] for params in method_parameters}
    assert len(task_kps) == 1
    task_kp = next(iter(task_kps))
    assert len(solver_parameters) == 1, 'Do not mix optimizer settings in one dataset'
    solver_settings = json.loads(next(iter(solver_parameters)))
    if selection_by_key is not None:
        assert len(entries) == len(selection_by_key) == search['totals']['original_episodes']
    output.mkdir(parents=True, exist_ok=False)
    task_files = []
    for case, episodes in sorted(tasks.items()):
        task = selected[case]
        path = output / task['suite'] / f"{task['task_id']:02d}" / 'dataset.hdf5'
        path.parent.mkdir(parents=True)
        successes = [e for e in episodes if e['success']]
        with h5py.File(task['source_hdf5'], 'r') as source, h5py.File(path, 'w') as file:
            file.attrs.update(schema='libero_tasktol_state_reconstruction_v1', action_contract=json.dumps(CONTRACT),
                              method_parameters=json.dumps(sorted(method_parameters)), task_name=task['task_name'], language=task['language'])
            data, attempts, references, targets = [file.create_group(key) for key in ('data', 'attempts', 'references', 'controller_targets')]
            for key, value in source['data'].attrs.items():
                data.attrs[key] = value
            source_env_args = source['data'].attrs['env_args']
            env_args = json.loads(source_env_args)
            env_args['env_kwargs'].update(control_freq=20, camera_heights=256, camera_widths=256)
            env_args['env_kwargs']['controller_configs']['kp'] = task_kp
            env_args['tasktol_client'] = dict(joint_control_mode='osc_task_joint_null', task_kp=task_kp, null_kp=10,
                                              action_contract=CONTRACT, direct_native_env_step_compatible=False)
            if solver_settings is not None:
                env_args['tasktol_client']['solver_settings'] = solver_settings
            data.attrs.update(num_demos=len(successes), total=sum(e['executed_frames'] for e in successes),
                              source_env_args=source_env_args, env_args=json.dumps(env_args),
                              action_contract=json.dumps(CONTRACT), tasktol_execution='LiberoTaskTolSession, osc_task_joint_null')
            for entry in episodes:
                demo = entry['demo']
                attempts[demo] = h5py.ExternalLink(entry['recording'], f'/attempts/{demo}')
                references[demo] = h5py.ExternalLink(entry['reference'], '/')
                if 'task_refinement' in entry:
                    trial_group = file.require_group('task_refinement_trials').create_group(demo)
                    for index, trial in enumerate(entry['task_refinement']['trials']):
                        trial_path = Path(trial['root']) / task['suite'] / f"{task['task_id']:02d}" / demo / 'ipopt_state_B10/recording.hdf5'
                        trial_group[f'trial_{index:02d}'] = h5py.ExternalLink(str(trial_path), f'/attempts/{demo}')
                    trial_group.attrs['selected_root'] = entry['task_refinement']['selected_root']
                if 'reconstruction_trials' in entry:
                    trial_group = file.require_group('reconstruction_trials').create_group(demo)
                    trial_group.attrs['selected_candidate'] = entry['selected_candidate'] or 'none'
                    for index, trial in enumerate(entry['reconstruction_trials']):
                        trial_group[f'trial_{index:02d}'] = h5py.ExternalLink(trial['recording'], f'/attempts/{demo}')
                        trial_group.attrs[f'trial_{index:02d}_candidate'] = trial['candidate']
                if 'refinement' in entry:
                    refined_job = entry['refinement']
                    trial_group = file.require_group('refinement_trials').create_group(demo)
                    trial_roots = [refined_job['baseline_root']] + [t['root'] for t in refined_job['trials']]
                    for index, trial_root in enumerate(trial_roots):
                        trial_path = Path(trial_root) / task['suite'] / f"{task['task_id']:02d}" / demo / 'ipopt_state_B10' / 'recording.hdf5'
                        trial_group[f'trial_{index:02d}'] = h5py.ExternalLink(str(trial_path), f'/attempts/{demo}')
                    trial_group.attrs['selection'] = json.dumps(refined_job)
                if entry['success']:
                    # Preserve recorder compatibility fields in attempts, but
                    # expose consistently pre-command robot state for training.
                    training = data.create_group(demo)
                    with h5py.File(entry['recording'], 'r') as recorded:
                        original = recorded['data'][demo]
                        for key, value in original.attrs.items():
                            training.attrs[key] = value
                        for key in original:
                            if key != 'robot_states':
                                training[key] = h5py.ExternalLink(entry['recording'], f'/data/{demo}/{key}')
                        training['robot_states'] = h5py.ExternalLink(entry['recording'], f'/data/{demo}/tasktol/pre_action_robot_states')
                        training['legacy_robot_states'] = h5py.ExternalLink(entry['recording'], f'/data/{demo}/robot_states')
                        training.create_dataset('policy_state', data=np.concatenate([
                            original['obs/ee_states'][()], original['obs/gripper_states'][()]], axis=1).astype(np.float32))
                        training.attrs['policy_state_contract'] = '8D native cached pre-command observation: xyz metres, axis-angle radians, two finger qpos metres'
                    rows = json.loads(Path(entry['report']).with_name('trajectory.json').read_text())
                    group = targets.create_group(demo)
                    group.create_dataset('joint_reference_rad', data=np.asarray([row['joint_target_rad'] for row in rows]))
                    group.create_dataset('nominal_position_m', data=np.asarray([row['nominal_target']['position_m'] for row in rows]))
                    group.create_dataset('nominal_rotation_vector_rad', data=np.asarray([row['nominal_target']['rotation_vector_rad'] for row in rows]))
            file.attrs['attempted_episodes'] = len(episodes)
        task_files.append(dict(case=case, suite=task['suite'], task_id=task['task_id'], path=str(path.resolve()),
                               attempts=len(episodes), successful=len(successes)))
    manifest = dict(schema='libero_tasktol_state_reconstruction_v1', action_contract=CONTRACT,
                    method_parameters=sorted(method_parameters), tasks=len(tasks), attempted_episodes=len(entries),
                    tasks_with_success=sum(task['successful'] > 0 for task in task_files),
                    tasks_without_success=[dict(case=task['case'], suite=task['suite'], task_id=task['task_id']) for task in task_files if not task['successful']],
                    successful_episodes=sum(e['success'] for e in entries),
                    successful_frames=sum(e['executed_frames'] for e in entries if e['success']),
                    rejected_episodes=sum(e['rejected'] for e in entries),
                    source_runs=[str(root.resolve()) for root in runs],
                    storage='HDF5 external links to immutable retained rollouts and reference sidecars; keep shards at saved addresses',
                    task_files=task_files, episodes=entries)
    if solver_settings is not None:
        manifest['solver_settings'] = solver_settings
    if reconstruction is not None:
        manifest.update(reconstruction_policy=str((reconstruction / 'policy.json').resolve()),
                        reconstruction_summary_sha256=sha(reconstruction / 'summary.json'),
                        reconstruction_totals=search['totals'],
                        method_scope='fixed offline candidate search; report first-trial success separately from recovered datasets')
    if refinement is not None:
        manifest.update(refinement_policy=str((refinement / 'policy.json').resolve()),
                        refinement_summary_sha256=sha(refinement / 'summary.json'),
                        method_scope='uniform offline reference refinement with fixed full-state cost and success-preserving acceptance; includes baseline and every rejected trial')
    if task_refinement is not None:
        manifest.update(task_refinement_policy=str((task_refinement / 'policy.json').resolve()),
                        task_refinement_plan=str((task_refinement / 'plan.json').resolve()),
                        task_refinement_summary_sha256=sha(task_refinement / 'summary.json'),
                        robot_joint_tracking_penalty=False, finger_joint_tracking_penalty=False,
                        optimizer_objective='unchanged library default',
                        method_scope='uniform task-only offline refinement; all 400 sampled episodes counted; no historical candidates')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ('tasks', 'tasks_with_success', 'attempted_episodes', 'successful_episodes', 'successful_frames', 'rejected_episodes')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', default=[])
    parser.add_argument('--reconstruction', type=Path)
    parser.add_argument('--refinement', type=Path)
    parser.add_argument('--task-refinement', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sum(map(bool, (args.runs, args.reconstruction, args.refinement, args.task_refinement))) == 1, 'Choose one recording selection policy'
    build(args.runs, args.output.resolve(), args.reconstruction.resolve() if args.reconstruction else None,
          args.refinement.resolve() if args.refinement else None,
          args.task_refinement.resolve() if args.task_refinement else None)


if __name__ == '__main__':
    main()
