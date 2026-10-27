"""Verify recordings and quantify success, solver coverage, and tracking lag."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = json.loads((root / 'provenance.json').read_text())
    progress = json.loads((root / 'summary.json').read_text())
    assert progress['complete'] and len(progress['results']) == len(provenance['plan'])
    assert not any('error' in job for job in progress['results'])
    snapshots = json.loads((root / 'source_snapshots.json').read_text()) if (root / 'source_snapshots.json').exists() else {}
    source_checks = {path: sha(path) == value or (path in snapshots and sha(root / snapshots[path]) == value) for path, value in provenance['source_sha256'].items()}
    assert all(source_checks.values()), source_checks
    aggregate = {mode: dict(episodes=0, final=0, ever=0, rejected=0, planned_frames=0, executed_frames=0) for mode in provenance['modes']}
    episodes = []
    for plan in provenance['plan']:
        time_scale = provenance.get('time_scale', 1)
        dest = root / plan['suite'] / f"{plan['task_id']:02d}" / plan['demo']
        reports = json.loads((dest / 'summary.json').read_text())['results']
        assert len({r['initial_state_sha256'] for r in reports}) == 1
        assert len({r['model_sha256'] for r in reports}) == 1
        assert len({sha(dest / r['mode'] / 'initial.png') for r in reports}) == 1
        with h5py.File(dest / 'state_reference.hdf5', 'r') as file:
            ref = {key: value[()] for key, value in file.items()}
            source_path = file.attrs['source']
            expected_input_sha = file.attrs['source_input_sha256']
        with h5py.File(source_path, 'r') as file:
            raw = file['data'][plan['demo']]
            states, actions = raw['states'][()], raw['actions'][()]
            actual_input_sha = hashlib.sha256(states.tobytes() + actions.tobytes() + str(raw.attrs['model_file']).encode()).hexdigest()
            assert actual_input_sha == expected_input_sha
            count = (len(states) - 1) * time_scale
            coordinate = np.arange(count + 1) / time_scale
            source_indices = np.arange(count) // time_scale
            expected_q = np.column_stack([np.interp(coordinate, np.arange(len(states)), states[:, axis + 1]) for axis in range(7)])
            np.testing.assert_array_equal(ref['q'], expected_q)
            expected_gripper = actions[source_indices, 6].copy()
            if provenance.get('gripper_policy') == 'native_integrator_pulse':
                expected_gripper[np.arange(count) % time_scale != provenance['pulse_phase']] = 0
            np.testing.assert_array_equal(ref['actions'][:, 6], expected_gripper)
        initial = states[0]
        item = dict(**plan, modes=[])
        if 'goal_joint_witness_rad' in ref:
            import mujoco
            with h5py.File(dest / 'ipopt_state_B10/recording.hdf5', 'r') as file:
                model = mujoco.MjModel.from_xml_string(file['attempts'][plan['demo']].attrs['model_file'])
            data = mujoco.MjData(model)
            site = model.site('gripper0_grip_site').id
            hand = model.body('robot0_right_hand').id
            # get_xml serializes the fixed hand quaternion at reduced precision.
            # Calibrate one constant tool-frame transform from the saved anchor;
            # every later witness must match with this same transform.
            data.qpos[:7] = ref['q'][0]
            mujoco.mj_kinematics(model, data)
            tool_frame = data.xmat[hand].reshape(3, 3).T @ ref['rotations'][0]
            item['serialized_tool_frame_correction_rad'] = float(np.linalg.norm(Rotation.from_matrix(tool_frame).as_rotvec()))
            assert item['serialized_tool_frame_correction_rad'] < 1e-5
            bounds = np.asarray([model.joint(f'robot0_joint{index}').range for index in range(1, 8)])
            for index, q in enumerate(ref['goal_joint_witness_rad']):
                assert np.all(q >= bounds[:, 0]) and np.all(q <= bounds[:, 1])
                data.qpos[:7] = q
                mujoco.mj_kinematics(model, data)
                np.testing.assert_allclose(data.site_xpos[site], ref['positions'][index + 1], atol=1e-10, rtol=0)
                np.testing.assert_allclose(data.xmat[hand].reshape(3, 3) @ tool_frame, ref['rotations'][index + 1], atol=1e-10, rtol=0)
            item['bounded_goal_witness_fk_validated'] = True
        for report in reports:
            mode = report['mode']
            folder = dest / mode
            rows = json.loads((folder / 'trajectory.json').read_text())
            executed = [row for row in rows if not row['aborted']]
            assert report['expected_frames'] == len(ref['actions']) == (len(states) - 1) * time_scale
            assert len(executed) == report['executed_frames']
            assert len(rows) == len(executed) + int(report['execution_rejected'])
            assert report['source_input_sha256'] == expected_input_sha
            assert all(row['physics_steps'] == (0 if row['aborted'] else 25) for row in rows)
            for index, row in enumerate(rows):
                assert row['raw_source_index'] == source_indices[index]
                if mode == 'ipopt_state_B10':
                    np.testing.assert_array_equal(row['action'], ref['actions'][index])
                    target = ref['target_indices'][index]
                    np.testing.assert_allclose(row['nominal_target']['position_m'], ref['positions'][target], atol=1e-10, rtol=0)
                    r = Rotation.from_rotvec(row['nominal_target']['rotation_vector_rad']).as_matrix()
                    assert np.linalg.norm(Rotation.from_matrix(r @ ref['rotations'][target].T).as_rotvec()) < 1e-9
                    if not row['aborted']:
                        assert row['optimizer']['feasible']
                        assert row['native_joint_goal_error_rad'] < 1e-10
            if mode == 'ipopt_state_B10':
                assert report['factor_statistics']['null_reference'] == 'joint'
                assert report['factor_statistics']['null_kp'] == 10
            with h5py.File(folder / 'recording.hdf5', 'r') as file:
                group = file['attempts'][plan['demo']]
                count = len(rows)
                np.testing.assert_array_equal(group['tasktol/source_step_indices'][()], np.arange(count))
                np.testing.assert_array_equal(group['tasktol/settled_state'][()], initial)
                np.testing.assert_array_equal(group['tasktol/pre_action_states'][0], initial)
                np.testing.assert_array_equal(group['actions'][()], actions[source_indices[:count]] if mode == 'osc_original' else ref['actions'][:count])
                if 'original_source_action_indices' in group['tasktol']:
                    np.testing.assert_array_equal(group['tasktol/original_source_action_indices'][()], source_indices[:count])
                assert len(group['obs/agentview_rgb']) == count
                assert len(group['obs/eye_in_hand_rgb']) == count
                assert (plan['demo'] in file['data']) == report['success']
                assert bool(np.any(group['dones'][()])) == report['success']
                expected_time = initial[0] + len(executed) * .05
                assert abs(group['tasktol/final_state'][0] - expected_time) < 1e-9
            frames = int(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                        '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(folder / 'main_rgb.mp4')], text=True).strip())
            assert frames == len(rows)
            if executed:
                actual = np.asarray([row['actual_position_m'] for row in executed])
                # Diagnostic lag estimate on the first 4s only; failure/contact
                # divergence later must not be misrepresented as pure delay.
                indices = np.arange(min(len(actual), 80))
                metric_positions = ref.get('source_positions_m', ref['positions'])
                lag_errors = [float(np.sqrt(np.mean(np.sum((actual[indices] - metric_positions[np.maximum(0, indices + 1 - lag)]) ** 2, axis=1))) * 1000) for lag in range(11)]
                source_errors = np.asarray([row['source_next_position_error_m'] for row in executed]) * 1000
                metrics = dict(mean_error_mm=float(np.mean(source_errors)), p95_error_mm=float(np.percentile(source_errors, 95)),
                               max_error_mm=float(np.max(source_errors)), early_best_lag_ms=int(np.argmin(lag_errors)) * 50,
                               early_lag_rmse_mm=lag_errors,
                               mean_orientation_error_deg=float(np.rad2deg(np.mean([row['source_next_rotation_error_rad'] for row in executed]))))
            else:
                metrics = {}
            value = dict(mode=mode, final=report['success'], ever=report['ever_success'], rejected=report['execution_rejected'],
                         rejected_source_index=report['rejected_source_index'], executed_frames=len(executed),
                         decoded_video_frames=frames, **metrics)
            item['modes'].append(value)
            agg = aggregate[mode]
            agg['episodes'] += 1
            agg['final'] += int(value['final'])
            agg['ever'] += int(value['ever'])
            agg['rejected'] += int(value['rejected'])
            agg['planned_frames'] += report['expected_frames']
            agg['executed_frames'] += len(executed)
        episodes.append(item)
    output = dict(scope='Fixed diagnostic sample, not a population success estimate',
                  primary='Entire N-1 known state transitions complete with final native task success',
                  source_checks=source_checks, aggregate=aggregate, episodes=episodes)
    (root / 'comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
