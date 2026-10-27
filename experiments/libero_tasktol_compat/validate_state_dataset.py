"""Independently validate packaged training episodes and cold replay evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(dataset, cold=None):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    expected = {(entry['case'], entry['demo']): entry for entry in manifest['episodes']}
    assert len(expected) == manifest['attempted_episodes']
    checks, frames, successes, attempts, trial_count = [], 0, 0, 0, 0
    for task in manifest['task_files']:
        with h5py.File(task['path'], 'r') as file:
            assert len(file['data']) == task['successful']
            assert len(file['attempts']) == task['attempts']
            env_args = json.loads(file['data'].attrs['env_args'])
            client = env_args['tasktol_client']
            assert client['joint_control_mode'] == 'osc_task_joint_null'
            assert client['task_kp'] == 150 and client['null_kp'] == 10
            if 'solver_settings' in manifest:
                assert client['solver_settings'] == manifest['solver_settings']
            assert env_args['env_kwargs']['control_freq'] == 20
            for demo, group in file['attempts'].items():
                attempts += 1
                entry = expected[(task['case'], demo)]
                assert sha(entry['recording']) == entry['recording_sha256']
                assert sha(entry['reference']) == entry['reference_sha256']
                assert sha(entry['report']) == entry['report_sha256']
                assert (demo in file['data']) == entry['success']
                actions = group['actions'][()]
                before = group['tasktol/pre_action_states'][()]
                count = len(actions)
                assert actions.shape == (count, 7) and np.all(np.isfinite(actions))
                assert set(np.unique(actions[:, 6])) <= {-1, 0, 1}
                np.testing.assert_array_equal(group['states'][()], before)
                np.testing.assert_array_equal(before[0], group['tasktol/initial_state'][()])
                np.testing.assert_allclose(np.diff(before[:, 0]), .05, rtol=0, atol=1e-9)
                for key in ('agentview_rgb', 'eye_in_hand_rgb'):
                    images = group['obs'][key]
                    assert images.shape == (count, 256, 256, 3)
                    assert images.dtype == np.uint8
                    assert images[0].std() > 0 and images[-1].std() > 0
                assert group['obs/joint_states'].shape == (count, 7)
                # Native step returns cached observables one 2ms tick behind
                # final qpos. Initial set_init_state explicitly forwards q0.
                model = mujoco.MjModel.from_xml_string(group.attrs['model_file'])
                observed_q = group['obs/joint_states'][()]
                cached_q = before[:, 1:8] - model.opt.timestep * before[:, 1 + model.nq:8 + model.nq]
                cached_q[0] = before[0, 1:8]
                np.testing.assert_allclose(observed_q, cached_q, rtol=0, atol=1e-12)
                if entry['success']:
                    successes += 1
                    frames += count
                    assert count == entry['executed_frames'] == entry['planned_frames']
                    assert group['dones'][-1] == 1 and group['rewards'][-1] == 1
                    assert not np.any(group['dones'][:-1])
                    reference = file['references'][demo]
                    training = file['data'][demo]
                    if 'policy_state' in training:
                        np.testing.assert_array_equal(training['robot_states'][()], group['tasktol/pre_action_robot_states'][()])
                        expected_policy_state = np.concatenate([group['obs/ee_states'][()], group['obs/gripper_states'][()]], axis=1).astype(np.float32)
                        np.testing.assert_array_equal(training['policy_state'][()], expected_policy_state)
                    np.testing.assert_array_equal(actions, reference['actions'][()])
                    p, r = reference['positions'][0], reference['rotations'][0]
                    target_p = file['controller_targets'][demo]['nominal_position_m'][()]
                    target_r = Rotation.from_rotvec(file['controller_targets'][demo]['nominal_rotation_vector_rad'][()]).as_matrix()
                    q = file['controller_targets'][demo]['joint_reference_rad'][()]
                    assert q.shape == (count, 7) and np.all(np.isfinite(q))
                    for index, action in enumerate(actions):
                        p = p + action[:3]
                        r = Rotation.from_rotvec(action[3:6]).as_matrix() @ r
                        np.testing.assert_allclose(p, target_p[index], rtol=0, atol=1e-9)
                        assert np.linalg.norm(Rotation.from_matrix(r @ target_r[index].T).as_rotvec()) < 1e-8
                    assert abs(group['tasktol/final_state'][0] - before[0, 0] - count * .05) < 1e-9
                else:
                    assert not np.any(group['dones'][()]) and not np.any(group['rewards'][()])
                if 'reconstruction_trials' in entry:
                    for index, trial in enumerate(entry['reconstruction_trials']):
                        trial_count += 1
                        assert sha(trial['recording']) == trial['recording_sha256']
                        assert sha(trial['report']) == trial['report_sha256']
                        trial_group = file['reconstruction_trials'][demo][f'trial_{index:02d}']
                        assert len(trial_group['actions']) == trial['executed_frames'] + int(trial['rejected'])
                        assert trial['success'] == (entry['success'] and index == len(entry['reconstruction_trials']) - 1)
                if 'refinement' in entry:
                    from closed_loop_state_metrics import evaluate_run
                    selected = entry['refinement']
                    current = evaluate_run(Path(selected['baseline_root']), write=False)
                    for trial in selected['trials']:
                        measured = evaluate_run(Path(trial['root']), write=False)
                        accepted = (measured['full_length'] and not measured['rejected']
                            and (not current['success'] or measured['success'])
                            and measured['objective'] < current['objective'])
                        assert accepted == trial['accepted']
                        if accepted:
                            current = measured
                    assert current['objective'] == selected['objective']
                    assert current['success'] == entry['success']
                    assert current['objective'] <= selected['baseline_objective']
                    for artifact in entry['refinement_artifacts']:
                        for item in artifact['files']:
                            assert sha(item['path']) == item['sha256']
                    assert len(file['refinement_trials'][demo]) == len(selected['trials']) + 1
                if 'task_refinement' in entry:
                    from experiments.libero_tasktol_compat.task_pose_object_metrics import evaluate_task_run
                    selected = entry['task_refinement']
                    current_root = Path(selected['baseline_root'])
                    current = evaluate_task_run(current_root, write=False)
                    for trial in selected['trials'][1:]:
                        assert Path(trial['parent']) == current_root
                        measured = evaluate_task_run(Path(trial['root']), write=False)
                        assert not measured['robot_joint_tracking_penalty'] and not measured['finger_joint_tracking_penalty']
                        accepted = (measured['full_length'] and not measured['rejected']
                            and (not current['success'] or measured['success'])
                            and measured['objective'] < current['objective'])
                        assert accepted == trial['accepted']
                        if accepted:
                            current_root, current = Path(trial['root']), measured
                    assert str(current_root) == selected['selected_root']
                    assert current['objective'] == selected['selected']['objective']
                    assert current['success'] == entry['success']
                    for artifact in entry['task_refinement_artifacts']:
                        for item in artifact['files']:
                            assert sha(item['path']) == item['sha256']
                    assert len(file['task_refinement_trials'][demo]) == len(selected['trials'])
                checks.append(dict(case=task['case'], demo=demo, successful=entry['success'], frames=count,
                                   native_pre_command_observation_timing_verified=True,
                                   observation_joint_cache_lag_s=float(model.opt.timestep),
                                   observation_vs_actual_joint_max_rad=float(np.max(np.abs(observed_q - before[:, 1:8]))),
                                   physical_reference_reconstruction=True,
                                   full_joint_reference_saved=True, input_shard_hashes=True))
    assert (attempts, successes, frames) == (manifest['attempted_episodes'], manifest['successful_episodes'], manifest['successful_frames'])
    result = dict(validated=True, tasks=len(manifest['task_files']), attempts=attempts, successful_episodes=successes,
                  successful_frames=frames, reconstruction_trials_checked=trial_count, episode_checks=checks)
    if cold is not None:
        replay = json.loads((cold / 'summary.json').read_text())
        assert replay['complete'] and replay['failures'] == 0
        includes_failures=replay.get('failures_included',False)
        planned=attempts if includes_failures else successes
        assert replay['planned'] == replay['replayed'] == planned
        assert len(replay['results']) == planned
        cold_plan = json.loads((cold / 'plan.json').read_text())
        assert Path(cold_plan['dataset']).resolve() == dataset.resolve()
        seen = set()
        for job in replay['results']:
            identity = (job['case'], job['demo'])
            assert identity not in seen and identity in expected
            seen.add(identity)
            if includes_failures:
                assert job['recorded_outcome_matches'] and job['reproduces_recording_bitwise']
                assert job['final_success']==expected[identity]['success']
                assert job['rejected']==expected[identity]['rejected']
                assert job['executed_frames']==expected[identity]['executed_frames']
            else:
                assert expected[identity]['success'] and job['final_success'] and not job['rejected']
            assert job['future_state_or_joint_witness_supplied_to_client'] is False
            assert job['states_bitwise_equal'] and job['final_bitwise_equal'] and job['joint_targets_bitwise_equal']
            if expected[identity]['success']:
                assert job['executed_frames'] == expected[identity]['planned_frames']
            if 'solver_settings' in manifest:
                assert job['solver_settings'] == manifest['solver_settings']
        result['cold_replay'] = dict(episodes=planned, successful_episodes=successes,
                                      includes_recorded_failures=includes_failures,
                                      recorded_failures_reproduced=attempts-successes if includes_failures else 0,
                                      every_recorded_outcome_reproduced=True,
                                      no_future_state_or_joint_witness_input=True, bitwise_state_and_joint_reference_match=True,
                                      summary_sha256=sha(cold / 'summary.json'))
    (dataset / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'episode_checks'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cold', type=Path)
    args = parser.parse_args()
    validate(args.dataset.resolve(), args.cold.resolve() if args.cold else None)
