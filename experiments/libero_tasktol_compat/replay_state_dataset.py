"""Cold replay of packaged physical actions through the shared IPOPT client.

No original future state, original action, or offline joint witness is supplied
to session.step. Saved controller targets/states are read only after execution
to compare with the recording.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'franka_mujoco'), str(ROOT / 'third_party/libero')]

from rerecord_state_reference import SELECTION, plain
from examples.libero import regenerate_libero_dataset as recorder
from examples.libero.tasktol_actions import decode_tasktol_action
import h5py
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tasktol.simulation.libero import LiberoTaskTolSession
from tasktol.control.ipopt import IpoptSettings
from experiments.libero_tasktol_compat.observe_online_ipopt import OnlineIpoptObserver


def replay(dataset, case, demo, output, gripper_noise=0, include_failure=False):
    selected = json.loads(SELECTION.read_text())['tasks'][case]
    task_file = dataset / selected['suite'] / f"{selected['task_id']:02d}" / 'dataset.hdf5'
    with h5py.File(task_file, 'r') as file:
        client_config = json.loads(file['data'].attrs['env_args'])['tasktol_client']
        expected_success = demo in file['data']
        group = file['attempts' if include_failure else 'data'][demo]
        actions = file['references'][demo]['actions'][()] if include_failure else group['actions'][()]
        initial = group['tasktol/initial_state'][()]
        anchor_p = file['references'][demo]['positions'][0]
        anchor_r = file['references'][demo]['rotations'][0]
    with h5py.File(selected['processed_hdf5'], 'r') as prior:
        configuration = json.loads(prior.attrs['configuration'])
        order = configuration['demo_ids']
        episode_index = order.index(demo)
        before_rng = None if not episode_index else json.loads(prior['regeneration/episodes'][order[episode_index - 1]].attrs['rng_state'])
        after_rng = json.loads(prior['regeneration/episodes'][demo].attrs['rng_state'])
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files')) / selected['suite'] / (selected['task_name'] + '.bddl')),
        controller='OSC_POSE', camera_names=['agentview', 'robot0_eye_in_hand'],
        camera_heights=256, camera_widths=256, control_freq=20,
    )
    output.mkdir(parents=True, exist_ok=False)
    rows, states = [], []
    try:
        env.seed(configuration['seed'])
        env.env.object_property_initializers *= episode_index + 1
        if before_rng is not None:
            recorder._restore_rng(before_rng)
        else:
            recorder._restore_rng(dict(numpy=recorder._rng_state()['numpy'], python=after_rng['python']))
        env.reset()
        env.set_init_state(initial)
        np.testing.assert_array_equal(env.sim.get_state().flatten(), initial)
        assert plain(recorder._rng_state()) == after_rng
        saved_settings = client_config.get('solver_settings')
        session = LiberoTaskTolSession(env, kp=client_config['task_kp'], joint_control_mode=client_config['joint_control_mode'],
                                      solver_settings=IpoptSettings(**saved_settings) if saved_settings is not None else None)
        actual_solver_settings = asdict(session.ik.settings)
        if saved_settings is not None:
            assert actual_solver_settings == saved_settings
        session.target.position = anchor_p.copy()
        session.target.rotation = anchor_r.copy()
        session.release.accepted_rotation = anchor_r.copy()
        observer = OnlineIpoptObserver()
        observer.ik = session.ik
        with observer:
            for index, action in enumerate(actions):
                states.append(env.sim.get_state().flatten().copy())
                policy_action = action.copy()
                if action[6] == 0:
                    policy_action[6] += gripper_noise * (1 if index % 2 else -1)
                _, row = session.step(decode_tasktol_action(policy_action), phase='outside_subtask')
                rows.append(row)
                if row['aborted']:
                    break
        final = env.sim.get_state().flatten().copy()
        success = bool(env.check_success()) and len(rows) == len(actions) and not any(row['aborted'] for row in rows)
    finally:
        env.close()
    # Ground truth recording is used below for evaluation only.
    with h5py.File(task_file, 'r') as file:
        group = file['attempts' if include_failure else 'data'][demo]
        expected_states = group['tasktol/pre_action_states'][()]
        expected_final = group['tasktol/final_state'][()]
        if include_failure:
            manifest = json.loads((dataset/'manifest.json').read_text())
            entry = next(e for e in manifest['episodes'] if e['case']==case and e['demo']==demo)
            expected_rows = json.loads(Path(entry['report']).with_name('trajectory.json').read_text())
            expected_q = np.asarray([r['joint_target_rad'] for r in expected_rows])
            expected_rejected = entry['rejected']
        else:
            expected_q = file['controller_targets'][demo]['joint_reference_rad'][()]
            expected_rejected = False
    states = np.asarray(states)
    actual_q = np.asarray([row['joint_target_rad'] for row in rows])
    count = len(rows)
    common_count = min(count,len(expected_states))
    rejected = any(row['aborted'] for row in rows)
    (output/'solver_observations.json').write_text(json.dumps(plain(dict(
        settings=actual_solver_settings, records=observer.records(rows))),indent=2)+'\n')
    result = dict(case=case, demo=demo, final_success=success, planned_frames=len(actions), executed_frames=sum(not row['aborted'] for row in rows),
                  rejected=rejected, expected_success=expected_success, expected_rejected=expected_rejected,
                  recorded_outcome_matches=success==expected_success and rejected==expected_rejected,
                  physical_action_units='metres/radians',
                  future_state_or_joint_witness_supplied_to_client=False,
                  injected_gripper_hold_noise=gripper_noise, gripper_deadband=.5,
                  solver_settings=actual_solver_settings,
                  pre_action_state_max_abs_error=float(np.max(np.abs(states[:common_count] - expected_states[:common_count]))),
                  final_state_max_abs_error=float(np.max(np.abs(final - expected_final))),
                  full_joint_reference_max_abs_error_rad=float(np.max(np.abs(actual_q[:common_count] - expected_q[:common_count]))),
                  states_bitwise_equal=bool(np.array_equal(states, expected_states)),
                  final_bitwise_equal=bool(np.array_equal(final, expected_final)),
                  joint_targets_bitwise_equal=bool(np.array_equal(actual_q, expected_q)))
    result['reproduces_recording_bitwise'] = (result['recorded_outcome_matches'] and result['states_bitwise_equal']
        and result['final_bitwise_equal'] and result['joint_targets_bitwise_equal'])
    (output / 'trajectory.json').write_text(json.dumps(plain(rows), indent=2) + '\n')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if include_failure:
        assert result['recorded_outcome_matches'], 'Outcome differs; retained trajectory and raw solves require diagnosis'
        assert len(states)==len(expected_states), 'Execution length differs from recording'
    else:
        assert success and not result['rejected']
    assert result['pre_action_state_max_abs_error'] < 1e-9
    assert result['final_state_max_abs_error'] < 1e-9
    assert result['full_joint_reference_max_abs_error_rad'] < 1e-9


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gripper-noise', type=float, default=0)
    parser.add_argument('--include-failure', action='store_true')
    args = parser.parse_args()
    replay(args.dataset.resolve(), args.case, args.demo, args.output.resolve(), args.gripper_noise, args.include_failure)
