"""Paired native-controller diagnostics; never a training-data recorder.

All execution uses LIBERO env.step. This script only instruments the native OSC
baseline against the frozen IPOPT benchmark inputs in the same pinned runtime.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL = ROOT.parent / 'franka_mujoco'
SELECTION = ROOT / 'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'


def replay(case, output, refresh_controller=False, source_reset_state=False, source_reset_history=False, full_context_dir=None,
           osc_reference_mode='feedback_delta', reference_actions_path=None):
    output.mkdir(parents=True, exist_ok=False)
    os.environ['NUMBA_CACHE_DIR'] = str(output / 'numba_cache')
    os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib')
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(ROOT / 'third_party/libero'))
    sys.path.insert(0, str(TASKTOL))
    import mujoco
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.control.task_space.kinematics import RobotKinematics
    from tasktol.geometry import rotation_vector
    from tasktol.tolerance.runtime import advance_reference_pose
    from tasktol.simulation.libero import libero_handles
    sys.path.insert(0, str(ROOT))
    from experiments.libero_tasktol_compat.osc_reference import (
        decode_source_goals, difference_reference_goals, integrate_reference_actions,
    )

    if osc_reference_mode not in ('feedback_delta', 'accumulated_reference'):
        raise ValueError('Unknown OSC reference mode')

    assert mujoco.__version__ == '3.2.3'
    selected = json.loads(SELECTION.read_text())['tasks'][case]
    with h5py.File(selected['processed_hdf5'], 'r') as f:
        demo = f['data'][selected['demo_id']]
        initial = demo['tasktol/pre_action_states'][0]
        states = demo['tasktol/pre_action_states'][()]
        final = demo['tasktol/final_state'][()]
        actions = demo['actions'][()]
        source_indices = demo['tasktol/source_step_indices'][()]
        configuration = json.loads(f.attrs['configuration'])
        order = configuration['demo_ids']
        previous = order[order.index(selected['demo_id'])-1] if order.index(selected['demo_id']) else None
        rng = json.loads(f['regeneration/episodes'][previous].attrs['rng_state']) if previous else None
        expected_rng = json.loads(f['regeneration/episodes'][selected['demo_id']].attrs['rng_state'])
    original = TASKTOL / 'experiments/libero_closed_loop/results/native_nominal40' / selected['suite'] / f"{selected['task_id']:02d}" / 'report.json'
    expected_hash = json.loads(original.read_text())['initial_state_sha256']
    actual_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    assert expected_hash == actual_hash
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files')) / selected['suite'] / (selected['task_name'] + '.bddl')),
        controller='OSC_POSE', camera_names=['agentview'], camera_heights=256,
        camera_widths=256, control_freq=20,
    )
    frames = []
    try:
        env.seed(0)
        if source_reset_history:
            # Native LIBERO appends property samplers at every hard reset.
            # Rehydrate the source's prior identical sampler entries; the
            # native reset still performs every sample and placement itself.
            env.env.object_property_initializers *= order.index(selected['demo_id']) + 1
        if source_reset_state and rng is not None:
            sys.path.insert(0, str(ROOT))
            from examples.libero.regenerate_libero_dataset import _restore_rng
            _restore_rng(rng)
        env.reset()
        reset_rng = np.random.get_state()
        rng_matches_source = (
            reset_rng[0] == expected_rng['numpy'][0]
            and reset_rng[1].tolist() == expected_rng['numpy'][1]
            and list(reset_rng[2:]) == expected_rng['numpy'][2:]
        )
        if full_context_dir is not None:
            with np.load(full_context_dir/selected['suite']/f"{selected['task_id']:02d}"/'initial_context.npz') as full:
                env.sim.model.body_pos[:]=full['body_pos']
                env.sim.model.body_quat[:]=full['body_quat']
                mujoco.mj_setState(env.sim.model._model,env.sim.data._data,full['integration_state'],int(full['state_spec']))
                ctrl=env.robots[0].controller
                ctrl.goal_pos=full['goal_pos'].copy();ctrl.goal_ori=full['goal_ori'].copy()
                ctrl.initial_joint=full['initial_joint'].copy()
                env.robots[0].gripper.current_action=full['gripper_action'].copy()
            env.sim.forward();env.env._update_observables(force=True)
            obs=env.env._get_observations()
        else:
            obs = env.set_init_state(initial)
        handles = libero_handles(env)
        kin = RobotKinematics(env.sim.model._model, handles)
        controller = env.robots[0].controller
        data = env.sim.data._data
        if refresh_controller:
            # A flattened physics snapshot does not invalidate robosuite's
            # controller cache. Let the normal first set_goal refresh it.
            controller.new_update = True
        if source_reset_state:
            # Source snapshots follow ten open-gripper settling controls.
            # Recover its native accumulator without adding physics steps.
            for _ in range(configuration['settle_steps'] * int(round(env.env.control_timestep / env.sim.model.opt.timestep))):
                env.robots[0].gripper.format_action(np.array([-1.0]))
        initial_gripper_action=env.robots[0].gripper.current_action.copy()
        qs = np.vstack((states[:, 1 + handles.qpos_ids], final[1 + handles.qpos_ids]))
        refs = [kin.evaluate(q, include_manipulability=False, include_link_points=False) for q in qs]
        # Decode the source goals before execution, using source *pre-action*
        # states. The held orientation belongs to the native controller reset,
        # not to the settled robot snapshot. No new rollout feedback enters
        # the independent nominal branch in accumulated_reference mode.
        if controller.position_limits is not None or controller.orientation_limits is not None:
            raise ValueError('Explicit adaptation required for limited OSC targets')
        source_positions = np.array([r.position for r in refs[:-1]])
        source_rotations = np.array([r.rotation for r in refs[:-1]])
        scaled_actions = np.array([controller.scale_action(a[:6]) for a in actions])
        goal_positions, goal_rotations = decode_source_goals(
            source_positions, source_rotations, scaled_actions,
            controller.goal_ori @ handles.ee_rotation_offset,
        )
        reference_actions = difference_reference_goals(
            goal_positions, goal_rotations, actions[:, 6], refs[0].position, refs[0].rotation,
        )
        anchor_position, anchor_rotation = refs[0].position.copy(), refs[0].rotation.copy()
        reference_metadata = None
        if reference_actions_path is not None:
            if osc_reference_mode != 'accumulated_reference':
                raise ValueError('Saved reference actions require accumulated_reference mode')
            from tasktol.benchmark.libero_reference_actions import load_reference_actions, verify_reference_anchor

            saved = load_reference_actions(
                reference_actions_path, selected['demo_id'], source_path=selected['processed_hdf5'],
                source_states=states, source_actions=actions,
            )
            verify_reference_anchor(saved, anchor_position, anchor_rotation)
            reference_actions = saved.actions
            anchor_position, anchor_rotation = saved.anchor_position.copy(), saved.anchor_rotation.copy()
            reference_metadata = saved.metadata
        reconstructed_p, reconstructed_r = integrate_reference_actions(
            reference_actions, anchor_position, anchor_rotation,
        )
        reconstruction_position = np.linalg.norm(reconstructed_p-goal_positions, axis=1)
        reconstruction_rotation = np.array([
            np.linalg.norm(rotation_vector(a @ b.T)) for a, b in zip(reconstructed_r, goal_rotations, strict=True)
        ])
        if reconstruction_position.max() > 1e-10 or reconstruction_rotation.max() > 1e-9:
            raise RuntimeError('Relabelled increments failed to reconstruct original OSC goals')
        np.savez_compressed(
            output/'relabelled_reference.npz',
            anchor_position_m=anchor_position, anchor_tool_rotation=anchor_rotation,
            source_actions=actions, source_step_indices=source_indices,
            source_pre_action_states=states, source_final_state=final,
            source_actual_positions_m=source_positions, source_actual_tool_rotations=source_rotations,
            source_goal_positions_m=goal_positions, source_goal_tool_rotations=goal_rotations,
            reference_delta_actions_m_rad_gripper=reference_actions,
        )
        if osc_reference_mode == 'accumulated_reference':
            # Native OSC's absolute target mode accepts physical position and
            # axis-angle orientation directly. It does not scale or clip the
            # pose as if it were an old normalized feedback-relative action.
            controller.use_delta = False
        execution_states = []
        reference_position, reference_rotation = anchor_position.copy(), anchor_rotation.copy()
        Image.fromarray(obs['agentview_image'][::-1, ::-1].copy()).save(output / 'initial.png')
        with imageio.get_writer(output / 'diagnostic_native_osc.mp4', fps=20, codec='libx264', quality=8) as video:
            for i, action in enumerate(actions):
                video.append_data(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                before_q = data.qpos[handles.qpos_ids].copy()
                before = kin.evaluate(before_q, include_manipulability=False, include_link_points=False)
                before_time = float(data.time)
                if osc_reference_mode == 'accumulated_reference':
                    reference_position, reference_rotation = advance_reference_pose(
                        reference_position, reference_rotation, reference_actions[i],
                    )
                    native_site_rotation = reference_rotation @ handles.ee_rotation_offset.T
                    native_action = np.r_[reference_position, rotation_vector(native_site_rotation), action[6]]
                else:
                    native_action = action
                obs, _, _, _ = env.step(native_action.tolist())
                execution_states.append(env.get_sim_state())
                after = kin.evaluate(data.qpos[handles.qpos_ids], include_manipulability=False, include_link_points=False)
                native_rot = controller.goal_ori @ handles.ee_rotation_offset
                row = dict(
                    index=i, pre_q=before_q.tolist(), post_q=data.qpos[handles.qpos_ids].tolist(),
                    source_position_error_m=float(np.linalg.norm(after.position - refs[i+1].position)),
                    source_rotation_error_rad=float(np.linalg.norm(rotation_vector(refs[i+1].rotation.T @ after.rotation))),
                    source_pre_position_error_m=float(np.linalg.norm(before.position - refs[i].position)),
                    actual_position_m=after.position.tolist(), actual_tool_rotation=after.rotation.tolist(),
                    native_goal_position_m=np.asarray(controller.goal_pos).tolist(),
                    native_goal_rotation=native_rot.tolist(), decoded_goal_position_m=goal_positions[i].tolist(),
                    decoded_goal_rotation=goal_rotations[i].tolist(),
                    goal_position_difference_m=float(np.linalg.norm(controller.goal_pos-goal_positions[i])),
                    goal_rotation_difference_rad=float(np.linalg.norm(rotation_vector(native_rot.T @ goal_rotations[i]))),
                    reference_delta_action=reference_actions[i].tolist(), native_action=native_action.tolist(),
                    physics_steps=int(round((float(data.time)-before_time)/env.sim.model.opt.timestep)),
                    tracking_position_error_m=float(np.linalg.norm(after.position-controller.goal_pos)),
                    dt=float(data.time)-before_time, success=bool(env.check_success()),
                )
                frames.append(row)
        Image.fromarray(obs['agentview_image'][::-1, ::-1].copy()).save(output / 'final.png')
        np.savez_compressed(output/'execution_states.npz', states=np.asarray(execution_states), nq=env.sim.model.nq,
                            nv=env.sim.model.nv, arm_qpos_ids=handles.qpos_ids,
                            finger_qpos_ids=handles.finger_qpos_ids)
        errors = np.array([r['source_position_error_m'] for r in frames])
        summary = dict(
            suite=selected['suite'], task_id=selected['task_id'], demo=selected['demo_id'],
            diagnostic_only=True, controller='OSC_POSE', task_success=bool(env.check_success()),
            osc_reference_mode=osc_reference_mode, ipopt_solver_calls=0,
            reference_actions=reference_metadata,
            relabelled_action_units='metres; world-frame rotation-vector radians; unchanged gripper',
            relabelled_rotation_rule='Log(R_goal[t] @ R_goal[t-1].T); first difference uses explicit anchor',
            reconstruction_position_max_m=float(reconstruction_position.max()),
            reconstruction_rotation_max_rad=float(reconstruction_rotation.max()),
            source_action_steps=len(actions), original_hdf5_modified=False,
            reference_action_exceeds_old_normalized_scale_frames=int(np.sum(np.any(
                np.abs(reference_actions[:,:6]) > np.array([.05,.05,.05,.5,.5,.5]), axis=1))),
            initial_state_sha256=actual_hash, matches_ipopt_initial_state=True, seed=0,
            frames=len(frames), mean_source_error_mm=float(errors.mean()*1000),
            max_source_error_mm=float(errors.max()*1000),
            max_goal_rotation_difference_rad=max(r['goal_rotation_difference_rad'] for r in frames),
            max_goal_position_difference_mm=max(r['goal_position_difference_m']*1000 for r in frames),
            native_kp=controller.kp.tolist(), native_kd=controller.kd.tolist(),
            refresh_controller_after_restore=refresh_controller,
            source_reset_state=source_reset_state,
            source_reset_history=source_reset_history,
            full_context_dir=str(full_context_dir) if full_context_dir is not None else None,
            reset_rng_matches_source=rng_matches_source,
            property_sampler_count=len(env.env.object_property_initializers),
            initial_gripper_action=initial_gripper_action.tolist(),
            fixtures={name:dict(position=env.sim.model.body_pos[env.sim.model.body_name2id(obj.root_body)].tolist(),rotation=env.sim.model.body_quat[env.sim.model.body_name2id(obj.root_body)].tolist()) for name,obj in env.env.fixtures_dict.items()},
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reference_module_sha256=hashlib.sha256((ROOT/'experiments/libero_tasktol_compat/osc_reference.py').read_bytes()).hexdigest(),
        )
        (output / 'trace.json').write_text(json.dumps(frames) + '\n')
        (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=int)
    parser.add_argument('--cases', type=int, nargs='+', default=list(range(40)))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--refresh-controller', action='store_true')
    parser.add_argument('--source-reset-state', action='store_true')
    parser.add_argument('--source-reset-history', action='store_true')
    parser.add_argument('--full-context-dir',type=Path)
    parser.add_argument('--osc-reference-mode', choices=['feedback_delta','accumulated_reference'], default='feedback_delta')
    parser.add_argument('--reference-root',type=Path,help='Read the same saved physical action HDF5 as IPOPT')
    args = parser.parse_args()
    if args.case is not None:
        task=json.loads(SELECTION.read_text())['tasks'][args.case]
        reference=None if args.reference_root is None else args.reference_root/task['suite']/f"{task['task_id']:02d}"/'reference_actions.hdf5'
        print(json.dumps(replay(args.case, args.output, args.refresh_controller, args.source_reset_state, args.source_reset_history,args.full_context_dir,args.osc_reference_mode,reference)), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    selected = json.loads(SELECTION.read_text())['tasks']
    def run(case):
        t = selected[case]
        dest = args.output / t['suite'] / f"{t['task_id']:02d}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.with_suffix('.log').open('w') as log:
            command = [sys.executable, __file__, '--case', str(case), '--output', str(dest)]
            if args.refresh_controller:
                command.append('--refresh-controller')
            if args.source_reset_state:
                command.append('--source-reset-state')
            if args.source_reset_history:
                command.append('--source-reset-history')
            if args.full_context_dir is not None:
                command.extend(['--full-context-dir',str(args.full_context_dir)])
            command.extend(['--osc-reference-mode',args.osc_reference_mode])
            if args.reference_root is not None:
                command.extend(['--reference-root',str(args.reference_root.resolve())])
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        path = dest / 'report.json'
        return json.loads(path.read_text()) if path.exists() else dict(case=case, error='worker_failed', returncode=done.returncode)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, case) for case in args.cases]):
            result = future.result()
            results.append(result)
            (args.output / 'summary.json').write_text(json.dumps(dict(complete=len(results)==len(args.cases), results=results), indent=2)+'\n')
            print(f'{len(results)}/{len(args.cases)} {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
