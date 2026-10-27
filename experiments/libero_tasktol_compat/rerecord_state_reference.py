"""State-derived reference experiments using the maintained native replay loop.

No source joint targets or future states enter the online IK/controller. Future
poses are offline labels, differenced once and then cumulatively integrated.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'franka_mujoco'), str(ROOT / 'third_party/libero')]
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/tasktol_state_relabel_numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tasktol_state_relabel_mpl')

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from examples.libero import regenerate_libero_dataset as recorder
from experiments.libero_tasktol_compat.osc_reference import difference_reference_goals, integrate_reference_actions
from experiments.libero_tasktol_compat.rerecord_original_reference_pilot import SELECTION, TASKTOL, plain, sha
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tasktol.control.task_space.kinematics import RobotKinematics
from tasktol.geometry import rotation_vector
from tasktol.simulation.libero import LiberoTaskTolSession, libero_handles
from tasktol.tolerance.runtime import advance_reference_pose

BASE_PLAN = ROOT / 'experiments/libero_tasktol_compat/results/original_remap_rerecord_12_v1/provenance.json'
MODES = ['osc_original', 'osc_state', 'ipopt_state_B10']


def state_reference(states, actions, handles, kin, lookahead, time_scale=1):
    """All original 50ms intervals, including waits; final missing next state omitted."""
    count = min(len(actions), len(states) - 1) * time_scale
    coordinate = np.arange(count + 1, dtype=float) / time_scale
    q_source = states[:, 1 + handles.qpos_ids]
    q = np.column_stack([np.interp(coordinate, np.arange(len(states)), q_source[:, axis]) for axis in range(7)])
    poses = [kin.evaluate(value, include_manipulability=False, include_link_points=False) for value in q]
    positions = np.asarray([pose.position for pose in poses])
    rotations = np.asarray([pose.rotation for pose in poses])
    targets = np.minimum(np.arange(count) + lookahead, count)
    source_indices = np.arange(count) // time_scale
    increments = difference_reference_goals(positions[targets], rotations[targets], actions[source_indices, 6], positions[0], rotations[0])
    p, r = integrate_reference_actions(increments, positions[0], rotations[0])
    assert np.max(np.abs(p - positions[targets])) < 1e-10
    assert np.max(np.abs(r - rotations[targets])) < 1e-10
    outside = np.maximum(handles.joint_lower - q, q - handles.joint_upper)
    return dict(actions=increments, positions=positions, rotations=rotations, target_indices=targets,
                q=q, joint_outside_rad=np.maximum(0, np.max(outside, axis=1)),
                source_action_indices=source_indices, source_state_coordinate=coordinate)


def worker(job, output, lookahead, modes, time_scale, gripper_policy='preserve', pulse_phase=0, label_method='achieved_state_fk'):
    assert mujoco.__version__ == '3.2.3'
    selected = json.loads(SELECTION.read_text())['tasks'][job['case']]
    demo = job['demo']
    dest = output / selected['suite'] / f"{selected['task_id']:02d}" / demo
    dest.mkdir(parents=True, exist_ok=False)
    with h5py.File(selected['processed_hdf5'], 'r') as prior:
        configuration = json.loads(prior.attrs['configuration'])
        order = configuration['demo_ids']
        episode_index = order.index(demo)
        before_rng = None if not episode_index else json.loads(prior['regeneration/episodes'][order[episode_index - 1]].attrs['rng_state'])
        after_rng = json.loads(prior['regeneration/episodes'][demo].attrs['rng_state'])
    reports = []
    for mode in modes:
        folder = dest / mode
        folder.mkdir()
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path('bddl_files')) / selected['suite'] / (selected['task_name'] + '.bddl')),
            controller='OSC_POSE', camera_names=['agentview', 'robot0_eye_in_hand'],
            camera_heights=256, camera_widths=256, control_freq=20,
        )
        rows, context = [], {}
        started = time.monotonic()
        try:
            env.seed(configuration['seed'])
            env.env.object_property_initializers *= episode_index + 1
            if before_rng is not None:
                recorder._restore_rng(before_rng)
            else:
                recorder._restore_rng(dict(numpy=recorder._rng_state()['numpy'], python=after_rng['python']))
            with h5py.File(selected['source_hdf5'], 'r') as source:
                raw = source['data'][demo]
                states, original = raw['states'][()], raw['actions'][()]
                expected_frames = (len(states) - 1) * time_scale
                replay_actions = original[np.arange(expected_frames) // time_scale]
                source_input_sha256 = hashlib.sha256(states.tobytes() + original.tobytes() + str(raw.attrs['model_file']).encode()).hexdigest()
                assert np.allclose(np.diff(states[:, 0]), .05, rtol=0, atol=1e-10)
                with imageio.get_writer(folder / 'main_rgb.mp4', fps=20, codec='libx264', quality=7) as video:
                    def ready(native, obs, state):
                        assert plain(recorder._rng_state()) == after_rng
                        np.testing.assert_array_equal(state, states[0])
                        handles = libero_handles(native)
                        assert np.array_equal(handles.qpos_ids, np.arange(7))
                        assert np.array_equal(handles.finger_qpos_ids, np.arange(7, 9))
                        kin = RobotKinematics(native.sim.model._model, handles)
                        reference = state_reference(states, original, handles, kin, lookahead, time_scale)
                        if label_method == 'contact_hybrid':
                            from experiments.libero_tasktol_compat.state_labelers import add_contact_bias
                            config = json.loads(source['data'].attrs['env_args'])['env_kwargs']['controller_configs']
                            assert np.array_equal(config['output_max'], [.05, .05, .05, .5, .5, .5])
                            reference = add_contact_bias(reference, states, original, handles, kin)
                        if gripper_policy == 'native_integrator_pulse':
                            reference['actions'][np.arange(expected_frames) % time_scale != pulse_phase, 6] = 0
                        context.update(handles=handles, kin=kin, reference=reference,
                                       p=reference['positions'][0].copy(), r=reference['rotations'][0].copy(),
                                       initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                                       initial_gripper=native.robots[0].gripper.current_action.copy(),
                                       model_sha256=hashlib.sha256(b''.join(np.asarray(getattr(native.sim.model._model, key)).tobytes() for key in ['body_pos', 'body_quat', 'body_mass', 'body_inertia', 'geom_friction', 'jnt_range'])).hexdigest(),
                                       native_xml=native.sim.model.get_xml())
                        Image.fromarray(obs['agentview_image'][::-1, ::-1].copy()).save(folder / 'initial.png')
                        path = dest / 'state_reference.hdf5'
                        if not path.exists():
                            with h5py.File(path, 'w') as file:
                                file.attrs.update(source=str(selected['source_hdf5']), demo=demo,
                                                  source_input_sha256=source_input_sha256,
                                                  lookahead_steps=lookahead, dt_s=.05,
                                                  time_scale=time_scale,
                                                  gripper_policy=gripper_policy, pulse_phase=pulse_phase,
                                                  label_method=label_method,
                                                  contract='physical metres/world rotvec radians; cumulative independent reference; unchanged original gripper')
                                for key, value in reference.items():
                                    file.create_dataset(key, data=value)
                        if mode == 'ipopt_state_B10':
                            session = LiberoTaskTolSession(native, kp=150, joint_control_mode='osc_task_joint_null')
                            session.target.position = context['p'].copy()
                            session.target.rotation = context['r'].copy()
                            session.release.accepted_rotation = context['r'].copy()
                            context['session'] = session
                        elif mode == 'osc_state':
                            native.robots[0].controller.use_delta = False

                    def execute(native, index, action, obs):
                        assert index == len(rows)
                        video.append_data(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                        reference, kin, handles = context['reference'], context['kin'], context['handles']
                        before = kin.evaluate(native.sim.data.qpos[handles.qpos_ids])
                        started_time = float(native.sim.data.time)
                        if mode == 'ipopt_state_B10':
                            transitions = []
                            native_step = native.step

                            def observe(command):
                                transition = native_step(command)
                                transitions.append(transition)
                                return transition

                            native.step = observe
                            try:
                                _, row = context['session'].step(reference['actions'][index], phase='outside_subtask')
                            finally:
                                native.step = native_step
                            row['raw_source_index'] = int(reference['source_action_indices'][index])
                            if row['aborted']:
                                rows.append(row)
                                assert not transitions and row['physics_steps'] == 0
                                raise recorder.ReplayActionRejected(row['optimizer']['status'])
                            assert len(transitions) == 1
                            transition = transitions[0]
                        else:
                            if mode == 'osc_state':
                                context['p'], context['r'] = advance_reference_pose(context['p'], context['r'], reference['actions'][index])
                                command = np.r_[context['p'], rotation_vector(context['r'] @ handles.ee_rotation_offset.T), reference['actions'][index, 6]]
                            else:
                                command = action
                            transition = native.step(command.tolist())
                            row = dict(index=index, raw_source_index=int(reference['source_action_indices'][index]), success=bool(transition[2]), aborted=False,
                                       physics_steps=int(round((float(native.sim.data.time) - started_time) / native.sim.model.opt.timestep)))
                        after = kin.evaluate(native.sim.data.qpos[handles.qpos_ids])
                        target = reference['target_indices'][index]
                        source_p = reference.get('source_positions_m', reference['positions'])
                        source_r = reference.get('source_rotations', reference['rotations'])
                        row.update(pre_position_m=before.position, actual_position_m=after.position,
                                   actual_rotation=after.rotation,
                                   reference_position_m=reference['positions'][target],
                                   target_source_index=int(target),
                                   target_original_state_coordinate=float(reference['source_state_coordinate'][target]),
                                   target_q_witness_outside_rad=float(reference['joint_outside_rad'][target]),
                                   tracking_position_error_m=float(np.linalg.norm(after.position - reference['positions'][target])),
                                   tracking_rotation_error_rad=float(np.linalg.norm(rotation_vector(after.rotation @ reference['rotations'][target].T))),
                                   source_next_position_error_m=float(np.linalg.norm(after.position - source_p[index + 1])),
                                   source_next_rotation_error_rad=float(np.linalg.norm(rotation_vector(after.rotation @ source_r[index + 1].T))),
                                   source_next_object_qpos_l2=float(np.linalg.norm(native.sim.data.qpos[9:] - states[index + 1, 10:1 + native.sim.model.nq])) if time_scale == 1 else None)
                        assert row['physics_steps'] == 25
                        if mode == 'ipopt_state_B10':
                            np.testing.assert_allclose(row['nominal_target']['position_m'], reference['positions'][target], atol=1e-10, rtol=0)
                        rows.append(row)
                        return transition

                    result = recorder.replay_demo(env, raw, on_settled=ready, action_executor=execute,
                                                  retain_failed=True, skip_noops=False, settle_steps=0,
                                                  max_source_steps=expected_frames, replay_actions=replay_actions)
                executed = [row for row in rows if not row['aborted']]
                indices = result['arrays']['source_step_indices']
                original_actions = result['arrays']['actions'].copy()
                if mode != 'osc_original':
                    result['arrays']['actions'] = context['reference']['actions'][indices]
                with h5py.File(folder / 'recording.hdf5', 'w') as saved:
                    saved.attrs.update(mode=mode, success=bool(result['success']), lookahead_steps=lookahead,
                                       action_contract='normalized native OSC' if mode == 'osc_original' else 'physical independent pose increments [m, rad, gripper]')
                    group = saved.create_group('attempts').create_group(demo)
                    recorder._write_demo(group, raw, result, dict(compression='gzip', gzip_level=1))
                    group.attrs['source_model_file'] = raw.attrs['model_file']
                    group.attrs['model_file'] = context['native_xml']
                    group['tasktol'].create_dataset('source_osc_actions', data=original_actions)
                    group['tasktol'].attrs['source_step_indices_domain'] = 'retimed command grid at 20Hz'
                    group['tasktol'].create_dataset('original_source_action_indices', data=context['reference']['source_action_indices'][indices])
                    if not result['success']:
                        group['dones'][:] = 0
                        group['rewards'][:] = 0
                    saved.create_group('data')
                    if result['success']:
                        saved['data'][demo] = group
                obs = env.env._get_observations()
                Image.fromarray(obs['agentview_image'][::-1, ::-1].copy()).save(folder / 'final.png')
                report = dict(mode=mode, success=bool(result['success']), ever_success=any(row['success'] for row in executed),
                              expected_frames=expected_frames, executed_frames=len(executed),
                              execution_rejected=result['execution_rejected'], rejected_source_index=result['rejected_source_index'],
                              rejected_index_domain='retimed command grid',
                              initial_state_sha256=context['initial_state_sha256'], model_sha256=context['model_sha256'],
                              initial_gripper=context['initial_gripper'], lookahead_steps=lookahead,
                              time_scale=time_scale,
                              gripper_policy=gripper_policy, pulse_phase=pulse_phase,
                              label_method=label_method,
                              source_state_motion='actual FK plus projected source motion bias in contact' if label_method == 'contact_hybrid' else 'original actual FK, no source actions in motion labels',
                              source_input_sha256=source_input_sha256,
                              skip_noops=False, settle_steps=0, dropped_final_missing_next_state=True,
                              tolerance='off', full_joint_reference=mode == 'ipopt_state_B10',
                              mean_source_next_position_error_mm=float(np.mean([row['source_next_position_error_m'] for row in executed]) * 1000) if executed else None,
                              max_source_next_position_error_mm=float(np.max([row['source_next_position_error_m'] for row in executed]) * 1000) if executed else None,
                              elapsed_s=time.monotonic() - started,
                              factor_statistics=getattr(env.robots[0].controller, 'factor_statistics', None))
                (folder / 'trajectory.json').write_text(json.dumps(plain(rows), indent=2) + '\n')
                (folder / 'report.json').write_text(json.dumps(plain(report), indent=2) + '\n')
                reports.append(report)
                print(json.dumps(plain(dict(case=job['case'], demo=demo, **report))), flush=True)
        finally:
            env.close()
    assert len({report['initial_state_sha256'] for report in reports}) == 1
    assert len({report['model_sha256'] for report in reports}) == 1
    assert all(np.array_equal(report['initial_gripper'], reports[0]['initial_gripper']) for report in reports)
    summary = dict(**job, paired_initialization_verified=True, results=reports)
    (dest / 'summary.json').write_text(json.dumps(plain(summary), indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=BASE_PLAN)
    parser.add_argument('--lookahead', type=int, default=1)
    parser.add_argument('--time-scale', type=int, default=1)
    parser.add_argument('--gripper-policy', choices=['preserve', 'native_integrator_pulse'], default='preserve')
    parser.add_argument('--pulse-phase', type=int, default=0)
    parser.add_argument('--label-method', choices=['achieved_state_fk', 'contact_hybrid'], default='achieved_state_fk')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--case', type=int)
    parser.add_argument('--demo')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.lookahead < 1 or args.time_scale < 1:
        parser.error('lookahead and time-scale must be positive')
    if args.time_scale != 1 and 'osc_original' in args.modes:
        parser.error('Raw original actions are only a baseline at their original time scale')
    if not 0 <= args.pulse_phase < args.time_scale:
        parser.error('pulse phase must be within the retimed interval')
    plan = json.loads(args.plan.read_text())['plan']
    plan = [job for job in plan if (args.case is None or job['case'] == args.case) and (args.demo is None or job['demo'] == args.demo)]
    assert plan
    if args.worker:
        assert len(plan) == 1
        worker(plan[0], args.output, args.lookahead, args.modes, args.time_scale, args.gripper_policy, args.pulse_phase, args.label_method)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(plan=plan, modes=args.modes, lookahead_steps=args.lookahead, time_scale=args.time_scale,
                      gripper_policy=args.gripper_policy, pulse_phase=args.pulse_phase,
                      label_method=args.label_method,
                      source='original states, not original OSC goals or regenerated states',
                      time_contract='20Hz execution; uniform joint-path interpolation and synchronized gripper retiming; all waits retained; zero settling; omit missing terminal state',
                      source_sha256={str(path): sha(path) for path in [Path(__file__), Path(recorder.__file__),
                          ROOT / 'experiments/libero_tasktol_compat/osc_reference.py',
                          ROOT / 'experiments/libero_tasktol_compat/state_labelers.py',
                          TASKTOL / 'tasktol/simulation/libero.py', TASKTOL / 'tasktol/simulation/libero_joint_controller.py',
                          TASKTOL / 'tasktol/control/ipopt/controller.py', TASKTOL / 'tasktol/control/task_space/kinematics.py']})
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    snapshots = {}
    (args.output / 'source_snapshot').mkdir()
    for index, path in enumerate(provenance['source_sha256']):
        target = args.output / 'source_snapshot' / f'{index}_{Path(path).name}'
        shutil.copyfile(path, target)
        snapshots[path] = str(target.relative_to(args.output))
    (args.output / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')

    def run(job):
        log = args.output / f"case_{job['case']:02d}_{job['demo']}.log"
        env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--case', str(job['case']), '--demo', job['demo'], '--plan', str(args.plan.resolve()), '--output', str(args.output.resolve()), '--lookahead', str(args.lookahead), '--time-scale', str(args.time_scale), '--gripper-policy', args.gripper_policy, '--pulse-phase', str(args.pulse_phase), '--label-method', args.label_method, '--modes', *args.modes]
        with log.open('w') as stream:
            done = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        path = args.output / job['suite'] / f"{job['task_id']:02d}" / job['demo'] / 'summary.json'
        return json.loads(path.read_text()) if path.exists() else dict(**job, error='worker_failed', returncode=done.returncode, log=str(log))

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in plan]):
            completed.append(future.result())
            aggregate = {mode: dict(final=sum(report['success'] for job in completed for report in job.get('results', []) if report['mode'] == mode),
                                    ever=sum(report['ever_success'] for job in completed for report in job.get('results', []) if report['mode'] == mode),
                                    rejected=sum(report['execution_rejected'] for job in completed for report in job.get('results', []) if report['mode'] == mode)) for mode in args.modes}
            (args.output / 'summary.json').write_text(json.dumps(dict(complete=len(completed) == len(plan), aggregate=aggregate, results=completed), indent=2) + '\n')
            print(json.dumps(dict(completed=len(completed), aggregate=aggregate)), flush=True)
    if any('error' in job for job in completed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
