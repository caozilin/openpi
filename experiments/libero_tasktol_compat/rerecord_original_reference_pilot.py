"""Paired small-sample recording through the existing OpenVLA replay recorder.

Only callbacks differ: raw native OSC, frozen original-goal OSC, or the maintained
IPOPT/B10 session. All modes use replay_demo's one recording/action loop and the
same native env.step. No fallback, additional hold, or relaxed constraints.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL = ROOT.parent / 'franka_mujoco'
sys.path[:0] = [str(ROOT), str(TASKTOL), str(ROOT / 'third_party/libero')]
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/tasktol_rerecord_numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tasktol_rerecord_mpl')

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.binding_utils import MjSim

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from examples.libero import regenerate_libero_dataset as recorder
from tasktol.benchmark.libero_reference_actions import load_reference_actions, verify_reference_anchor
from tasktol.control.task_space.kinematics import RobotKinematics
from tasktol.geometry import rotation_vector
from tasktol.simulation.libero import LiberoTaskTolSession, libero_handles
from tasktol.simulation.robot import find_handles
from tasktol.tolerance.runtime import advance_reference_pose
from experiments.libero_tasktol_compat.osc_reference import difference_reference_goals
from experiments.libero_tasktol_compat.audit_original_osc_samples import remap_xml

MODES = ['osc_original_actions', 'osc_original_remap', 'ipopt_original_remap_B10']
SELECTION = ROOT / 'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def retained_indices(actions):
    retained = []
    previous = None
    for index, action in enumerate(actions):
        if not recorder.is_noop(action, previous):
            retained.append(index)
            previous = action
    return np.asarray(retained, dtype=int)


def original_goals(group, env):
    """Use archived XML and native OSC, then express its tool frame locally."""
    states, actions = group['states'][()], group['actions'][()]
    indices = retained_indices(actions)
    assert np.any(actions[indices[0], 3:6] != 0), 'Unknown initial held rotation is excluded by selection'
    current = libero_handles(env)
    xml, mapping = remap_xml(str(group.attrs['model_file']))
    model = mujoco.MjModel.from_xml_string(xml)
    h = find_handles(model, current.spec)
    sim = MjSim(model)
    assert states.shape[1] == 1 + model.nq + model.nv
    sim.data.qpos[:] = states[0, 1:1 + model.nq]
    sim.forward()
    hand = model.body('robot0_right_hand').id
    offset = sim.data.site_xmat[h.ee_site_id].reshape(3, 3).T @ sim.data.body_xmat[hand].reshape(3, 3)
    h = replace(h, ee_rotation_offset=offset)
    raw_kin = RobotKinematics(model, h)
    local_kin = RobotKinematics(env.sim.model._model, current)
    q0 = states[0, 1 + h.qpos_ids]
    raw0, local0 = raw_kin.evaluate(q0), local_kin.evaluate(q0)
    fixed_rotation = raw0.rotation.T @ local0.rotation
    config = json.loads(group.file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
    ctrl = OperationalSpaceController(sim=sim, eef_name=h.spec.ee_site,
        joint_indexes=dict(joints=h.joint_ids, qpos=h.qpos_ids, qvel=h.dof_ids),
        actuator_range=(model.actuator_ctrlrange[h.actuator_ids, 0], model.actuator_ctrlrange[h.actuator_ids, 1]), **config)
    positions, rotations = [], []
    max_goal_so3_projection = 0.0
    max_position_crosscheck = 0.0
    max_rotation_crosscheck = 0.0
    for state, action in zip(states, actions, strict=True):
        sim.data.qpos[:] = state[1:1 + model.nq]
        sim.data.qvel[:] = state[1 + model.nq:]
        sim.forward()
        ctrl.new_update = True
        ctrl.set_goal(action[:6])
        positions.append(ctrl.goal_pos.copy())
        native_rotation = ctrl.goal_ori @ h.ee_rotation_offset @ fixed_rotation
        # robosuite's quaternion helper leaves ~1e-9 nonorthogonality. Rigid
        # rotation increments require SO(3); record the roundoff-only correction.
        u, _, vh = np.linalg.svd(native_rotation)
        rotation = u @ vh
        assert np.linalg.det(rotation) > 0
        max_goal_so3_projection = max(max_goal_so3_projection, float(np.max(np.abs(rotation - native_rotation))))
        rotations.append(rotation)
        q = state[1 + h.qpos_ids]
        raw, local = raw_kin.evaluate(q), local_kin.evaluate(q)
        max_position_crosscheck = max(max_position_crosscheck, float(np.linalg.norm(raw.position - local.position)))
        max_rotation_crosscheck = max(max_rotation_crosscheck, float(np.max(np.abs(raw.rotation @ fixed_rotation - local.rotation))))
    assert max_position_crosscheck < 1e-12 and max_rotation_crosscheck < 1e-12
    return dict(states=states, actions=actions, indices=indices,
                goal_positions=np.asarray(positions)[indices], goal_rotations=np.asarray(rotations)[indices],
                config=config, archived_xml_sha256=hashlib.sha256(str(group.attrs['model_file']).encode()).hexdigest(),
                remapped_xml=xml, asset_mappings=mapping,
                native_goal_so3_projection_max_element=max_goal_so3_projection,
                robot_geometry_position_crosscheck_m=max_position_crosscheck,
                robot_geometry_rotation_crosscheck=max_rotation_crosscheck)


def write_reference(path, demo, source, inputs, anchor_p, anchor_r):
    idx = inputs['indices']
    actions = difference_reference_goals(inputs['goal_positions'], inputs['goal_rotations'],
                                         inputs['actions'][idx, 6], anchor_p, anchor_r)
    with h5py.File(path, 'w') as file:
        file.attrs.update(schema='libero_osc_independent_reference_actions_v1', source_hdf5=str(Path(source).resolve()),
                          action_contract=json.dumps(dict(position='metres, adjacent frozen ORIGINAL-state OSC goals',
                              rotation='world rotation-vector radians', first='from common settled actual anchor',
                              source_sampling='OpenVLA no-op selection; original source indices retained',
                              gripper='unchanged original normalized command')))
        group = file.create_group('data').create_group(demo)
        group.attrs['input_state_action_sha256'] = hashlib.sha256(inputs['states'][idx].tobytes() + inputs['actions'][idx].tobytes()).hexdigest()
        for key, value in dict(actions=actions, anchor_position_m=anchor_p, anchor_tool_rotation=anchor_r,
                               goal_positions_m=inputs['goal_positions'], goal_tool_rotations=inputs['goal_rotations'],
                               original_source_indices=idx).items():
            group.create_dataset(key, data=value)


def worker(case, demo, output):
    assert mujoco.__version__ == '3.2.3'
    selected = json.loads(SELECTION.read_text())['tasks'][case]
    dest = output / selected['suite'] / f"{selected['task_id']:02d}" / demo
    dest.mkdir(parents=True, exist_ok=False)
    modes = []
    source = Path(selected['source_hdf5'])
    with h5py.File(selected['processed_hdf5'], 'r') as prior:
        configuration = json.loads(prior.attrs['configuration'])
        order = configuration['demo_ids']
        episode_index = order.index(demo)
        before_rng = None if not episode_index else json.loads(prior['regeneration/episodes'][order[episode_index - 1]].attrs['rng_state'])
        after_rng = json.loads(prior['regeneration/episodes'][demo].attrs['rng_state'])
        prior_success = bool(json.loads(prior['regeneration/episodes'][demo].attrs['metadata'])['success'])
    for mode in MODES:
        folder = dest / mode
        folder.mkdir()
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / selected['suite'] / (selected['task_name'] + '.bddl')),
                                 controller='OSC_POSE', camera_names=['agentview', 'robot0_eye_in_hand'],
                                 camera_heights=256, camera_widths=256, control_freq=20)
        rows = []
        context = {}
        start = time.monotonic()
        try:
            env.seed(configuration['seed'])
            env.env.object_property_initializers *= episode_index + 1
            if before_rng is not None:
                recorder._restore_rng(before_rng)
            else:
                # The original worker seeded NumPy through env.seed, but its
                # unused Python RNG was not seeded. Restore the logged state;
                # equality after reset verifies that it is indeed unused.
                recorder._restore_rng(dict(numpy=recorder._rng_state()['numpy'], python=after_rng['python']))
            with h5py.File(source, 'r') as file:
                raw = file['data'][demo]
                # Native reset supplies program history; archived state supplies
                # physics. This exactly retains the existing recorder lifecycle.
                # Reference decoding is delayed until AFTER settling, so helper
                # construction cannot consume/reset the recording RNG.
                with imageio.get_writer(folder / 'main_rgb.mp4', fps=20, codec='libx264', quality=7) as video:
                    def ready(native, obs, state):
                        actual_rng = plain(recorder._rng_state())
                        assert actual_rng == after_rng, 'Reset history differs from original regeneration ledger'
                        handles = libero_handles(native)
                        kin = RobotKinematics(native.sim.model._model, handles)
                        fk = kin.evaluate(native.sim.data.qpos[handles.qpos_ids])
                        inputs = original_goals(raw, native)
                        reference_path = dest / 'original_reference_actions.hdf5'
                        if not reference_path.exists():
                            write_reference(reference_path, demo, source, inputs, fk.position, fk.rotation)
                            (dest / 'original_model_local_paths.xml').write_text(inputs['remapped_xml'])
                        reference = load_reference_actions(reference_path, demo, source_path=source,
                            source_states=inputs['states'][inputs['indices']], source_actions=inputs['actions'][inputs['indices']])
                        verify_reference_anchor(reference, fk.position, fk.rotation)
                        context.update(handles=handles, kin=kin, reference=reference, inputs=inputs,
                            reference_p=reference.anchor_position.copy(), reference_r=reference.anchor_rotation.copy(),
                            initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                            model_physical_sha256=hashlib.sha256(b''.join(np.asarray(getattr(native.sim.model._model, k)).tobytes() for k in ['body_pos', 'body_quat', 'body_mass', 'body_inertia', 'geom_friction', 'jnt_range'])).hexdigest(),
                            initial_gripper=native.robots[0].gripper.current_action.copy(),
                            native_xml=native.sim.model.get_xml())
                        Image.fromarray(obs['agentview_image'][::-1, ::-1].copy()).save(folder / 'initial.png')
                        if mode == 'ipopt_original_remap_B10':
                            context['session'] = LiberoTaskTolSession(native, kp=150, joint_control_mode='osc_task_joint_null')
                            # Settling ends with cached site kinematics one
                            # physics step behind qpos. Use the saved FK anchor,
                            # as the canonical relabelled-action benchmark does.
                            context['session'].target.position = reference.anchor_position.copy()
                            context['session'].target.rotation = reference.anchor_rotation.copy()
                            context['session'].release.accepted_rotation = reference.anchor_rotation.copy()
                        elif mode == 'osc_original_remap':
                            native.robots[0].controller.use_delta = False

                    def execute(native, raw_index, action, obs):
                        index = len(rows)
                        reference = context['reference']
                        assert raw_index == context['inputs']['indices'][index]
                        video.append_data(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                        if mode == 'ipopt_original_remap_B10':
                            transitions = []
                            native_step = native.step

                            def observe(command):
                                transition = native_step(command)
                                transitions.append(transition)
                                return transition

                            native.step = observe
                            try:
                                next_obs, row = context['session'].step(reference.actions[index], phase='outside_subtask')
                            finally:
                                native.step = native_step
                            row['raw_source_index'] = raw_index
                            rows.append(row)
                            if row['aborted']:
                                assert not transitions and row['physics_steps'] == 0
                                raise recorder.ReplayActionRejected(row['optimizer']['status'])
                            assert len(transitions) == 1
                            assert np.linalg.norm(np.array(row['nominal_target']['position_m']) - reference.goal_positions[index]) < 1e-10
                            return transitions[0]
                        before_time = float(native.sim.data.time)
                        if mode == 'osc_original_remap':
                            context['reference_p'], context['reference_r'] = advance_reference_pose(
                                context['reference_p'], context['reference_r'], reference.actions[index])
                            assert np.linalg.norm(context['reference_p'] - reference.goal_positions[index]) < 1e-10
                            command = np.r_[context['reference_p'], rotation_vector(context['reference_r'] @ context['handles'].ee_rotation_offset.T), action[6]]
                        else:
                            command = action
                        transition = native.step(command.tolist())
                        rows.append(dict(index=index, raw_source_index=raw_index, success=bool(transition[2]), aborted=False,
                                         physics_steps=int(round((float(native.sim.data.time) - before_time) / native.sim.model.opt.timestep)),
                                         actual_q=native.sim.data.qpos[context['handles'].qpos_ids].copy()))
                        return transition

                    result = recorder.replay_demo(env, raw, on_settled=ready, action_executor=execute, retain_failed=True)
                indices = result['arrays']['source_step_indices']
                positions_in_reference = np.searchsorted(context['inputs']['indices'], indices)
                assert np.array_equal(context['inputs']['indices'][positions_in_reference], indices)
                original_actions = result['arrays']['actions'].copy()
                if mode != 'osc_original_actions':
                    result['arrays']['actions'] = context['reference'].actions[positions_in_reference]
                with h5py.File(folder / 'recording.hdf5', 'w') as saved:
                    saved.attrs.update(mode=mode, task=selected['task_name'], success=bool(result['success']),
                                       action_contract='normalized native OSC' if mode == 'osc_original_actions' else 'physical independent reference increments [m, rad, original gripper]',
                                       reference_actions_sha256=context['reference'].metadata['sha256'])
                    group = saved.create_group('attempts').create_group(demo)
                    recorder._write_demo(group, raw, result, dict(compression='gzip', gzip_level=1))
                    group.attrs['source_model_file'] = raw.attrs['model_file']
                    group.attrs['model_file'] = context['native_xml']
                    group['tasktol'].create_dataset('source_osc_actions', data=original_actions)
                    if not result['success']:
                        group['dones'][:] = 0
                        group['rewards'][:] = 0
                    saved.create_group('data')
                    if result['success']:
                        saved['data'][demo] = group
                obs = env.env._get_observations()
                Image.fromarray(obs['agentview_image'][::-1, ::-1].copy()).save(folder / 'final.png')
                report = dict(mode=mode, success=bool(result['success']), ever_success=any(row['success'] for row in rows if not row['aborted']),
                              expected_frames=len(context['inputs']['indices']), attempted_frames=len(rows),
                              executed_frames=sum(not r['aborted'] for r in rows), execution_rejected=result['execution_rejected'],
                              rejection=result['execution_rejection'], rejected_source_index=result['rejected_source_index'],
                              discarded_noops_total=len(context['inputs']['actions']) - len(context['inputs']['indices']),
                              initial_state_sha256=context['initial_state_sha256'], model_physical_sha256=context['model_physical_sha256'],
                              initial_gripper=context['initial_gripper'], reference=context['reference'].metadata,
                              reset_rng_matches_prior_ledger=True, prior_OpenVLA_success=prior_success,
                              physics_steps=sum(row['physics_steps'] for row in rows), elapsed_s=time.monotonic() - start,
                              tolerance='off', native_kp=env.robots[0].controller.kp.copy(),
                              native_goal_so3_projection_max_element=context['inputs']['native_goal_so3_projection_max_element'],
                              factor_statistics=getattr(env.robots[0].controller, 'factor_statistics', None))
                (folder / 'trajectory.json').write_text(json.dumps(plain(rows), indent=2) + '\n')
                (folder / 'report.json').write_text(json.dumps(plain(report), indent=2) + '\n')
                modes.append(report)
                print(json.dumps(plain(dict(case=case, demo=demo, mode=mode, success=report['success'], executed=report['executed_frames'], rejected=report['rejected_source_index']))), flush=True)
        finally:
            env.close()
    assert len({m['initial_state_sha256'] for m in modes}) == 1
    assert len({m['model_physical_sha256'] for m in modes}) == 1
    assert len({m['reference']['sha256'] for m in modes}) == 1
    assert all(np.array_equal(m['initial_gripper'], modes[0]['initial_gripper']) for m in modes)
    summary = dict(case=case, suite=selected['suite'], task_id=selected['task_id'], task_name=selected['task_name'],
                   demo=demo, prior_OpenVLA_success=prior_success, paired_initialization_verified=True, results=modes)
    (dest / 'summary.json').write_text(json.dumps(plain(summary), indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--case', type=int)
    parser.add_argument('--demo')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        worker(args.case, args.demo, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    selection = json.loads(SELECTION.read_text())['tasks']
    plan = []
    for case in [4, 10, 22, 38]:
        selected = selection[case]
        with h5py.File(selected['source_hdf5'], 'r') as source:
            eligible = []
            for demo in sorted(source['data'], key=lambda x: int(x.split('_')[-1])):
                actions = source['data'][demo]['actions'][()]
                idx = retained_indices(actions)
                if len(idx) and np.any(actions[idx[0], 3:6] != 0):
                    eligible.append(demo)
            demos = list(dict.fromkeys(eligible[:2] + [selected['demo_id']]))
            assert selected['demo_id'] in eligible
            while len(demos) < 3:
                demos.append(next(d for d in eligible if d not in demos))
            for demo in demos:
                plan.append(dict(case=case, demo=demo, suite=selected['suite'], task_id=selected['task_id']))
    provenance = dict(selection='4 suites, one diagnostic task per suite; first two original numbered demos with known first orientation plus fixed representative; no success-based selection',
                      diagnostics_not_population_estimate=True, modes=MODES, plan=plan,
                      identical_OpenVLA_settling_and_noop_filter=True, source_reference='archived original-state OSC goals, not regenerated-state goals',
                      source_sha256={str(p): sha(p) for p in [Path(__file__), Path(recorder.__file__), TASKTOL / 'tasktol/simulation/libero.py',
                          TASKTOL / 'tasktol/simulation/libero_joint_controller.py', TASKTOL / 'tasktol/control/ipopt/controller.py']})
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    completed = []

    def run(job):
        log = args.output / f"case_{job['case']:02d}_{job['demo']}.log"
        env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with log.open('w') as stream:
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--case', str(job['case']), '--demo', job['demo'], '--output', str(args.output.resolve())], env=env, stdout=stream, stderr=subprocess.STDOUT)
        summary = args.output / job['suite'] / f"{job['task_id']:02d}" / job['demo'] / 'summary.json'
        return json.loads(summary.read_text()) if summary.exists() else dict(**job, error='worker_failed', returncode=done.returncode, log=str(log))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in plan]):
            completed.append(future.result())
            (args.output / 'summary.json').write_text(json.dumps(dict(complete=len(completed) == len(plan), results=completed), indent=2) + '\n')
            print(f'{len(completed)}/{len(plan)} ' + json.dumps({k: v for k, v in completed[-1].items() if k != 'results'}), flush=True)
    if any('error' in result for result in completed):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
