"""Native LIBERO environment compatibility probe; no TaskTol rollout implementation.

Uses LIBERO's own OffScreenRenderEnv and env.step. TaskTol is imported from its
existing checkout. This probe does not install a controller or alter annotations.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True

OPENPI = Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL = Path('/media/czl/sata/franka_my_code/franka_mujoco')
LIBERO = OPENPI / 'third_party/libero'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--suite', default='libero_spatial')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--steps', type=int, default=0, help='0: complete saved demonstration')
    parser.add_argument('--skip-tasktol', action='store_true', help='Old LIBERO baseline environment only')
    parser.add_argument('--compat', action='store_true', help='Opt-in inspected robosuite 1.4.1 / MuJoCo 3.10 API bridge')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
    os.environ.setdefault('NUMBA_CACHE_DIR', str(out / 'numba_cache'))
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib'))
    sys.path[:0] = [str(TASKTOL), str(LIBERO)]
    report = dict(status='running', python=sys.version, executable=sys.executable,
                  output=str(out), stages=[], packages={}, tasktol_checkout=str(TASKTOL),
                  libero_checkout=str(LIBERO), renderer=os.environ['MUJOCO_GL'],
                  scope='Environment compatibility only. Original OSC controller; TaskTol optimizer not controlling rollout.')
    report['probe_sha256'] = sha(Path(__file__))
    report['compatibility_sha256'] = sha(Path(__file__).with_name('compatibility.py'))
    for name in ['mujoco', 'numpy', 'robosuite', 'numba', 'scipy', 'cyipopt', 'h5py', 'gym', 'bddl']:
        try:
            report['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report['packages'][name] = None

    def save():
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    @contextmanager
    def stage(name):
        record = dict(name=name, status='running')
        report['stages'].append(record)
        save()
        print('START ' + name, flush=True)
        start = time.monotonic()
        try:
            yield record
        except Exception as exc:
            record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise
        else:
            record['status'] = 'passed'
        finally:
            record['elapsed_s'] = time.monotonic() - start
            save()
            print(record['status'].upper() + ' ' + name, flush=True)

    env = None
    try:
        with stage('imports') as record:
            import h5py
            import imageio.v2 as imageio
            import mujoco
            import numpy as np
            from PIL import Image
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            if args.compat:
                from compatibility import install_mass_matrix_bridge
                report['compatibility_bridges'] = install_mass_matrix_bridge()
            if not args.skip_tasktol:
                import cyipopt
                from tasktol.control.ipopt import IpoptIK
                from tasktol.control.task_space.kinematics import RobotKinematics
                from tasktol.tolerance.runtime import advance_reference_pose
                record['tasktol_imports'] = [IpoptIK.__module__, RobotKinematics.__module__, advance_reference_pose.__module__]
                report['tasktol_source_fingerprints'] = {str(p.relative_to(TASKTOL)): sha(p) for p in [
                    TASKTOL/'pyproject.toml', TASKTOL/'tasktol/control/ipopt/controller.py',
                    TASKTOL/'tasktol/control/task_space/kinematics.py', TASKTOL/'tasktol/tolerance/runtime.py']}
            report['libero_paths'] = {key: get_libero_path(key) for key in ['assets', 'bddl_files', 'init_states', 'datasets']}
            assert Path(report['libero_paths']['assets']).resolve() == LIBERO/'libero/libero/assets'
            record['assets_referenced_in_place'] = True

        with stage('load_demonstration_and_task') as record:
            selection_path = OPENPI/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'
            selected = next(t for t in json.loads(selection_path.read_text())['tasks']
                            if t['suite'] == args.suite and t['task_id'] == args.task_id)
            bddl = Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')
            with h5py.File(selected['source_hdf5'], 'r') as file:
                group = file['data'][selected['demo_id']]
                initial_state = group['states'][0]
                source_actions = group['actions'][()]
            # Exactly reuse the existing preprocessing selection of non-noop actions.
            indices = np.asarray(selected['source_step_indices'], dtype=int)
            actions = source_actions[indices]
            if args.steps:
                actions = actions[:args.steps]
            report['task'] = dict(suite=args.suite, task_id=args.task_id, name=selected['task_name'],
                demo=selected['demo_id'], bddl=str(bddl), source_hdf5=selected['source_hdf5'],
                bddl_sha256=sha(bddl), initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
                actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(), retained_source_indices=indices[:len(actions)].tolist(),
                initial_state_source='First saved state of the original demonstration, not torch benchmark .pruned_init',
                seed=7, settling_steps=10, control_freq_hz=20)
            assert bddl.is_file()
            record['action_count'] = len(actions)

        with stage('construct_native_libero_environment') as record:
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=['agentview'],
                                     camera_heights=256, camera_widths=256, control_freq=20)
            env.seed(7)
            record['controller'] = type(env.robots[0].controller).__name__

        with stage('reset_restore_and_settle') as record:
            env.reset()
            env.set_init_state(initial_state)
            for _ in range(10):
                obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
            record['state_finite'] = bool(np.all(np.isfinite(env.sim.data.qpos)))
            record['initial_success'] = bool(env.check_success())
            assert record['state_finite']
            Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])).save(out/'initial.png')

        with stage('native_model_data_and_jacobian') as record:
            model, data = env.sim.model._model, env.sim.data._data
            assert isinstance(model, mujoco.MjModel) and isinstance(data, mujoco.MjData)
            robot = env.robots[0]
            site_id = int(robot.eef_site_id)
            # mj_step integrates qpos after computing position-dependent fields.
            # Refresh a COPY for FK/Jacobian comparison at identical qpos; never
            # advance or rewrite the live simulator during the diagnostic.
            reference_data = copy.copy(data)
            mujoco.mj_forward(model, reference_data)
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, reference_data, jp, jr, site_id)
            dofs = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
            jac = np.vstack([jp[:, dofs], jr[:, dofs]])
            assert jac.shape == (6, 7) and np.all(np.isfinite(jac))
            record.update(native_model=True, native_data=True, arm_jacobian_shape=list(jac.shape),
                eef_site=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id),
                position_observation_error_m=float(np.linalg.norm(data.site_xpos[site_id]-obs['robot0_eef_pos'])),
                live_derived_pose_lag_m=float(np.linalg.norm(data.site_xpos[site_id]-reference_data.site_xpos[site_id])),
                diagnostic_reference='mj_forward on independent MjData copy at current qpos; no live state mutation',
                joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in robot._ref_joint_indexes],
                nq=model.nq, nv=model.nv, nu=model.nu, physics_dt=float(model.opt.timestep))
            if not args.skip_tasktol:
                # Only the FK query interface is mapped here. No actuator or
                # gripper-controller mapping is claimed by this probe.
                from types import SimpleNamespace
                probe_handles = SimpleNamespace(
                    ee_site_id=site_id,
                    qpos_ids=np.asarray(robot._ref_joint_pos_indexes, dtype=int),
                    dof_ids=dofs,
                    link_body_ids=model.jnt_bodyid[np.asarray(robot._ref_joint_indexes, dtype=int)])
                qpos_before = data.qpos.copy()
                kin = RobotKinematics(model, probe_handles)
                state = kin.evaluate(data.qpos[probe_handles.qpos_ids],
                                     include_manipulability=False, include_link_points=False)
                fk_error = float(np.linalg.norm(state.position-reference_data.site_xpos[site_id]))
                angular_world = state.rotation @ state.jacobian[3:]
                jac_error = float(np.max(np.abs(np.vstack([state.jacobian[:3], angular_world])-jac)))
                record.update(tasktol_fk_position_error_m=fk_error,
                              tasktol_jacobian_max_error=jac_error,
                              tasktol_fk_preserves_simulation_state=True)
                assert fk_error < 1e-8 and jac_error < 1e-8
                assert np.array_equal(qpos_before, data.qpos)

        with stage('native_osc_demonstration_replay') as record:
            positions, joints, sim_times, successes = [], [], [], []
            expected_dt = 1./20
            started = float(env.sim.data.time)
            with imageio.get_writer(out/'replay.mp4', fps=20, codec='libx264', quality=8) as writer:
                for index, action in enumerate(actions):
                    positions.append(np.asarray(obs['robot0_eef_pos']).copy())
                    joints.append(np.asarray(obs['robot0_joint_pos']).copy())
                    sim_times.append(float(env.sim.data.time))
                    writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                    obs, reward, done, info = env.step(action.tolist())
                    success = bool(env.check_success())
                    successes.append(success)
                    assert np.all(np.isfinite(env.sim.data.qpos)) and np.all(np.isfinite(env.sim.data.qvel))
                    assert abs(float(env.sim.data.time)-(started+(index+1)*expected_dt)) < 1e-7
                    if index in [len(actions)//2, len(actions)-1]:
                        Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])).save(out/f'after_{index+1:04d}.png')
                    if (index+1)%25 == 0:
                        print(f'REPLAY {index+1}/{len(actions)}', flush=True)
            np.savez_compressed(out/'observations.npz', eef_position=np.array(positions),
                joint_position=np.array(joints), sim_time=np.array(sim_times), success=np.array(successes))
            record.update(executed_steps=len(actions), physics_advanced_s=float(env.sim.data.time)-started,
                          expected_control_dt=expected_dt, success_any=any(successes),
                          success_final=bool(env.check_success()), finite_states=True,
                          tasktol_controls_enabled=False)
            report['task_success'] = record['success_final']
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report['close_error'] = repr(exc)
        save()
    print(json.dumps({k:report[k] for k in ['status','packages','output']}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
