"""Offline FK / IPOPT audit of saved failures; no rollout or physics steps.

Rebuild the native task model, verify FK against recorded physical poses, and
restore the three accepted-reference history vectors before isolated re-solves.
Saved failure joint_target is a held accepted command, NOT the rejected iterate.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'franka_mujoco'), str(ROOT / 'third_party/libero')]
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/tasktol_failure_geometry_numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tasktol_failure_geometry_mpl')

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from tasktol.control.ipopt import IpoptIK, IpoptSettings
from tasktol.control.ipopt.adapter import IpoptAdapter
from tasktol.control.pose import TargetPose
from tasktol.control.task_space.objective import ObjectiveSettings
from tasktol.geometry import rotation_vector
from tasktol.simulation.libero import libero_handles

BASE = ROOT / 'experiments/libero_tasktol_compat/results'
SOURCE = BASE / 'ipopt_factorized_B10_all40_v1/kp_150'
OUTPUT = BASE / 'failed_ik_geometry_audit_v1'


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def audit(rel):
    source = SOURCE / rel
    dest = OUTPUT / rel
    dest.mkdir(parents=True, exist_ok=False)
    report = json.loads((source / 'report.json').read_text())
    rows = json.loads((source / 'trajectory.json').read_text())['frames']
    failed = rows[-1]
    assert failed['aborted'] and failed['physics_steps'] == 0
    assert np.array_equal(failed['joint_target_rad'], rows[-2]['joint_target_rad'])
    # final.png is the terminal physical state: failure does not advance physics.
    shutil.copyfile(source / 'final.png', dest / 'failure_actual.png')
    env = ControlEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files')) / report['task']['suite'] / (report['task']['name'] + '.bddl')),
        controller='OSC_POSE', use_camera_obs=False, has_offscreen_renderer=False, control_freq=20,
    )
    try:
        env.seed(0)
        env.reset()
        handles = libero_handles(env)
        model = env.sim.model._model
        ik = IpoptIK(model, handles, IpoptSettings(max_iterations=60, max_cpu_time_s=0.5),
                     ObjectiveSettings.delta_q_squared_conditioning_safeguard())
        target = TargetPose(np.array(failed['nominal_target']['position_m']),
                            Rotation.from_rotvec(failed['nominal_target']['rotation_vector_rad']).as_matrix())

        def geometry(q):
            q = np.asarray(q)
            fk = ik.kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
            jac = fk.jacobian
            sv = np.linalg.svd(jac, compute_uv=False)
            scaled = jac.copy()
            scaled[:3] /= 0.3  # explicit characteristic length, not an absolute singularity threshold
            sv_scaled = np.linalg.svd(scaled, compute_uv=False)
            margin = np.minimum(q - handles.joint_lower, handles.joint_upper - q)
            err = np.r_[fk.position - target.position, rotation_vector(target.rotation.T @ fk.rotation)]
            return dict(q_rad=q, limit_margin_deg=np.rad2deg(margin),
                        closest_joint=int(np.argmin(margin)) + 1, min_limit_margin_deg=np.rad2deg(margin.min()),
                        singular_values_geometric=sv, sigma_min_geometric=sv[-1], condition_geometric=sv[0] / sv[-1],
                        singular_values_length_scaled=sv_scaled, condition_length_scaled=sv_scaled[0] / sv_scaled[-1],
                        position_error_xyz_m=err[:3], position_error_max_m=np.max(np.abs(err[:3])),
                        rotation_error_xyz_rad=err[3:], rotation_error_max_rad=np.max(np.abs(err[3:])),
                        position_error_norm_mm=np.linalg.norm(err[:3]) * 1000,
                        rotation_error_norm_deg=np.rad2deg(np.linalg.norm(err[3:])))

        actual = np.array(failed['pre_action']['joint_position_rad'])
        fk = ik.kinematics.evaluate(actual, include_manipulability=False, include_link_points=False)
        recorded_r = Rotation.from_rotvec(failed['pre_action']['rotation_vector_rad']).as_matrix()
        checks = dict(actual_fk_position_error_m=float(np.linalg.norm(fk.position - failed['pre_action']['position_m'])),
                      actual_fk_rotation_error_rad=float(np.linalg.norm(rotation_vector(fk.rotation.T @ recorded_r))),
                      rotation_offset_max_difference=float(np.max(np.abs(handles.ee_rotation_offset - report['ee_rotation_offset']))))
        assert checks['actual_fk_position_error_m'] < 1e-10 and checks['actual_fk_rotation_error_rad'] < 1e-10
        assert checks['rotation_offset_max_difference'] < 1e-10
        data = mujoco.MjData(model)
        data.qpos[handles.qpos_ids] = actual

        def restored(settings):
            trial = IpoptIK(model, handles, settings, ObjectiveSettings.delta_q_squared_conditioning_safeguard())
            trial._xopt = np.array(rows[-2]['joint_target_rad'])
            trial._previous_state = np.array(rows[-3]['joint_target_rad'])
            trial._previous_state2 = np.array(rows[-4]['joint_target_rad'])
            return trial

        solves = []
        for label, settings, seed in [
            ('original_budget_previous_reference_seed', IpoptSettings(max_iterations=60, max_cpu_time_s=0.5), None),
            ('expanded_budget_same_seed', IpoptSettings(max_iterations=300, max_cpu_time_s=5.0), None),
            ('expanded_budget_actual_state_seed', IpoptSettings(max_iterations=300, max_cpu_time_s=5.0), actual),
        ]:
            trial = restored(settings)
            if seed is not None:
                trial.set_warm_start(seed)
            raw = []
            native_solve = IpoptAdapter.solve

            def observed(adapter, *args, **kwargs):
                result = native_solve(adapter, *args, **kwargs)
                raw.append(result)
                return result

            IpoptAdapter.solve = observed
            try:
                q, diag = trial.solve(data, target, np.ones(6, dtype=bool))
            finally:
                IpoptAdapter.solve = native_solve
            result = raw[0]
            solves.append(dict(label=label, feasible=bool(diag.feasible), status=diag.status,
                               iterations=diag.iterations, elapsed_ms=diag.elapsed_ms,
                               raw_status_code=result.status_code, raw_candidate=geometry(result.q),
                               published_command=geometry(q)))
        prefix = [dict(frame=row['index'], actual=geometry(row['pre_action']['joint_position_rad']),
                       accepted_reference=geometry(row['joint_target_rad'])) for row in rows[:-1]]
        # Errors in prefix geometry are relative to the FAILED target, only
        # conditioning/limit quantities are compared across these historical rows.
        output = dict(task=report['task'], failed_frame=failed['index'], source_index=failed['source_index'],
                      time_s=failed['time_s'], original_solver=failed['optimizer'],
                      screenshot=str(dest / 'failure_actual.png'),
                      screenshot_sha256=hashlib.sha256((dest / 'failure_actual.png').read_bytes()).hexdigest(),
                      model_fk_checks=checks, joint_lower_rad=handles.joint_lower, joint_upper_rad=handles.joint_upper,
                      actual=geometry(actual), previous_accepted_reference=geometry(failed['joint_target_rad']),
                      target=failed['nominal_target'], isolated_resolves=solves, prefix_geometry=prefix,
                      methodology=dict(physics_steps=0, goal_and_bounds_unchanged=True,
                                       saved_failure_target_is_previous_accepted_reference=True,
                                       original_raw_iterate_was_not_saved=True,
                                       raw_candidates_are_isolated_resolves_not_historical_iterates=True,
                                       scaled_jacobian='v/0.3m and angular velocity; all seven joint columns',
                                       prefix_pose_errors='relative to failed target, not historical frame target'))
        (dest / 'audit.json').write_text(json.dumps(plain(output), indent=2) + '\n')
        brief = {k: output[k] for k in ['task', 'failed_frame', 'screenshot', 'actual', 'previous_accepted_reference', 'isolated_resolves']}
        print(json.dumps(plain(brief)), flush=True)
        return output
    finally:
        env.close()


def main():
    OUTPUT.mkdir(exist_ok=False)
    results = [audit(rel) for rel in ['libero_spatial/04', 'libero_goal/02', 'libero_10/08']]
    (OUTPUT / 'summary.json').write_text(json.dumps(plain(results), indent=2) + '\n')


if __name__ == '__main__':
    main()
