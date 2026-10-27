"""Original-speed, unchanged-gripper preview / PD feedforward label probes.

Only offline labels change. The maintained recorder, strict IPOPT session,
seven-joint references, native physics and original 50ms time grid are reused.
"""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, TASKTOL, SELECTION, plain, sha

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from experiments.libero_tasktol_compat.osc_reference import difference_reference_goals, integrate_reference_actions
from tasktol.geometry import rotation_matrix, rotation_vector


def make_reference(base, actions, handles, kin, method, preview_steps, lead_s, posture_weight=1e-5):
    positions, rotations, joints = base['positions'], base['rotations'], base['q']
    count = len(positions) - 1
    dt = .05
    if method == 'preview':
        indices = np.minimum(np.arange(count) + preview_steps, count)
        desired_p, desired_r = positions[indices], rotations[indices]
    else:
        # For xdd = kp(r-x)-kd*xd, the ideal continuous-time reference is
        # r=x_des+(kd/kp)*xd_des+xdd_des/kp, not merely x_des(t+tau).
        window = min(7, len(positions) if len(positions) % 2 else len(positions) - 1)
        velocity = savgol_filter(positions, window, 3, deriv=1, delta=dt, axis=0)
        acceleration = savgol_filter(positions, window, 3, deriv=2, delta=dt, axis=0)
        omega = np.asarray([rotation_vector(b @ a.T) / dt for a, b in zip(rotations[:-1], rotations[1:])])
        omega = np.vstack([omega, omega[-1]])
        omega = savgol_filter(omega, window, 3, axis=0)
        angular_acceleration = np.gradient(omega, dt, axis=0)
        desired_p = positions[:-1] + lead_s * velocity[:-1] + acceleration[:-1] / 150
        desired_r = np.asarray([rotation_matrix(lead_s * w + a / 150) @ r
                                for r, w, a in zip(rotations[:-1], omega[:-1], angular_acceleration[:-1])])
    lower, upper = handles.joint_lower + 1e-4, handles.joint_upper - 1e-4
    goals_p, goals_r, witnesses, residuals = [], [], [], []
    for index, (position, rotation) in enumerate(zip(desired_p, desired_r, strict=True)):
        seed_index = min(index + preview_steps, count) if method == 'preview' else index
        seed = np.clip(joints[seed_index], lower + 1e-8, upper - 1e-8)

        def residual(q):
            pose = kin.evaluate(q, include_manipulability=False, include_link_points=False)
            return np.r_[pose.position - position, .1 * rotation_vector(pose.rotation @ rotation.T),
                         posture_weight * (q - seed)]

        solved = least_squares(residual, seed, bounds=(lower, upper), max_nfev=40,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        pose = kin.evaluate(solved.x, include_manipulability=False, include_link_points=False)
        goals_p.append(pose.position)
        goals_r.append(pose.rotation)
        witnesses.append(solved.x)
        residuals.append([np.linalg.norm(pose.position - position),
                          np.linalg.norm(rotation_vector(pose.rotation @ rotation.T))])
    increments = difference_reference_goals(goals_p, goals_r, actions[:count, 6], positions[0], rotations[0])
    rebuilt_p, rebuilt_r = integrate_reference_actions(increments, positions[0], rotations[0])
    np.testing.assert_allclose(rebuilt_p, goals_p, atol=1e-10, rtol=0)
    np.testing.assert_allclose(rebuilt_r, goals_r, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(increments[:, 6], actions[:count, 6])
    assert len(increments) == count and np.all(np.asarray(witnesses) >= lower) and np.all(np.asarray(witnesses) <= upper)
    base.update(source_positions_m=positions.copy(), source_rotations=rotations.copy(),
                positions=np.concatenate([positions[:1], goals_p]),
                rotations=np.concatenate([rotations[:1], goals_r]),
                target_indices=np.arange(1, count + 1), actions=increments,
                goal_joint_witness_rad=np.asarray(witnesses), projection_pose_residual=np.asarray(residuals),
                desired_goal_positions_m=desired_p, desired_goal_rotations=desired_r,
                joint_outside_rad=np.zeros(count + 1))
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--method', choices=['preview', 'pd'], required=True)
    parser.add_argument('--preview-steps', type=int, default=4)
    parser.add_argument('--lead-s', type=float, default=2 / np.sqrt(150))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.preview_steps >= 1 and np.isfinite(args.lead_s) and args.lead_s >= 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    job = dict(case=args.case, demo=args.demo, suite=task['suite'], task_id=task['task_id'])
    original = runner.state_reference

    def reference_factory(states, actions, handles, kin, lookahead, time_scale=1):
        assert time_scale == lookahead == 1
        return make_reference(original(states, actions, handles, kin, 1, 1), actions, handles, kin,
                              args.method, args.preview_steps, args.lead_s)

    runner.state_reference = reference_factory
    sources = [Path(__file__), Path(runner.__file__), ROOT / 'examples/libero/regenerate_libero_dataset.py',
               TASKTOL / 'tasktol/simulation/libero.py', TASKTOL / 'tasktol/simulation/libero_joint_controller.py']
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshots / f'{index}_{path.name}')
    (output / 'source_snapshots.json').write_text(json.dumps(
        {str(p): f'source_snapshot/{i}_{p.name}' for i, p in enumerate(sources)}, indent=2) + '\n')
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=1, lookahead_steps=1,
                      label_method=f'original_speed_{args.method}_bounded_FK', preview_steps=args.preview_steps,
                      lead_s=args.lead_s, gripper_policy='preserve', task_kp=150, null_kp=10,
                      original_motion_actions_used=False, offline_future_only=True,
                      source_sha256={str(p): sha(p) for p in sources})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    result = runner.worker(job, output, 1, ['ipopt_state_B10'], 1, 'preserve')
    folder = output / job['suite'] / f"{job['task_id']:02d}" / job['demo'] / 'ipopt_state_B10'
    with runner.h5py.File(folder.parent / 'state_reference.hdf5', 'r+') as file:
        file.attrs.update(label_method=provenance['label_method'], preview_steps=args.preview_steps,
                          lead_s=args.lead_s, source_state_motion='offline state trajectory with bounded projected preview or PD feedforward')
    report = result['results'][0]
    report.update(label_method=provenance['label_method'], preview_steps=args.preview_steps, lead_s=args.lead_s,
                  source_state_motion='offline state trajectory with bounded projected preview or PD feedforward')
    rows = json.loads((folder / 'trajectory.json').read_text())
    executed = [r for r in rows if not r['aborted']]
    for field, scale, suffix in [('source_next_position_error_m', 1000, 'position_mm'),
                                  ('source_next_rotation_error_rad', 180 / np.pi, 'rotation_deg')]:
        values = [r[field] * scale for r in executed]
        report[f'source_{suffix}'] = dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                                         max=float(np.max(values))) if values else None
    (folder / 'report.json').write_text(json.dumps(plain(report), indent=2) + '\n')
    (folder.parent / 'summary.json').write_text(json.dumps(plain(result), indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(plain(dict(complete=True, results=[result])), indent=2) + '\n')
    print(json.dumps(plain(report)), flush=True)


if __name__ == '__main__':
    main()
