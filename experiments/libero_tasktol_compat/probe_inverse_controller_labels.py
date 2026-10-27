"""Experimental state-only inverse-dynamics / inverse-B10 action labels.

Full source qpos/qvel and finite-difference accelerations determine inverse
dynamics torques. A bounded joint witness is fitted through the unchanged B10
torque map. Only its FK pose increments are sent to the normal online IPOPT
session. This is an offline label experiment, not a replacement physics loop.
"""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, SELECTION, TASKTOL, plain, sha

import h5py
import mujoco
import numpy as np
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.binding_utils import MjSim
from scipy.optimize import least_squares

from experiments.libero_tasktol_compat.osc_reference import difference_reference_goals
from tasktol.geometry import rotation_vector
from tasktol.simulation.libero_joint_controller import FactorizedReferenceController


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--loss', choices=['torque', 'task_acceleration'], default='torque')
    args = parser.parse_args()
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(task['source_hdf5'], 'r') as file:
        config = json.loads(file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
    original_reference = runner.state_reference

    def inverse_reference(states, actions, handles, kin, lookahead, time_scale=1):
        assert time_scale == lookahead == 1
        reference = original_reference(states, actions, handles, kin, lookahead, time_scale)
        model = kin.model
        sim = MjSim(model)
        sim.data.qpos[:] = states[0, 1:1 + model.nq]
        sim.data.qvel[:] = states[0, 1 + model.nq:]
        sim.forward()
        native = OperationalSpaceController(sim=sim, eef_name=handles.spec.ee_site,
            joint_indexes=dict(joints=handles.joint_ids, qpos=handles.qpos_ids, qvel=handles.dof_ids),
            actuator_range=(model.actuator_ctrlrange[handles.actuator_ids, 0], model.actuator_ctrlrange[handles.actuator_ids, 1]), **config)
        ctrl = FactorizedReferenceController.from_native(native, task_feedback='osc', null_reference='joint', null_kp=10)
        ctrl.use_delta = False
        velocity = states[:, 1 + model.nq:]
        acceleration = np.gradient(velocity, .05, axis=0)
        qgoals, positions, rotations, diagnostics = [], [], [], []
        lower = handles.joint_lower + 1e-4
        upper = handles.joint_upper - 1e-4
        torque_scale = np.maximum(1, np.max(np.abs(model.actuator_ctrlrange[handles.actuator_ids]), axis=1))
        for index in range(len(actions) - 1):
            state = states[index]
            sim.data.qpos[:] = state[1:1 + model.nq]
            sim.data.qvel[:] = velocity[index]
            sim.forward()
            sim.data.qacc[:] = acceleration[index]
            mujoco.mj_inverse(model, sim.data._data)
            required = sim.data.qfrc_inverse[handles.dof_ids].copy()
            assert np.all(np.isfinite(required))
            ctrl.new_update = True
            ctrl.update(force=True)
            projection = ctrl.J_full @ np.linalg.inv(ctrl.mass_matrix)
            acceleration_scale = np.maximum(1, np.abs(projection) @ torque_scale)
            fit_dimensions = 7 if args.loss == 'torque' else 6
            actual = state[1 + handles.qpos_ids]
            lo, hi = np.maximum(lower, actual - .5), np.minimum(upper, actual + .5)
            initial = np.clip(actual + (2 / np.sqrt(150)) * velocity[index, handles.dof_ids]
                              + acceleration[index, handles.dof_ids] / 150, lo + 1e-8, hi - 1e-8)

            def residual(q):
                fk = kin.evaluate(q, include_manipulability=False, include_link_points=False)
                ctrl.set_joint_reference(q)
                ctrl.new_update = True
                ctrl.set_goal(np.r_[fk.position, rotation_vector(fk.rotation @ handles.ee_rotation_offset.T)])
                torque = np.clip(ctrl.run_controller(), ctrl.actuator_min, ctrl.actuator_max)
                difference = torque - required
                fit_error = difference / torque_scale if args.loss == 'torque' else (projection @ difference) / acceleration_scale
                return np.r_[fit_error, .01 * (q - actual)]

            initial_residual = residual(initial)
            solved = least_squares(residual, initial, bounds=(lo, hi), max_nfev=25,
                                   ftol=1e-6, xtol=1e-6, gtol=1e-6)
            assert np.all(solved.x >= lower) and np.all(solved.x <= upper)
            fk = kin.evaluate(solved.x, include_manipulability=False, include_link_points=False)
            qgoals.append(solved.x)
            positions.append(fk.position)
            rotations.append(fk.rotation)
            diagnostics.append(dict(index=index, optimizer_success=bool(solved.success), nfev=int(solved.nfev),
                                    required_torque_nm=required, loss=args.loss,
                                    normalized_fit_residual_before=float(np.linalg.norm(initial_residual[:fit_dimensions])),
                                    normalized_fit_residual_after=float(np.linalg.norm(residual(solved.x)[:fit_dimensions]))))
        qgoals = np.asarray(qgoals)
        initial_p, initial_r = reference['positions'][0], reference['rotations'][0]
        reference.update(source_positions_m=reference['positions'].copy(), source_rotations=reference['rotations'].copy(),
                         goal_joint_witness_rad=qgoals,
                         actions=difference_reference_goals(positions, rotations, actions[:-1, 6], initial_p, initial_r),
                         positions=np.concatenate([initial_p[None], np.asarray(positions)]),
                         rotations=np.concatenate([initial_r[None], np.asarray(rotations)]),
                         source_state_coordinate=np.r_[0, np.arange(len(qgoals))],
                         joint_outside_rad=np.r_[0., np.maximum(0, np.max(np.maximum(lower - qgoals, qgoals - upper), axis=1))])
        (output / 'inverse_label_fit.json').write_text(json.dumps(plain(diagnostics), indent=2) + '\n')
        return reference

    runner.state_reference = inverse_reference
    sources = [Path(__file__), Path(runner.__file__), ROOT / 'examples/libero/regenerate_libero_dataset.py',
               TASKTOL / 'tasktol/simulation/libero_joint_controller.py', TASKTOL / 'tasktol/simulation/libero.py']
    (output / 'source_snapshot').mkdir()
    checks, snapshots = {}, {}
    for index, path in enumerate(sources):
        path = path.resolve()
        checks[str(path)] = sha(path)
        dest = output / 'source_snapshot' / f'{index}_{path.name}'
        shutil.copyfile(path, dest)
        snapshots[str(path)] = str(dest.relative_to(output))
    job = dict(case=args.case, demo=args.demo, suite=task['suite'], task_id=task['task_id'])
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=1, lookahead_steps=1,
                      label_method='state_inverse_dynamics_inverse_B10', original_motion_actions_used=False,
                      inverse_controller_loss=args.loss,
                      acceleration='central finite difference of full source qvel at 20Hz', source_sha256=checks)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    result = runner.worker(job, output, 1, ['ipopt_state_B10'], 1)
    dest = output / task['suite'] / f"{task['task_id']:02d}" / args.demo
    with h5py.File(dest / 'state_reference.hdf5', 'r+') as file:
        file.attrs['label_method'] = provenance['label_method']
        source_positions, source_rotations = file['source_positions_m'][()], file['source_rotations'][()]
    folder = dest / 'ipopt_state_B10'
    rows = json.loads((folder / 'trajectory.json').read_text())
    for row in rows:
        if row['aborted']:
            continue
        index = row['index']
        row['source_next_position_error_m'] = float(np.linalg.norm(np.asarray(row['actual_position_m']) - source_positions[index + 1]))
        row['source_next_rotation_error_rad'] = float(np.linalg.norm(rotation_vector(np.asarray(row['actual_rotation']) @ source_rotations[index + 1].T)))
    report = result['results'][0]
    report['label_method'] = provenance['label_method']
    errors = [row['source_next_position_error_m'] for row in rows if not row['aborted']]
    report['mean_source_next_position_error_mm'] = float(np.mean(errors) * 1000) if errors else None
    report['max_source_next_position_error_mm'] = float(np.max(errors) * 1000) if errors else None
    (folder / 'trajectory.json').write_text(json.dumps(rows, indent=2) + '\n')
    (folder / 'report.json').write_text(json.dumps(plain(report), indent=2) + '\n')
    (dest / 'summary.json').write_text(json.dumps(plain(result), indent=2) + '\n')
    aggregate = dict(ipopt_state_B10=dict(final=int(report['success']), ever=int(report['ever_success']), rejected=int(report['execution_rejected'])))
    (output / 'summary.json').write_text(json.dumps(plain(dict(complete=True, aggregate=aggregate, results=[result])), indent=2) + '\n')
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
