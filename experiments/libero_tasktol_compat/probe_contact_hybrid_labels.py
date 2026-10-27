"""Hybrid relabel: achieved states in free space, source bias near contact.

All output targets are FK of bounded joint witnesses. Original motion commands
are used only in geometry-detected contact windows, never as executed actions.
The online client remains strict IPOPT/B10 with seven-joint references.
"""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, SELECTION, TASKTOL, sha
from finalize_inverse_probe import finalize

import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares

from experiments.libero_tasktol_compat.osc_reference import difference_reference_goals
from tasktol.geometry import rotation_matrix, rotation_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(task['source_hdf5']) as file:
        config = json.loads(file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
    assert np.array_equal(config['output_max'], [.05, .05, .05, .5, .5, .5])
    original_reference = runner.state_reference

    def hybrid_reference(states, actions, handles, kin, lookahead, time_scale=1):
        reference = original_reference(states, actions, handles, kin, lookahead, time_scale)
        model, data = kin.model, mujoco.MjData(kin.model)
        names = [model.geom(index).name or '' for index in range(model.ngeom)]
        contact = np.zeros(len(states), dtype=bool)
        for index, state in enumerate(states):
            data.qpos[:] = state[1:1 + model.nq]
            data.qvel[:] = state[1 + model.nq:]
            mujoco.mj_forward(model, data)
            for collision in data.contact:
                first, second = names[collision.geom1], names[collision.geom2]
                finger_external = (first.startswith('gripper0') and not second.startswith(('robot0', 'gripper0'))) or (second.startswith('gripper0') and not first.startswith(('robot0', 'gripper0')))
                if finger_external and collision.dist <= .001:
                    contact[index] = True
                    break
        alpha = contact.astype(float)
        for offset in range(1, 4):
            weight = 1 - offset / 4
            alpha[offset:] = np.maximum(alpha[offset:], weight * contact[:-offset])
            alpha[:-offset] = np.maximum(alpha[:-offset], weight * contact[offset:])
        coordinate = reference['source_state_coordinate'][:-1]
        blend = np.interp(coordinate, np.arange(len(states)), alpha)
        lower, upper = handles.joint_lower + 1e-4, handles.joint_upper - 1e-4
        positions, rotations, witnesses, diagnostics = [], [], [], []
        for index, weight in enumerate(blend):
            q = reference['q'][index]
            next_q = reference['q'][index + 1]
            current_p, current_r = reference['positions'][index], reference['rotations'][index]
            next_p, next_r = reference['positions'][index + 1], reference['rotations'][index + 1]
            action = actions[reference['source_action_indices'][index]]
            source_goal_p = current_p + .05 * action[:3]
            source_goal_r = rotation_matrix(.5 * action[3:6]) @ current_r
            desired_p = (1 - weight) * next_p + weight * source_goal_p
            desired_r = rotation_matrix(weight * rotation_vector(source_goal_r @ next_r.T)) @ next_r
            if weight == 0:
                witness = np.clip(next_q, lower, upper)
                status, nfev = True, 0
            else:
                def residual(candidate):
                    pose = kin.evaluate(candidate, include_manipulability=False, include_link_points=False)
                    return np.r_[pose.position - desired_p, .1 * rotation_vector(pose.rotation @ desired_r.T), 1e-5 * (candidate - q)]
                result = least_squares(residual, np.clip(q, lower + 1e-8, upper - 1e-8), bounds=(lower, upper),
                                       max_nfev=40, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                witness, status, nfev = result.x, bool(result.success), int(result.nfev)
            pose = kin.evaluate(witness, include_manipulability=False, include_link_points=False)
            witnesses.append(witness)
            positions.append(pose.position)
            rotations.append(pose.rotation)
            diagnostics.append(dict(index=index, source_coordinate=float(coordinate[index]), contact_blend=float(weight),
                                    projection_converged=status, nfev=nfev,
                                    position_adjustment_m=float(np.linalg.norm(pose.position - desired_p)),
                                    rotation_adjustment_rad=float(np.linalg.norm(rotation_vector(pose.rotation @ desired_r.T)))))
        initial_p, initial_r = reference['positions'][0], reference['rotations'][0]
        reference.update(source_positions_m=reference['positions'].copy(), source_rotations=reference['rotations'].copy(),
                         goal_joint_witness_rad=np.asarray(witnesses), contact_blend=blend, original_contact_flags=contact,
                         actions=difference_reference_goals(positions, rotations, reference['actions'][:, 6], initial_p, initial_r),
                         positions=np.concatenate([initial_p[None], np.asarray(positions)]),
                         rotations=np.concatenate([initial_r[None], np.asarray(rotations)]),
                         joint_outside_rad=np.zeros(len(witnesses) + 1))
        (output / 'contact_label_projection.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        return reference

    runner.state_reference = hybrid_reference
    sources = [Path(__file__), Path(runner.__file__), Path(__file__).with_name('finalize_inverse_probe.py'),
               ROOT / 'examples/libero/regenerate_libero_dataset.py', TASKTOL / 'tasktol/simulation/libero.py',
               TASKTOL / 'tasktol/simulation/libero_joint_controller.py']
    (output / 'source_snapshot').mkdir()
    checks, snapshots = {}, {}
    for index, path in enumerate(sources):
        path = path.resolve()
        checks[str(path)] = sha(path)
        target = output / 'source_snapshot' / f'{index}_{path.name}'
        shutil.copyfile(path, target)
        snapshots[str(path)] = str(target.relative_to(output))
    job = dict(case=args.case, demo=args.demo, suite=task['suite'], task_id=task['task_id'])
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=2, lookahead_steps=1,
                      gripper_policy='native_integrator_pulse', pulse_phase=0, task_kp=150,
                      label_method='state_free_source_bias_contact_bounded_FK', original_motion_actions_used='only detected contact windows, padded 3 source frames',
                      source_sha256=checks)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    runner.worker(job, output, 1, ['ipopt_state_B10'], 2, 'native_integrator_pulse', 0)
    finalize(output)


if __name__ == '__main__':
    main()
