"""Offline, controller-aware additions to achieved-state reference labels."""

import mujoco
import numpy as np
from scipy.optimize import least_squares

from experiments.libero_tasktol_compat.osc_reference import difference_reference_goals
from tasktol.geometry import rotation_matrix, rotation_vector


def add_contact_bias(reference, states, actions, handles, kin):
    """Use source motion intent near contact; project all targets into joint bounds.

    The common achieved-state reference already encodes time scaling. Source
    contact detection never advances physics. Returned joint witnesses certify
    offline geometry only; the online client receives pose increments alone.
    """
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
    positions, rotations, witnesses, residuals, evaluations = [], [], [], [], []
    for index, weight in enumerate(blend):
        q, next_q = reference['q'][index:index + 2]
        current_p, current_r = reference['positions'][index], reference['rotations'][index]
        next_p, next_r = reference['positions'][index + 1], reference['rotations'][index + 1]
        source_action = np.clip(actions[reference['source_action_indices'][index], :6], -1, 1)
        source_goal_p = current_p + .05 * source_action[:3]
        source_goal_r = rotation_matrix(.5 * source_action[3:6]) @ current_r
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
        assert np.all(witness >= lower) and np.all(witness <= upper)
        pose = kin.evaluate(witness, include_manipulability=False, include_link_points=False)
        witnesses.append(witness)
        positions.append(pose.position)
        rotations.append(pose.rotation)
        residuals.append([float(np.linalg.norm(pose.position - desired_p)), float(np.linalg.norm(rotation_vector(pose.rotation @ desired_r.T)))])
        evaluations.append([int(status), nfev])
    initial_p, initial_r = reference['positions'][0], reference['rotations'][0]
    reference.update(source_positions_m=reference['positions'].copy(), source_rotations=reference['rotations'].copy(),
                     goal_joint_witness_rad=np.asarray(witnesses), contact_blend=blend, original_contact_flags=contact,
                     projection_pose_residual=np.asarray(residuals), projection_converged_nfev=np.asarray(evaluations),
                     actions=difference_reference_goals(positions, rotations, reference['actions'][:, 6], initial_p, initial_r),
                     positions=np.concatenate([initial_p[None], np.asarray(positions)]),
                     rotations=np.concatenate([initial_r[None], np.asarray(rotations)]),
                     joint_outside_rad=np.zeros(len(witnesses) + 1))
    return reference
