"""Time-aligned physical-state metrics; never compare quaternion coordinates."""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np


# Fixed physical scales/weights used for every task and every offline trial.
# Pose errors use MuJoCo tangent coordinates. No time warping or state resets.
GROUPS = {
    'arm_position_rad': (.1, 1.), 'arm_velocity_rad_s': (1., .2),
    'finger_position_m': (.005, .1), 'finger_velocity_m_s': (.05, .05),
    'object_translation_m': (.01, 2.), 'object_rotation_rad': (.1, .5),
    'object_linear_velocity_m_s': (.1, .2), 'object_angular_velocity_rad_s': (1., .1),
}


def state_groups(model):
    groups = {name: [] for name in GROUPS}
    objects = []
    for j in range(model.njnt):
        name = model.joint(j).name or f'joint_{j}'
        start, kind = int(model.jnt_dofadr[j]), int(model.jnt_type[j])
        size = 6 if kind == mujoco.mjtJoint.mjJNT_FREE else 3 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
        ids = list(range(start, start + size))
        if name.startswith('robot0_joint'):
            groups['arm_position_rad'] += ids
            groups['arm_velocity_rad_s'] += ids
        elif name.startswith(('gripper0', 'robot0_gripper')):
            groups['finger_position_m'] += ids
            groups['finger_velocity_m_s'] += ids
        else:
            translation = ids[:3] if kind == mujoco.mjtJoint.mjJNT_FREE else ids if kind == mujoco.mjtJoint.mjJNT_SLIDE else []
            rotation = ids[3:] if kind == mujoco.mjtJoint.mjJNT_FREE else ids if kind != mujoco.mjtJoint.mjJNT_SLIDE else []
            groups['object_translation_m'] += translation
            groups['object_linear_velocity_m_s'] += translation
            groups['object_rotation_rad'] += rotation
            groups['object_angular_velocity_rad_s'] += rotation
            objects.append(dict(name=name, translation_dofs=translation, rotation_dofs=rotation))
    assert len(groups['arm_position_rad']) == 7
    return groups, objects


def compare_states(model, actual, source):
    actual, source = np.asarray(actual), np.asarray(source)
    assert actual.shape == source.shape and actual.shape[1] == 1 + model.nq + model.nv
    np.testing.assert_allclose(actual[:, 0], source[:, 0], atol=1e-9, rtol=0)
    pose = np.empty((len(actual), model.nv))
    for i, (a, b) in enumerate(zip(actual, source, strict=True)):
        mujoco.mj_differentiatePos(model, pose[i], 1., b[1:1 + model.nq], a[1:1 + model.nq])
    velocity = actual[:, 1 + model.nq:] - source[:, 1 + model.nq:]
    groups, objects = state_groups(model)
    terms, trajectory_cost = {}, np.zeros(len(actual))
    for name, ids in groups.items():
        scale, weight = GROUPS[name]
        data = velocity if 'velocity' in name else pose
        if not ids:
            continue
        error = data[:, ids]
        cost = np.mean((error / scale) ** 2, axis=1) * weight
        trajectory_cost += cost
        terms[name] = dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                           p95_abs=float(np.percentile(np.abs(error), 95)), max_abs=float(np.max(np.abs(error))),
                           terminal_rmse=float(np.sqrt(np.mean(error[-1] ** 2))),
                           normalized_weighted_cost=float(np.mean(cost)), scale=scale, weight=weight, dimensions=len(ids))
    object_metrics = []
    for item in objects:
        detail = dict(name=item['name'])
        for key in ('translation', 'rotation'):
            ids = item[f'{key}_dofs']
            if ids:
                norm = np.linalg.norm(pose[:, ids], axis=1)
                detail[key] = dict(mean=float(norm.mean()), p95=float(np.percentile(norm, 95)),
                                   max=float(norm.max()), terminal=float(norm[-1]))
        object_metrics.append(detail)
    # Include a modest terminal penalty without replacing trajectory tracking.
    objective = float(trajectory_cost.mean() + .25 * trajectory_cost[-1])
    return dict(objective=objective, trajectory_mean_cost=float(trajectory_cost.mean()),
                terminal_cost=float(trajectory_cost[-1]), frames=len(actual), groups=terms,
                objects=object_metrics, time_aligned=True, time_warping=False), pose, velocity


def evaluate_run(root, write=True):
    root = Path(root)
    provenance = json.loads((root / 'provenance.json').read_text())
    assert provenance.get('time_scale', 1) == 1 and len(provenance['plan']) == 1
    job = provenance['plan'][0]
    dest = root / job['suite'] / f"{job['task_id']:02d}" / job['demo']
    folder = dest / 'ipopt_state_B10'
    report = json.loads((folder / 'report.json').read_text())
    with h5py.File(dest / 'state_reference.hdf5') as f:
        source_path = f.attrs['source']
    with h5py.File(source_path) as f:
        source = f['data'][job['demo']]['states'][()]
        grip = f['data'][job['demo']]['actions'][:, 6]
    with h5py.File(folder / 'recording.hdf5') as f:
        g = f['attempts'][job['demo']]
        before, final = g['tasktol/pre_action_states'][()], g['tasktol/final_state'][()]
        np.testing.assert_array_equal(g['actions'][:, 6], grip[:len(before)])
        model = mujoco.MjModel.from_xml_string(g.attrs['model_file'])
    executed = report['executed_frames']
    actual = np.concatenate([before[:executed], final[None]])
    result, pose, velocity = compare_states(model, actual, source[:len(actual)])
    result.update(**job, success=report['success'], rejected=report['execution_rejected'],
                  full_length=executed == len(source) - 1 and not report['execution_rejected'],
                  gripper_unchanged=True, source_hdf5=str(source_path))
    if not result['full_length']:
        result['objective'] = 1e12 + result['objective']
    if write:
        (root / 'physical_state_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(root / 'physical_state_errors.npz', position_tangent=pose, velocity=velocity)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(evaluate_run(args.root), indent=2))
