"""Task-only reconstruction costs: no robot/finger joint tracking penalties.

Default IPOPT's own objective is entirely independent of this offline score.
"""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.libero_tasktol_compat.closed_loop_state_metrics import compare_states, state_groups, GROUPS


def task_cost(model, actual_states, source_states, actual_p, actual_r, source_p, source_r):
    diagnostic, pose, velocity = compare_states(model, actual_states, source_states)
    groups, _ = state_groups(model)
    trajectory = np.zeros(len(actual_states))
    terms = {}
    for name, ids in groups.items():
        if not name.startswith('object_') or not ids:
            continue
        scale, weight = GROUPS[name]
        error = (velocity if 'velocity' in name else pose)[:, ids]
        cost = weight * np.mean((error / scale) ** 2, axis=1)
        trajectory += cost
        terms[name] = dict(mean_cost=float(cost.mean()), scale=scale, weight=weight)
    position = np.asarray(actual_p) - source_p
    rotation = Rotation.from_matrix(np.asarray(actual_r) @ np.swapaxes(source_r, 1, 2)).as_rotvec()
    for name, error, scale, weight in [('ee_position_m', position, .01, 1.), ('ee_rotation_rad', rotation, .1, .5)]:
        cost = weight * np.mean((error / scale) ** 2, axis=1)
        trajectory += cost
        terms[name] = dict(mean_cost=float(cost.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                           mean_norm=float(np.mean(np.linalg.norm(error,axis=1))), scale=scale,weight=weight)
    return dict(objective=float(trajectory.mean()+.25*trajectory[-1]),
                mean_cost=float(trajectory.mean()),terminal_cost=float(trajectory[-1]),
                terms=terms,frames=len(actual_states),robot_joint_tracking_penalty=False,
                finger_joint_tracking_penalty=False,objects=diagnostic['objects'])


def evaluate_task_run(root, write=True):
    root=Path(root)
    provenance=json.loads((root/'provenance.json').read_text());job=provenance['plan'][0]
    dest=root/job['suite']/f"{job['task_id']:02d}"/job['demo'];folder=dest/'ipopt_state_B10'
    report=json.loads((folder/'report.json').read_text())
    with h5py.File(dest/'state_reference.hdf5') as f:
        source_path=f.attrs['source']
        source_p=f['source_positions_m'][()] if 'source_positions_m' in f else f['positions'][()]
        source_r=f['source_rotations'][()] if 'source_rotations' in f else f['rotations'][()]
    with h5py.File(source_path) as f:source=f['data'][job['demo']]['states'][()]
    with h5py.File(folder/'recording.hdf5') as f:
        g=f['attempts'][job['demo']];before=g['tasktol/pre_action_states'][()];final=g['tasktol/final_state'][()]
        model=mujoco.MjModel.from_xml_string(g.attrs['model_file'])
    rows=json.loads((folder/'trajectory.json').read_text())
    rows=[r for r in rows if not r['aborted']]
    actual=np.concatenate([before[:len(rows)],final[None]])
    actual_p=np.asarray([source_p[0]]+[r['actual_position_m'] for r in rows])
    actual_r=np.asarray([source_r[0]]+[r['actual_rotation'] for r in rows])
    result=task_cost(model,actual,source[:len(actual)],actual_p,actual_r,source_p[:len(actual)],source_r[:len(actual)])
    result.update(**job,success=report['success'],rejected=report['execution_rejected'],
                  full_length=len(rows)==len(source)-1 and not report['execution_rejected'])
    if not result['full_length']:result['objective']+=1e12
    if write:(root/'task_pose_object_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(evaluate_task_run(p.parse_args().root),indent=2))
