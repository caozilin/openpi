"""Uniform task-only reference generation with untouched default online IPOPT.

Uses the maintained recorder and LiberoTaskTolSession; no extra dynamics loop.
The offline FK projection has zero posture weight. Source q is only a numerical
initialization and source FK input, never an online joint tracking target.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, SELECTION, TASKTOL, plain, sha
from probe_original_speed_reference import make_reference

import h5py
import numpy as np
from scipy.signal import savgol_filter

from experiments.libero_tasktol_compat.osc_reference import decode_source_goals
from experiments.libero_tasktol_compat.task_pose_object_metrics import evaluate_task_run
from experiments.libero_tasktol_compat.observe_online_ipopt import OnlineIpoptObserver
from tasktol.control.ipopt import IpoptSettings
from tasktol.control.task_space.objective import ObjectiveSettings
from tasktol.geometry import rotation_matrix, rotation_vector


def limited_rows(values, limit):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values * np.minimum(1., limit / np.maximum(norms, 1e-15))


def corrected_goals(reference, rows, gain):
    assert len(rows) == len(reference['actions']) and all(not r['aborted'] for r in rows)
    error_p = reference['source_positions_m'][1:] - np.asarray([r['actual_position_m'] for r in rows])
    error_r = np.asarray([rotation_vector(target @ np.asarray(row['actual_rotation']).T)
                          for target, row in zip(reference['source_rotations'][1:], rows, strict=True)])
    window = min(5, len(rows) if len(rows) % 2 else len(rows)-1)
    assert window >= 3
    error_p = savgol_filter(error_p, window, 2, axis=0)
    error_r = savgol_filter(error_r, window, 2, axis=0)
    indices = np.minimum(np.arange(len(rows))+1, len(rows)-1)
    correction_p = limited_rows(gain*error_p[indices], .005)
    correction_r = limited_rows(gain*error_r[indices], .03)
    p = reference['positions'][1:] + correction_p
    r = np.asarray([rotation_matrix(delta) @ old for delta, old in
                    zip(correction_r, reference['rotations'][1:], strict=True)])
    return p, r, correction_p, correction_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--gain', type=float)
    args = parser.parse_args()
    assert (args.parent is None) == (args.gain is None)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    job = dict(case=args.case, suite=task['suite'], task_id=task['task_id'], demo=args.demo)
    parent_reference = None
    if args.parent:
        assert args.gain in (.5, .25)
        parent = args.parent.resolve()
        prior_job = json.loads((parent/'provenance.json').read_text())['plan'][0]
        assert all(prior_job[k] == v for k, v in job.items())
        dest = parent/job['suite']/f"{job['task_id']:02d}"/job['demo']
        with h5py.File(dest/'state_reference.hdf5') as file:
            parent_reference = {k: v[()] for k, v in file.items()}
        previous_rows = json.loads((dest/'ipopt_state_B10/trajectory.json').read_text())
        goals_p, goals_r, correction_p, correction_r = corrected_goals(parent_reference, previous_rows, args.gain)
    else:
        with h5py.File(task['source_hdf5'], 'r') as file:
            config = json.loads(file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
            action_scale = np.asarray(config['output_max'])
            np.testing.assert_array_equal(action_scale, [.05, .05, .05, .5, .5, .5])
    original_reference = runner.state_reference

    def reference_factory(states, actions, handles, kin, lookahead, time_scale=1):
        assert lookahead == time_scale == 1
        base = original_reference(states, actions, handles, kin, 1, 1)
        source_p, source_r, source_q = (base[key].copy() for key in ('positions', 'rotations', 'q'))
        if parent_reference is None:
            target_p, target_r = decode_source_goals(source_p[:-1], source_r[:-1],
                np.clip(actions[:len(source_p)-1, :6], -1, 1)*action_scale, source_r[0])
        else:
            target_p, target_r = goals_p, goals_r
            # Previous reference q is only the pose solver's initial guess.
            base['q'] = np.concatenate([source_q[:1], parent_reference['goal_joint_witness_rad']])
        base.update(positions=np.concatenate([source_p[:1], target_p]),
                    rotations=np.concatenate([source_r[:1], target_r]))
        result = make_reference(base, actions, handles, kin, 'preview', 1, 0, posture_weight=0.)
        result.update(source_positions_m=source_p, source_rotations=source_r, q=source_q)
        if parent_reference is not None:
            result.update(iteration_correction_position_m=correction_p,
                          iteration_correction_rotation_rad=correction_r)
        return result

    runner.state_reference = reference_factory
    session_type = runner.LiberoTaskTolSession
    observed = []
    solver_observer = OnlineIpoptObserver()

    def default_session(env, *, kp, joint_control_mode):
        session = session_type(env, kp=kp, joint_control_mode=joint_control_mode, solver_settings=IpoptSettings())
        assert session.ik.settings == IpoptSettings()
        assert session.ik.objective_settings == ObjectiveSettings.delta_q_squared_conditioning_safeguard()
        observed.append(dict(settings=asdict(session.ik.settings), objective=plain(asdict(session.ik.objective_settings))))
        solver_observer.ik = session.ik
        return session

    runner.LiberoTaskTolSession = default_session
    sources = [Path(__file__), Path(runner.__file__), Path(__file__).with_name('probe_original_speed_reference.py'),
        Path(__file__).with_name('task_pose_object_metrics.py'), Path(__file__).with_name('closed_loop_state_metrics.py'),
        ROOT/'experiments/libero_tasktol_compat/osc_reference.py', ROOT/'examples/libero/regenerate_libero_dataset.py',
        TASKTOL/'tasktol/control/ipopt/controller.py', TASKTOL/'tasktol/control/ipopt/adapter.py',
        TASKTOL/'tasktol/control/task_space/objective.py', TASKTOL/'tasktol/control/task_space/kinematics.py',
        TASKTOL/'tasktol/simulation/libero.py', TASKTOL/'tasktol/simulation/libero_joint_controller.py',
        Path(__file__).with_name('observe_online_ipopt.py')]
    (output/'source_snapshot').mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, output/'source_snapshot'/f'{i}_{path.name}')
    label = 'task_only_pose_projection' if parent_reference is None else 'task_only_IRLC_pose_projection'
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=1, lookahead_steps=1,
        label_method=label, gripper_policy='preserve', task_kp=150, null_kp=10,
        parent=str(args.parent.resolve()) if args.parent else None, gain=args.gain,
        projection_posture_weight=0., robot_joint_tracking_penalty=False, finger_joint_tracking_penalty=False,
        solver_settings=asdict(IpoptSettings()), optimizer_objective='unmodified library default',
        source_goal_contract='source actual FK plus native scaled original action; held rotation starts from source FK',
        source_sha256={str(p): sha(p) for p in sources})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (output/'source_snapshots.json').write_text(json.dumps(
        {str(p): f'source_snapshot/{i}_{p.name}' for i, p in enumerate(sources)}, indent=2)+'\n')
    with solver_observer:
        result = runner.worker(job, output, 1, ['ipopt_state_B10'], 1, 'preserve')
    folder = output/job['suite']/f"{job['task_id']:02d}"/job['demo']/'ipopt_state_B10'
    rows = json.loads((folder/'trajectory.json').read_text())
    (output/'solver_observations.json').write_text(json.dumps(plain(dict(
        settings=asdict(IpoptSettings()), physics_or_optimizer_modified=False,
        timing='process_time and perf_counter around original calls; observation bookkeeping overhead included',
        records=solver_observer.records(rows))), indent=2)+'\n')
    with h5py.File(folder.parent/'state_reference.hdf5', 'r+') as file:
        file.attrs.update(label_method=label, projection_posture_weight=0., robot_joint_tracking_penalty=False)
    result['results'][0].update(label_method=label, solver_settings=asdict(IpoptSettings()),
        source_state_motion='source intent projection with no joint penalty; unchanged default IPOPT',
        projection_posture_weight=0., robot_joint_tracking_penalty=False)
    (folder/'report.json').write_text(json.dumps(plain(result['results'][0]), indent=2)+'\n')
    (folder.parent/'summary.json').write_text(json.dumps(plain(result), indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(plain(dict(complete=True, results=[result])), indent=2)+'\n')
    (output/'optimizer_default_audit.json').write_text(json.dumps(
        dict(defaults=observed, source_sha256=provenance['source_sha256']), indent=2)+'\n')
    print(json.dumps(evaluate_task_run(output)), flush=True)


if __name__ == '__main__':
    main()
