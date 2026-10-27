"""Single-variable bandwidth calibration for fixed state-derived references."""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, SELECTION, TASKTOL, plain, sha

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--task-kp', type=float, default=600)
    parser.add_argument('--time-scale', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert np.isfinite(args.task_kp) and args.task_kp > 0
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    job = dict(case=args.case, demo=args.demo, suite=task['suite'], task_id=task['task_id'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    session_type = runner.LiberoTaskTolSession

    def calibrated_session(env, *, kp, joint_control_mode):
        assert kp == 150 and joint_control_mode == 'osc_task_joint_null'
        session = session_type(env, kp=args.task_kp, joint_control_mode=joint_control_mode)
        np.testing.assert_array_equal(env.robots[0].controller.kp, np.full(6, args.task_kp))
        assert env.robots[0].controller.null_kp == 10
        return session

    runner.LiberoTaskTolSession = calibrated_session
    sources = [Path(__file__), Path(runner.__file__), ROOT / 'examples/libero/regenerate_libero_dataset.py',
               TASKTOL / 'tasktol/simulation/libero.py', TASKTOL / 'tasktol/simulation/libero_joint_controller.py']
    (output / 'source_snapshot').mkdir()
    checks, snapshots = {}, {}
    for index, path in enumerate(sources):
        path = path.resolve()
        checks[str(path)] = sha(path)
        target = output / 'source_snapshot' / f'{index}_{path.name}'
        shutil.copyfile(path, target)
        snapshots[str(path)] = str(target.relative_to(output))
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=args.time_scale, lookahead_steps=1,
                      gripper_policy='native_integrator_pulse', pulse_phase=0,
                      task_kp=args.task_kp, task_kd=float(2 * np.sqrt(args.task_kp)), null_kp=10,
                      label_method='achieved_state_fk', source_sha256=checks)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    result = runner.worker(job, output, 1, ['ipopt_state_B10'], args.time_scale, 'native_integrator_pulse', 0)
    report = result['results'][0]
    report.update(task_kp=args.task_kp, task_kd=float(2 * np.sqrt(args.task_kp)), null_kp=10)
    dest = output / job['suite'] / f"{job['task_id']:02d}" / job['demo']
    folder = dest / 'ipopt_state_B10'
    with h5py.File(folder / 'recording.hdf5', 'r+') as file:
        file.attrs.update(task_kp=args.task_kp, task_kd=float(2 * np.sqrt(args.task_kp)), null_kp=10)
    (folder / 'report.json').write_text(json.dumps(plain(report), indent=2) + '\n')
    (dest / 'summary.json').write_text(json.dumps(plain(result), indent=2) + '\n')
    aggregate = dict(ipopt_state_B10=dict(final=int(report['success']), ever=int(report['ever_success']), rejected=int(report['execution_rejected'])))
    (output / 'summary.json').write_text(json.dumps(plain(dict(complete=True, aggregate=aggregate, results=[result])), indent=2) + '\n')
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
