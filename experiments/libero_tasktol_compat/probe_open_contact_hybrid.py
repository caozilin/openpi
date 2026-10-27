"""Validate open-hand contact bias without blanket relabeling of grasps."""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, SELECTION, TASKTOL, sha
from finalize_inverse_probe import finalize
from open_contact_labeler import add_open_contact_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original_reference = runner.state_reference

    def reference_factory(states, actions, handles, kin, lookahead, time_scale=1):
        base = original_reference(states, actions, handles, kin, lookahead, time_scale)
        return add_open_contact_bias(base, states, actions, handles, kin)

    runner.state_reference = reference_factory
    sources = [Path(__file__), Path(runner.__file__), Path(__file__).with_name('open_contact_labeler.py'),
               Path(__file__).with_name('finalize_inverse_probe.py'), ROOT / 'examples/libero/regenerate_libero_dataset.py',
               TASKTOL / 'tasktol/simulation/libero.py', TASKTOL / 'tasktol/simulation/libero_joint_controller.py']
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
                      label_method='state_free_source_bias_sustained_open_contact_bounded_FK',
                      source_state_motion='actual FK plus source bias only in sustained open-hand contact',
                      original_motion_actions_used='only sustained open-hand contact windows', source_sha256=checks)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    runner.worker(job, output, 1, ['ipopt_state_B10'], 2, 'native_integrator_pulse', 0)
    finalize(output)


if __name__ == '__main__':
    main()
