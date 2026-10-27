"""Test native gripper integrator retiming without changing the gripper model.

Panda's format_action uses sign(command) on every 2ms tick. Repeating +/-1
through a slowed trajectory does not slow finger-reference integration. Zero
holds its internal command; one pulse per original action preserves its count.
"""

import argparse
import json
from pathlib import Path
import shutil

from rerecord_state_reference import ROOT, SELECTION, plain, sha, state_reference, worker
import rerecord_state_reference as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-scale', type=int, default=2)
    parser.add_argument('--lookahead', type=int, default=1)
    parser.add_argument('--pulse-phase', type=int, default=0)
    args = parser.parse_args()
    assert args.time_scale > 1 and 0 <= args.pulse_phase < args.time_scale
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    job = dict(case=args.case, demo=args.demo, suite=task['suite'], task_id=task['task_id'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    old = state_reference

    def pulse_reference(states, actions, handles, kin, lookahead, time_scale=1):
        reference = old(states, actions, handles, kin, lookahead, time_scale)
        for index, action in enumerate(reference['actions']):
            if index % time_scale != args.pulse_phase:
                action[6] = 0
        return reference

    runner.state_reference = pulse_reference
    sources = [Path(__file__), Path(runner.__file__), ROOT / 'examples/libero/regenerate_libero_dataset.py']
    (output / 'source_snapshot').mkdir()
    checks, snapshots = {}, {}
    for index, path in enumerate(sources):
        path = path.resolve()
        checks[str(path)] = sha(path)
        dest = output / 'source_snapshot' / f'{index}_{path.name}'
        shutil.copyfile(path, dest)
        snapshots[str(path)] = str(dest.relative_to(output))
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=args.time_scale,
                      lookahead_steps=args.lookahead, gripper_policy='native_integrator_pulse', pulse_phase=args.pulse_phase,
                      source_sha256=checks)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source_snapshots.json').write_text(json.dumps(snapshots, indent=2) + '\n')
    result = worker(job, output, args.lookahead, ['ipopt_state_B10'], args.time_scale)
    report = result['results'][0]
    aggregate = dict(ipopt_state_B10=dict(final=int(report['success']), ever=int(report['ever_success']), rejected=int(report['execution_rejected'])))
    (output / 'summary.json').write_text(json.dumps(plain(dict(complete=True, aggregate=aggregate, results=[result])), indent=2) + '\n')
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
