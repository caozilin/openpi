"""Uniform original-speed source-intent projection; no contact-stage rules."""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, SELECTION, TASKTOL, plain, sha
from probe_original_speed_reference import make_reference

import h5py
import numpy as np

from experiments.libero_tasktol_compat.osc_reference import decode_source_goals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--posture-weight', type=float, default=1e-5)
    args = parser.parse_args()
    assert np.isfinite(args.posture_weight) and args.posture_weight > 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = json.loads(SELECTION.read_text())['tasks'][args.case]
    job = dict(case=args.case, demo=args.demo, suite=task['suite'], task_id=task['task_id'])
    with h5py.File(task['source_hdf5']) as file:
        config = json.loads(file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
        scale = np.asarray(config['output_max'])
        np.testing.assert_array_equal(scale, [.05, .05, .05, .5, .5, .5])
    original = runner.state_reference

    def reference_factory(states, actions, handles, kin, lookahead, time_scale=1):
        assert lookahead == time_scale == 1
        reference = original(states, actions, handles, kin, 1, 1)
        source_p, source_r = reference['positions'].copy(), reference['rotations'].copy()
        count = len(source_p) - 1
        goals_p, goals_r = decode_source_goals(source_p[:-1], source_r[:-1],
            np.clip(actions[:count, :6], -1, 1) * scale, source_r[0])
        reference.update(positions=np.concatenate([source_p[:1], goals_p]),
                         rotations=np.concatenate([source_r[:1], goals_r]))
        reference = make_reference(reference, actions, handles, kin, 'preview', 1, 0, args.posture_weight)
        reference.update(source_positions_m=source_p, source_rotations=source_r)
        return reference

    runner.state_reference = reference_factory
    sources = [Path(__file__), Path(runner.__file__), Path(__file__).with_name('probe_original_speed_reference.py'),
               ROOT / 'experiments/libero_tasktol_compat/osc_reference.py',
               ROOT / 'examples/libero/regenerate_libero_dataset.py',
               TASKTOL / 'tasktol/simulation/libero.py', TASKTOL / 'tasktol/simulation/libero_joint_controller.py']
    (output / 'source_snapshot').mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, output / 'source_snapshot' / f'{index}_{path.name}')
    (output / 'source_snapshots.json').write_text(json.dumps(
        {str(p): f'source_snapshot/{i}_{p.name}' for i, p in enumerate(sources)}, indent=2) + '\n')
    label = 'original_speed_global_source_intent_bounded_FK'
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=1, lookahead_steps=1,
                      label_method=label, gripper_policy='preserve', task_kp=150, null_kp=10,
                      original_motion_actions_used=True, contact_stage_rules=False, posture_weight=args.posture_weight,
                      source_goal_contract='source full-state FK plus native-scaled source action; zero rotation holds history initialized at first FK, not exact cached native OSC goal equivalence',
                      source_sha256={str(p): sha(p) for p in sources})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    result = runner.worker(job, output, 1, ['ipopt_state_B10'], 1, 'preserve')
    dest = output / job['suite'] / f"{job['task_id']:02d}" / job['demo']
    folder = dest / 'ipopt_state_B10'
    with h5py.File(dest / 'state_reference.hdf5', 'r+') as file:
        file.attrs.update(label_method=label, source_goal_contract=provenance['source_goal_contract'])
    report = result['results'][0]
    report.update(label_method=label, source_state_motion=provenance['source_goal_contract'])
    rows = json.loads((folder / 'trajectory.json').read_text())
    executed = [r for r in rows if not r['aborted']]
    for field, scale, suffix in [('source_next_position_error_m', 1000, 'position_mm'),
                                  ('source_next_rotation_error_rad', 180 / np.pi, 'rotation_deg')]:
        values = [r[field] * scale for r in executed]
        report[f'source_{suffix}'] = dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                                         max=float(np.max(values))) if values else None
    (folder / 'report.json').write_text(json.dumps(plain(report), indent=2) + '\n')
    (dest / 'summary.json').write_text(json.dumps(plain(result), indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(plain(dict(complete=True, results=[result])), indent=2) + '\n')
    print(json.dumps(plain(report)), flush=True)


if __name__ == '__main__':
    main()
