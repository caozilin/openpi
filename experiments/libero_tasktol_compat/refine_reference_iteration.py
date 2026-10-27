"""Offline IRLC-style update with full native IPOPT rerecording and state costs.

References: IROS24 IRLC; Yamane et al. arXiv:2604.16850. This implementation
uses a filtered, bounded pose-error update plus geometric projection; it does
not inherit those papers' convergence guarantees through nonlinear IPOPT.
"""

import argparse
import json
from pathlib import Path
import shutil

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT, TASKTOL, plain, sha
from probe_original_speed_reference import make_reference
from closed_loop_state_metrics import evaluate_run

import h5py
import numpy as np
from scipy.signal import savgol_filter
from tasktol.geometry import rotation_matrix, rotation_vector


def limited_rows(values, limit):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values * np.minimum(1., limit / np.maximum(norms, 1e-15))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gain', type=float, default=.5)
    parser.add_argument('--anticipation', type=int, default=1)
    parser.add_argument('--proposal', type=Path, help='Offline sampling proposal: bounded position/rotation corrections')
    args = parser.parse_args()
    assert 0 < args.gain <= 1 and 0 <= args.anticipation <= 3
    parent, output = args.parent.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((parent / 'provenance.json').read_text())
    job = previous['plan'][0]
    parent_dest = parent / job['suite'] / f"{job['task_id']:02d}" / job['demo']
    with h5py.File(parent_dest / 'state_reference.hdf5') as file:
        reference = {k: v[()] for k, v in file.items()}
    rows = json.loads((parent_dest / 'ipopt_state_B10/trajectory.json').read_text())
    assert len(rows) == len(reference['actions']) and not any(r['aborted'] for r in rows)
    error_p = reference['source_positions_m'][1:] - np.asarray([r['actual_position_m'] for r in rows])
    error_r = np.asarray([rotation_vector(b @ np.asarray(r['actual_rotation']).T)
                          for b, r in zip(reference['source_rotations'][1:], rows, strict=True)])
    window = min(5, len(rows) if len(rows) % 2 else len(rows) - 1)
    error_p = savgol_filter(error_p, window, 2, axis=0)
    error_r = savgol_filter(error_r, window, 2, axis=0)
    indices = np.minimum(np.arange(len(rows)) + args.anticipation, len(rows) - 1)
    correction_p = limited_rows(args.gain * error_p[indices], .005)
    correction_r = limited_rows(args.gain * error_r[indices], .03)
    if args.proposal:
        with np.load(args.proposal) as proposal:
            correction_p, correction_r = proposal['position'].copy(), proposal['rotation'].copy()
        assert correction_p.shape == correction_r.shape == (len(rows), 3)
        assert np.all(np.isfinite(correction_p)) and np.all(np.isfinite(correction_r))
        assert np.max(np.linalg.norm(correction_p, axis=1)) <= .005 + 1e-12
        assert np.max(np.linalg.norm(correction_r, axis=1)) <= .03 + 1e-12
    goals_p = reference['positions'][1:] + correction_p
    goals_r = np.asarray([rotation_matrix(e) @ r for e, r in zip(correction_r, reference['rotations'][1:], strict=True)])
    old_reference = runner.state_reference

    def reference_factory(states, actions, handles, kin, lookahead, time_scale=1):
        assert lookahead == time_scale == 1
        base = old_reference(states, actions, handles, kin, 1, 1)
        source_p, source_r, source_q = base['positions'].copy(), base['rotations'].copy(), base['q'].copy()
        # Regularize the projection around the previous feasible reference
        # witness: zero correction should preserve that target, not shrink it.
        base.update(positions=np.concatenate([source_p[:1], goals_p]),
                    rotations=np.concatenate([source_r[:1], goals_r]),
                    q=np.concatenate([source_q[:1], reference['goal_joint_witness_rad']]))
        result = make_reference(base, actions, handles, kin, 'preview', 1, 0, .001)
        result.update(source_positions_m=source_p, source_rotations=source_r, q=source_q,
                      iteration_correction_position_m=correction_p, iteration_correction_rotation_rad=correction_r)
        return result

    runner.state_reference = reference_factory
    sources = [Path(__file__), Path(runner.__file__), Path(__file__).with_name('probe_original_speed_reference.py'),
               Path(__file__).with_name('closed_loop_state_metrics.py'),
               ROOT / 'examples/libero/regenerate_libero_dataset.py',
               TASKTOL / 'tasktol/simulation/libero.py', TASKTOL / 'tasktol/simulation/libero_joint_controller.py']
    (output / 'source_snapshot').mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, output / 'source_snapshot' / f'{i}_{path.name}')
    provenance = dict(plan=[job], modes=['ipopt_state_B10'], time_scale=1, lookahead_steps=1,
                      label_method='offline_sampled_reference_update' if args.proposal else 'offline_IRLC_filtered_bounded_reference_update', gripper_policy='preserve',
                      parent=str(parent), gain=args.gain, anticipation=args.anticipation,
                      position_update_limit_m=.005, rotation_update_limit_rad=.03,
                      task_kp=150, null_kp=10, online_future_input=False,
                      source_sha256={str(p): sha(p) for p in sources},
                      parent_inputs_sha256={str(p): sha(p) for p in [parent_dest/'state_reference.hdf5', parent_dest/'ipopt_state_B10/trajectory.json']})
    if args.proposal:
        provenance['proposal'] = dict(path=str(args.proposal.resolve()), sha256=sha(args.proposal))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'source_snapshots.json').write_text(json.dumps({str(p):f'source_snapshot/{i}_{p.name}' for i,p in enumerate(sources)}, indent=2)+'\n')
    result = runner.worker(job, output, 1, ['ipopt_state_B10'], 1, 'preserve')
    folder = output / job['suite'] / f"{job['task_id']:02d}" / job['demo'] / 'ipopt_state_B10'
    with h5py.File(folder.parent / 'state_reference.hdf5', 'r+') as file:
        file.attrs['label_method'] = provenance['label_method']
    result['results'][0].update(label_method=provenance['label_method'],
                               source_state_motion='parent reference plus previous trial tracking error; original motion intent retained by initialization')
    (folder/'report.json').write_text(json.dumps(plain(result['results'][0]), indent=2)+'\n')
    (folder.parent/'summary.json').write_text(json.dumps(plain(result), indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(plain(dict(complete=True, results=[result])), indent=2)+'\n')
    metrics = evaluate_run(output)
    previous_metrics = evaluate_run(parent)
    accepted = (metrics['full_length'] and not metrics['rejected']
                and (not previous_metrics['success'] or metrics['success'])
                and metrics['objective'] < previous_metrics['objective'])
    decision = dict(accepted=accepted, previous_objective=previous_metrics['objective'],
                    objective=metrics['objective'], previous_success=previous_metrics['success'], success=metrics['success'],
                    criterion='complete feasible rollout; retain task success; strictly decrease fixed full-state cost')
    (output/'acceptance.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision), flush=True)


if __name__ == '__main__':
    main()
