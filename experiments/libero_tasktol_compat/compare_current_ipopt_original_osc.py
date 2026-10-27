"""Paired original-speed comparison against the original achieved EE trajectory."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'experiments/libero_tasktol_compat/results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(position, rotation):
    return dict(frames=len(position), position_mean_mm=float(np.mean(position)*1000),
                position_p95_mm=float(np.percentile(position, 95)*1000),
                position_max_mm=float(np.max(position)*1000),
                rotation_mean_deg=float(np.degrees(np.mean(rotation))),
                rotation_p95_deg=float(np.degrees(np.percentile(rotation, 95))),
                rotation_max_deg=float(np.degrees(np.max(rotation))))


def main():
    manifest = ROOT/'data/libero_tasktol_state/20261024_task_pose_default_ipopt11_v1/manifest.json'
    dataset = json.loads(manifest.read_text())
    out = RESULTS/'current_default_ipopt_vs_original_osc11_v1'
    out.mkdir(exist_ok=False)
    baseline_roots = [RESULTS/name for name in (
        'state_reference_12_L1_v1', 'state_reference_contact4_original_v1',
        'state_reference_spatial07_original_v1', 'osc_original_default_comparison_missing1_v1')]
    rows, evidence = [], []
    pooled = {mode: [[], []] for mode in ('osc_original', 'ipopt_current')}
    for entry in dataset['episodes']:
        current_path = Path(entry['report'])
        current = json.loads(current_path.read_text())
        relative = Path(entry['suite'])/f"{entry['task_id']:02d}"/entry['demo']/'osc_original/report.json'
        candidates = [root/relative for root in baseline_roots if (root/relative).is_file()]
        assert len(candidates) == 1
        baseline_path = candidates[0]
        baseline = json.loads(baseline_path.read_text())
        fields = ('initial_state_sha256', 'model_sha256', 'initial_gripper', 'source_input_sha256',
                  'executed_frames', 'expected_frames', 'skip_noops', 'settle_steps')
        assert all(baseline[key] == current[key] for key in fields)
        assert baseline['skip_noops'] is False and baseline['settle_steps'] == 0
        assert not baseline['execution_rejected'] and not current['execution_rejected']
        assert current['time_scale'] == baseline.get('time_scale', 1) == 1
        with h5py.File(entry['reference'], 'r') as reference:
            p = reference['source_positions_m'][()]
            r = reference['source_rotations'][()]
            source = Path(reference.attrs['source'])
        with h5py.File(source, 'r') as file:
            actions = file['data'][entry['demo']]['actions'][()]
            state = file['data'][entry['demo']]['states'][()]
        count = current['executed_frames']
        assert len(p) == len(r) == len(state) == count+1
        for mode, report_path, report in [('osc_original', baseline_path, baseline),
                                          ('ipopt_current', current_path, current)]:
            folder = report_path.parent
            trajectory = json.loads((folder/'trajectory.json').read_text())
            assert len(trajectory) == count and all(not row['aborted'] for row in trajectory)
            actual_p = np.asarray([row['actual_position_m'] for row in trajectory])
            actual_r = np.asarray([row['actual_rotation'] for row in trajectory])
            position = np.linalg.norm(actual_p-p[1:], axis=1)
            rotation = np.linalg.norm(Rotation.from_matrix(actual_r @ r[1:].transpose(0, 2, 1)).as_rotvec(), axis=1)
            np.testing.assert_allclose(position, [row['source_next_position_error_m'] for row in trajectory], atol=1e-12, rtol=0)
            np.testing.assert_allclose(rotation, [row['source_next_rotation_error_rad'] for row in trajectory], atol=1e-10, rtol=0)
            recording = folder/'recording.hdf5'
            with h5py.File(recording, 'r') as file:
                group = file['attempts'][entry['demo']]
                np.testing.assert_array_equal(group['tasktol/initial_state'][()], state[0])
                np.testing.assert_allclose(np.diff(group['tasktol/pre_action_states'][:, 0]), .05, atol=1e-9, rtol=0)
                np.testing.assert_array_equal(group['actions'][:, 6], actions[:count, 6])
                if mode == 'osc_original':
                    np.testing.assert_array_equal(group['actions'][()], actions[:count])
            pooled[mode][0].append(position)
            pooled[mode][1].append(rotation)
            rows.append(dict(case=entry['case'], suite=entry['suite'], task_id=entry['task_id'], demo=entry['demo'],
                             mode=mode, success=report['success'], **stats(position, rotation)))
            np.savez_compressed(out/f"case{entry['case']:02d}_{entry['demo']}_{mode}.npz",
                                source_state_index=np.arange(1, count+1), time_s=state[1:, 0],
                                position_error_m=position, rotation_error_rad=rotation)
            evidence.append(dict(case=entry['case'], demo=entry['demo'], mode=mode,
                                 report=str(report_path), report_sha256=sha(report_path),
                                 trajectory_sha256=sha(folder/'trajectory.json'), recording_sha256=sha(recording)))
    summary = dict(complete=True, episodes=11, tasks=dataset['tasks'],
                   comparison='actual post-step EE vs original next-state achieved FK at identical 20Hz timestamps',
                   includes_failed_episodes=True, excludes_initial_zero_error_state=True,
                   paired_initial_state_model_gripper_and_source_hash_verified=True,
                   same_original_gripper_no_noop_filter_or_settling=True,
                   ipopt_protocol='offline historical candidate selection then untouched default IPOPT rerender',
                   osc_protocol='one replay of unchanged original normalized OSC actions',
                   manifest=str(manifest), manifest_sha256=sha(manifest), evidence=evidence)
    for mode, (positions, rotations) in pooled.items():
        successful = sum(row['success'] for row in rows if row['mode'] == mode)
        summary[mode] = dict(successful=successful, attempted=11, success_rate=successful/11,
                             **stats(np.concatenate(positions), np.concatenate(rotations)))
    summary['failed_osc_episodes'] = [row for row in rows if row['mode'] == 'osc_original' and not row['success']]
    both_successful = {(row['case'], row['demo']) for row in rows
                       if row['mode'] == 'osc_original' and row['success']}
    summary['both_successful_subset'] = {}
    for mode in pooled:
        positions, rotations = [], []
        for row in rows:
            if row['mode'] != mode or (row['case'], row['demo']) not in both_successful:
                continue
            with np.load(out/f"case{row['case']:02d}_{row['demo']}_{mode}.npz") as saved:
                positions.append(saved['position_error_m'])
                rotations.append(saved['rotation_error_rad'])
        summary['both_successful_subset'][mode] = dict(episodes=len(both_successful),
            **stats(np.concatenate(positions), np.concatenate(rotations)))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (out/'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in summary.items() if key not in ('evidence', 'failed_osc_episodes')}, indent=2))


if __name__ == '__main__':
    main()
