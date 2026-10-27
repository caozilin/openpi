"""Verify paired recording evidence and summarize the fixed 12-episode pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    provenance = json.loads((root / 'provenance.json').read_text())
    progress = json.loads((root / 'summary.json').read_text())
    assert progress['complete'] and len(progress['results']) == len(provenance['plan']) == 12
    assert not any('error' in result for result in progress['results'])
    source_checks = {path: sha(path) == expected for path, expected in provenance['source_sha256'].items()}
    source_checks.update({path: sha(path) == expected for path, expected in json.loads((root / 'helper_source_snapshot.json').read_text()).items()})
    assert all(source_checks.values())
    aggregates = {mode: dict(attempted=0, full_recording_success=0, ever_task_success=0, native_success_at_stop=0,
                            rejected=0, rejected_after_earlier_success=0, completed_without_task_success=0,
                            planned_frames=0, executed_frames=0) for mode in provenance['modes']}
    tasks = {}
    episodes = []
    checks = []
    for plan in provenance['plan']:
        dest = root / plan['suite'] / f"{plan['task_id']:02d}" / plan['demo']
        summary = json.loads((dest / 'summary.json').read_text())
        reports = summary['results']
        assert len({r['initial_state_sha256'] for r in reports}) == 1
        assert len({r['model_physical_sha256'] for r in reports}) == 1
        assert len({r['reference']['sha256'] for r in reports}) == 1
        assert len({sha(dest / r['mode'] / 'initial.png') for r in reports}) == 1
        with h5py.File(dest / 'original_reference_actions.hdf5', 'r') as file:
            reference = file['data'][plan['demo']]
            reference_actions = reference['actions'][()]
            source_indices = reference['original_source_indices'][()]
            target_p = reference['goal_positions_m'][()]
            target_r = reference['goal_tool_rotations'][()]
        episode = dict(**plan, prior_OpenVLA_success=summary['prior_OpenVLA_success'], results=[])
        task = f"{plan['suite']}/{plan['task_id']:02d}"
        if task not in tasks:
            tasks[task] = {mode: 0 for mode in provenance['modes']}
        for report in reports:
            mode = report['mode']
            folder = dest / mode
            rows = json.loads((folder / 'trajectory.json').read_text())
            attempted = report['attempted_frames']
            assert len(rows) == attempted
            assert report['expected_frames'] == len(source_indices)
            assert report['executed_frames'] == attempted - int(report['execution_rejected'])
            assert report['physics_steps'] == report['executed_frames'] * 25
            with h5py.File(folder / 'recording.hdf5', 'r') as file:
                group = file['attempts'][plan['demo']]
                assert np.array_equal(group['tasktol/source_step_indices'][()], source_indices[:attempted])
                assert sha(report['reference']['path']) == report['reference']['sha256']
                settled = group['tasktol/settled_state'][()]
                assert hashlib.sha256(settled.tobytes()).hexdigest() == report['initial_state_sha256']
                expected_actions = group['tasktol/source_osc_actions'][()] if mode == 'osc_original_actions' else reference_actions[:attempted]
                assert np.array_equal(group['actions'][()], expected_actions)
                assert (plan['demo'] in file['data']) == bool(report['success'])
                assert bool(np.any(group['dones'][()])) == bool(report['success'])
                assert group['obs/agentview_rgb'].shape[0] == attempted
                assert group['obs/eye_in_hand_rgb'].shape[0] == attempted
            if mode == 'ipopt_original_remap_B10':
                for index, row in enumerate(rows):
                    assert row['raw_source_index'] == source_indices[index]
                    assert np.array_equal(row['action'], reference_actions[index])
                    assert np.linalg.norm(np.asarray(row['nominal_target']['position_m']) - target_p[index]) < 1e-10
                    r = Rotation.from_rotvec(row['nominal_target']['rotation_vector_rad']).as_matrix()
                    assert np.linalg.norm(Rotation.from_matrix(r @ target_r[index].T).as_rotvec()) < 1e-9
                    assert row['physics_steps'] == (0 if row['aborted'] else 25)
                    if not row['aborted']:
                        assert row['optimizer']['feasible']
                        assert row['native_joint_goal_error_rad'] < 1e-10
                assert report['factor_statistics']['task_feedback'] == 'osc'
                assert report['factor_statistics']['null_reference'] == 'joint'
                assert report['factor_statistics']['null_kp'] == 10
            frames = int(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(folder / 'main_rgb.mp4')], text=True).strip())
            assert frames == attempted
            agg = aggregates[mode]
            agg['attempted'] += 1
            agg['full_recording_success'] += int(report['success'])
            agg['ever_task_success'] += int(report['ever_success'])
            agg['native_success_at_stop'] += int(rows[-1]['success'])
            agg['rejected'] += int(report['execution_rejected'])
            agg['rejected_after_earlier_success'] += int(report['execution_rejected'] and report['ever_success'])
            agg['completed_without_task_success'] += int(not report['execution_rejected'] and not report['success'])
            agg['planned_frames'] += report['expected_frames']
            agg['executed_frames'] += report['executed_frames']
            tasks[task][mode] += int(report['success'])
            item = dict(mode=mode, success=report['success'], ever_success=report['ever_success'], native_success_at_stop=bool(rows[-1]['success']),
                        expected_frames=report['expected_frames'], executed_frames=report['executed_frames'],
                        rejected_source_index=report['rejected_source_index'], rejection=report['rejection'])
            episode['results'].append(item)
            checks.append(dict(task=task, demo=plan['demo'], mode=mode, paired_inputs_and_anchors=True,
                               saved_actions_and_source_indices=True, strict_ipopt_execution=True,
                               video_decoded_frames=frames, report_sha256=sha(folder / 'report.json')))
        episodes.append(episode)
    output = dict(scope=provenance['selection'], diagnostic_sample=True,
                  primary='Full original-length filtered recording completes and native success is true at final frame',
                  secondary='Native task predicate true at any actually executed frame; does not turn a solver-aborted recording into a completed one',
                  episodes=12, aggregates=aggregates, task_success_counts=tasks, episode_results=episodes,
                  source_checks=source_checks, evidence_checks=checks, evidence_shards=len(checks))
    (root / 'comparison.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(aggregates=aggregates, task_success_counts=tasks, shards=len(checks)), indent=2))


if __name__ == '__main__':
    main()
