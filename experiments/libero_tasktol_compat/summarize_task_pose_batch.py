"""Summarize fixed sampled episodes, separating full-rollout and prefix errors."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'franka_mujoco')]

import h5py
import numpy as np

from tasktol.control.ipopt import IpoptSettings
from tasktol.control.task_space.objective import ObjectiveSettings


def stats(position, rotation):
    if not position:
        return dict(frames=0)
    p, r = np.concatenate(position)*1000, np.degrees(np.concatenate(rotation))
    return dict(frames=len(p), position_mean_mm=float(p.mean()), position_p95_mm=float(np.percentile(p, 95)),
                position_max_mm=float(p.max()), rotation_mean_deg=float(r.mean()),
                rotation_p95_deg=float(np.percentile(r, 95)), rotation_max_deg=float(r.max()))


def latency_stats(values):
    if not values:
        return dict(calls=0)
    values = np.asarray(values)
    return dict(calls=len(values), mean_ms=float(values.mean()), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                max_ms=float(values.max()), over_50_ms=int(np.count_nonzero(values > 50)),
                over_50_fraction=float(np.mean(values > 50)))


def summarize(output):
    summary = json.loads((output/'summary.json').read_text())
    policy = json.loads((output/'policy.json').read_text())
    plan = json.loads((output/'plan.json').read_text())
    assert len(plan['plan']) == 400 and plan['tasks'] == 40
    counts = {}
    for job in plan['plan']:
        key = (job['case'], job['suite'], job['task_id'])
        counts[key] = counts.get(key, 0)+1
    assert len(counts) == 40 and set(counts.values()) == {10}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in policy['source_sha256'].items())
    groups = {}
    rows, audits = [], []
    for result in summary['results']:
        if 'error' in result:
            continue
        for stage in ('baseline', 'selected'):
            root = Path(result[f'{stage}_root'])
            folder = root/result['suite']/f"{result['task_id']:02d}"/result['demo']/'ipopt_state_B10'
            report = json.loads((folder/'report.json').read_text())
            metric = json.loads((root/'task_pose_object_metrics.json').read_text())
            rows_raw = json.loads((folder/'trajectory.json').read_text())
            executed = [row for row in rows_raw if not row['aborted']]
            elapsed = [row['optimizer']['elapsed_ms'] for row in rows_raw if row['optimizer']['solver_called']]
            position = np.asarray([row['source_next_position_error_m'] for row in executed])
            rotation = np.asarray([row['source_next_rotation_error_rad'] for row in executed])
            assert len(executed) == report['executed_frames']
            assert report['expected_frames'] == result['planned_frames']
            assert report['solver_settings'] == asdict(IpoptSettings())
            assert report['projection_posture_weight'] == 0.
            assert report['robot_joint_tracking_penalty'] is False
            audit = json.loads((root/'optimizer_default_audit.json').read_text())
            assert audit['defaults'] == [dict(settings=asdict(IpoptSettings()),
                objective=asdict(ObjectiveSettings.delta_q_squared_conditioning_safeguard()))]
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in audit['source_sha256'].items())
            with h5py.File(folder.parent/'state_reference.hdf5', 'r') as file:
                assert file.attrs['projection_posture_weight'] == 0.
            audits.append(dict(case=result['case'], demo=result['demo'], stage=stage, root=str(root), defaults_verified=True))
            reason = rows_raw[-1]['optimizer']['status'] if report['execution_rejected'] else 'none'
            row = dict(case=result['case'], suite=result['suite'], task_id=result['task_id'], demo=result['demo'], stage=stage,
                success=report['success'], rejected=report['execution_rejected'], rejection_status=reason,
                full_length=metric['full_length'], planned_frames=result['planned_frames'],
                executed_frames=len(executed), task_cost=metric['objective'], root=str(root),
                **{k:v for k,v in stats([position], [rotation]).items() if k != 'frames'} if len(position) else {})
            # Keep a uniform CSV schema even if a solver rejects frame zero.
            if not len(position):
                row = dict(case=result['case'], suite=result['suite'], task_id=result['task_id'], demo=result['demo'], stage=stage,
                    success=report['success'], rejected=report['execution_rejected'], rejection_status=reason,
                    full_length=False, planned_frames=result['planned_frames'], executed_frames=0,
                    task_cost=metric['objective'], root=str(root), position_mean_mm=None, position_p95_mm=None,
                    position_max_mm=None, rotation_mean_deg=None, rotation_p95_deg=None, rotation_max_deg=None)
            rows.append(row)
            for scope in ('all', result['suite'], f"case{result['case']:02d}"):
                key = (stage, scope)
                group = groups.setdefault(key, dict(episodes=0, successes=0, rejected=0, full_length=0,
                    planned_frames=0, executed_frames=0, full_positions=[], full_rotations=[],
                    success_positions=[], success_rotations=[], prefix_positions=[], prefix_rotations=[],
                    rejection_statuses={}, solver_elapsed_ms=[]))
                group['episodes'] += 1; group['successes'] += int(report['success'])
                group['rejected'] += int(report['execution_rejected']); group['full_length'] += int(metric['full_length'])
                group['planned_frames'] += result['planned_frames']; group['executed_frames'] += len(executed)
                group['solver_elapsed_ms'].extend(elapsed)
                if report['execution_rejected']:
                    group['rejection_statuses'][reason] = group['rejection_statuses'].get(reason, 0)+1
                if len(position):
                    subset = 'full' if metric['full_length'] else 'prefix'
                    group[f'{subset}_positions'].append(position); group[f'{subset}_rotations'].append(rotation)
                    if report['success']:
                        group['success_positions'].append(position); group['success_rotations'].append(rotation)
    aggregated = {}
    for (stage, scope), group in groups.items():
        metrics = {kind: stats(group.pop(f'{kind}_positions'), group.pop(f'{kind}_rotations')) for kind in ('full', 'success', 'prefix')}
        group.update(full_rollout_error=metrics['full'], successful_rollout_error=metrics['success'], rejected_prefix_error=metrics['prefix'])
        group['adapter_solve_wall_time'] = latency_stats(group.pop('solver_elapsed_ms'))
        group['execution_coverage'] = group['executed_frames']/group['planned_frames'] if group['planned_frames'] else 0.
        aggregated.setdefault(stage, {})[scope] = group
    result = dict(complete=summary['complete'], planned=400, completed=summary['completed'], process_errors=summary['process_errors'],
        all_40_tasks_have_10_presampled_source_episodes=True, unchanged_default_optimizer_and_zero_outer_joint_penalty_verified=True,
        scope='offline reconstruction, fixed random sample; not VLA success rate',
        error_contract='post-step actual FK vs original next-state actual FK; same timestamps; initial zeros excluded; no time alignment',
        timing_contract='adapter problem.solve wall time, excluding other client work; 6 parallel offline workers, not isolated deployment benchmark',
        truncated_rollouts='prefix errors reported separately, never mixed into full-rollout tracking statistics',
        aggregates=aggregated, default_audits=audits)
    (output/'tracking_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    if rows:
        with (output/'tracking_episodes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    task_rows = []
    for (case, suite, task_id), expected in sorted(counts.items()):
        row = dict(case=case, suite=suite, task_id=task_id, planned=expected)
        for stage in ('baseline', 'selected'):
            values = aggregated.get(stage, {}).get(f'case{case:02d}', {})
            for key in ('episodes', 'successes', 'rejected', 'full_length'):
                row[f'{stage}_{key}'] = values.get(key, 0)
            for key in ('position_mean_mm', 'position_p95_mm', 'rotation_mean_deg', 'rotation_p95_deg'):
                row[f'{stage}_{key}'] = values.get('full_rollout_error', {}).get(key)
        task_rows.append(row)
    with (output/'tasks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0])); writer.writeheader(); writer.writerows(task_rows)
    print(json.dumps(dict(complete=result['complete'], completed=result['completed'], process_errors=result['process_errors'],
        baseline=aggregated.get('baseline', {}).get('all'), selected=aggregated.get('selected', {}).get('all')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output.resolve())
