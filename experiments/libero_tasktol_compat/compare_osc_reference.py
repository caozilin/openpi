"""Numerical evidence for native OSC feedback actions versus relabelled references."""

import argparse
import json
from pathlib import Path

import numpy as np

from tasktol.geometry import rotation_vector


def stats(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def compare(feedback, accumulated):
    tasks, pooled_position, pooled_rotation = [], [], []
    pooled_feedback_source, pooled_accumulated_source = [], []
    checks = []
    for report_path in sorted(feedback.glob('*/*/report.json')):
        rel = report_path.parent.relative_to(feedback)
        first, second = report_path.parent, accumulated/rel
        reports = [json.loads((p/'report.json').read_text()) for p in (first, second)]
        traces = [json.loads((p/'trace.json').read_text()) for p in (first, second)]
        if len(traces[0]) != len(traces[1]):
            raise ValueError(f'Unpaired episode lengths: {rel}')
        position = np.array([np.linalg.norm(np.array(a['actual_position_m'])-b['actual_position_m'])*1000
                             for a, b in zip(*traces, strict=True)])
        rotation = np.array([np.linalg.norm(rotation_vector(
            np.array(a['actual_tool_rotation']) @ np.array(b['actual_tool_rotation']).T))
            for a, b in zip(*traces, strict=True)])
        joint = np.array([np.max(np.abs(np.array(a['post_q'])-b['post_q']))
                         for a, b in zip(*traces, strict=True)])
        source_errors = [np.array([row['source_position_error_m']*1000 for row in trace]) for trace in traces]
        with np.load(first/'execution_states.npz') as a, np.load(second/'execution_states.npz') as b:
            state_max = float(np.max(np.abs(a['states']-b['states'])))
            nq = int(a['nq'])
            arm_ids = a['arm_qpos_ids']
            other_ids = np.setdiff1d(np.arange(nq), arm_ids)
            other_qpos_max = float(np.max(np.abs(a['states'][:,1+other_ids]-b['states'][:,1+other_ids])))
        with np.load(first/'relabelled_reference.npz') as a, np.load(second/'relabelled_reference.npz') as b:
            labels_equal = all(np.array_equal(a[k], b[k]) for k in a.files)
        task_checks = dict(
            initial_state_identical=reports[0]['initial_state_sha256']==reports[1]['initial_state_sha256'],
            fixtures_identical=reports[0]['fixtures']==reports[1]['fixtures'],
            reset_rng_verified=all(r['reset_rng_matches_source'] for r in reports),
            gains_identical=reports[0]['native_kp']==reports[1]['native_kp'] and reports[0]['native_kd']==reports[1]['native_kd'],
            reference_labels_identical=labels_equal,
            native_step_count=all(row['physics_steps']==25 for trace in traces for row in trace),
            ipopt_calls_zero=all(r['ipopt_solver_calls']==0 for r in reports),
            reference_reconstruction=all(r['reconstruction_position_max_m']<1e-10 and r['reconstruction_rotation_max_rad']<1e-9 for r in reports),
            absolute_target_position_match=reports[1]['max_goal_position_difference_mm']<1e-7,
        )
        checks.append(task_checks)
        tasks.append(dict(task=str(rel), demo=reports[0]['demo'], frames=len(position),
                          feedback_success=reports[0]['task_success'], accumulated_success=reports[1]['task_success'],
                          paired_eef_position_difference_mm=stats(position), paired_eef_rotation_difference_rad=stats(rotation),
                          paired_arm_joint_max_rad=float(joint.max()),
                          paired_flat_state_max_absolute_component=state_max,
                          paired_non_arm_qpos_max_absolute_component=other_qpos_max,
                          flat_state_component_units='mixed SI metres/radians/velocities; not a distance norm',
                          feedback_source_position_error_mm=stats(source_errors[0]),
                          accumulated_source_position_error_mm=stats(source_errors[1]),
                          accumulated_native_goal_rotation_difference_max_rad=reports[1]['max_goal_rotation_difference_rad'],
                          checks=task_checks))
        pooled_position.extend(position); pooled_rotation.extend(rotation)
        pooled_feedback_source.extend(source_errors[0]); pooled_accumulated_source.extend(source_errors[1])
    if not tasks:
        raise ValueError('No paired episodes found')
    expected = len(json.loads((feedback/'summary.json').read_text())['results'])
    return dict(
        experiment='native OSC only: source state/action -> goals -> differences -> online independent accumulation -> native absolute OSC',
        scope='Fixed selected demonstrations; low-level state-feedback control; not a learned VLA policy experiment',
        paired_tasks=len(tasks), expected_tasks=expected, frames=len(pooled_position),
        feedback_success=sum(t['feedback_success'] for t in tasks), accumulated_success=sum(t['accumulated_success'] for t in tasks),
        outcome_changes=[t['task'] for t in tasks if t['feedback_success']!=t['accumulated_success']],
        checks_passed=sum(sum(c.values()) for c in checks), checks_total=sum(len(c) for c in checks),
        paired_eef_position_difference_mm=stats(pooled_position), paired_eef_rotation_difference_rad=stats(pooled_rotation),
        feedback_source_position_error_mm=stats(pooled_feedback_source),
        accumulated_source_position_error_mm=stats(pooled_accumulated_source),
        task_mean_paired_position_difference_mm=stats([t['paired_eef_position_difference_mm']['mean'] for t in tasks]),
        tasks=tasks,
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feedback',type=Path,required=True)
    parser.add_argument('--accumulated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.feedback,args.accumulated)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2))
    if result['checks_passed']!=result['checks_total'] or result['paired_tasks']!=result['expected_tasks']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
