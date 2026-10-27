"""Validate implemented torque correction and compare closed-loop baselines."""

import argparse
import json
from pathlib import Path

import numpy as np

from tasktol.simulation.libero_joint_controller import matched_joint_inertia


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    matched = root / 'tracking_inertia_matched_kp150_v1'
    trace_results = []
    for path in sorted(matched.glob('*/*/control_trace.npz')):
        rel = path.parent.relative_to(matched)
        with np.load(path) as a, np.load(root/'tracking_ipopt_kp150_v1'/rel/'control_trace.npz') as b, np.load(root/'tracking_osc_v1'/rel/'control_trace.npz') as c:
            a, b, c = ({k: item[k] for k in item.files} for item in (a, b, c))
        initial_equal = all(np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][0], c[k][0])
                            for k in ['q', 'v', 'mass', 'jac', 'bias'])
        first_target_equal = np.array_equal(a['q_goal'][0], b['q_goal'][0])
        effective = matched_joint_inertia(a['mass'][0], a['jac'][0])
        expected = effective @ (a['kp'][0]*(a['q_goal'][0]-a['q'][0])-a['kd'][0]*a['v'][0]) + a['bias'][0]
        torque_error = float(np.max(np.abs(a['tau'][0]-expected)))
        trace_results.append(dict(
            task=str(rel), same_initial_state=initial_equal, same_first_ipopt_joint_reference=first_target_equal,
            implemented_torque_max_difference_nm=torque_error,
            old_first_torque_difference_from_osc_nm=float(np.linalg.norm(b['tau'][0]-c['tau'][0])),
            new_first_torque_difference_from_osc_nm=float(np.linalg.norm(a['tau'][0]-c['tau'][0])),
            old_torque_clipped_calls=int(np.sum(np.any(b['tau'] != b['tau_clipped'], axis=1))),
            new_torque_clipped_calls=int(np.sum(np.any(a['tau'] != a['tau_clipped'], axis=1))),
            physics_period_s=float(np.median(np.diff(a['time_s']))),
            joint_error_rms_rad=float(np.sqrt(np.mean((a['q_goal']-a['q'])**2))),
        ))
    old = json.loads((root/'ipopt_relabelled_comparison_v1.json').read_text())
    new = json.loads((root/'ipopt_inertia_matched_comparison_v1.json').read_text())
    before = {r['task']: r for r in old['results']}
    tasks = []
    diagnostics = []
    common_old = []
    common_new = []
    for row in new['results']:
        report_path = root/'ipopt_inertia_matched_kp150_all40_v1/kp_150'/row['task']/'report.json'
        report = json.loads(report_path.read_text())
        diagnostics.append(report['inertia_diagnostics'])
        prior = before[row['task']]
        old_frames=json.loads((root/'ipopt_relabelled_kp150_all40_v1/kp_150'/row['task']/'trajectory.json').read_text())['frames']
        new_frames=json.loads((report_path.parent/'trajectory.json').read_text())['frames']
        osc_frames=json.loads((root/'osc_reference_accumulated40_v1'/row['task']/'trace.json').read_text())
        paired=[]
        for i,(a,b) in enumerate(zip(old_frames,new_frames)):
            if a['aborted'] or b['aborted']:
                break
            actual=np.asarray(osc_frames[i]['actual_position_m'])
            common_old.append(float(1000*np.linalg.norm(np.asarray(a['post_action']['position_m'])-actual)))
            common_new.append(float(1000*np.linalg.norm(np.asarray(b['post_action']['position_m'])-actual)))
            paired.append(i)
        tasks.append(dict(task=row['task'], old_success=prior['ipopt_success'], new_success=row['ipopt_success'],
                          old_ever_success=prior['ipopt_ever_success'], new_ever_success=row['ipopt_ever_success'],
                          old_mean_paired_error_mm=prior['paired_eef_position_difference_mm']['mean'],
                          new_mean_paired_error_mm=row['paired_eef_position_difference_mm']['mean'],
                          common_executed_frames=len(paired),
                          failure_type=row['failure_type'], rejected_targets=row['rejected_targets']))
    calls = sum(d['calls'] for d in diagnostics)
    result = dict(
        scope='Fixed kp150; same 40 demonstrations and relabelled independent targets; full joint references; no tolerance; no VLA',
        initial_torque_validation=trace_results,
        baseline=dict(final_success=old['ipopt_success'], ever_success=old['ipopt_ever_success'],
                      paired_error_mm=old['paired_eef_position_difference_mm']),
        matched=dict(final_success=new['ipopt_success'], ever_success=new['ipopt_ever_success'],
                     paired_error_mm=new['paired_eef_position_difference_mm']),
        gained_tasks=[r['task'] for r in tasks if r['new_success'] and not r['old_success']],
        lost_tasks=[r['task'] for r in tasks if r['old_success'] and not r['new_success']],
        note='Global mean errors pair each run with OSC on its executed prefix; differing abort lengths must be considered',
        common_executed_prefix=dict(frames=len(common_old),old_mean_mm=float(np.mean(common_old)),new_mean_mm=float(np.mean(common_new)),
                                    old_p95_mm=float(np.percentile(common_old,95)),new_p95_mm=float(np.percentile(common_new,95)),
                                    old_max_mm=max(common_old),new_max_mm=max(common_new)),
        controller_calls=calls, min_effective_eigenvalue=min(d['min_effective_eigenvalue'] for d in diagnostics),
        max_symmetry_error=max(d['max_symmetry_error'] for d in diagnostics),
        clipped_calls=sum(d['torque_clipped_calls'] for d in diagnostics),
        joint_error_rms_rad=float(np.sqrt(sum(d['joint_error_squared_sum'] for d in diagnostics)/(7*calls))),
        max_joint_error_rad=max(d['max_joint_error_rad'] for d in diagnostics),
        tasks=tasks,
    )
    (root/'inertia_matched_comparison_summary_v1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'}, indent=2))
    assert len(trace_results)==3
    assert all(t['same_initial_state'] and t['same_first_ipopt_joint_reference'] and t['implemented_torque_max_difference_nm']<1e-9 for t in trace_results)
    assert new['checks_passed']==new['checks_total']


if __name__=='__main__':
    main()
