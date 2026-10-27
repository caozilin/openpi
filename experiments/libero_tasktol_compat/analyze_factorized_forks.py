"""Validate native recovery and summarize same-state factor effects."""

import argparse
import json
from pathlib import Path

import numpy as np

from tasktol.geometry import rotation_vector


def stats(values):
    return dict(mean=float(np.mean(values)),max=float(np.max(values)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);args=parser.parse_args()
    summary=json.loads((args.batch/'summary.json').read_text());assert summary['complete']
    spots=[];per_mode={};source_checks=[]
    for task in summary['results']:
        directory=args.batch/task['task']
        original=Path('/media/czl/sata/franka_my_code/openpi/experiments/libero_tasktol_compat/results/osc_reference_hdf5_for_pair40_v1')/task['task']/'trace.json'
        source=json.loads((directory/'source_osc/trace.json').read_text());before=json.loads(original.read_text())
        source_checks.append(dict(task=task['task'],q_trajectory_bitwise_identical=[r['post_q'] for r in source]==[r['post_q'] for r in before],eef_trajectory_bitwise_identical=[r['actual_position_m'] for r in source]==[r['actual_position_m'] for r in before]))
        groups={}
        for row in task['records']:groups.setdefault(row['frame'],{})[row['mode']]=row
        for frame,group in sorted(groups.items()):
            native=group['native'];values={}
            for mode,row in group.items():
                if 'skipped' in row:continue
                delta=np.asarray(row['final_position'])-native['final_position']
                angle=np.degrees(np.linalg.norm(rotation_vector(np.asarray(row['final_rotation']) @ np.asarray(native['final_rotation']).T)))
                values[mode]=dict(position_difference_mm=float(1000*np.linalg.norm(delta)),rotation_difference_deg=float(angle),
                                  linear_velocity_difference_mm_s=float(1000*np.linalg.norm(np.asarray(row['final_velocity'])[:3]-native['final_velocity'][:3])),
                                  first_torque_difference_norm_nm=float(np.linalg.norm(np.asarray(row['first_tau'])-native['first_tau'])),
                                  normal_impulse_difference_ns=float(row['contact_normal_impulse_abs_ns']-native['contact_normal_impulse_abs_ns']),
                                  normal_impulse_absolute_difference_ns=float(abs(row['contact_normal_impulse_abs_ns']-native['contact_normal_impulse_abs_ns'])),
                                  torque_clipped_calls=row['torque_clipped_calls'])
                if mode!='native':per_mode.setdefault(mode,[]).append(values[mode])
            with np.load(Path(native['data_directory'])/'physics_trace.npz') as a,np.load(Path(group['D10']['data_directory'])/'physics_trace.npz') as b:
                d_q=float(np.max(np.abs(a['q_post']-b['q_post'])))
                d_tau=float(np.max(np.abs(a['tau']-b['tau'])))
            checks=dict(same_initial_integration_state=len({r['source_state_sha256'] for r in group.values() if 'skipped' not in r})==1,
                        native_D_recovery=d_q<1e-8 and d_tau<1e-7,
                        native_physics_steps=all(r.get('control_calls',25)==25 for r in group.values()),
                        fifty_ms=all(abs(r.get('duration_s',.05)-.05)<1e-9 for r in group.values()),
                        native_formula_reconstruction=all(r.get('native_reconstruction_max_residual_nm',0)<1e-9 for r in group.values()))
            if all(m in group and 'skipped' not in group[m] for m in ['A10','B10','C10','D10']):
                checks.update(
                    A_B_same_null=np.max(np.abs(np.asarray(group['A10']['first_null_torque'])-group['B10']['first_null_torque']))<1e-10,
                    C_D_same_null=np.max(np.abs(np.asarray(group['C10']['first_null_torque'])-group['D10']['first_null_torque']))<1e-10,
                    A_C_same_task=np.max(np.abs(np.asarray(group['A10']['first_task_torque'])-group['C10']['first_task_torque']))<1e-10,
                    B_D_same_task=np.max(np.abs(np.asarray(group['B10']['first_task_torque'])-group['D10']['first_task_torque']))<1e-10,
                )
            spots.append(dict(task=task['task'],frame=frame,events=native['events'],D_native_max_joint_difference_rad=d_q,D_native_max_torque_difference_nm=d_tau,
                              values=values,checks={k:bool(v) for k,v in checks.items()}))
    result=dict(scope='Four selected demonstrations, same native OSC states, one 50ms step per branch; no full-episode success claim',
                tasks=len(summary['results']),snapshots=len(spots),
                mode_summary={mode:{key:stats([r[key] for r in values]) for key in values[0] if key!='torque_clipped_calls'} for mode,values in per_mode.items()},
                source_replay_read_only_validation=source_checks,
                checks_passed=sum(sum(r['checks'].values()) for r in spots),checks_total=sum(len(r['checks']) for r in spots),
                results=spots)
    (args.batch/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    assert result['checks_passed']==result['checks_total']
    assert all(r['q_trajectory_bitwise_identical'] and r['eef_trajectory_bitwise_identical'] for r in source_checks)


if __name__=='__main__':main()
