"""Cross-check displayed pre-action metrics against independent post-state pairs."""

import argparse
import json
from pathlib import Path

import numpy as np

from tasktol.geometry import rotation_matrix, rotation_vector


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--videos',type=Path,required=True)
    args=parser.parse_args()
    base=Path('/media/czl/sata/franka_my_code/openpi/experiments/libero_tasktol_compat/results')
    records=[]
    for path in sorted(args.videos.glob('*/*/eef_frame_metrics.json')):
        rel=path.parent.relative_to(args.videos)
        metrics=json.loads(path.read_text())['frames']
        left=json.loads((base/'osc_reference_hdf5_for_pair40_v1'/rel/'trace.json').read_text())
        right=json.loads((base/'ipopt_relabelled_kp150_all40_v1/kp_150'/rel/'trajectory.json').read_text())['frames']
        position_errors=[]
        rotation_errors=[]
        left_tracking_errors=[]
        right_tracking_errors=[]
        issued_indices=[]
        for index,row in enumerate(right):
            if row['aborted']:
                break
            shown=metrics[index+1]
            position=1000*np.linalg.norm(np.asarray(row['post_action']['position_m'])-left[index]['actual_position_m'])
            rotation=np.degrees(np.linalg.norm(rotation_vector(
                rotation_matrix(np.asarray(row['post_action']['rotation_vector_rad'])) @ np.asarray(left[index]['actual_tool_rotation']).T
            )))
            position_errors.append(abs(shown['actual_position_difference_mm']-position))
            rotation_errors.append(abs(shown['actual_rotation_difference_deg']-rotation))
            if 'left_tracking_error_6d' in shown:
                lp=np.asarray(left[index]['actual_position_m'])
                lr=np.asarray(left[index]['actual_tool_rotation'])
                lgoal_p=np.asarray(left[index]['native_goal_position_m'])
                lgoal_r=np.asarray(left[index]['native_goal_rotation'])
                rp=np.asarray(row['post_action']['position_m'])
                rr=rotation_matrix(np.asarray(row['post_action']['rotation_vector_rad']))
                rgoal_p=np.asarray(row['nominal_target']['position_m'])
                rgoal_r=rotation_matrix(np.asarray(row['nominal_target']['rotation_vector_rad']))
                lexpected=np.r_[(lp-lgoal_p)*1000,np.degrees(rotation_vector(lr @ lgoal_r.T))]
                rexpected=np.r_[(rp-rgoal_p)*1000,np.degrees(rotation_vector(rr @ rgoal_r.T))]
                left_tracking_errors.append(float(np.max(np.abs(np.asarray(shown['left_tracking_error_6d'])-lexpected))))
                right_tracking_errors.append(float(np.max(np.abs(np.asarray(shown['right_tracking_error_6d'])-rexpected))))
                issued_indices.append(shown['left_reference_index']==shown['right_reference_index']==index)
        abort=next((i for i,row in enumerate(right) if row['aborted']),None)
        checks=dict(
            initial_actual_positions_match=metrics[0]['actual_position_difference_mm']<1e-7,
            initial_actual_rotations_match=metrics[0]['actual_rotation_difference_deg']<1e-7,
            position_values_match_executed_post_states=max(position_errors)<1e-7,
            rotation_values_match_executed_post_states=max(rotation_errors)<1e-7,
            frame_count=len(metrics)==len(left)+40,
            finite=all(np.isfinite(x['actual_position_difference_mm']) and np.isfinite(x['actual_rotation_difference_deg']) for x in metrics),
            synchronized_timestamps=all(abs(x['left_pose_time_s']-x['right_pose_time_s'])<1e-10 for x in metrics if x['synchronized']),
            abort_marked=all(x['synchronized']==(abort is None or x['video_frame']<=abort) for x in metrics),
        )
        if left_tracking_errors:
            checks.update(
                left_6d_matches_actual_vs_own_issued_reference=max(left_tracking_errors)<1e-7,
                right_6d_matches_actual_vs_own_issued_reference=max(right_tracking_errors)<1e-7,
                own_reference_index_matches_executed_action=all(issued_indices),
                initial_reference_anchor=metrics[0]['left_reference_index'] is None and metrics[0]['right_reference_index'] is None,
                rejected_candidate_never_used=abort is None or all(x['right_reference_index'] is None or x['right_reference_index']<abort for x in metrics),
                right_after_abort_error_held=abort is None or all(np.allclose(x['right_tracking_error_6d'],metrics[abort]['right_tracking_error_6d'],atol=1e-10,rtol=0) for x in metrics[abort:]),
            )
        records.append(dict(task=str(rel),checks={k:bool(v) for k,v in checks.items()},executed_samples=len(position_errors),
                            max_position_validation_difference_mm=float(max(position_errors)),max_rotation_validation_difference_deg=float(max(rotation_errors))))
    result=dict(tasks=len(records),executed_samples=sum(r['executed_samples'] for r in records),
                checks_passed=sum(sum(r['checks'].values()) for r in records),checks_total=sum(len(r['checks']) for r in records),results=records)
    (args.videos/'eef_metrics_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    assert len(records)==40 and result['checks_passed']==result['checks_total']


if __name__=='__main__':
    main()
