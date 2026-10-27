"""Actual end-effector differences aligned with pre-action video observations."""

import numpy as np

from tasktol.geometry import rotation_matrix, rotation_vector


def aligned_eef_metrics(left, right, anchor_position, anchor_rotation, *, fps=20, tail=40):
    """Use the previous OSC post-state for video frame i, never post-state i.

    OSC frame zero uses the verified shared actual initial pose. IPOPT supplies
    explicit pre-action poses. Added terminal PNGs use post-action final poses.
    After an IPOPT abort, differences describe held pictures, not synchronous
    execution. Pose timestamps are relative to the shared restored snapshot.
    """
    count = len(left)
    if not count or not right or len(right) > count:
        raise ValueError('Invalid paired trajectory lengths')
    aborts = [i for i, row in enumerate(right) if row['aborted']]
    abort = aborts[0] if aborts else None
    if aborts and aborts != [len(right)-1]:
        raise ValueError('IPOPT abort must be the final recorded reference frame')
    if abort is None and len(right) != count:
        raise ValueError('Short IPOPT trajectory without recorded abort')
    if any(row['physics_steps'] != 25 for row in left) or any(
        row['physics_steps'] != (0 if row['aborted'] else 25) for row in right
    ):
        raise ValueError('Unexpected physical-step timing')
    for i in range(1,count):
        if not np.array_equal(left[i]['pre_q'], left[i-1]['post_q']):
            raise ValueError('OSC state continuity does not match video indexing')
    for i in range(1,len(right)):
        if not np.allclose(right[i]['pre_action']['position_m'], right[i-1]['post_action']['position_m'], atol=1e-12, rtol=0):
            raise ValueError('IPOPT state continuity does not match video indexing')
    if not np.allclose(anchor_position, right[0]['pre_action']['position_m'], atol=1e-10, rtol=0):
        raise ValueError('Shared initial position not verified')
    initial_rotation=rotation_matrix(np.asarray(right[0]['pre_action']['rotation_vector_rad']))
    if np.linalg.norm(rotation_vector(initial_rotation @ np.asarray(anchor_rotation).T)) > 1e-9:
        raise ValueError('Shared initial tool rotation not verified')
    result=[]
    for index in range(count+tail):
        if index == 0:
            left_p, left_r = np.asarray(anchor_position), np.asarray(anchor_rotation)
        else:
            source=left[min(index-1,count-1)]
            left_p, left_r = np.asarray(source['actual_position_m']), np.asarray(source['actual_tool_rotation'])
        left_time=min(index,count)/fps
        if index < len(right):
            source=right[index]['pre_action']
            right_time=index/fps
        else:
            source=right[-1]['post_action']
            right_time=(abort if abort is not None else count)/fps
        right_p=np.asarray(source['position_m'])
        right_r=rotation_matrix(np.asarray(source['rotation_vector_rad']))
        delta=(right_p-left_p)*1000
        kind=('held_after_ipopt_abort' if abort is not None and index>abort else
              'synchronized_final' if index>=count else 'synchronized_pre_action')
        result.append(dict(
            video_frame=index, video_time_s=index/fps,
            left_pose_time_s=left_time, right_pose_time_s=right_time,
            comparison_kind=kind, synchronized=kind!='held_after_ipopt_abort',
            actual_position_difference_mm=float(np.linalg.norm(delta)),
            actual_rotation_difference_deg=float(np.degrees(np.linalg.norm(rotation_vector(right_r @ left_r.T)))),
            delta_position_ipopt_minus_osc_mm=delta.tolist(),
            left_actual_position_m=left_p.tolist(),right_actual_position_m=right_p.tolist(),
            left_actual_tool_rotation=left_r.tolist(),right_actual_tool_rotation=right_r.tolist(),
        ))
    return result


def aligned_tracking_metrics(left, right, anchor_position, anchor_rotation, *, fps=20, tail=40):
    """Signed 6D actual-minus-own-reference error in world coordinates.

    At pre-action picture i, command i has NOT been issued: use reference i-1.
    Frame zero uses the independent reference's actual-pose initial anchor.
    A rejected IPOPT candidate is never treated as an issued reference.
    Orientation is Log(R_actual R_reference.T), not Euler-angle subtraction.
    """
    rows=aligned_eef_metrics(left,right,anchor_position,anchor_rotation,fps=fps,tail=tail)
    executed=[i for i,row in enumerate(right) if not row['aborted']]
    last_right=executed[-1] if executed else -1
    for row in rows:
        index=row['video_frame']
        reference_indices=dict(left=min(index-1,len(left)-1),right=min(index-1,last_right))
        for side,ref_index in reference_indices.items():
            if ref_index<0:
                ref_p,ref_r=np.asarray(anchor_position),np.asarray(anchor_rotation)
                source='initial independent-reference actual-pose anchor'
            elif side=='left':
                target=left[ref_index]
                ref_p=np.asarray(target['native_goal_position_m'])
                ref_r=np.asarray(target['native_goal_rotation'])
                source='native OSC issued target (physical tool frame)'
            else:
                target=right[ref_index]['nominal_target']
                ref_p=np.asarray(target['position_m'])
                ref_r=rotation_matrix(np.asarray(target['rotation_vector_rad']))
                source='issued IPOPT accumulated nominal tool target (strict no-tolerance solve)'
            actual_p=np.asarray(row[f'{side}_actual_position_m'])
            actual_r=np.asarray(row[f'{side}_actual_tool_rotation'])
            position=(actual_p-ref_p)*1000
            rotation=np.degrees(rotation_vector(actual_r @ ref_r.T))
            row.update({
                f'{side}_reference_index':None if ref_index<0 else ref_index,
                f'{side}_reference_source':source,
                f'{side}_reference_position_m':ref_p.tolist(),
                f'{side}_reference_tool_rotation':ref_r.tolist(),
                f'{side}_tracking_error_6d':np.r_[position,rotation].tolist(),
            })
    return rows
