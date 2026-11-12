"""Apply immutable offline labels at their original source-frame indices."""
import math

PHASES={'pregrasp':'APP','grasp':'ENG','postgrasp':'EXEC','release':'DIS'}

def label_policy(labels):
    frames=labels['frames']
    if not frames or labels['control_hz']!=20:
        raise ValueError('Require nonempty 20Hz labels')
    for i,frame in enumerate(frames):
        bounds=frame['rotation_tolerance_rad']
        if frame['source_frame_index']!=i or frame['phase'] not in PHASES:
            raise ValueError('Labels must cover every source frame in order')
        if len(bounds)!=6 or any(not math.isfinite(x) or x<0 or x>math.pi/6+1e-12 for x in bounds):
            raise ValueError('Invalid six-direction radian label')
        if frame['phase'] in ('grasp','release') and any(bounds):
            raise ValueError('ENG/DIS labels must be strict')
    def resolve(index,observation,action):
        frame=frames[index];bounds=list(frame['rotation_tolerance_rad'])
        mask=''.join('1' if bounds[2*j]>0 or bounds[2*j+1]>0 else '0' for j in range(3))
        return dict(phase=PHASES[frame['phase']],semantic_mask=mask,bounds_rad=bounds,
            label_subtask_index=frame['subtask_index'],semantic_subtask_index=frame['subtask_index'],
            label_source_frame_index=frame['source_frame_index'],label_source_phase=frame['phase'],
            semantic_assignment_source='frozen per-source-frame offline annotation, not online phase prediction')
    return resolve
