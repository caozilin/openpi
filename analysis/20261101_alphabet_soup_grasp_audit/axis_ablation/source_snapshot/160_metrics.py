"""Evaluation-only stage-relative SO(3) release and scalar annotation score."""
from __future__ import annotations

import math

import numpy as np

from ..geometry import rotation_vector


class StageRotationReleaseRecorder:
    """Track true angle peaks, not an envelope of separate Euler extrema.

    Input rotations must belong to the accepted optimizer realization and its
    transported stage reference. Physical tracking lag is not tolerance use.
    This recorder never changes the controller state or its objective.
    """

    stages = ('pregrasp','postgrasp')

    def __init__(self,subtask_count: int):
        if type(subtask_count) is not int or subtask_count<1:
            raise ValueError('Positive subtask count required')
        self.peaks = np.zeros((subtask_count,2))
        self.counts = np.zeros((subtask_count,2),dtype=int)

    def sample(self,subtask_index: int,stage: str,relative_rotation: np.ndarray) -> float:
        matrix=np.asarray(relative_rotation,dtype=float)
        if matrix.shape!=(3,3) or not np.all(np.isfinite(matrix)):
            raise ValueError('Finite relative rotation matrix required')
        if not 0<=subtask_index<len(self.peaks):
            raise ValueError('Invalid release subtask')
        stage_index=self.stages.index(stage)
        # Reuse the project's stable principal log at both zero and pi.
        angle=float(np.linalg.norm(rotation_vector(matrix)))
        self.peaks[subtask_index,stage_index]=max(self.peaks[subtask_index,stage_index],angle)
        self.counts[subtask_index,stage_index]+=1
        return angle

    def summary(self,*,trajectory_complete: bool,sampling_stride: int=1) -> dict:
        subtasks=[]
        for index,(peaks,counts) in enumerate(zip(self.peaks,self.counts)):
            stages={stage:dict(samples=int(count),peak_angle_deg=float(np.degrees(peak)) if count else None)
                for stage,peak,count in zip(self.stages,peaks,counts)}
            subtasks.append(dict(subtask_index=index,**stages,
                observed_stage_peak_sum_deg=float(np.degrees(np.sum(peaks)))))
        observed=float(np.degrees(np.sum(self.peaks)))
        complete=bool(trajectory_complete and sampling_stride==1)
        return dict(definition='sum over semantic stages of maximum principal SO(3) rotation angle',
            reference='accepted optimizer rotation relative to transported stage reference',
            units='degree',sampling_stride=sampling_stride,
            sample_scope='complete 10 Hz trajectory' if complete else 'sampled/executed prefix only',
            trajectory_complete=bool(trajectory_complete),
            whole_episode_release_deg=observed if complete else None,
            observed_stage_peak_sum_deg=observed,subtasks=subtasks)


def joint_tolerance_score(velocity_rad_s: float,release_deg: float,*,
                         velocity_reference_rad_s: float,release_weight: float,
                         release_scale_deg: float=30.) -> float:
    """Lower is better. Scale and weight are frozen before comparing points.

    Feasibility remains a hard, separately checked condition. Full objectives
    of complete table-clearance failures may train a surrogate, not win a label.
    """
    values=(velocity_rad_s,release_deg,velocity_reference_rad_s,release_weight,release_scale_deg)
    try:
        finite=all(math.isfinite(v) for v in values)
    except TypeError:
        finite=False
    if not finite or velocity_rad_s<0 or release_deg<0:
        raise ValueError('Finite nonnegative objective observations required')
    if velocity_reference_rad_s<=0 or release_scale_deg<=0 or release_weight<0:
        raise ValueError('Positive fixed scales and nonnegative release weight required')
    return velocity_rad_s/velocity_reference_rad_s+release_weight*release_deg/release_scale_deg
