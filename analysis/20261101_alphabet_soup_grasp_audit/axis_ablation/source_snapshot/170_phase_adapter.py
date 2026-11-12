"""Replay the shared MuJoCo causal classifier over saved LIBERO observations.

No physics steps, planned labels, future commands, or semantic-mask assignment.
LIBERO fingers have opposite qpos signs; normalize to two physical openings.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import numpy as np

SHARED = Path('/media/czl/sata/franka_my_code/franka_mujoco')


@lru_cache(maxsize=1)
def shared_classifier():
    sys.path.insert(0, str(SHARED))
    import mujoco
    from tasktol.tasks.phases import PhaseClassifier
    from tasktol.config import TELEOP_DT
    # Constructor-only finger metadata. No scene, robot, or dynamics is loaded.
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="finger_joint1" type="slide" range="0 .04"/>
        <geom type="sphere" size=".01" mass=".01"/></body>
      <body><joint name="finger_joint2" type="slide" range="0 .04"/>
        <geom type="sphere" size=".01" mass=".01"/></body>
    </worldbody></mujoco>''')
    return PhaseClassifier, model, float(TELEOP_DT)


def classify_rows(rows, frequency_hz=20):
    classifier_type, model, native_dt = shared_classifier()
    classifier = classifier_type(model)
    if frequency_hz <= 0:
        raise ValueError('Sampling frequency must be positive')
    result=[]; events=[]
    for i,row in enumerate(rows):
        command = bool(row['action'][6] > 0)
        raw=np.asarray(row['pre_action']['finger_qpos_m'],dtype=float)
        openings=np.clip(raw * [1., -1.], 0., .04)
        observation=classifier.update(openings,command)
        value=dict(phase=observation.phase.key,observed_subtask_index=observation.subtask_index,
            phase_sample_frame=i,phase_sampled=True,
            phase_source=f'shared_mujoco_PhaseClassifier_pre_action_{frequency_hz:g}Hz',
            phase_aperture_m=observation.aperture_m,phase_motion_m=observation.motion_m)
        if not result or value['phase']!=result[-1]['phase'] or value['observed_subtask_index']!=result[-1]['observed_subtask_index']:
            events.append(dict(frame=i,time_s=i/frequency_hz,phase=value['phase'],
                observed_subtask_index=value['observed_subtask_index'],role='causal_observation_not_semantic_annotation'))
        result.append(value)
    rule=dict(classifier='tasktol.tasks.phases.PhaseClassifier',shared_root=str(SHARED),
        frequency_hz=frequency_hz,video_frequency_hz=frequency_hz,stable_frames=classifier.stable_frames,
        native_config_frequency_hz=1/native_dt,
        motion_threshold_m=classifier.motion_threshold_m,phase_override=False,future_commands=False,
        sampling='every pre-action video frame at 20Hz; shared default 3 stable samples; no resampling or held intermediate labels',
        fingers='clip([q_left, -q_right], 0, .04) physical openings in metres',
        semantic_scope='observed subtask index is not mapped onto frozen visual semantic subtasks',
        source_sha256={str(SHARED/p):hashlib.sha256((SHARED/p).read_bytes()).hexdigest()
            for p in ['tasktol/tasks/phases.py','tasktol/config.py']})
    return result,events,rule
