"""Pair learned signed tolerance bounds with actions in the existing LIBERO session."""
import numpy as np

TOLERANCE_ORDER = ['rx-', 'rx+', 'ry-', 'ry+', 'rz-', 'rz+']
MAX_BOUND_RAD = np.pi / 6
PHASES = {'pregrasp': 'APP', 'grasp': 'ENG', 'postgrasp': 'EXEC', 'release': 'DIS'}
POLICY = 'model_joint13_directional_bounds_with_shared_phase_guard'


def decode_prediction(prediction):
    from examples.libero.tasktol_actions import decode_tasktol_action
    actions = np.asarray(prediction['actions'], dtype=float)
    bounds = np.asarray(prediction['rotation_tolerance'], dtype=float)
    if actions.shape != (10, 7) or bounds.shape != (10, 6):
        raise ValueError('Require paired horizon10 actions7 and directional tolerance6')
    if not np.isfinite(actions).all() or not np.isfinite(bounds).all():
        raise ValueError('Nonfinite joint13 prediction is an infrastructure error')
    return [(a.copy(), b.copy(), raw.copy()) for a, b, raw in
            zip(decode_tasktol_action(actions[:5]), bounds[:5], actions[:5], strict=True)]


class PredictedTolerancePolicy:
    def __init__(self, classifier=None):
        if classifier is None:
            from examples.libero.mask_viewer.phase_adapter import shared_classifier
            cls, model, _ = shared_classifier()
            classifier = cls(model)
        self.classifier = classifier

    def resolve(self, fingers, action, raw_bounds):
        raw = np.asarray(raw_bounds, dtype=float)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError('Require six finite directional bounds in radians')
        bounded = np.clip(raw, 0., MAX_BOUND_RAD)
        observed = self.classifier.update(
            np.clip(np.asarray(fingers) * [1., -1.], 0., .04), bool(action[6] > 0))
        phase = PHASES[observed.phase.key]
        active = np.any(bounded.reshape(3, 2) > 0., axis=1)
        if phase not in ('APP', 'EXEC'):
            active[:] = False
        return dict(phase=phase, subtask_index=observed.subtask_index,
                    semantic_mask=''.join('1' if bit else '0' for bit in active),
                    bounds_rad=bounded.tolist(), predicted_bounds_raw_rad=raw.tolist(),
                    predicted_bounds_clipped=bool(np.any(raw != bounded)),
                    tolerance_bounds_policy=POLICY, observed_phase=observed.phase.key,
                    observed_subtask_index=observed.subtask_index,
                    phase_aperture_m=observed.aperture_m, phase_motion_m=observed.motion_m)


def step_with_prediction(session, action, selection):
    obs, row = session.step(action, **{k: selection[k] for k in
                           ('phase', 'subtask_index', 'semantic_mask', 'bounds_rad')})
    if row['semantic_mask'] != selection['semantic_mask'] or not np.allclose(
            row['bounds_rad'], selection['bounds_rad'], atol=1e-15, rtol=0):
        raise RuntimeError('The native session did not receive the predicted bounds')
    row.update({k: v for k, v in selection.items() if k not in
                ('phase', 'subtask_index', 'semantic_mask', 'bounds_rad')})
    return obs, row
