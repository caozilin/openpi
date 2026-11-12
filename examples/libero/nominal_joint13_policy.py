"""Ignore the joint13 tolerance output while retaining its diagnostic values."""
import numpy as np

from examples.libero.predicted_tolerance_policy import decode_prediction

POLICY = 'disabled_model_bounds_diagnostic_only'


class NominalDiagnosticPolicy:
    def resolve(self, fingers, action, raw_bounds):
        raw = np.asarray(raw_bounds, dtype=float)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError('Require six finite model tolerance outputs for diagnostics')
        return dict(phase='APP', subtask_index=0, semantic_mask='000', bounds_rad=None,
                    predicted_bounds_raw_rad=raw.tolist(),
                    predicted_bounds_unused=True, tolerance_bounds_policy=POLICY)


def step_without_tolerance(session, action, selection):
    # None selects the canonical strict branch. A zero-width tolerance box
    # would still invoke the tolerance path and is deliberately not used here.
    obs, row = session.step(action)
    assert row['bounds_rad'] is None
    assert row['semantic_mask'] == row['effective_mask'] == '000'
    row.update({k: selection[k] for k in
                ('predicted_bounds_raw_rad', 'predicted_bounds_unused', 'tolerance_bounds_policy')})
    return obs, row
