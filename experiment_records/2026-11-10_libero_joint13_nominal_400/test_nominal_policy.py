import numpy as np
import pytest
from examples.libero.nominal_joint13_policy import NominalDiagnosticPolicy, step_without_tolerance, decode_prediction


def test_tolerance_output_cannot_change_control_call():
    class Session:
        def step(self, action, **options):
            assert options == {}
            self.action = action.copy()
            return 'observation', dict(bounds_rad=None, semantic_mask='000', effective_mask='000')
    action = np.arange(7, dtype=float)
    policy = NominalDiagnosticPolicy()
    session = Session()
    for bounds in [np.zeros(6), np.full(6, np.pi / 6), np.arange(6) - 3.]:
        selection = policy.resolve(None, action, bounds)
        _, row = step_without_tolerance(session, action, selection)
        np.testing.assert_array_equal(session.action, action)
        assert row['predicted_bounds_unused'] and row['bounds_rad'] is None
        np.testing.assert_array_equal(row['predicted_bounds_raw_rad'], bounds)


@pytest.mark.parametrize('bad', [np.ones(5), np.full(6, np.nan), np.full(6, np.inf)])
def test_invalid_diagnostics_rejected(bad):
    with pytest.raises(ValueError):
        NominalDiagnosticPolicy().resolve(None, np.zeros(7), bad)


def test_same_paired_action_decoding_as_tolerance_group():
    from examples.libero.predicted_tolerance_policy import decode_prediction as previous
    pred = dict(actions=np.arange(70, dtype=float).reshape(10, 7) / 1000,
                rotation_tolerance=np.arange(60, dtype=float).reshape(10, 6) / 100)
    a, b = decode_prediction(pred), previous(pred)
    assert len(a) == len(b) == 5
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            np.testing.assert_array_equal(x, y)
