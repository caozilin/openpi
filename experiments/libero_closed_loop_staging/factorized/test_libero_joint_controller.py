import numpy as np
import pytest

from tasktol.simulation.libero_joint_controller import matched_joint_inertia


def test_local_task_response_matches_split_inertia_and_preserves_null_direction():
    rng = np.random.default_rng(42)
    factor = rng.normal(size=(7, 7))
    mass = factor.T @ factor + np.eye(7)
    jac = rng.normal(size=(6, 7))
    effective = matched_joint_inertia(mass, jac)
    gram = jac @ np.linalg.solve(mass, jac.T)
    split = np.zeros((6, 6))
    split[:3, :3] = np.linalg.pinv(gram[:3, :3])
    split[3:, 3:] = np.linalg.pinv(gram[3:, 3:])
    acceleration = rng.normal(size=7)
    np.testing.assert_allclose(
        jac @ np.linalg.solve(mass, effective @ acceleration),
        gram @ split @ jac @ acceleration, atol=1e-10,
    )
    null_vector = np.linalg.svd(jac, full_matrices=True)[2][-1]
    np.testing.assert_allclose(effective @ null_vector, mass @ null_vector, atol=1e-10)
    np.testing.assert_allclose(effective, effective.T, atol=1e-10)
    assert np.linalg.eigvalsh(effective).min() > 0


@pytest.mark.parametrize("rank", [0, 3, 5])
def test_rank_loss_is_finite_and_keeps_positive_joint_inertia(rank):
    jac = np.zeros((6, 7))
    jac[:rank, :rank] = np.eye(rank)
    mass = np.diag(np.arange(1, 8, dtype=float))
    effective = matched_joint_inertia(mass, jac)
    assert np.all(np.isfinite(effective))
    assert np.linalg.eigvalsh(effective).min() > 0


def test_task_null_decomposition_preserves_current_inertia_correction():
    from robosuite.utils.control_utils import opspace_matrices

    rng=np.random.default_rng(2026)
    for _ in range(15):
        matrix=rng.normal(size=(7,7))
        mass=matrix.T @ matrix+np.eye(7)
        jac=rng.normal(size=(6,7))
        _,lp,lr,null=opspace_matrices(mass,jac,jac[:3],jac[3:])
        split=np.zeros((6,6));split[:3,:3]=lp;split[3:,3:]=lr
        uq=rng.normal(size=7)
        np.testing.assert_allclose(matched_joint_inertia(mass,jac) @ uq,
                                   jac.T @ split @ jac @ uq+null.T @ mass @ uq,atol=1e-10)


def test_factorized_clone_rejects_bound_instrumentation_override():
    from robosuite.controllers.osc import OperationalSpaceController
    from tasktol.simulation.libero_joint_controller import FactorizedReferenceController

    native=OperationalSpaceController.__new__(OperationalSpaceController)
    native.run_controller=lambda: None
    with pytest.raises(ValueError,match="instrumentation"):
        FactorizedReferenceController.from_native(native,task_feedback="osc",null_reference="joint")
