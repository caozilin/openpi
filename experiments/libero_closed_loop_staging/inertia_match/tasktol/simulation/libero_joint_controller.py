"""Optional OSC-inertia matching for full-reference native joint control.

This changes only the inertia mapping inside native run_controller. The native
JointPositionController still owns target decoding, interpolation, state updates
and cleanup; the native robot still clips and applies torques and steps physics.
"""

from __future__ import annotations

import numpy as np
from robosuite.controllers.joint_pos import JointPositionController
from robosuite.utils.control_utils import opspace_matrices


def matched_joint_inertia(mass, jacobian):
    """Match native uncoupled OSC inertia; retain all seven joint errors.

    Uses native pseudoinverse semantics, including at rank loss. Local task-error
    equivalence assumes uniform scalar gains; this is not an exact nonlinear
    equivalence to OSC or a global stability guarantee.
    """
    full, position, rotation, _ = opspace_matrices(
        mass, jacobian, jacobian[:3], jacobian[3:]
    )
    split = np.zeros((6, 6))
    split[:3, :3] = position
    split[3:, 3:] = rotation
    effective = mass + jacobian.T @ (split - full) @ jacobian
    if not np.all(np.isfinite(effective)):
        raise FloatingPointError("Nonfinite matched joint inertia")
    return effective


class InertiaMatchedJointPositionController(JointPositionController):
    """Native joint controller with an explicit, opt-in inertia correction."""

    @classmethod
    def from_native(cls, controller):
        # Adopt initialized state without running constructor/update/reset again.
        # No dependency source, global factory or other robot is modified.
        if type(controller) is not JointPositionController:
            raise TypeError("Expected an unmodified native JointPositionController")
        if controller.impedance_mode != "fixed":
            raise ValueError("Inertia matching requires fixed impedance")
        if not (np.allclose(controller.kp, controller.kp[0], rtol=0, atol=1e-12)
                and np.allclose(controller.kd, controller.kd[0], rtol=0, atol=1e-12)):
            raise ValueError("Inertia matching requires uniform scalar kp and kd")
        result = cls.__new__(cls)
        result.__dict__.update(controller.__dict__)
        result.inertia_diagnostics = dict(
            calls=0, min_effective_eigenvalue=None, max_symmetry_error=0.0,
            correction_norm_sum_nm=0.0, max_correction_norm_nm=0.0,
            torque_clipped_calls=0, joint_error_squared_sum=0.0,
            max_joint_error_rad=0.0,
        )
        return result

    def run_controller(self):
        native_torque = super().run_controller()
        effective = matched_joint_inertia(self.mass_matrix, self.J_full)
        # Recover native PD acceleration after native interpolation. This avoids
        # maintaining a second implementation of its setpoint handling.
        acceleration = np.linalg.solve(
            self.mass_matrix, native_torque - self.torque_compensation
        )
        correction = (effective - self.mass_matrix) @ acceleration
        self.torques = native_torque + correction
        if not np.all(np.isfinite(self.torques)):
            raise FloatingPointError("Nonfinite inertia-matched joint torque")
        eigenvalue = float(np.linalg.eigvalsh((effective + effective.T) / 2).min())
        if eigenvalue <= 0:
            raise FloatingPointError("Matched joint inertia is not positive definite")
        stats = self.inertia_diagnostics
        stats["calls"] += 1
        previous = stats["min_effective_eigenvalue"]
        stats["min_effective_eigenvalue"] = eigenvalue if previous is None else min(previous, eigenvalue)
        stats["max_symmetry_error"] = max(stats["max_symmetry_error"], float(np.max(np.abs(effective - effective.T))))
        norm = float(np.linalg.norm(correction))
        stats["correction_norm_sum_nm"] += norm
        stats["max_correction_norm_nm"] = max(stats["max_correction_norm_nm"], norm)
        stats["torque_clipped_calls"] += int(np.any(
            (self.torques < self.actuator_min) | (self.torques > self.actuator_max)
        ))
        error = self.goal_qpos - self.joint_pos
        stats["joint_error_squared_sum"] += float(error @ error)
        stats["max_joint_error_rad"] = max(stats["max_joint_error_rad"], float(np.max(np.abs(error))))
        return self.torques
