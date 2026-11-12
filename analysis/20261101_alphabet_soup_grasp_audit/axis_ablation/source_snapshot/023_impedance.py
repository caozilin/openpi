"""Direct operational-space impedance control for a configured serial arm."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .pose import TargetPose
from ..geometry import (
    rotation_tolerance_coordinate_jacobian,
    rotation_tolerance_coordinates,
    rotation_vector,
)
from ..simulation.robot import RobotHandles, ee_pose


@dataclass(frozen=True)
class ImpedanceDiagnostics:
    """Live diagnostics for the torque-domain Cartesian controller."""

    sigma_min: float
    manipulability: float
    nullspace_speed: float
    wrench_norm: float
    torque_norm: float
    controller_mode: str = "impedance"


class OperationalSpaceImpedance:
    """Six-dimensional Cartesian spring-damper with direct torque output.

    This controller deliberately uses ``tau = J.T @ wrench`` rather than a
    pseudoinverse or an IK update.  ``qfrc_bias`` provides gravity and velocity
    bias compensation; joint-velocity damping is only dissipative and does not
    choose a posture in the Panda's redundant seventh degree of freedom.
    """

    _POSITION_STIFFNESS = np.array((140.0, 140.0, 140.0))       # N / m
    _POSITION_DAMPING = np.array((28.0, 28.0, 28.0))            # N s / m
    _ROTATION_STIFFNESS = np.array((14.0, 14.0, 14.0))          # N m / rad
    _ROTATION_DAMPING = np.array((2.8, 2.8, 2.8))               # N m s / rad
    _JACOBIAN_SCALE = np.array((1.0, 1.0, 1.0, 0.25, 0.25, 0.25))

    def __init__(self, model: mujoco.MjModel, handles: RobotHandles) -> None:
        self.model = model
        self.handles = handles
        self._servo_gain = model.actuator_gainprm[handles.actuator_ids].copy()
        self._servo_bias = model.actuator_biasprm[handles.actuator_ids].copy()
        self._joint_damping = np.linspace(1.5, 0.5, handles.spec.arm_dof)
        force_ranges = np.abs(model.actuator_forcerange[handles.actuator_ids])
        self._torque_limits = np.max(force_ranges, axis=1)
        self._torque_limits[self._torque_limits <= 0.0] = 50.0
        # Backwards-compatible diagnostic name used by the existing checks.
        self._TORQUE_LIMITS = self._torque_limits
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        """Disable/re-enable the XML joint-position servos on mode changes."""
        if enabled == self._enabled:
            return
        if enabled:
            self.model.actuator_gainprm[self.handles.actuator_ids] = 0.0
            self.model.actuator_biasprm[self.handles.actuator_ids] = 0.0
        else:
            self.model.actuator_gainprm[self.handles.actuator_ids] = self._servo_gain
            self.model.actuator_biasprm[self.handles.actuator_ids] = self._servo_bias
        self._enabled = enabled

    def torque(self, data: mujoco.MjData, target: TargetPose, active_dofs: np.ndarray, tolerance_rotation: np.ndarray | None = None) -> tuple[np.ndarray, ImpedanceDiagnostics]:
        """Return arm torques from the current 6D task-space error at 1 kHz."""
        tolerance_rotation = (
            tolerance_rotation
            if not np.all(np.asarray(active_dofs, dtype=bool)[3:])
            else None
        )
        position, rotation = ee_pose(data, self.handles)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, self.handles.ee_site_id)
        angular_world = jac_rot[:, self.handles.dof_ids]
        if tolerance_rotation is None:
            rotation_jacobian = angular_world
            rotation_error = rotation_vector(target.rotation @ rotation.T)
        else:
            rotation_jacobian = rotation_tolerance_coordinate_jacobian(
                rotation,
                target.rotation,
                tolerance_rotation,
                angular_world,
            )
            rotation_error = -rotation_tolerance_coordinates(
                rotation, target.rotation, tolerance_rotation,
            )
        jacobian = np.vstack((
            jac_pos[:, self.handles.dof_ids], rotation_jacobian,
        ))
        qvel = data.qvel[self.handles.dof_ids]
        twist = jacobian @ qvel

        # Translation stays in base. In tolerance mode, rotational error,
        # rate and generalized wrench share the fixed-axis XYZ chart.
        position_error = target.position - position
        wrench = np.concatenate((
            self._POSITION_STIFFNESS * position_error - self._POSITION_DAMPING * twist[:3],
            self._ROTATION_STIFFNESS * rotation_error - self._ROTATION_DAMPING * twist[3:],
        ))
        wrench *= np.asarray(active_dofs, dtype=float)

        torque = jacobian.T @ wrench
        torque += data.qfrc_bias[self.handles.dof_ids]
        torque -= self._joint_damping * qvel
        torque = np.clip(torque, -self._torque_limits, self._torque_limits)

        singular_values = np.linalg.svd(jacobian * self._JACOBIAN_SCALE[:, None], compute_uv=False)
        manipulability = float(np.prod(singular_values))
        diagnostics = ImpedanceDiagnostics(
            float(singular_values[-1]), manipulability, 0.0,
            float(np.linalg.norm(wrench)), float(np.linalg.norm(torque)),
        )
        return torque, diagnostics
