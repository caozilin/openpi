from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ...geometry import (
    rotation_tolerance_coordinate_jacobian,
    rotation_tolerance_coordinates,
    rotation_vector,
)
from ...simulation.robot import RobotHandles


@dataclass(frozen=True)
class KinematicState:
    position: np.ndarray
    rotation: np.ndarray
    jacobian: np.ndarray
    manipulability: float | None
    link_points: np.ndarray | None


class RobotKinematics:
    """Side-effect-free MuJoCo FK queries used by the optimizer."""

    def __init__(self, model: mujoco.MjModel, handles: RobotHandles) -> None:
        self.model = model
        self.handles = handles
        self._data = mujoco.MjData(model)
        # The author implementation represents a serial arm as joint-frame
        # points and measures distances between non-adjacent frame segments.
        self._link_body_ids = handles.link_body_ids.copy()

    def evaluate(self, q: np.ndarray, *, include_manipulability: bool = True, include_link_points: bool = True) -> KinematicState:
        self._data.qpos[self.handles.qpos_ids] = q
        mujoco.mj_forward(self.model, self._data)
        position = self._data.site_xpos[self.handles.ee_site_id].copy()
        rotation = self._data.site_xmat[self.handles.ee_site_id].reshape(3, 3) @ getattr(
            self.handles, "ee_rotation_offset", np.eye(3),
        )
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self._data, jac_pos, jac_rot, self.handles.ee_site_id)
        # The paper evaluates pose errors in the goal/tool frame.  Tool-frame
        # angular rows also preserve the existing teleoperator convention.
        jacobian = np.vstack((jac_pos[:, self.handles.dof_ids], rotation.T @ jac_rot[:, self.handles.dof_ids]))
        manipulability = None
        if include_manipulability:
            # Match RelaxedIK Core's ``get_manipulability_immutable``:
            # sqrt(clamp(det(J J^T), 0, 1)), using the unscaled base-frame
            # 6-D geometric Jacobian.  The pose Jacobian above remains in
            # the goal frame because that is a separate pose-error convention.
            base_jacobian = np.vstack((jac_pos[:, self.handles.dof_ids], jac_rot[:, self.handles.dof_ids]))
            determinant = float(np.linalg.det(base_jacobian @ base_jacobian.T))
            manipulability = float(np.sqrt(np.clip(determinant, 0.0, 1.0)))
        link_points = self._data.xpos[self._link_body_ids].copy() if include_link_points else None
        return KinematicState(position, rotation, jacobian, manipulability, link_points)

    @staticmethod
    def pose_error(
        state: KinematicState,
        goal_position: np.ndarray,
        goal_rotation: np.ndarray,
        tolerance_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        # Translation tasks are always the base/world X, Y and Z axes.
        position_error = state.position - goal_position
        if tolerance_rotation is None:
            rotation_error = rotation_vector(goal_rotation.T @ state.rotation)
        else:
            rotation_error = rotation_tolerance_coordinates(
                state.rotation, goal_rotation, tolerance_rotation,
            )
        return np.concatenate((position_error, rotation_error))

    @staticmethod
    def pose_error_jacobian(
        state: KinematicState,
        goal_rotation: np.ndarray,
        tolerance_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Analytic Jacobian of :meth:`pose_error`.

        MuJoCo supplies world-frame geometric angular velocity. Without a
        tolerance frame, the SO(3) left-Jacobian inverse maps it through the
        logarithm. With a tolerance frame, the ordered XYZ/RPY rate inverse is
        used instead; it is exact away from pitch = +/- pi/2.
        """
        goal_rotation = np.asarray(goal_rotation, dtype=float)
        # ``state.jacobian`` stores tool-frame angular rows.
        angular_world = state.rotation @ state.jacobian[3:]
        if tolerance_rotation is not None:
            rotation_jacobian = rotation_tolerance_coordinate_jacobian(
                state.rotation,
                goal_rotation,
                tolerance_rotation,
                angular_world,
            )
        else:
            error_goal = rotation_vector(goal_rotation.T @ state.rotation)
            angle = float(np.linalg.norm(error_goal))
            skew = np.array((
                (0.0, -error_goal[2], error_goal[1]),
                (error_goal[2], 0.0, -error_goal[0]),
                (-error_goal[1], error_goal[0], 0.0),
            ))
            if angle < 1e-5:
                coefficient = 1.0 / 12.0 + angle * angle / 720.0
            else:
                coefficient = (
                    1.0 - 0.5 * angle / np.tan(0.5 * angle)
                ) / (angle * angle)
            log_left_inverse = (
                np.eye(3)
                - 0.5 * skew
                + coefficient * (skew @ skew)
            )
            rotation_jacobian = (
                log_left_inverse @ goal_rotation.T @ angular_world
            )
        return np.vstack((state.jacobian[:3], rotation_jacobian))

    @staticmethod
    def rotation_increment_and_jacobian(
        candidate_state: KinematicState,
        anchor_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the accepted-anchor increment and its joint Jacobian."""
        frame = np.asarray(tolerance_frame, dtype=float)
        anchor = np.asarray(anchor_rotation, dtype=float)
        if frame.shape != (3, 3) or anchor.shape != (3, 3):
            raise ValueError("Anchor and tolerance frame must be 3x3 rotations")
        angular_world = candidate_state.rotation @ candidate_state.jacobian[3:]
        relative = frame.T @ candidate_state.rotation @ anchor.T @ frame
        increment = rotation_vector(relative)
        angle = float(np.linalg.norm(increment))
        skew = np.array((
            (0.0, -increment[2], increment[1]),
            (increment[2], 0.0, -increment[0]),
            (-increment[1], increment[0], 0.0),
        ))
        if angle < 1e-5:
            coefficient = 1.0 / 12.0 + angle * angle / 720.0
        else:
            coefficient = (
                1.0 - 0.5 * angle / np.tan(0.5 * angle)
            ) / (angle * angle)
        left_inverse = np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)
        return increment, left_inverse @ (frame.T @ angular_world)


# Backwards-compatible public name used by existing tests and imports.
PandaKinematics = RobotKinematics

rotation_increment_and_jacobian = RobotKinematics.rotation_increment_and_jacobian
