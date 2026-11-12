"""Direct top-grasp plan for a 90-degree clockwise knob turn."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ...geometry import axis_angle_matrix, rotation_vector
from ...simulation.robot import RobotHandles
from .pick_place import (
    _append_coordinated_pregrasp,
    _append_final_grasp_descent,
    append_blended_motion,
    append_dwell,
    gripper_close_dwell_ticks,
    gripper_preclose_dwell_ticks,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame


def _active_root(model: mujoco.MjModel) -> str:
    names: list[str] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith("rotate_knob_") or name.endswith("_knob"):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            names.append(name)
    if len(names) != 1:
        raise RuntimeError(f"Expected one active rotate knob, found {names}")
    return names[0]


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Rotate Knob object is missing: {name}")
    return int(object_id)


class RotateKnobClockwise:
    """Pinch the upright knob and rotate it around the robot-base Z axis."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
    ) -> None:
        self.model = model
        self.handles = handles

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        root_name = _active_root(self.model)
        geom_id = _required_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{root_name}_geom",
        )
        knob_body_id = _required_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{root_name}_knob",
        )
        center = data.geom_xpos[geom_id].copy()
        half_height = float(self.model.geom_size[geom_id, 2])
        gripper_offset_z = gripper_vertical_envelope(
            self.model, self.handles,
        ).upper_box_center_offset(
            half_height,
            height_bias_m=0.006,
        )
        grasp_position = center.copy()
        grasp_position[2] += gripper_offset_z
        object_rotation = data.xmat[knob_body_id].reshape(3, 3)
        closing_axis = object_rotation[:, 1].copy()
        closing_axis[2] = 0.0
        closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-12)
        tool_z = np.array((0.0, 0.0, -1.0))
        # The knob turn is a base-frame rotation.  Left multiplication keeps
        # the axis fixed in the robot base instead of rotating around the
        # end-effector's current local Z axis.
        base_body_id = _required_id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.handles.spec.base_body,
        )
        base_rotation = data.xmat[base_body_id].reshape(3, 3)
        base_z = base_rotation[:, 2].copy()
        base_z /= max(float(np.linalg.norm(base_z)), 1e-12)
        quarter_turn = axis_angle_matrix(
            base_z,
            -0.5 * math.pi,
        )
        half_turn = axis_angle_matrix(
            base_z,
            -0.25 * math.pi,
        )

        def candidate_rotation(axis_sign: float) -> np.ndarray:
            candidate_closing = axis_sign * closing_axis
            candidate_x = np.cross(candidate_closing, tool_z)
            candidate_x /= max(float(np.linalg.norm(candidate_x)), 1e-12)
            return np.column_stack((candidate_x, candidate_closing, tool_z))

        candidates = (
            candidate_rotation(1.0),
            candidate_rotation(-1.0),
        )

        # Both signs pinch the same thin face.  Choose the one whose complete
        # 90-degree turn is centred nearest the current wrist orientation.
        grasp_rotation = min(
            candidates,
            key=lambda rotation: float(np.linalg.norm(rotation_vector(
                current_rotation.T @ half_turn @ rotation,
            ))),
        )
        clockwise_rotation = (
            quarter_turn @ grasp_rotation
        )
        approach_limits = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.120),
            angular_speed_rad_s=math.inf,
            command_rate_hz=limits.command_rate_hz,
        )
        precision_limits = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.045),
            angular_speed_rad_s=math.inf,
            command_rate_hz=limits.command_rate_hz,
        )

        frames: list[TaskFrame] = []
        handoff_position, handoff_speed = _append_coordinated_pregrasp(
            frames,
            current_position,
            current_rotation,
            grasp_position,
            grasp_rotation,
            "knob",
            approach_limits,
        )
        _append_final_grasp_descent(
            frames,
            handoff_position,
            grasp_position,
            grasp_rotation,
            "knob",
            approach_limits,
            handoff_speed,
        )
        append_dwell(
            frames, grasp_position, grasp_rotation,
            False, "Align at knob", gripper_preclose_dwell_ticks(limits),
        )
        append_dwell(
            frames, grasp_position, grasp_rotation,
            True, "Close gripper on knob", gripper_close_dwell_ticks(limits),
        )
        append_blended_motion(
            frames,
            grasp_position,
            grasp_rotation,
            grasp_position,
            clockwise_rotation,
            True,
            "Rotate knob clockwise 90 degrees",
            precision_limits,
            start_speed_m_s=0.0,
            end_speed_m_s=0.0,
            fixed_duration_s=3.0,
            use_smooth_speed_profile=False,
        )
        append_dwell(
            frames, grasp_position, clockwise_rotation,
            True, "Hold knob at 90 degrees", 2,
        )
        return tuple(frames)


__all__ = ("RotateKnobClockwise",)
