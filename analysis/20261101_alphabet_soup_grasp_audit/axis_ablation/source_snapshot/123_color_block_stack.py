"""Build the fixed yellow/red/purple cube stack."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .pick_place import (
    PickBox,
    append_blended_motion,
    append_dwell,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame
from ...simulation.robot import RobotHandles
from ..catalog import COLOR_BLOCK_STACK_TASK_SPECS


# Lower into a shallow solver contact before opening.  A centimetre-scale
# release gap makes these small cubes fall and convert tiny tilt errors into
# lateral impact; half a millimetre leaves only contact-solver penetration.
_RELEASE_CLEARANCE_M = 0.0005


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Color-stack object is missing: {name}")
    return int(object_id)


def _limits(limits: MotionLimits, linear: float, angular_deg: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, linear),
        angular_speed_rad_s=min(limits.angular_speed_rad_s, math.radians(angular_deg)),
        command_rate_hz=limits.command_rate_hz,
    )


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip((np.trace(first.T @ second) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


class StackColoredBlocks:
    """Build one direct top-pick/top-place sequence per requested color."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        task_name: str,
    ) -> None:
        self.model = model
        self.handles = handles
        try:
            self.order = COLOR_BLOCK_STACK_TASK_SPECS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown color-stack task: {task_name}") from error

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *,
        _level: int | None = None,
        candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        body_names = {
            color: f"stack_cube_{color}" for color in self.order
        }
        if candidate_index is not None and (_level is None or type(candidate_index) is not int
                                           or not 0 <= candidate_index < 16):
            raise ValueError("Explicit stack candidates require one level and index 0..15")
        geom_names = {}
        for color, body_name in body_names.items():
            body_id = _required_id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name,
            )
            active = [
                int(geom_id)
                for geom_id in np.flatnonzero(self.model.geom_bodyid == body_id)
                if self.model.geom_contype[geom_id] != 0
            ]
            if len(active) != 1:
                raise RuntimeError(
                    f"Expected one active geom for {body_name}, found {active}",
                )
            geom_names[color] = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, active[0],
            )
        base_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_names[self.order[0]],
        )
        table_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        table_geom = int(self.model.body_geomadr[table_body])
        table_top = float(
            data.geom_xpos[table_geom, 2]
            + self.model.geom_size[table_geom, 2]
        )
        base_center = data.xpos[base_body].copy()
        base_rotation = data.xmat[base_body].reshape(3, 3).copy()
        base_geom = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM,
            geom_names[self.order[0]],
        )
        cube_edge_m = 2.0 * float(self.model.geom_size[base_geom, 0])

        frames: list[TaskFrame] = []
        start_position = np.asarray(current_position, dtype=float).copy()
        start_rotation = np.asarray(current_rotation, dtype=float).copy()
        gripper_envelope = gripper_vertical_envelope(
            self.model, self.handles,
        )
        long_fingers = gripper_envelope.finger_contact_height_m >= 0.030
        transfer_limits = _limits(limits, 0.150, 20.0)
        lower_limits = _limits(
            limits,
            0.035 if self.handles.spec.uid == "xarm7"
            else (0.045 if long_fingers else 0.080),
            25.0 if self.handles.spec.uid == "xarm7" else 30.0,
        )
        retreat_limits = _limits(
            limits, 0.090 if long_fingers else 0.130, 30.0,
        )
        placement_hold_frames = (
            28 if self.handles.spec.uid == "xarm7"
            else (28 if long_fingers else 16)
        )
        release_hold_frames = (
            1 if self.handles.spec.uid == "xarm7"
            else (24 if long_fingers else 18)
        )
        release_clearance_m = _RELEASE_CLEARANCE_M

        for level, color in enumerate(self.order[1:], start=1):
            if _level is not None and level != _level:
                continue
            body_name = body_names[color]
            geom_name = geom_names[color]
            geom_id = _required_id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name,
            )
            pick = PickBox(
                self.model,
                body_name=body_name,
                geom_name=geom_name,
                approach_clearance_m=0.080,
                lift_clearance_m=0.090,
                gripper_envelope=gripper_envelope,
                # Pinch the upper-middle portion so the fingertip collision
                # envelope remains above the supporting block or table.
                grasp_height_bias_m=0.0,
            ).plan(
                data, start_position, start_rotation, limits,
                grasp_index=None if candidate_index is None else candidate_index // 4,
            )
            pick_frames = list(pick.frames)
            frames.extend(pick_frames)

            cube_center = data.geom_xpos[geom_id].copy()
            cube_rotation = data.geom_xmat[geom_id].reshape(3, 3).copy()
            # Preserve the rigid grasp transform while aligning the moved
            # cube's square axes with the randomized bottom cube.
            cube_rotation_in_gripper = (
                pick.grasp_rotation.T @ cube_rotation
            )
            # All four quarter-turns keep the cube edges mutually parallel.
            # Select the equivalent solution requiring the
            # least wrist rotation; square-axis identity is not observable.
            placement_candidates = []
            for quarter_turn in range(4):
                angle = 0.5 * np.pi * quarter_turn
                in_plane = np.array((
                    (np.cos(angle), -np.sin(angle), 0.0),
                    (np.sin(angle), np.cos(angle), 0.0),
                    (0.0, 0.0, 1.0),
                ))
                desired_cube_rotation = base_rotation @ in_plane
                placement_candidates.append(
                    desired_cube_rotation @ cube_rotation_in_gripper.T
                )
            placement_rotation = min(
                placement_candidates,
                key=lambda candidate: _rotation_distance(
                    pick.grasp_rotation, candidate,
                ),
            )
            if candidate_index is not None:
                placement_rotation = placement_candidates[candidate_index % 4]
            object_in_gripper = (
                pick.grasp_rotation.T
                @ (cube_center - pick.grasp_position)
            )
            desired_center = base_center.copy()
            desired_center[2] = (
                table_top
                + (level + 0.5) * cube_edge_m
                + release_clearance_m
            )
            place_position = (
                desired_center
                - placement_rotation @ object_in_gripper
            )
            transfer_height = max(
                float(pick.lifted_position[2]),
                float(place_position[2] + 0.100),
                0.515 + level * 0.010,
            )
            raised = pick.lifted_position.copy()
            raised[2] = transfer_height
            above = place_position.copy()
            above[2] = transfer_height

            if raised[2] > pick.lifted_position[2] + 1e-9:
                append_blended_motion(
                    frames,
                    pick.lifted_position,
                    pick.grasp_rotation,
                    raised,
                    pick.grasp_rotation,
                    True,
                    f"Raise {color} cube",
                    retreat_limits,
                    start_speed_m_s=retreat_limits.linear_speed_m_s,
                    end_speed_m_s=retreat_limits.linear_speed_m_s,
                )
            append_blended_motion(
                frames,
                raised,
                pick.grasp_rotation,
                above,
                placement_rotation,
                True,
                f"Move {color} cube above stack",
                transfer_limits,
                start_speed_m_s=retreat_limits.linear_speed_m_s,
                end_speed_m_s=lower_limits.linear_speed_m_s,
            )
            append_blended_motion(
                frames,
                above,
                placement_rotation,
                place_position,
                placement_rotation,
                True,
                f"Lower {color} cube onto level {level + 1}",
                lower_limits,
                start_speed_m_s=lower_limits.linear_speed_m_s,
                end_speed_m_s=0.0,
            )
            append_dwell(
                frames, place_position, placement_rotation, True,
                f"Hold {color} cube on stack", placement_hold_frames,
            )
            append_dwell(
                frames, place_position, placement_rotation, False,
                f"Release {color} cube", release_hold_frames,
            )
            retreat = place_position.copy()
            retreat[2] += 0.090
            append_blended_motion(
                frames,
                place_position,
                placement_rotation,
                retreat,
                placement_rotation,
                False,
                f"Retract after placing {color} cube",
                retreat_limits,
                start_speed_m_s=0.0,
                end_speed_m_s=0.0,
            )
            start_position = retreat
            start_rotation = placement_rotation

        return tuple(frames)

    def plan_subtask(
        self,
        index: int,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *, candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        """Plan one stack level from the live boundary pose and scene."""
        if not 0 <= int(index) < len(self.order) - 1:
            raise IndexError("Color-stack subtask index is out of range")
        return self.plan(
            data,
            current_position,
            current_rotation,
            limits,
            _level=int(index) + 1,
            candidate_index=candidate_index,
        )


__all__ = ("StackColoredBlocks",)
