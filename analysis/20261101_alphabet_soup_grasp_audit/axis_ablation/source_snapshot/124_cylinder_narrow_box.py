"""Pick a lying cylinder, align it with, and release it into a narrow box."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ...config import GRIPPER_TRAVEL_TIME
from ...simulation.robot import RobotHandles
from .paper_placement import PlaceBlockOnPaper
from .pick_place import append_blended_motion, append_dwell
from .types import MotionLimits, TaskFrame


_BOX_HEIGHT_M = 0.052
_DROP_CLEARANCE_ABOVE_RIM_M = 0.010
_POST_RELEASE_SETTLE_TIME_S = 1.0


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Cylinder-to-box planner object is missing: {name}")
    return int(object_id)


def _active_pair(
    model: mujoco.MjModel,
    object_prefix: str,
    box_prefix: str,
) -> tuple[str, str, str]:
    active_objects: list[str] = []
    active_boxes: list[str] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id,
        ) or ""
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if not np.any(model.geom_contype[geom_ids] != 0):
            continue
        if name.startswith(object_prefix):
            active_objects.append(name)
        elif name.startswith(box_prefix):
            active_boxes.append(name)
    if len(active_objects) != 1 or len(active_boxes) != 1:
        raise RuntimeError(
            "Expected one active object and one matching narrow box, found "
            f"{active_objects=} and {active_boxes=}",
        )
    size = active_objects[0].removeprefix(object_prefix)
    expected_box = f"{box_prefix}{size}"
    if active_boxes[0] != expected_box:
        raise RuntimeError(
            f"Object {active_objects[0]!r} requires {expected_box!r}, "
            f"not {active_boxes[0]!r}",
        )
    return active_objects[0], active_boxes[0], size


def _limited(limits: MotionLimits, linear_speed_m_s: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, linear_speed_m_s),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


def _horizontal_alignment_delta(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> float:
    """Return the nearest yaw aligning two unoriented horizontal axes."""
    source = float(np.arctan2(source_axis[1], source_axis[0]))
    target = float(np.arctan2(target_axis[1], target_axis[0]))
    return float((target - source + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


class PlaceCylinderInNarrowBox:
    """Plan the task from live object and narrow-box poses."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        *,
        object_prefix: str = "narrow_box_cylinder_",
        box_prefix: str = "narrow_box_",
        object_label: str = "cylinder",
    ) -> None:
        self.model = model
        self.handles = handles
        self.object_prefix = object_prefix
        self.box_prefix = box_prefix
        self.object_label = object_label

    def _placement_frames(
        self,
        data: mujoco.MjData,
        pick,
        cylinder_name: str,
        box_name: str,
        limits: MotionLimits,
        *,
        placement_index: int = 0,
    ) -> tuple[TaskFrame, ...]:
        cylinder_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_name,
        )
        cylinder_geom = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM,
            f"{cylinder_name}_geom",
        )
        box_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, box_name,
        )
        interior_site = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE,
            f"{box_name}_interior_site",
        )

        object_rotation = data.xmat[cylinder_body].reshape(3, 3).copy()
        box_rotation = data.xmat[box_body].reshape(3, 3).copy()
        yaw = _horizontal_alignment_delta(
            object_rotation[:, 2], box_rotation[:, 0],
        ) + placement_index * math.pi
        cosine, sine = math.cos(yaw), math.sin(yaw)
        yaw_rotation = np.array((
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ))
        placement_rotation = yaw_rotation @ pick.grasp_rotation

        object_center = data.geom_xpos[cylinder_geom].copy()
        object_in_gripper = pick.grasp_rotation.T @ (
            object_center - pick.grasp_position
        )
        radius = float(self.model.geom_size[cylinder_geom, 0])
        desired_center = data.site_xpos[interior_site].copy()
        # The interior site is the geometric centre of the matched narrow
        # box. Release directly above it so both cylinders and blocks are
        # placed at the requested centre rather than using a biased offset.
        desired_center[2] = (
            float(data.xpos[box_body, 2])
            + _BOX_HEIGHT_M
            + radius
            + _DROP_CLEARANCE_ABOVE_RIM_M
        )
        release_position = (
            desired_center - placement_rotation @ object_in_gripper
        )

        transfer_height = max(
            float(pick.lifted_position[2]),
            float(release_position[2] + 0.100),
            float(data.xpos[box_body, 2] + _BOX_HEIGHT_M + 0.160),
            0.510,
        )
        raised = pick.lifted_position.copy()
        raised[2] = transfer_height
        above = release_position.copy()
        above[2] = transfer_height
        lift_limits = _limited(limits, 0.100)
        transfer_limits = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.080),
            angular_speed_rad_s=min(
                limits.angular_speed_rad_s, math.radians(30.0),
            ),
            command_rate_hz=limits.command_rate_hz,
        )
        lower_limits = _limited(limits, 0.045)

        frames: list[TaskFrame] = []
        if transfer_height > pick.lifted_position[2] + 1e-9:
            append_blended_motion(
                frames,
                pick.lifted_position,
                pick.grasp_rotation,
                raised,
                pick.grasp_rotation,
                True,
                f"Raise {self.object_label} to transfer height",
                lift_limits,
                start_speed_m_s=lift_limits.linear_speed_m_s,
                end_speed_m_s=lift_limits.linear_speed_m_s,
            )
        append_blended_motion(
            frames,
            raised,
            pick.grasp_rotation,
            above,
            placement_rotation,
            True,
            f"Align {self.object_label} above narrow box",
            transfer_limits,
            start_speed_m_s=lift_limits.linear_speed_m_s,
            end_speed_m_s=lower_limits.linear_speed_m_s,
        )
        append_blended_motion(
            frames,
            above,
            placement_rotation,
            release_position,
            placement_rotation,
            True,
            f"Lower {self.object_label} over narrow box",
            lower_limits,
            start_speed_m_s=lower_limits.linear_speed_m_s,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            release_position,
            placement_rotation,
            True,
            f"Hold {self.object_label} aligned over narrow box",
            8,
        )
        release_phase = f"Release {self.object_label} into narrow box"
        release_and_settle_ticks = max(1, int(math.ceil(
            (GRIPPER_TRAVEL_TIME + _POST_RELEASE_SETTLE_TIME_S)
            * limits.command_rate_hz
        )))
        append_dwell(
            frames,
            release_position,
            placement_rotation,
            False,
            release_phase,
            release_and_settle_ticks,
        )
        return tuple(frames)

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *,
        candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        cylinder_name, box_name, _ = _active_pair(
            self.model, self.object_prefix, self.box_prefix,
        )
        cylinder_geom_name = f"{cylinder_name}_geom"
        selector = PlaceBlockOnPaper(
            self.model, self.handles, object_label=self.object_label,
        )
        picks = selector._lying_pick_candidates(
            data,
            cylinder_name,
            cylinder_geom_name,
            self.object_label,
            current_position,
            current_rotation,
            limits,
        )
        if candidate_index is not None:
            if type(candidate_index) is not int or not 0 <= candidate_index < 2 * len(picks):
                raise ValueError("Invalid narrow-box candidate index")
            pick = picks[candidate_index // 2]
            return (*pick.frames, *self._placement_frames(
                data, pick, cylinder_name, box_name, limits,
                placement_index=candidate_index % 2))
        pick, placement = selector.select_lying_pick_for_suffix(
            data,
            picks,
            lambda candidate: self._placement_frames(
                data, candidate, cylinder_name, box_name, limits,
            ),
        )
        return (*pick.frames, *placement)


class PlaceBoxInNarrowBox(PlaceCylinderInNarrowBox):
    """Plan placement of a lying rectangular block in its matched narrow box."""

    def __init__(self, model: mujoco.MjModel, handles: RobotHandles) -> None:
        super().__init__(
            model,
            handles,
            object_prefix="narrow_box_block_",
            box_prefix="block_narrow_box_",
            object_label="rectangular block",
        )


__all__ = ("PlaceBoxInNarrowBox", "PlaceCylinderInNarrowBox")
