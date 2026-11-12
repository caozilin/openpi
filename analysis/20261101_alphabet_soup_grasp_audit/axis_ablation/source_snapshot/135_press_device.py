"""Simple generated trajectories for top-down button pressing."""

from __future__ import annotations

import mujoco
import numpy as np

from ...simulation.robot import RobotHandles
from .pick_place import (
    append_blended_motion,
    append_dwell,
    append_rest_to_rest_motion,
    gripper_close_dwell_ticks,
)
from .types import PREGRASP_CLEARANCE_M, MotionLimits, TaskFrame


_HOVER_CLEARANCE_M = 0.018
_APPROACH_CLEARANCE_M = 0.090
# A Cartesian command must continue beyond the nominal slide range so all
# supported joint-space controllers maintain contact through the full stroke.
# The articulated joint and contact solver determine the physical endpoint.
_PRESS_OVERTRAVEL_M = 0.008
_NON_PANDA_POWER_STRIP_PRESS_FRACTION = 0.80
_POWER_STRIP_TRACKING_HEIGHT_ALLOWANCE_M = {
    "xarm7": {
        "default": 0.016,
        "power_strip_5225": 0.0158,
        "power_strip_555": 0.010,
    },
    "ur5e": {
        "default": 0.019,
        "power_strip_5225": 0.0155,
        "power_strip_555": 0.012,
    },
}
_APPROACH_SPEED_M_S = 0.120
# Press tasks use a direct descent.  Run it 50% faster than the former
# 25 mm/s setting while retaining the same smooth start/stop profile.
_PRECISION_SPEED_M_S = 0.0375


def _descendants(model: mujoco.MjModel, root_id: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(model.nbody):
        current = candidate
        while current > root_id:
            current = int(model.body_parentid[current])
        if current == root_id:
            result.add(candidate)
    return result


def _active_root(model: mujoco.MjModel, prefix: str) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith(prefix) or name.endswith("_button"):
            continue
        subtree = _descendants(model, body_id)
        geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, list(subtree)))
        if np.any(model.geom_contype[geom_ids] != 0):
            candidates.append((name, body_id))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one active press device with prefix {prefix!r}, "
            f"found {[name for name, _ in candidates]}",
        )
    return candidates[0]


def _geom_support(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    direction: np.ndarray,
) -> float:
    """Return the farthest point of one geom along a world-frame direction."""
    local_direction = (
        data.geom_xmat[geom_id].reshape(3, 3).T @ direction
    )
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        radial = float(np.abs(local_direction) @ size)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radial = float(size[0])
    elif geom_type in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        radial = float(
            abs(local_direction[2]) * size[1]
            + np.linalg.norm(local_direction[:2]) * size[0]
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radial = float(np.linalg.norm(local_direction * size))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        first = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        radial = float(np.max(
            model.mesh_vert[first:first + count] @ local_direction,
        ))
    else:
        raise RuntimeError(
            "Unsupported fingertip collision geometry for button pressing: "
            f"type={geom_type}",
        )
    return float(data.geom_xpos[geom_id] @ direction + radial)


def _press_contact_projection(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    ee_position: np.ndarray,
    ee_rotation: np.ndarray,
) -> float:
    """Measure the signed fingertip extent along the robot's tool Z axis."""
    finger_bodies: set[int] = set()
    for joint_name in handles.spec.finger_joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
        )
        if joint_id < 0:
            raise RuntimeError(f"Gripper finger joint not found: {joint_name}")
        finger_bodies.update(_descendants(
            model, int(model.jnt_bodyid[joint_id]),
        ))
    direction = np.asarray(ee_rotation[:, 2], dtype=float)
    endpoint = float(ee_position @ direction)
    projections = [
        _geom_support(model, data, geom_id, direction) - endpoint
        for geom_id in range(model.ngeom)
        if (
            int(model.geom_bodyid[geom_id]) in finger_bodies
            and int(model.geom_contype[geom_id]) != 0
        )
    ]
    if not projections:
        raise RuntimeError("Gripper has no fingertip collision geometry")
    return max(projections)


def _press_rotation(
    current_rotation: np.ndarray,
    outward_normal: np.ndarray,
) -> np.ndarray:
    normal = np.asarray(outward_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    tool_z = -normal
    closing_axis = current_rotation[:, 1].copy()
    closing_axis -= tool_z * float(closing_axis @ tool_z)
    if np.linalg.norm(closing_axis) < 1e-9:
        closing_axis = np.array((0.0, 1.0, 0.0))
        closing_axis -= tool_z * float(closing_axis @ tool_z)
    closing_axis /= np.linalg.norm(closing_axis)
    tool_x = np.cross(closing_axis, tool_z)
    tool_x /= np.linalg.norm(tool_x)
    closing_axis = np.cross(tool_z, tool_x)
    return np.column_stack((tool_x, closing_axis, tool_z))


def _limited(limits: MotionLimits, linear_speed_m_s: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, linear_speed_m_s),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


class PressDevice:
    """Generate one direct approach that ends as soon as the button is down."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        *,
        root_prefix: str,
        label: str,
    ) -> None:
        self.model = model
        self.handles = handles
        self.root_prefix = root_prefix
        self.label = label

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        root_name, _ = _active_root(self.model, self.root_prefix)
        site_name = f"{root_name}_button_site"
        joint_name = f"{root_name}_button_joint"
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
        )
        if site_id < 0 or joint_id < 0:
            raise RuntimeError(
                f"Press device is missing {site_name!r} or {joint_name!r}",
            )
        button_surface = data.site_xpos[site_id].copy()
        site_rotation = data.site_xmat[site_id].reshape(3, 3)
        outward_normal = site_rotation[:, 2].copy()
        outward_normal /= np.linalg.norm(outward_normal)
        rotation = _press_rotation(current_rotation, outward_normal)
        tool_z = rotation[:, 2]

        travel = float(self.model.jnt_range[joint_id, 1])
        contact_offset = (
            _press_contact_projection(
                self.model,
                data,
                self.handles,
                current_position,
                current_rotation,
            )
            if self.handles.spec.uid == "panda"
            # For coupled xArm/Robotiq fingers the global gripper support point
            # is not above the button centre. Use the calibrated closed centre
            # pad projection, which agrees with the actual contact points.
            else self.handles.spec.gripper_closed_finger_bottom_z_offset_m
        )
        if self.root_prefix == "power_strip_":
            allowances = _POWER_STRIP_TRACKING_HEIGHT_ALLOWANCE_M.get(
                self.handles.spec.uid,
            )
            if allowances is not None:
                contact_offset += allowances.get(
                    root_name, allowances["default"],
                )
        unpressed_contact = button_surface - tool_z * contact_offset
        hover = (
            unpressed_contact
            + outward_normal * _HOVER_CLEARANCE_M
        )
        approach = (
            unpressed_contact + outward_normal * PREGRASP_CLEARANCE_M
            if self.root_prefix == "press_button_"
            else hover + outward_normal * _APPROACH_CLEARANCE_M
        )
        press_depth = 0.90 * travel + _PRESS_OVERTRAVEL_M
        if (
            self.root_prefix == "power_strip_"
            and self.handles.spec.uid != "panda"
        ):
            button_qpos = int(self.model.jnt_qposadr[joint_id])
            current_press = float(np.clip(
                data.qpos[button_qpos], 0.0, travel,
            ))
            press_depth = max(
                0.0,
                _NON_PANDA_POWER_STRIP_PRESS_FRACTION * travel
                - current_press,
            )
        press = unpressed_contact - outward_normal * press_depth

        frames: list[TaskFrame] = []
        approach_limits = _limited(limits, _APPROACH_SPEED_M_S)
        precision_limits = _limited(limits, _PRECISION_SPEED_M_S)
        precision_speed = precision_limits.linear_speed_m_s
        append_blended_motion(
            frames,
            current_position,
            current_rotation,
            approach,
            rotation,
            False,
            f"Move above {self.label}",
            approach_limits,
            start_speed_m_s=0.0,
            end_speed_m_s=(
                0.0
                if self.root_prefix in ("click_bell_", "power_strip_")
                else precision_speed
            ),
            use_smooth_speed_profile=False,
        )
        append_dwell(
            frames, approach, rotation, True, "Close gripper",
            gripper_close_dwell_ticks(limits),
        )
        if self.root_prefix in ("click_bell_", "power_strip_"):
            append_rest_to_rest_motion(
                frames,
                approach,
                rotation,
                press,
                rotation,
                True,
                f"Press {self.label}",
                precision_limits,
            )
            return tuple(frames)
        append_blended_motion(
            frames,
            approach,
            rotation,
            hover,
            rotation,
            True,
            f"Descend above {self.label}",
            precision_limits,
            start_speed_m_s=precision_speed,
            end_speed_m_s=0.0,
            use_smooth_speed_profile=False,
        )
        append_blended_motion(
            frames,
            hover,
            rotation,
            press,
            rotation,
            True,
            f"Press {self.label}",
            precision_limits,
            start_speed_m_s=0.0,
            end_speed_m_s=0.0,
            use_smooth_speed_profile=False,
        )
        return tuple(frames)


__all__ = ("PressDevice",)
