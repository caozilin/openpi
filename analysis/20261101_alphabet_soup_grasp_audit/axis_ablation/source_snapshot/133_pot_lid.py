"""Seed-variant-aware trajectories for removing and replacing a pot lid."""

from __future__ import annotations

import mujoco
import numpy as np

from ...simulation.robot import RobotHandles
from ...simulation.randomization.objects import (
    body_planar_footprint,
    polygon_distance,
)
from .pick_place import (
    _plan_top_pick,
    append_blended_motion,
    append_dwell,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame


_LID_RADIUS_M = 0.0824
_LID_HALF_HEIGHT_M = 0.0032
_SEATED_LID_Z_OFFSET_M = 0.072
_TRANSFER_CLEARANCE_M = 0.100
_TABLE_RELEASE_CLEARANCE_M = 0.002
_NON_PANDA_GRASP_BOTTOM_CLEARANCE_M = 0.002
_POT_LID_PLANAR_CLEARANCE_M = 0.035
# Avoid the folded inner workspace where a Cartesian target can be nominally
# solvable but poorly tracked while carrying the large lid.  This radius is
# comfortably inside all three supported arms' tabletop workspaces.
_PREFERRED_RELEASE_RADIUS_M = 0.58


def active_pot_lid_variant(model: mujoco.MjModel) -> tuple[str, str, str]:
    """Return ``(kind, pot body, lid body)`` for the one visible variant."""
    variants = (
        ("cylindrical", "teleop_pot", "teleop_pot_lid"),
        ("handle", "teleop_pot_lift", "teleop_pot_lift_lid"),
    )
    active: list[tuple[str, str, str]] = []
    for variant in variants:
        lid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, variant[2])
        if lid_id < 0:
            continue
        descendants = [
            body_id for body_id in range(model.nbody)
            if _is_descendant(model, body_id, int(lid_id))
        ]
        geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, descendants))
        if np.any(model.geom_contype[geom_ids] != 0):
            active.append(variant)
    if len(active) != 1:
        raise RuntimeError(f"Expected one active pot/lid variant, found {active}")
    return active[0]


def _is_descendant(model: mujoco.MjModel, candidate: int, root: int) -> bool:
    while candidate > root:
        candidate = int(model.body_parentid[candidate])
    return candidate == root


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = int(mujoco.mj_name2id(model, object_type, name))
    if result < 0:
        raise RuntimeError(f"Pot-lid task object is missing: {name}")
    return result


def _downward_rotation(current_rotation: np.ndarray, closing_axis: np.ndarray, sign_index=None) -> np.ndarray:
    axis = np.asarray(closing_axis, dtype=float).copy()
    axis[2] = 0.0
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    if (sign_index == 1 if sign_index is not None else axis @ current_rotation[:, 1] < 0.0):
        axis *= -1.0
    tool_z = np.array((0.0, 0.0, -1.0))
    tool_x = np.cross(axis, tool_z)
    tool_x /= np.linalg.norm(tool_x)
    return np.column_stack((tool_x, axis, tool_z))


def _table_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    table = _id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
    geom = int(model.body_geomadr[table])
    center = data.geom_xpos[geom]
    half = model.geom_size[geom]
    return np.array((center[:2] - half[:2], center[:2] + half[:2])), float(
        center[2] + half[2]
    )


def _beside_pot_xy(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pot_id: int,
    pot_xy: np.ndarray,
    robot_base_xy: np.ndarray,
) -> np.ndarray:
    bounds, _ = _table_bounds(model, data)
    lower = bounds[0] + _LID_RADIUS_M + 0.010
    upper = bounds[1] - _LID_RADIUS_M - 0.010
    pot_footprint = body_planar_footprint(model, data, pot_id)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    unit_directions = np.column_stack((np.cos(angles), np.sin(angles)))
    lid_angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    lid_footprint_local = _LID_RADIUS_M * np.column_stack((
        np.cos(lid_angles), np.sin(lid_angles),
    ))
    feasible: list[tuple[float, np.ndarray]] = []
    # Start near the former nominal displacement, then expand only when the
    # full pot footprint (especially its long handle) blocks that ring.
    for offset in np.arange(0.24, 0.381, 0.02):
        for direction in unit_directions:
            point = pot_xy + float(offset) * direction
            if np.any(point < lower) or np.any(point > upper):
                continue
            lid_footprint = lid_footprint_local + point
            if polygon_distance(lid_footprint, pot_footprint) < _POT_LID_PLANAR_CLEARANCE_M:
                continue
            feasible.append((float(offset), point))
        if feasible:
            break
    if not feasible:
        raise RuntimeError(
            "No tabletop position beside the complete pot footprint can contain the lid",
        )
    # At equal pot clearance, prefer the arm's well-conditioned middle
    # workspace.  The former nearest-to-base rule selected folded-arm poses
    # that SQP could not track reliably while carrying the lid.
    return min(feasible, key=lambda item: (
        item[0],
        abs(float(np.linalg.norm(item[1] - robot_base_xy))
            - _PREFERRED_RELEASE_RADIUS_M),
        abs(float(item[1][1] - robot_base_xy[1])),
    ))[1].copy()


def lid_table_destination(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    *,
    expected_kind: str | None = None,
) -> np.ndarray:
    """Return the shared planner's upright lid pose beside the active pot."""
    kind, pot_name, _ = active_pot_lid_variant(model)
    if expected_kind is not None and kind != expected_kind:
        raise RuntimeError(
            f"Lid destination requires {expected_kind!r}, active variant is {kind!r}",
        )
    pot_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, pot_name)
    _, table_top = _table_bounds(model, data)
    robot_root = int(model.body_rootid[int(handles.link_body_ids[0])])
    return np.array((
        *_beside_pot_xy(
            model,
            data,
            pot_id,
            data.xpos[pot_id, :2],
            data.xpos[robot_root, :2],
        ),
        table_top + _LID_HALF_HEIGHT_M + _TABLE_RELEASE_CLEARANCE_M,
    ))


class ManipulatePotLid:
    """Top-grasp the active lid and either uncover or cover its pot."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        *,
        close_pot: bool,
        expected_kind: str | None = None,
    ) -> None:
        self.model = model
        self.handles = handles
        self.close_pot = bool(close_pot)
        self.expected_kind = expected_kind

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *, candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        kind, pot_name, lid_name = active_pot_lid_variant(self.model)
        if self.expected_kind is not None and kind != self.expected_kind:
            raise RuntimeError(
                f"Pot-lid plan requires {self.expected_kind!r}, active variant is {kind!r}",
            )
        pot_id = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, pot_name)
        lid_id = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, lid_name)
        grasp_geom_name = (
            "teleop_pot_lid_knob"
            if kind == "cylindrical"
            else "teleop_pot_lift_lid_handle_bar"
        )
        grasp_geom = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, grasp_geom_name)
        grasp_center = data.geom_xpos[grasp_geom].copy()
        grasp_radius = float(self.model.geom_size[grasp_geom, 0])
        grasp_vertical_half_extent = (
            float(self.model.geom_size[grasp_geom, 1])
            if kind == "cylindrical"
            else grasp_radius
        )
        closing_axis = (
            current_rotation[:, 1]
            if kind == "cylindrical"
            else data.geom_xmat[grasp_geom].reshape(3, 3)[:, 1]
        )
        rotation = _downward_rotation(current_rotation, closing_axis)
        if candidate_index is not None:
            if kind != "handle" or type(candidate_index) is not int or candidate_index not in (0, 1):
                raise ValueError("Explicit lid candidates require a handle lid and index 0/1")
            # Use a fixed object-axis sign, not a sign selected from live EE yaw.
            rotation = _downward_rotation(current_rotation, closing_axis, candidate_index)
        envelope = gripper_vertical_envelope(self.model, self.handles)
        # Panda's short pads use the shallowest legal U-handle alignment.  The
        # xArm and Robotiq fingers are much longer and their TCPs sit higher in
        # the gripper: that same formula lowers the EE until the fingers hit
        # the lid.  For those grippers, place the closed fingertip bottom just
        # inside the grasp entity's lower edge, independent of TCP geometry.
        if self.handles.spec.uid != "panda":
            grasp_offset = (
                envelope.closed_finger_bottom_z_offset_m
                - grasp_vertical_half_extent
                + _NON_PANDA_GRASP_BOTTOM_CLEARANCE_M
            )
        elif kind == "handle":
            grasp_offset = envelope.object_center_offset_range(
                grasp_vertical_half_extent
            )[1]
        else:
            grasp_offset = None
        pick = _plan_top_pick(
            grasp_center,
            grasp_radius,
            grasp_vertical_half_extent,
            envelope,
            rotation,
            current_position,
            current_rotation,
            limits,
            0.100,
            _TRANSFER_CLEARANCE_M,
            "pot lid",
            lid_id,
            gripper_offset_z_m=grasp_offset,
        )

        lid_position = data.xpos[lid_id].copy()
        pot_position = data.xpos[pot_id].copy()
        handle_offset = grasp_center - lid_position
        ee_offset = pick.grasp_position - grasp_center
        if self.close_pot:
            destination_lid = np.array((
                pot_position[0],
                pot_position[1],
                pot_position[2] + _SEATED_LID_Z_OFFSET_M,
            ))
            final_phase = "Lower lid onto pot"
        else:
            destination_lid = lid_table_destination(
                self.model,
                data,
                self.handles,
                expected_kind=kind,
            )
            final_phase = "Lower lid onto table"
        place_position = destination_lid + handle_offset + ee_offset
        transfer = pick.lifted_position.copy()
        transfer[:2] = place_position[:2]
        transfer[2] = max(
            pick.lifted_position[2],
            place_position[2] + _TRANSFER_CLEARANCE_M,
            pot_position[2] + 0.152,
        )

        frames = list(pick.frames)
        append_blended_motion(
            frames, pick.lifted_position, pick.grasp_rotation,
            transfer, pick.grasp_rotation, True, "Move lid to destination", limits,
            start_speed_m_s=limits.linear_speed_m_s,
            end_speed_m_s=limits.linear_speed_m_s,
        )
        append_blended_motion(
            frames, transfer, pick.grasp_rotation,
            place_position, pick.grasp_rotation, True, final_phase, limits,
            start_speed_m_s=limits.linear_speed_m_s,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames, place_position, pick.grasp_rotation,
            False, "Release pot lid", 12,
        )
        return tuple(frames)


__all__ = (
    "ManipulatePotLid",
    "active_pot_lid_variant",
    "lid_table_destination",
)
