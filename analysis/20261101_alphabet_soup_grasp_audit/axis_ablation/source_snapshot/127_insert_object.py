"""Generated grasp-align-insert trajectories for rods and rectangular bars."""

from __future__ import annotations

import mujoco
import numpy as np

from .pick_place import (
    _nearest_downward_rotation,
    _plan_top_pick,
    append_blended_motion,
    append_dwell,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame
from ...simulation.robot import RobotHandles


_GRASP_REAR_OFFSET_FRACTION = 0.30
_CAVITY_DEPTH_M = 0.060
_CAVITY_BOTTOM_CLEARANCE_M = 0.002
_UP_GRIPPER_RIM_CLEARANCE_M = 0.015
_SIDE_GRIPPER_RIM_CLEARANCE_M = 0.025
_TRANSFER_CLEARANCE_M = 0.100
_PREINSERT_CLEARANCE_M = 0.022
_PICK_DESCENT_SPEED_M_S = 0.100
_PICK_LIFT_SPEED_M_S = 0.100
_TRANSFER_SPEED_M_S = 0.140
_ALIGN_SPEED_M_S = 0.060
_INSERT_SPEED_M_S = 0.040


def _active_body(model: mujoco.MjModel, prefix: str) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith(prefix):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            candidates.append((name, body_id))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one active body with prefix {prefix!r}, "
            f"found {[name for name, _ in candidates]}",
        )
    return candidates[0]


def _limited(limits: MotionLimits, speed: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, speed),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


def _downward_grasp_rotation(
    object_rotation: np.ndarray,
    current_rotation: np.ndarray,
) -> np.ndarray:
    closing_axis = object_rotation[:, 1].copy()
    closing_axis[2] = 0.0
    closing_axis /= max(np.linalg.norm(closing_axis), 1e-12)
    if closing_axis @ current_rotation[:, 1] < 0.0:
        closing_axis *= -1.0
    tool_z = np.array((0.0, 0.0, -1.0))
    tool_x = np.cross(closing_axis, tool_z)
    tool_x /= max(np.linalg.norm(tool_x), 1e-12)
    return np.column_stack((tool_x, closing_axis, tool_z))


class InsertObject:
    """Pick the active bar and insert its local +Z end into the matched hole."""

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
        object_name, object_body = _active_body(
            self.model, "insert_object_",
        )
        container_name, _ = _active_body(
            self.model, "insert_container_",
        )
        geom_name = f"{object_name}_geom"
        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name,
        )
        site_name = f"{container_name}_hole_site"
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        if geom_id < 0 or site_id < 0:
            raise RuntimeError(
                f"Insertion model is missing {geom_name!r} or {site_name!r}",
            )

        object_position = data.xpos[object_body].copy()
        object_rotation = data.xmat[object_body].reshape(3, 3).copy()
        object_axis = object_rotation[:, 2].copy()
        object_starts_upright = abs(float(object_axis[2])) > 0.8
        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half_size = self.model.geom_size[geom_id].copy()
            half_length = float(half_size[2])
            cross_width = 2.0 * float(half_size[0])
            vertical_half_extent = float(
                np.abs(object_rotation[2]) @ half_size
            )
            horizontal_radius = (
                float(np.linalg.norm(half_size[:2]))
                if object_starts_upright
                else float(np.linalg.norm(half_size))
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            radius = float(self.model.geom_size[geom_id, 0])
            half_length = float(self.model.geom_size[geom_id, 1])
            cross_width = 2.0 * radius
            axis_vertical = abs(float(object_axis[2]))
            vertical_half_extent = float(
                axis_vertical * half_length
                + np.sqrt(max(1.0 - axis_vertical**2, 0.0)) * radius
            )
            horizontal_radius = (
                radius if object_starts_upright else half_length + radius
            )
        else:
            raise RuntimeError(
                f"Unsupported insertion object geom type: {geom_type}",
            )
        length = 2.0 * half_length
        pickup_envelope_extent = vertical_half_extent
        if object_starts_upright:
            grasp_rear_fraction = -_GRASP_REAR_OFFSET_FRACTION
            pickup_envelope_extent = 0.25 * length
            lift_clearance = 0.09
        elif 0.014 <= vertical_half_extent < 0.020:
            # The 3 cm bars are most stable slightly higher between the
            # Panda pads; narrower variants stay lower to retain contact.
            pickup_envelope_extent = 0.022
            lift_clearance = 0.075
        elif vertical_half_extent < 0.014:
            # Raise the Panda grasp slightly for the smallest variants so the
            # fingertips do not brush the tabletop.
            pickup_envelope_extent = 0.017
            lift_clearance = 0.075
        else:
            lift_clearance = 0.075
        if not container_name.endswith("_up"):
            # For a side-facing holder, lift only until the object's centre
            # reaches the hole-centre Z height.
            lift_clearance = max(
                0.0,
                float(data.site_xpos[site_id, 2] - object_position[2]),
            )
        if object_starts_upright:
            pass
        elif container_name.endswith("_up"):
            grasp_rear_fraction = _GRASP_REAR_OFFSET_FRACTION
        else:
            grasp_rear_fraction = _GRASP_REAR_OFFSET_FRACTION
        grasp_center = (
            object_position
            - object_axis * (grasp_rear_fraction * length)
        )
        if (
            geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            and object_starts_upright
        ):
            # A vertical cylinder has no meaningful yaw.  Preserve the
            # current horizontal gripper heading instead of following the
            # object's randomized local-X/local-Y frame.
            grasp_rotation = _nearest_downward_rotation(current_rotation)
        else:
            grasp_rotation = _downward_grasp_rotation(
                object_rotation,
                current_rotation,
            )
        pick = _plan_top_pick(
            grasp_center,
            # Keep the fingertips clear of the tabletop for the smallest
            # variants while retaining enough pad overlap to hold them.
            pickup_envelope_extent,
            horizontal_radius,
            gripper_vertical_envelope(self.model, self.handles),
            grasp_rotation,
            current_position,
            current_rotation,
            limits,
            0.095,
            lift_clearance,
            "insertion object",
            object_body,
            descent_speed_m_s=_PICK_DESCENT_SPEED_M_S,
            lift_speed_m_s=_PICK_LIFT_SPEED_M_S,
        )

        object_in_gripper_rotation = (
            pick.grasp_rotation.T @ object_rotation
        )
        object_in_gripper_position = (
            pick.grasp_rotation.T
            @ (object_position - pick.grasp_position)
        )
        site_position = data.site_xpos[site_id].copy()
        site_rotation = data.site_xmat[site_id].reshape(3, 3).copy()
        outward_normal = site_rotation[:, 2].copy()
        outward_normal /= np.linalg.norm(outward_normal)
        # Both supported cross-sections are symmetric about the insertion
        # axis.  Choose an equivalent in-plane orientation with the smallest
        # wrist turn instead of forcing an arbitrary 180-degree roll.
        candidate_count = (
            4
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX)
            else 8
        )
        if outward_normal[2] > 0.7:
            if object_starts_upright:
                preferred_tool_z = np.array((0.0, 0.0, -1.0))
            else:
                preferred_tool_z = np.array((1.0, 0.0, 0.0))
        else:
            preferred_tool_z = np.array((0.0, 0.0, -1.0))
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        insertion_axis_mapping = (
            np.eye(3)
            if object_starts_upright and outward_normal[2] > 0.7
            else np.diag((-1.0, 1.0, -1.0))
        )
        for index in range(candidate_count):
            angle = 2.0 * np.pi * index / candidate_count
            cosine, sine = np.cos(angle), np.sin(angle)
            in_plane = np.array((
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ))
            object_candidate = (
                site_rotation @ in_plane @ insertion_axis_mapping
            )
            gripper_candidate = (
                object_candidate @ object_in_gripper_rotation.T
            )
            relative = pick.grasp_rotation.T @ gripper_candidate
            wrist_turn = float(np.arccos(np.clip(
                0.5 * (np.trace(relative) - 1.0),
                -1.0,
                1.0,
            )))
            direction_error = float(np.arccos(np.clip(
                gripper_candidate[:, 2] @ preferred_tool_z,
                -1.0,
                1.0,
            )))
            candidates.append((
                direction_error + 0.15 * wrist_turn,
                object_candidate,
                gripper_candidate,
            ))
        _, inserted_object_rotation, inserted_gripper_rotation = min(
            candidates,
            key=lambda item: item[0],
        )
        is_cylinder = geom_type == int(
            mujoco.mjtGeom.mjGEOM_CYLINDER,
        )
        if is_cylinder and object_starts_upright:
            # A vertical cylinder is already aligned with an upward-facing
            # circular hole.  Preserve its grasped axial angle instead of
            # adding an arbitrary, visually meaningless twist about local Z.
            inserted_object_rotation = object_rotation.copy()
            inserted_gripper_rotation = pick.grasp_rotation.copy()
        # Insert as deeply as the 6 cm blind tunnel and the far-end grasp
        # permit.  Stop just before the closed bottom and keep the finger
        # grasp centre clear of the rim instead of using a shallow fraction.
        insertion_depth = min(
            _CAVITY_DEPTH_M - _CAVITY_BOTTOM_CLEARANCE_M,
            (
                half_length
                + abs(grasp_rear_fraction * length)
                - (
                    _UP_GRIPPER_RIM_CLEARANCE_M
                    if outward_normal[2] > 0.7
                    else _SIDE_GRIPPER_RIM_CLEARANCE_M
                )
            ),
        )
        # Panda's closed asymmetric fingers hold a horizontal object a few
        # millimetres toward -X under load.  Compensate that repeatable grip
        # offset before entering an upward-facing hole; the cavity geometry
        # and the success tolerance remain unchanged.
        alignment_bias = np.zeros(3, dtype=float)
        if outward_normal[2] > 0.7:
            # The pad deflection is repeatable for each cross-section size.
            # Compensate it in the table-X direction without changing the
            # requested hole pose or using feedback from the rollout.
            if object_starts_upright:
                if (
                    geom_type == int(mujoco.mjtGeom.mjGEOM_BOX)
                    and cross_width > 0.045
                ):
                    # The wide square bar's load deflection follows the
                    # selected equivalent gripper orientation, not world XY.
                    alignment_bias = inserted_gripper_rotation @ np.array(
                        (-0.0060, -0.0023, 0.0),
                    )
                else:
                    alignment_bias[0] = (
                        -0.002
                        if 0.025 < cross_width < 0.040
                        else (-0.003 if cross_width >= 0.040 else 0.0)
                    )
            else:
                alignment_bias[0] = (
                    0.011
                    if 0.025 < cross_width < 0.040
                    else (0.005 if cross_width >= 0.040 else 0.0)
                )
        elif np.isclose(cross_width, 0.040):
            # The 4 cm side-facing variant settles a few millimetres toward
            # world -X/-Z in the asymmetric Panda grasp.  Centre it on the
            # physical 4.5 cm tunnel before the straight insertion segment.
            alignment_bias[0] = 0.0025
            alignment_bias[2] = 0.0025
        inserted_object_position = (
            site_position
            + alignment_bias
            + outward_normal * (half_length - insertion_depth)
        )

        def gripper_for_object(
            desired_object_position: np.ndarray,
        ) -> np.ndarray:
            return (
                desired_object_position
                - inserted_gripper_rotation @ object_in_gripper_position
            )

        above_object = (
            inserted_object_position
            + outward_normal * _TRANSFER_CLEARANCE_M
        )
        preinsert_object = (
            inserted_object_position
            + outward_normal * _PREINSERT_CLEARANCE_M
        )
        above_gripper = gripper_for_object(above_object)
        preinsert_gripper = gripper_for_object(preinsert_object)
        inserted_gripper = gripper_for_object(inserted_object_position)
        transfer_limits = _limited(limits, _TRANSFER_SPEED_M_S)
        align_limits = _limited(limits, _ALIGN_SPEED_M_S)
        insert_limits = _limited(limits, _INSERT_SPEED_M_S)
        orient_limits = MotionLimits(
            linear_speed_m_s=transfer_limits.linear_speed_m_s,
            angular_speed_rad_s=min(
                limits.angular_speed_rad_s,
                0.50,
            ),
            command_rate_hz=limits.command_rate_hz,
        )

        frames = list(pick.frames)
        safe_transfer_gripper = pick.lifted_position.copy()
        transfer_start = safe_transfer_gripper
        transfer_start_rotation = pick.grasp_rotation
        needs_transfer_orientation = not (
            is_cylinder and object_starts_upright
        )
        if outward_normal[2] <= 0.7:
            high_above_gripper = above_gripper.copy()
            high_above_gripper[2] = safe_transfer_gripper[2]
            append_blended_motion(
                frames,
                safe_transfer_gripper,
                transfer_start_rotation,
                high_above_gripper,
                inserted_gripper_rotation,
                True,
                (
                    "Move and orient object above matching hole"
                    if needs_transfer_orientation
                    else "Traverse above matching holder"
                ),
                (
                    orient_limits
                    if needs_transfer_orientation
                    else transfer_limits
                ),
                start_speed_m_s=min(
                    0.0525, transfer_limits.linear_speed_m_s,
                ),
                end_speed_m_s=0.0,
            )
            transfer_start = high_above_gripper
            transfer_start_rotation = inserted_gripper_rotation
        append_blended_motion(
            frames,
            transfer_start,
            transfer_start_rotation,
            above_gripper,
            inserted_gripper_rotation,
            True,
            (
                "Move and orient object above matching hole"
                if needs_transfer_orientation
                else "Move object above matching hole"
            ),
            (
                orient_limits
                if needs_transfer_orientation
                else transfer_limits
            ),
            start_speed_m_s=0.0,
            end_speed_m_s=align_limits.linear_speed_m_s,
        )
        append_blended_motion(
            frames,
            above_gripper,
            inserted_gripper_rotation,
            preinsert_gripper,
            inserted_gripper_rotation,
            True,
            "Align object with hole",
            align_limits,
            start_speed_m_s=align_limits.linear_speed_m_s,
            end_speed_m_s=insert_limits.linear_speed_m_s,
        )
        append_blended_motion(
            frames,
            preinsert_gripper,
            inserted_gripper_rotation,
            inserted_gripper,
            inserted_gripper_rotation,
            True,
            "Insert object",
            insert_limits,
            start_speed_m_s=insert_limits.linear_speed_m_s,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            inserted_gripper,
            inserted_gripper_rotation,
            True,
            "Object inserted",
            5,
        )
        return tuple(frames)
