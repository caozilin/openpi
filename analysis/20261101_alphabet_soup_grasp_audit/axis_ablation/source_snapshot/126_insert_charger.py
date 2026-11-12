"""Simple pick-align-insert trajectory for rectangular charger heads."""

from __future__ import annotations

import mujoco
import numpy as np

from .pick_place import (
    GripperVerticalEnvelope,
    PickBox,
    append_blended_motion,
    append_dwell,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame
from ...simulation.robot import RobotHandles


_TRANSFER_CLEARANCE_M = 0.085
_PREINSERT_CLEARANCE_M = 0.018
_INSERT_SURFACE_GAP_M = 0.001
_TRANSFER_SPEED_M_S = 0.120
_ALIGN_SPEED_M_S = 0.050
_INSERT_SPEED_M_S = 0.025
_NON_PANDA_GRASP_LIFT_M = {
    "xarm7": 0.025,
    "ur5e": 0.018,
}


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


def _limited(limits: MotionLimits, linear_speed_m_s: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, linear_speed_m_s),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


class InsertCharger:
    """Top-pick a charger, align its pins, then end at full insertion."""

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
        strip_name, _ = _active_body(self.model, "insert_strip_")
        charger_name, charger_body_id = _active_body(
            self.model, "charger_",
        )
        pin_count = 2 if charger_name.endswith("2pin") else 3
        core_name = f"{charger_name}_body_geom"
        core_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, core_name,
        )
        site_name = f"{strip_name}_socket_{pin_count}pin_site"
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        if core_id < 0 or site_id < 0:
            raise RuntimeError(
                f"Insertion model is missing {core_name!r} or {site_name!r}",
            )

        gripper_envelope = gripper_vertical_envelope(
            self.model, self.handles,
        )
        grasp_lift = _NON_PANDA_GRASP_LIFT_M.get(
            self.handles.spec.uid, 0.0,
        )
        if grasp_lift:
            gripper_envelope = GripperVerticalEnvelope(
                gripper_envelope.base_above_ee_m - grasp_lift,
                gripper_envelope.contact_upper_above_ee_m - grasp_lift,
            )
        pick = PickBox(
            self.model,
            body_name=charger_name,
            geom_name=core_name,
            approach_clearance_m=0.090,
            lift_clearance_m=0.060,
            gripper_envelope=gripper_envelope,
        ).plan(
            data,
            current_position,
            current_rotation,
            limits,
        )

        object_position = data.xpos[charger_body_id].copy()
        object_rotation = data.xmat[charger_body_id].reshape(3, 3).copy()
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
        # Local charger +Z is the pin direction.  This proper rotation maps
        # it into the socket while keeping local Y aligned with the outlet.
        inserted_object_rotation = (
            site_rotation @ np.diag((-1.0, 1.0, -1.0))
        )
        inserted_gripper_rotation = (
            inserted_object_rotation @ object_in_gripper_rotation.T
        )
        core_half_depth = float(self.model.geom_size[core_id, 2])
        inserted_object_position = (
            site_position
            + outward_normal * (core_half_depth + _INSERT_SURFACE_GAP_M)
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
        frames = list(pick.frames)
        append_blended_motion(
            frames,
            pick.lifted_position,
            pick.grasp_rotation,
            above_gripper,
            inserted_gripper_rotation,
            True,
            "Move charger above socket",
            transfer_limits,
            start_speed_m_s=min(
                0.0525, transfer_limits.linear_speed_m_s,
            ),
            end_speed_m_s=align_limits.linear_speed_m_s,
        )
        append_blended_motion(
            frames,
            above_gripper,
            inserted_gripper_rotation,
            preinsert_gripper,
            inserted_gripper_rotation,
            True,
            "Align charger pins",
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
            "Insert charger plug",
            insert_limits,
            start_speed_m_s=insert_limits.linear_speed_m_s,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            inserted_gripper,
            inserted_gripper_rotation,
            True,
            "Charger fully inserted",
            4,
        )
        return tuple(frames)
