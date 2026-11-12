"""Authoritative evaluation for placing a lying cylinder in a narrow box."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .predicates import (
    PickPlaceContactMonitor,
    body_free_joint_speeds,
    is_blocking_collision,
    speeds_within_limits,
)


_BOX_HEIGHT_M = 0.052
_BOX_FLOOR_TOP_M = 0.007
_MIN_INSERTION_DEPTH_M = 0.005


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Cylinder-to-box evaluator object is missing: {name}")
    return int(object_id)


def _active_pair(
    model: mujoco.MjModel,
    object_prefix: str,
    box_prefix: str,
) -> tuple[str, str]:
    objects: list[str] = []
    boxes: list[str] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id,
        ) or ""
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if not np.any(model.geom_contype[geom_ids] != 0):
            continue
        if name.startswith(object_prefix):
            objects.append(name)
        elif name.startswith(box_prefix):
            boxes.append(name)
    if len(objects) != 1 or len(boxes) != 1:
        raise RuntimeError(
            "Expected one active narrow-box task pair, found "
            f"{objects=} and {boxes=}",
        )
    size = objects[0].removeprefix(object_prefix)
    if boxes[0] != f"{box_prefix}{size}":
        raise RuntimeError("Active object and narrow box sizes do not match")
    return objects[0], boxes[0]


def _object_world_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half_extents = model.geom_size[geom_id, :3]
        local = np.array([
            (x, y, z)
            for x in (-half_extents[0], half_extents[0])
            for y in (-half_extents[1], half_extents[1])
            for z in (-half_extents[2], half_extents[2])
        ])
    else:
        radius = float(model.geom_size[geom_id, 0])
        half_length = float(model.geom_size[geom_id, 1])
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        local = np.array([
            (
                radius * np.cos(angle),
                radius * np.sin(angle),
                axial,
            )
            for axial in (-half_length, half_length)
            for angle in angles
        ])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return data.geom_xpos[geom_id] + local @ rotation.T


class CylinderNarrowBoxEvaluator:
    """Require a grasped, lifted, released and stable object inside its box."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        stable_frames: int = 5,
        linear_speed_limit_m_s: float = 0.03,
        angular_speed_limit_rad_s: float = 0.30,
        object_prefix: str = "narrow_box_cylinder_",
        box_prefix: str = "narrow_box_",
        object_label: str = "cylinder",
    ) -> None:
        if stable_frames <= 0:
            raise ValueError("stable_frames must be positive")
        self.model = model
        self.object_label = object_label
        self.cylinder_name, self.box_name = _active_pair(
            model, object_prefix, box_prefix,
        )
        self.cylinder_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.cylinder_name,
        )
        self.cylinder_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.cylinder_name}_geom",
        )
        self.box_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.box_name,
        )
        self.interior_site = _required_id(
            model, mujoco.mjtObj.mjOBJ_SITE,
            f"{self.box_name}_interior_site",
        )
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "link0",
        ) >= 0:
            robot_root = "link0"
            fingers = ("left_finger", "right_finger")
        elif mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "link_base",
        ) >= 0:
            robot_root = "link_base"
            fingers = ("left_finger", "right_finger")
        else:
            robot_root = "base"
            fingers = ("rb_left_pad", "rb_right_pad")
        self.contacts = PickPlaceContactMonitor(
            model,
            target_body_name=self.cylinder_name,
            receptacle_body_name=self.box_name,
            distractor_body_names=(),
            robot_root_name=robot_root,
            finger_body_names=fingers,
        )
        self.initial_z = (
            float(data.geom_xpos[self.cylinder_geom, 2])
            if data is not None else 0.0
        )
        self.stable_frames = int(stable_frames)
        self.linear_speed_limit_m_s = float(linear_speed_limit_m_s)
        self.angular_speed_limit_rad_s = float(angular_speed_limit_rad_s)
        self._ever_grasped = False
        self._ever_lifted = False
        self._stable_count = 0
        self._success = False
        self._subgoals = {
            "grasped": False,
            "lifted": False,
            "placed": False,
        }
        self._last_conditions: dict[str, Any] = {}

    @property
    def collision(self):
        return self.contacts.collision

    @property
    def success(self) -> bool:
        return self._success and not is_blocking_collision(self.collision)

    @property
    def should_abort(self) -> bool:
        return is_blocking_collision(self.collision)

    def observe_contacts(
        self,
        data: mujoco.MjData,
        frame_index: int,
        time_s: float,
    ) -> None:
        self.contacts.observe(data, frame_index, time_s)
        if len(self.contacts.finger_target_contacts(data)) >= 2:
            self._ever_grasped = True
            self._subgoals["grasped"] = True

    def _support_contact(self, data: mujoco.MjData) -> bool:
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > 0.001:
                continue
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            if self.cylinder_body in bodies and self.box_body in bodies:
                return True
        return False

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        del frame_index, time_s
        if float(data.geom_xpos[self.cylinder_geom, 2]) >= self.initial_z + 0.030:
            self._ever_lifted = True
            self._subgoals["lifted"] = True

        points = _object_world_points(
            self.model, data, self.cylinder_geom,
        )
        box_center = data.xpos[self.box_body].copy()
        box_rotation = data.xmat[self.box_body].reshape(3, 3)
        local_points = (points - box_center) @ box_rotation
        interior_half_xy = self.model.site_size[self.interior_site, :2]
        maximum_xy = np.max(np.abs(local_points[:, :2]), axis=0)
        local_minimum = np.min(local_points, axis=0)
        local_maximum = np.max(local_points, axis=0)
        cylinder_bottom = float(local_minimum[2])
        cylinder_top = float(local_maximum[2])
        cylinder_center_local = (
            data.geom_xpos[self.cylinder_geom] - box_center
        ) @ box_rotation
        cylinder_center_z = float(cylinder_center_local[2])
        center_of_mass_local = (
            data.xipos[self.cylinder_body] - box_center
        ) @ box_rotation
        center_inside_xy = bool(np.all(
            np.abs(center_of_mass_local[:2]) <= interior_half_xy,
        ))
        center_of_mass_inside = bool(
            center_inside_xy
            and _BOX_FLOOR_TOP_M <= center_of_mass_local[2] <= _BOX_HEIGHT_M
        )
        insertion_depth = _BOX_HEIGHT_M - cylinder_bottom
        overlap_x = min(local_maximum[0], interior_half_xy[0]) - max(
            local_minimum[0], -interior_half_xy[0],
        )
        overlap_y = min(local_maximum[1], interior_half_xy[1]) - max(
            local_minimum[1], -interior_half_xy[1],
        )
        overlap_z = min(cylinder_top, _BOX_HEIGHT_M) - max(
            cylinder_bottom, _BOX_FLOOR_TOP_M,
        )
        partly_inside = bool(
            center_of_mass_inside
            and overlap_x >= _MIN_INSERTION_DEPTH_M
            and overlap_y >= _MIN_INSERTION_DEPTH_M
            and overlap_z >= _MIN_INSERTION_DEPTH_M
        )
        on_floor = bool(
            _BOX_FLOOR_TOP_M - 0.003
            <= cylinder_bottom
            <= _BOX_FLOOR_TOP_M + 0.008
        )
        center_below_rim = cylinder_center_z <= _BOX_HEIGHT_M + 0.002
        support_contact = self._support_contact(data)
        finger_contacts = self.contacts.finger_target_contacts(data)
        released = not gripper_closed and not finger_contacts
        linear_speed, angular_speed = body_free_joint_speeds(
            self.model, data, self.cylinder_body,
        )
        stable = speeds_within_limits(
            linear_speed,
            angular_speed,
            self.linear_speed_limit_m_s,
            self.angular_speed_limit_rad_s,
        )
        valid_final = bool(
            self._ever_grasped
            and self._ever_lifted
            and partly_inside
            and released
            and stable
            and not is_blocking_collision(self.collision)
        )
        self._stable_count = self._stable_count + 1 if valid_final else 0
        self._success = self._stable_count >= self.stable_frames
        self._subgoals["placed"] = self._success
        self._last_conditions = {
            "inside_box": partly_inside,
            "center_inside_box_xy": center_inside_xy,
            "center_of_mass_inside_box": center_of_mass_inside,
            "center_of_mass_local_x_m": float(center_of_mass_local[0]),
            "center_of_mass_local_y_m": float(center_of_mass_local[1]),
            "center_of_mass_local_z_m": float(center_of_mass_local[2]),
            "insertion_depth_m": float(insertion_depth),
            "minimum_insertion_depth_m": _MIN_INSERTION_DEPTH_M,
            "interior_overlap_x_m": float(overlap_x),
            "interior_overlap_y_m": float(overlap_y),
            "interior_overlap_z_m": float(overlap_z),
            "on_box_floor": on_floor,
            # Retain floor/support diagnostics for dataset analysis; partial
            # insertion no longer requires either condition.
            "center_below_rim": bool(center_below_rim),
            "support_contact": support_contact,
            "released": released,
            "stable": stable,
            "stable_frames": self._stable_count,
            "maximum_local_x_m": float(maximum_xy[0]),
            "maximum_local_y_m": float(maximum_xy[1]),
            "interior_half_x_m": float(interior_half_xy[0]),
            "interior_half_y_m": float(interior_half_xy[1]),
            "cylinder_bottom_local_z_m": cylinder_bottom,
            "cylinder_center_local_z_m": cylinder_center_z,
            "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
        }

    def result(
        self,
        executor_finished: bool,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        task_success = self.success and executor_finished
        if is_blocking_collision(self.collision):
            reason = self.collision.category
        elif task_success:
            reason = "success"
        elif timed_out:
            reason = "timeout"
        elif not executor_finished:
            reason = "planner_incomplete"
        elif not self._ever_grasped:
            reason = "grasp_failed"
        elif not self._ever_lifted:
            reason = "lift_failed"
        else:
            reason = "placement_failed"
        return {
            "success": task_success,
            "termination_reason": reason,
            "subgoals": dict(self._subgoals),
            "final_conditions": dict(self._last_conditions),
            "collision": (
                None if self.collision is None else self.collision.as_dict()
            ),
            "knocked_over": None,
        }


class BoxNarrowBoxEvaluator(CylinderNarrowBoxEvaluator):
    """Evaluate placement of a lying rectangular block in a narrow box."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model,
            data,
            object_prefix="narrow_box_block_",
            box_prefix="block_narrow_box_",
            object_label="rectangular block",
            **kwargs,
        )


__all__ = ("BoxNarrowBoxEvaluator", "CylinderNarrowBoxEvaluator")
