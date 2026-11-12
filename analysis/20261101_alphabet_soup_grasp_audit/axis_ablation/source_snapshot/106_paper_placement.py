"""Privileged evaluation for placing one cube inside its paper decal."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .predicates import (
    PickPlaceContactMonitor,
    body_free_joint_speeds,
    box_world_corners,
    is_blocking_collision,
    speeds_within_limits,
)


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Paper placement evaluator object is missing: {name}")
    return int(object_id)


def _active_object(model: mujoco.MjModel) -> str:
    names: list[str] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith("paper_object_"):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            names.append(name)
    if len(names) != 1:
        raise RuntimeError(
            f"Expected one active paper-placement object, found {names}",
        )
    return names[0]


def _object_world_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
        return box_world_corners(model, data, geom_id)
    if int(model.geom_type[geom_id]) != int(
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    ):
        raise ValueError("Paper placement supports boxes and cylinders only")
    radius = float(model.geom_size[geom_id, 0])
    half_length = float(model.geom_size[geom_id, 1])
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
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


class PaperPlacementEvaluator:
    """Require the complete cube footprint to lie within the white square."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
        *,
        stable_frames: int = 5,
        extra_distractor_body_names: tuple[str, ...] = (),
        object_body_name: str | None = None,
        object_geom_name: str | None = None,
        decal_body_name: str | None = None,
        decal_geom_name: str | None = None,
        object_kind: str | None = None,
        upright: bool | None = None,
    ) -> None:
        self.model = model
        self.cube_name = object_body_name or _active_object(model)
        if object_body_name is None:
            variant_name = self.cube_name.removeprefix("paper_object_")
            inferred_kind, self.size_name, *pose_parts = variant_name.split("_")
            inferred_upright = pose_parts == ["upright"]
            if pose_parts not in ([], ["upright"]):
                raise RuntimeError(f"Unknown paper-placement pose: {variant_name}")
            self.object_kind = object_kind or inferred_kind
            self.upright = inferred_upright if upright is None else bool(upright)
            self.decal_name = decal_body_name or f"paper_decal_{variant_name}"
        else:
            if object_kind is None or upright is None or decal_body_name is None:
                raise ValueError(
                    "Explicit paper objects require object_kind, upright and decal_body_name",
                )
            self.size_name = "custom"
            self.object_kind = object_kind
            self.upright = bool(upright)
            self.decal_name = decal_body_name
        self.cube_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.cube_name,
        )
        self.cube_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            object_geom_name or f"{self.cube_name}_geom",
        )
        self.decal_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.decal_name,
        )
        self.decal_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            decal_geom_name or f"{self.decal_name}_geom",
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
        distractors = tuple(
            name
            for name in (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                )
                for body_id in range(model.nbody)
            )
            if name and name.startswith("paper_object_")
            and name != self.cube_name
        ) + tuple(extra_distractor_body_names)
        self.contacts = PickPlaceContactMonitor(
            model,
            target_body_name=self.cube_name,
            receptacle_body_name=self.decal_name,
            distractor_body_names=distractors,
            robot_root_name=robot_root,
            finger_body_names=fingers,
        )
        self.table_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        self.initial_z = (
            float(data.xpos[self.cube_body, 2])
            if data is not None else 0.0
        )
        self.stable_frames = int(stable_frames)
        self._stable_count = 0
        self._success = False
        self._grasp_offset_world: np.ndarray | None = None
        self._subgoals = {"placed": False}
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

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        del frame_index, time_s
        corners = _object_world_points(
            self.model, data, self.cube_geom,
        )
        decal_center = data.geom_xpos[self.decal_geom]
        decal_rotation = data.geom_xmat[self.decal_geom].reshape(3, 3)
        local_corners = (corners - decal_center) @ decal_rotation
        object_center_local = (
            data.geom_xpos[self.cube_geom] - decal_center
        ) @ decal_rotation
        decal_half = self.model.geom_size[self.decal_geom, :2]
        maximum_xy = np.max(np.abs(local_corners[:, :2]), axis=0)
        inside = bool(np.all(
            np.abs(object_center_local[:2]) <= decal_half + 0.0005,
        ))

        table_geom = int(self.model.body_geomadr[self.table_body])
        table_top = float(
            data.geom_xpos[table_geom, 2]
            + self.model.geom_size[table_geom, 2]
        )
        cube_bottom = float(np.min(corners[:, 2]))
        valid_height = table_top - 0.003 <= cube_bottom <= table_top + 0.010
        placed = inside and valid_height
        self._success = bool(placed and not is_blocking_collision(self.collision))
        self._subgoals["placed"] = self._success
        self._last_conditions = {
            "inside_paper_region": inside,
            "valid_height": bool(valid_height),
            "maximum_local_x_m": float(maximum_xy[0]),
            "maximum_local_y_m": float(maximum_xy[1]),
            "object_center_local_x_m": float(object_center_local[0]),
            "object_center_local_y_m": float(object_center_local[1]),
            "paper_half_x_m": float(decal_half[0]),
            "paper_half_y_m": float(decal_half[1]),
            "cube_bottom_z_m": cube_bottom,
            "table_top_z_m": table_top,
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
