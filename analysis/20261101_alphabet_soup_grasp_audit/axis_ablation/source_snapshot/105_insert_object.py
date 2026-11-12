"""Privileged-state evaluation for matched-shape object insertion."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .predicates import RobotSafetyMonitor, is_blocking_collision
from .types import CollisionEvent


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(model.nbody):
        current = candidate
        while current > root:
            current = int(model.body_parentid[current])
        if current == root:
            result.add(candidate)
    return result


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Insert-object evaluator is missing {name!r}")
    return int(object_id)


def _active_body(model: mujoco.MjModel, prefix: str) -> tuple[str, int]:
    active: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith(prefix):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            active.append((name, body_id))
    if len(active) != 1:
        raise RuntimeError(
            f"Expected one active body with prefix {prefix!r}, "
            f"found {[name for name, _ in active]}",
        )
    return active[0]


def _robot_parts(
    model: mujoco.MjModel,
) -> tuple[set[int], set[int]]:
    choices = (
        ("link0", ("left_finger", "right_finger")),
        ("link_base", ("left_finger", "right_finger")),
        ("base", ("rb_left_pad", "rb_right_pad")),
    )
    for root_name, finger_names in choices:
        root = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, root_name,
        )
        if root < 0:
            continue
        fingers = set().union(*(
            _subtree(
                model,
                _required_id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name,
                ),
            )
            for name in finger_names
        ))
        return _subtree(model, int(root)), fingers
    raise RuntimeError("Cannot identify robot bodies for object insertion")


class InsertObjectEvaluator:
    """Require a held, aligned object tip to enter the matching cavity."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        stable_frames: int = 3,
        lateral_tolerance_m: float = 0.004,
        angular_tolerance_rad: float = np.deg2rad(12.0),
        depth_fraction: float = 0.15,
    ) -> None:
        self.model = model
        self.object_name, self.object_body = _active_body(
            model, "insert_object_",
        )
        self.container_name, self.container_body = _active_body(
            model, "insert_container_",
        )
        self.object_geom = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.object_name}_geom",
        )
        self.hole_site = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"{self.container_name}_hole_site",
        )
        geom_type = int(model.geom_type[self.object_geom])
        self.half_length = float(
            model.geom_size[self.object_geom, 2]
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX)
            else model.geom_size[self.object_geom, 1]
        )
        self.required_depth = 2.0 * self.half_length * float(
            depth_fraction,
        )
        self.object_bodies = _subtree(model, self.object_body)
        self.container_bodies = _subtree(model, self.container_body)
        del stable_frames
        self.safety = RobotSafetyMonitor(model)
        self.lateral_tolerance_m = float(lateral_tolerance_m)
        self.angular_tolerance_rad = float(angular_tolerance_rad)
        self._success = False
        self._subgoals = {"inserted": False}
        self._last_conditions: dict[str, Any] = {}

    @property
    def collision(self) -> CollisionEvent | None:
        return self.safety.collision

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
        self.safety.observe(data, frame_index, time_s)

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        del frame_index, time_s
        object_position = data.xpos[self.object_body]
        object_rotation = data.xmat[self.object_body].reshape(3, 3)
        axis = object_rotation[:, 2]
        hole_position = data.site_xpos[self.hole_site]
        hole_rotation = data.site_xmat[self.hole_site].reshape(3, 3)
        outward_normal = hole_rotation[:, 2]
        axis_dot_outward = float(axis @ outward_normal)
        insertion_end_sign = 1.0 if axis_dot_outward >= 0.0 else -1.0
        tip = (
            object_position
            - insertion_end_sign * axis * self.half_length
        )
        delta = tip - hole_position
        signed_gap = float(delta @ outward_normal)
        insertion_depth = -signed_gap
        lateral = delta - signed_gap * outward_normal
        lateral_error = float(np.linalg.norm(lateral))
        alignment_angle = float(np.arccos(np.clip(
            abs(axis_dot_outward), -1.0, 1.0,
        )))
        aligned = bool(
            lateral_error <= self.lateral_tolerance_m
            and alignment_angle <= self.angular_tolerance_rad
        )
        inserted = bool(
            aligned
            and insertion_depth >= self.required_depth
            and insertion_depth <= 0.105
        )
        del gripper_closed
        self._success = bool(inserted and not is_blocking_collision(self.collision))
        self._subgoals["inserted"] = self._success
        self._last_conditions = {
            "lateral_error_m": lateral_error,
            "lateral_vector_world_m": tuple(
                float(value) for value in lateral
            ),
            "lateral_tolerance_m": self.lateral_tolerance_m,
            "alignment_angle_rad": alignment_angle,
            "angular_tolerance_rad": self.angular_tolerance_rad,
            "insertion_depth_m": insertion_depth,
            "required_depth_m": self.required_depth,
        }

    def result(
        self,
        executor_finished: bool,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        task_success = bool(self.success and executor_finished)
        if is_blocking_collision(self.collision):
            reason = self.collision.category
        elif task_success:
            reason = "success"
        elif timed_out:
            reason = "timeout"
        elif not executor_finished:
            reason = "planner_incomplete"
        else:
            reason = "insertion_failed"
        return {
            "success": task_success,
            "termination_reason": reason,
            "subgoals": dict(self._subgoals),
            "final_conditions": dict(self._last_conditions),
            "collision": (
                None
                if self.collision is None
                else self.collision.as_dict()
            ),
            "knocked_over": None,
        }
