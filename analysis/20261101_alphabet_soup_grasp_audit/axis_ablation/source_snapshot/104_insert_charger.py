"""Privileged-state evaluation for charger-plug insertion."""

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
        raise RuntimeError(f"Insert-charger evaluator is missing {name!r}")
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
        root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, root_name,
        )
        if root_id < 0:
            continue
        finger_bodies = set().union(*(
            _subtree(
                model,
                _required_id(
                    model, mujoco.mjtObj.mjOBJ_BODY, finger_name,
                ),
            )
            for finger_name in finger_names
        ))
        return _subtree(model, int(root_id)), finger_bodies
    raise RuntimeError("Cannot identify robot bodies for charger insertion")


class InsertChargerEvaluator:
    """Require pose-aligned physical insertion while the charger is held."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        stable_frames: int = 3,
        lateral_tolerance_m: float = 0.007,
        surface_gap_tolerance_m: float = 0.008,
        angular_tolerance_rad: float = np.deg2rad(12.0),
    ) -> None:
        self.model = model
        self.strip_name, self.strip_body = _active_body(
            model, "insert_strip_",
        )
        self.charger_name, self.charger_body = _active_body(
            model, "charger_",
        )
        self.pin_count = 2 if self.charger_name.endswith("2pin") else 3
        self.core_geom = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.charger_name}_body_geom",
        )
        self.target_site = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"{self.strip_name}_socket_{self.pin_count}pin_site",
        )
        self.strip_bodies = _subtree(model, self.strip_body)
        self.charger_bodies = _subtree(model, self.charger_body)
        del stable_frames, data
        self.safety = RobotSafetyMonitor(model)
        self.lateral_tolerance_m = float(lateral_tolerance_m)
        self.surface_gap_tolerance_m = float(surface_gap_tolerance_m)
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
        charger_position = data.xpos[self.charger_body]
        charger_rotation = data.xmat[self.charger_body].reshape(3, 3)
        site_position = data.site_xpos[self.target_site]
        site_rotation = data.site_xmat[self.target_site].reshape(3, 3)
        outward_normal = site_rotation[:, 2]
        pin_axis = charger_rotation[:, 2]
        core_half_depth = float(self.model.geom_size[self.core_geom, 2])
        core_front = (
            charger_position + pin_axis * core_half_depth
        )
        delta = core_front - site_position
        surface_gap = float(delta @ outward_normal)
        lateral_vector = delta - surface_gap * outward_normal
        lateral_error = float(np.linalg.norm(lateral_vector))
        alignment_angle = float(np.arccos(np.clip(
            pin_axis @ (-outward_normal), -1.0, 1.0,
        )))
        aligned = bool(
            lateral_error <= self.lateral_tolerance_m
            and alignment_angle <= self.angular_tolerance_rad
        )
        inserted = bool(
            aligned
            and -0.004 <= surface_gap <= self.surface_gap_tolerance_m
        )
        del gripper_closed
        self._success = bool(inserted and not is_blocking_collision(self.collision))
        self._subgoals["inserted"] = self._success
        self._last_conditions = {
            "pin_count": self.pin_count,
            "lateral_error_m": lateral_error,
            "lateral_tolerance_m": self.lateral_tolerance_m,
            "surface_gap_m": surface_gap,
            "surface_gap_tolerance_m": self.surface_gap_tolerance_m,
            "alignment_angle_rad": alignment_angle,
            "angular_tolerance_rad": self.angular_tolerance_rad,
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
