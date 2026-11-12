"""Privileged-state evaluation for articulated push devices."""

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
        raise RuntimeError(f"Press-device evaluator is missing {name!r}")
    return int(object_id)


def _active_root(model: mujoco.MjModel, prefix: str) -> tuple[str, int]:
    active: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith(prefix) or name.endswith("_button"):
            continue
        subtree = _subtree(model, body_id)
        geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, list(subtree)))
        if np.any(model.geom_contype[geom_ids] != 0):
            active.append((name, body_id))
    if len(active) != 1:
        raise RuntimeError(
            f"Expected one active device with prefix {prefix!r}, "
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
                    model, mujoco.mjtObj.mjOBJ_BODY, finger_name,
                ),
            )
            for finger_name in finger_names
        ))
        return _subtree(model, int(root)), fingers
    raise RuntimeError("Cannot identify robot bodies for press-device task")


class PressDeviceEvaluator:
    """Require a physical button stroke and reject unintended arm collisions."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        root_prefix: str,
        press_fraction: float = 0.80,
    ) -> None:
        del data
        if not 0.0 < press_fraction <= 1.0:
            raise ValueError("press_fraction must be in (0, 1]")
        self.model = model
        self.root_name, self.root_id = _active_root(model, root_prefix)
        self.device_bodies = _subtree(model, self.root_id)
        self.button_geom = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.root_name}_button_geom",
        )
        self.button_joint = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{self.root_name}_button_joint",
        )
        self.button_qpos = int(model.jnt_qposadr[self.button_joint])
        self.button_travel = float(model.jnt_range[self.button_joint, 1])
        self.press_threshold = press_fraction * self.button_travel
        self.robot_bodies, self.finger_bodies = _robot_parts(model)
        table = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        self.table_bodies = _subtree(model, table)
        self.maximum_press_m = 0.0
        self.pressed = False
        self.safety = RobotSafetyMonitor(model)

    @property
    def collision(self) -> CollisionEvent | None:
        return self.safety.collision

    def _observe_press(self, data: mujoco.MjData) -> None:
        displacement = float(data.qpos[self.button_qpos])
        self.maximum_press_m = max(self.maximum_press_m, displacement)
        self.pressed = self.pressed or (
            displacement >= self.press_threshold
        )

    def _name(
        self,
        object_type: mujoco.mjtObj,
        object_id: int,
    ) -> str:
        return (
            mujoco.mj_id2name(self.model, object_type, object_id)
            or f"id_{object_id}"
        )

    def observe_contacts(
        self,
        data: mujoco.MjData,
        frame_index: int,
        time_s: float,
    ) -> None:
        self._observe_press(data)
        self.safety.observe(data, frame_index, time_s)

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        del gripper_closed, frame_index, time_s
        self._observe_press(data)

    @property
    def success(self) -> bool:
        return self.pressed and not is_blocking_collision(self.collision)

    @property
    def should_abort(self) -> bool:
        return is_blocking_collision(self.collision)

    def result(
        self,
        executor_finished: bool,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        success = self.success and executor_finished
        if is_blocking_collision(self.collision):
            reason = self.collision.category
        elif success:
            reason = "success"
        elif timed_out:
            reason = "timeout"
        elif not executor_finished:
            reason = "planner_incomplete"
        else:
            reason = "button_not_pressed"
        return {
            "success": success,
            "termination_reason": reason,
            "subgoals": {"pressed": self.pressed},
            "final_conditions": {
                "active_device": self.root_name,
                "maximum_press_m": self.maximum_press_m,
                "required_press_m": self.press_threshold,
                "press_fraction": (
                    self.maximum_press_m / self.button_travel
                    if self.button_travel > 0.0
                    else 0.0
                ),
            },
            "collision": (
                None
                if self.collision is None
                else self.collision.as_dict()
            ),
            "knocked_over": None,
        }


__all__ = ("PressDeviceEvaluator",)
