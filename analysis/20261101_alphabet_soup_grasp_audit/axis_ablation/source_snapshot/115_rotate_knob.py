"""Privileged evaluation for a clockwise quarter-turn knob task."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from ...simulation.robots import ROBOT_SPECS, RobotSpec
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
        raise RuntimeError(f"Rotate Knob evaluator is missing {name!r}")
    return int(object_id)


def _active_root(model: mujoco.MjModel) -> tuple[str, int]:
    active: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith("rotate_knob_") or name.endswith("_knob"):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            active.append((name, body_id))
    if len(active) != 1:
        raise RuntimeError(
            f"Expected one active rotate knob, found {[x[0] for x in active]}",
        )
    return active[0]


_FINGER_CONTACT_ROOTS = {
    "panda": ("left_finger", "right_finger"),
    "xarm7": ("left_finger", "right_finger"),
    "ur5e": ("rb_left_pad", "rb_right_pad"),
}


def _active_robot_spec(model: mujoco.MjModel) -> RobotSpec:
    matches = [
        spec
        for spec in ROBOT_SPECS.values()
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site,
        ) >= 0
        and all(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
            ) >= 0
            for joint_name in spec.finger_joint_names
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot identify robot for Rotate Knob: "
            f"{[spec.uid for spec in matches]}",
        )
    return matches[0]


def _robot_parts(
    model: mujoco.MjModel,
    spec: RobotSpec,
) -> tuple[set[int], set[int]]:
    root = _required_id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.base_body,
    )
    fingers = set().union(*(
        _subtree(
            model,
            _required_id(
                model, mujoco.mjtObj.mjOBJ_BODY, finger_name,
            ),
        )
        for finger_name in _FINGER_CONTACT_ROOTS[spec.uid]
    ))
    return _subtree(model, root), fingers


class RotateKnobEvaluator:
    """Require a final clockwise rotation between 70 and 110 degrees."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
        *,
        target_angle_deg: float = 90.0,
        tolerance_deg: float = 20.0,
    ) -> None:
        self.model = model
        self.robot_spec = _active_robot_spec(model)
        self.root_name, self.root_body = _active_root(model)
        self.knob_body = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{self.root_name}_knob",
        )
        self.knob_geom = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.root_name}_geom",
        )
        self.ee_site = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            self.robot_spec.ee_site,
        )
        self.finger_joints = tuple(
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.robot_spec.finger_joint_names
        )
        self.finger_qpos = tuple(
            int(model.jnt_qposadr[joint]) for joint in self.finger_joints
        )
        self.joint = _required_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{self.root_name}_joint",
        )
        self.qpos_id = int(model.jnt_qposadr[self.joint])
        self.initial_angle = (
            float(data.qpos[self.qpos_id])
            if data is not None else float(model.qpos0[self.qpos_id])
        )
        self.target_angle = math.radians(target_angle_deg)
        self.tolerance = math.radians(tolerance_deg)
        self.robot_bodies, self.finger_bodies = _robot_parts(
            model, self.robot_spec,
        )
        table = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        self.table_bodies = _subtree(model, table)
        self.device_bodies = _subtree(model, self.root_body)
        self.safety = RobotSafetyMonitor(model)
        self._success = False
        self._subgoals = {"rotated_clockwise": False}
        self._last_conditions: dict[str, Any] = {}

    def _finger_knob_contacts(
        self, data: mujoco.MjData,
    ) -> set[int]:
        result: set[int] = set()
        contact_margin = float(self.model.geom_margin[self.knob_geom])
        for index in range(data.ncon):
            contact = data.contact[index]
            if float(contact.dist) > contact_margin + 1e-6:
                continue
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            if self.knob_geom not in (geom_a, geom_b):
                continue
            bodies = {
                int(self.model.geom_bodyid[geom_a]),
                int(self.model.geom_bodyid[geom_b]),
            }
            result.update(bodies & self.finger_bodies)
        return result

    def _name(
        self,
        object_type: mujoco.mjtObj,
        object_id: int,
    ) -> str:
        return (
            mujoco.mj_id2name(self.model, object_type, object_id)
            or f"id_{object_id}"
        )

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
        del gripper_closed, frame_index, time_s
        clockwise = self.initial_angle - float(data.qpos[self.qpos_id])
        within_target = (
            abs(clockwise - self.target_angle) <= self.tolerance
        )
        self._success = bool(
            within_target and not is_blocking_collision(self.collision)
        )
        self._subgoals["rotated_clockwise"] = self._success
        self._last_conditions = {
            "active_knob": self.root_name,
            "initial_angle_deg": math.degrees(self.initial_angle),
            "clockwise_rotation_deg": math.degrees(clockwise),
            "target_rotation_deg": math.degrees(self.target_angle),
            "tolerance_deg": math.degrees(self.tolerance),
            "within_target": within_target,
        }

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
        elif not self._subgoals["rotated_clockwise"]:
            reason = "clockwise_rotation_incomplete"
        else:
            reason = "final_angle_out_of_tolerance"
        return {
            "success": success,
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


__all__ = ("RotateKnobEvaluator",)
