"""Privileged evaluation for fruit-to-plate and fruit-to-bowl tasks."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .predicates import PickPlaceContactMonitor, is_blocking_collision
from ..planning.pick_place import _geom_world_vertical_half_extent
from ..catalog import FRUIT_TRAY_SEQUENCE_TASK_SPECS


def _active_name(model: mujoco.MjModel, prefix: str) -> str:
    names: list[str] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith(prefix):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            names.append(name)
    if len(names) != 1:
        raise RuntimeError(
            f"Expected one active body with prefix {prefix!r}, found {names}",
        )
    return names[0]


def _required_body(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, name,
    )
    if body_id < 0:
        raise RuntimeError(f"Fruit evaluator body is missing: {name}")
    return int(body_id)


class FruitReceptacleEvaluator:
    """Judge only the fruit's terminal center/contact with its receptacle."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
        *,
        fruit: str,
        receptacle: str,
        fruit_body_name: str | None = None,
        receptacle_body_name: str | None = None,
        distractor_body_names: tuple[str, ...] | None = None,
        plate_radius_limit_m: float = 0.105,
        bowl_radius_limit_m: float = 0.070,
        tray_edge_margin_m: float = 0.010,
        stable_frames: int = 5,
    ) -> None:
        self.model = model
        self.fruit_name = fruit_body_name or f"fruit_task_{fruit}"
        self.fruit_body = _required_body(model, self.fruit_name)
        self.receptacle_kind = receptacle
        self.receptacle_name = (
            receptacle_body_name
            or _active_name(
                model,
                (
                    "tray_khaki_"
                    if receptacle == "tray"
                    else f"fruit_task_{receptacle}_"
                ),
            )
        )
        self.receptacle_body = _required_body(
            model, self.receptacle_name,
        )
        if distractor_body_names is None:
            other_fruits = tuple(
                body_name
                for name in ("banana", "strawberry", "apple")
                if name != fruit
                if mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name := f"fruit_task_{name}",
                ) >= 0
            )
            if receptacle == "tray":
                distractor_body_names = other_fruits
            else:
                other_receptacle = _active_name(
                    model,
                    "fruit_task_bowl_" if receptacle == "plate"
                    else "fruit_task_plate_",
                )
                distractor_body_names = (*other_fruits, other_receptacle)
        self.radius_limit_m = (
            float(plate_radius_limit_m)
            if receptacle == "plate" else float(bowl_radius_limit_m)
        )
        self.tray_edge_margin_m = float(tray_edge_margin_m)
        self.tray_interior_site = (
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"{self.receptacle_name}_interior_site",
            )
            if receptacle == "tray" else -1
        )
        if receptacle == "tray" and self.tray_interior_site < 0:
            raise RuntimeError(
                f"Tray interior site is missing: {self.receptacle_name}"
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
            target_body_name=self.fruit_name,
            receptacle_body_name=self.receptacle_name,
            distractor_body_names=distractor_body_names,
            robot_root_name=robot_root,
            finger_body_names=fingers,
        )
        del stable_frames, data
        self._success = False
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
        fruit_position = data.xpos[self.fruit_body]
        target_position = data.xpos[self.receptacle_body]
        radial_error = float(np.linalg.norm(
            fruit_position[:2] - target_position[:2],
        ))
        radius_limit = self.radius_limit_m
        if self.receptacle_kind == "tray":
            site_rotation = data.site_xmat[
                self.tray_interior_site
            ].reshape(3, 3)
            local_xy = (
                (fruit_position - data.site_xpos[self.tray_interior_site])
                @ site_rotation
            )[:2]
            tray_half_xy = self.model.site_size[
                self.tray_interior_site, :2
            ]
            over_target = bool(np.all(
                np.abs(local_xy)
                <= tray_half_xy - self.tray_edge_margin_m
            ))
        else:
            local_xy = None
            tray_half_xy = None
            over_target = radial_error <= radius_limit

        bottom_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM,
            f"{self.receptacle_name}_bottom",
        )
        if bottom_geom < 0:
            geom_ids = np.flatnonzero(
                self.model.geom_bodyid == self.receptacle_body,
            )
            collision_ids = geom_ids[self.model.geom_contype[geom_ids] != 0]
            if collision_ids.size == 0:
                raise RuntimeError(
                    f"Receptacle has no collision geometry: {self.receptacle_name}",
                )
            bottom_geom = int(collision_ids[np.argmin(
                data.geom_xpos[collision_ids, 2]
            )])
        base_top = float(
            data.geom_xpos[bottom_geom, 2]
            + _geom_world_vertical_half_extent(
                self.model, data, bottom_geom,
            )
        )
        if self.receptacle_kind == "bowl":
            geom_ids = np.flatnonzero(
                self.model.geom_bodyid == self.receptacle_body,
            )
            collision_ids = geom_ids[
                self.model.geom_contype[geom_ids] != 0
            ]
            rim_top = max(
                float(
                    data.geom_xpos[geom_id, 2]
                    + _geom_world_vertical_half_extent(
                        self.model, data, int(geom_id),
                    )
                )
                for geom_id in collision_ids
            )
            valid_height = (
                base_top - 0.010
                <= float(fruit_position[2])
                <= rim_top + 0.045
            )
        else:
            rim_top = base_top
            valid_height = (
                base_top - 0.010
                <= float(fruit_position[2])
                <= base_top + 0.090
            )

        support_contact = any(
            float(data.contact[index].dist) <= 1e-6
            and {
                int(self.model.geom_bodyid[int(data.contact[index].geom1)]),
                int(self.model.geom_bodyid[int(data.contact[index].geom2)]),
            } == {self.fruit_body, self.receptacle_body}
            for index in range(data.ncon)
        )
        placed = over_target and support_contact
        self._success = bool(placed and not is_blocking_collision(self.collision))
        self._subgoals["placed"] = self._success
        self._last_conditions = {
            "radial_error_m": radial_error,
            "radial_limit_m": radius_limit,
            "valid_height": bool(valid_height),
            "base_top_z_m": base_top,
            "rim_top_z_m": rim_top,
            "fruit_center_z_m": float(fruit_position[2]),
            "support_contact": support_contact,
            "inside_tray": bool(over_target) if self.receptacle_kind == "tray" else None,
            **(
                {}
                if local_xy is None or tray_half_xy is None
                else {
                    "fruit_center_local_xy_m": local_xy.tolist(),
                    "tray_interior_half_xy_m": tray_half_xy.tolist(),
                }
            ),
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


class FruitSequenceEvaluator:
    """Evaluate the fixed ordered fruit sequence into a shared tray."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
        task_name: str,
    ) -> None:
        try:
            self.order = FRUIT_TRAY_SEQUENCE_TASK_SPECS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown fruit sequence task: {task_name}") from error
        self.labels = tuple(f"{fruit}_to_tray" for fruit in self.order)
        self.evaluators = tuple(
            FruitReceptacleEvaluator(
                model, data, fruit=fruit, receptacle="tray",
            )
            for fruit in self.order
        )
        self.active_subtask = 0
        self.completed = [False, False]
        self.order_violation = False

    @property
    def collision(self):
        return next(
            (evaluator.collision for evaluator in self.evaluators
             if evaluator.collision is not None),
            None,
        )

    @property
    def success(self) -> bool:
        return (
            all(self.completed)
            and all(evaluator.success for evaluator in self.evaluators)
            and not self.order_violation
        )

    @property
    def should_abort(self) -> bool:
        return any(
            is_blocking_collision(evaluator.collision)
            for evaluator in self.evaluators
        )

    def observe_contacts(
        self,
        data: mujoco.MjData,
        frame_index: int,
        time_s: float,
    ) -> None:
        for evaluator in self.evaluators:
            evaluator.observe_contacts(data, frame_index, time_s)

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        for evaluator in self.evaluators:
            evaluator.observe_frame(
                data, gripper_closed, frame_index, time_s,
            )
        if self.active_subtask < len(self.evaluators):
            if any(
                evaluator.success
                for evaluator in self.evaluators[self.active_subtask + 1:]
            ) and not self.evaluators[self.active_subtask].success:
                self.order_violation = True
            if self.evaluators[self.active_subtask].success:
                self.completed[self.active_subtask] = True
                self.active_subtask += 1

    def result(
        self,
        executor_finished: bool,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        task_success = self.success and executor_finished
        collision = self.collision
        if is_blocking_collision(collision):
            reason = collision.category
        elif task_success:
            reason = "success"
        elif timed_out:
            reason = "timeout"
        elif not executor_finished:
            reason = "planner_incomplete"
        else:
            reason = "subtask_failed"
        subtask_results = {
            label: evaluator.result(
                self.completed[index] or executor_finished,
                timed_out,
            )
            for index, (label, evaluator) in enumerate(zip(
                self.labels, self.evaluators, strict=True,
            ))
        }
        return {
            "success": task_success,
            "termination_reason": reason,
            "subgoals": {
                "completed_subtasks": list(self.completed),
                "active_subtask": min(
                    self.active_subtask, len(self.completed) - 1,
                ),
                "order_violation": self.order_violation,
                "subtasks": {
                    label: result["subgoals"]
                    for label, result in subtask_results.items()
                },
            },
            "final_conditions": {
                label: result["final_conditions"]
                for label, result in subtask_results.items()
            },
            "collision": None if collision is None else collision.as_dict(),
            "knocked_over": None,
        }


__all__ = ("FruitReceptacleEvaluator", "FruitSequenceEvaluator")
