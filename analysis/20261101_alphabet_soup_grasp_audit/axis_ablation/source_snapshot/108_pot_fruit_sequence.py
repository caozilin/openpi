"""Privileged evaluation for opening a pot, adding a pear, and reclosing it."""

from __future__ import annotations

from typing import Any

import mujoco

from .fruit_receptacles import FruitReceptacleEvaluator
from .pot_lid import PotLidEvaluator
from .predicates import is_blocking_collision


class RoundHandlePotPearSequenceEvaluator:
    """Require open-lid, pear-in-pot, and closed-lid events in that order."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
    ) -> None:
        self.open_lid = PotLidEvaluator(
            model,
            data,
            close_pot=False,
            expected_kind="cylindrical",
        )
        self.pear = FruitReceptacleEvaluator(
            model,
            data,
            fruit="pear",
            receptacle="bowl",
            fruit_body_name="fruit_task_pear",
            receptacle_body_name="teleop_pot",
            distractor_body_names=("teleop_pot_lid",),
            # The pear may settle against the pot wall after the lid is
            # replaced; its body centre must remain inside the 75.2 mm cavity.
            bowl_radius_limit_m=0.071,
        )
        self.close_lid = PotLidEvaluator(
            model,
            data,
            close_pot=True,
            expected_kind="cylindrical",
        )
        self.evaluators = (self.open_lid, self.pear, self.close_lid)
        self.completed = [False, False, False]
        self.active_subtask = 0
        self.order_violation = False

    @property
    def collision(self):
        return next((
            evaluator.collision
            for evaluator in self.evaluators
            if evaluator.collision is not None
        ), None)

    @property
    def success(self) -> bool:
        return bool(
            all(self.completed)
            and self.pear.success
            and self.close_lid.success
            and not self.order_violation
            and not is_blocking_collision(self.collision)
        )

    @property
    def should_abort(self) -> bool:
        return any(evaluator.should_abort for evaluator in self.evaluators)

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
                data,
                gripper_closed,
                frame_index,
                time_s,
            )
        if self.active_subtask == 0:
            if self.pear.success:
                self.order_violation = True
            if self.open_lid.success:
                self.completed[0] = True
                self.active_subtask = 1
        elif self.active_subtask == 1:
            if self.close_lid.success and not self.pear.success:
                self.order_violation = True
            if self.pear.success:
                self.completed[1] = True
                self.active_subtask = 2
        elif self.close_lid.success:
            self.completed[2] = True
            self.active_subtask = 3

    def result(
        self,
        executor_finished: bool,
        timed_out: bool = False,
    ) -> dict[str, Any]:
        task_success = bool(self.success and executor_finished)
        collision = self.collision
        if is_blocking_collision(collision):
            reason = collision.category
        elif task_success:
            reason = "success"
        elif timed_out:
            reason = "timeout"
        elif not executor_finished:
            reason = "planner_incomplete"
        elif self.order_violation:
            reason = "subtask_order_violation"
        else:
            reason = "subtask_failed"
        open_result = self.open_lid.result(self.completed[0], timed_out)
        pear_result = self.pear.result(self.completed[1], timed_out)
        close_result = self.close_lid.result(executor_finished, timed_out)
        return {
            "success": task_success,
            "termination_reason": reason,
            "subgoals": {
                "completed_subtasks": list(self.completed),
                "active_subtask": min(self.active_subtask, 2),
                "order_violation": self.order_violation,
            },
            "final_conditions": {
                "lid_placed_beside_pot": open_result["final_conditions"],
                "pear_in_pot": pear_result["final_conditions"],
                "lid_closed": close_result["final_conditions"],
            },
            "collision": None if collision is None else collision.as_dict(),
            "knocked_over": None,
        }


__all__ = ("RoundHandlePotPearSequenceEvaluator",)
