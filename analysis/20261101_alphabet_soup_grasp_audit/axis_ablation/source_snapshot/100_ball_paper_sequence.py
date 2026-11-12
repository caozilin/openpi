"""Ordered evaluation for ball-to-plate then cube-to-paper."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .paper_placement import PaperPlacementEvaluator
from .pick_place import BlockToPlateEvaluator
from .predicates import is_blocking_collision


def _active_paper_object(model: mujoco.MjModel) -> str:
    names: list[str] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith("paper_object_cube_"):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            names.append(name)
    if len(names) != 1:
        raise RuntimeError(
            f"Expected one active cube paper object, found {names}",
        )
    return names[0]


class BallPaperSequenceEvaluator:
    """Require both placements to succeed in their declared order."""

    _LABELS = ("ball_to_plate", "cube_to_paper")

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
    ) -> None:
        paper_object = _active_paper_object(model)
        self.evaluators = (
            BlockToPlateEvaluator(
                model,
                data,
                target_body_name="teleop_ball",
                target_geom_name="teleop_ball_geom",
                distractor_body_names=(paper_object,),
                topple_body_names=(),
                require_target_upright=False,
                require_stability=False,
            ),
            PaperPlacementEvaluator(
                model,
                data,
                extra_distractor_body_names=(
                    "teleop_ball", "teleop_plate",
                ),
            ),
        )
        self.active_subtask = 0
        self.completed = [False, False]

    @property
    def collision(self):
        return next(
            (evaluator.collision for evaluator in self.evaluators
             if evaluator.collision is not None),
            None,
        )

    @property
    def success(self) -> bool:
        return all(evaluator.success for evaluator in self.evaluators)

    @property
    def should_abort(self) -> bool:
        return is_blocking_collision(self.collision)

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
        for index, evaluator in enumerate(self.evaluators):
            evaluator.observe_frame(
                data, gripper_closed, frame_index, time_s,
            )
            self.completed[index] = evaluator.success
        self.active_subtask = next(
            (index for index, done in enumerate(self.completed) if not done),
            len(self.completed) - 1,
        )

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
        results = {
            label: evaluator.result(
                self.completed[index] or executor_finished,
                timed_out,
            )
            for index, (label, evaluator) in enumerate(zip(
                self._LABELS, self.evaluators, strict=True,
            ))
        }
        return {
            "success": task_success,
            "termination_reason": reason,
            "subgoals": {
                "completed_subtasks": list(self.completed),
                "active_subtask": self.active_subtask,
                "subtasks": {
                    label: result["subgoals"]
                    for label, result in results.items()
                },
            },
            "final_conditions": {
                label: result["final_conditions"]
                for label, result in results.items()
            },
            "collision": None if collision is None else collision.as_dict(),
            "knocked_over": None,
        }


__all__ = ("BallPaperSequenceEvaluator",)
