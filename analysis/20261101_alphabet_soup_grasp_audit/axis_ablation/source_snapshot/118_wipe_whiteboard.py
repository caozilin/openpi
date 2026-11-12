"""Erase visual ink only when the cloth physically contacts the board."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class WipeWhiteboardEvaluator:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData | None) -> None:
        del data
        self.model = model
        self.board_geom = int(model.geom("wipe_board_surface").id)
        self.eraser_body = int(model.body("board_eraser").id)
        self.cloth_geoms = tuple(int(model.geom(name).id) for name in (
            "board_eraser_cloth_center", "board_eraser_cloth_left",
            "board_eraser_cloth_right",
        ))
        self.marks = tuple(int(model.geom(f"wipe_mark_{row}_{point:02d}").id)
                           for row in range(3) for point in range(61))
        self.remaining = set(self.marks)
        self.contact_observed = False
        self.swipe_observed = False
        self._last_contact_xy: np.ndarray | None = None
        self._contact_travel_m = 0.0

    @property
    def success(self) -> bool:
        return self.contact_observed and not self.remaining

    @property
    def should_abort(self) -> bool:
        return False

    def observe_contacts(
        self, data: mujoco.MjData, frame_index: int, time_s: float,
    ) -> None:
        del frame_index, time_s
        touching = set()
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.board_geom in pair:
                touching.update(pair.intersection(self.cloth_geoms))
        touching = {
            geom_id for geom_id in touching
            if data.geom_xmat[geom_id].reshape(3, 3)[2, 2] >= 0.8
        }
        if not touching:
            self._last_contact_xy = None
            self._contact_travel_m = 0.0
            return
        self.contact_observed = True
        contact_xy = data.xpos[self.eraser_body, :2].copy()
        if self._last_contact_xy is not None:
            self._contact_travel_m += float(np.linalg.norm(
                contact_xy - self._last_contact_xy
            ))
        self._last_contact_xy = contact_xy
        self.swipe_observed |= self._contact_travel_m >= 0.003
        if not self.swipe_observed:
            return
        for cloth_id in touching:
            center = data.geom_xpos[cloth_id]
            rotation = data.geom_xmat[cloth_id].reshape(3, 3)
            half = self.model.geom_size[cloth_id]
            for mark_id in tuple(self.remaining):
                local = rotation.T @ (data.geom_xpos[mark_id] - center)
                if abs(local[0]) <= half[0] + 0.0018 and abs(local[1]) <= half[1] + 0.0018:
                    self.model.geom_rgba[mark_id, 3] = 0.0
                    self.remaining.remove(mark_id)

    def observe_frame(
        self, data: mujoco.MjData, gripper_closed: bool,
        frame_index: int = 0, time_s: float = 0.0,
    ) -> None:
        del data, gripper_closed, frame_index, time_s

    def result(self, executor_finished: bool, timed_out: bool = False) -> dict[str, Any]:
        success = bool(executor_finished and self.success)
        return {
            "success": success,
            "termination_reason": (
                "success" if success else "timeout" if timed_out
                else "planner_incomplete" if not executor_finished
                else "ink_remaining"
            ),
            "subgoals": {
                "cloth_contacted_board": self.contact_observed,
                "cloth_swiped_board": self.swipe_observed,
                "all_marks_erased": not self.remaining,
            },
            "final_conditions": {
                "marks_total": len(self.marks),
                "marks_remaining": len(self.remaining),
                "marks_remaining_by_stroke": tuple(
                    sum(mark in self.remaining for mark in self.marks[row * 61:(row + 1) * 61])
                    for row in range(3)
                ),
            },
            "collision": None,
            "knocked_over": None,
        }


__all__ = ("WipeWhiteboardEvaluator",)
