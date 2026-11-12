"""Privileged-state success checks for opening and closing a pot lid."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..planning.pot_lid import active_pot_lid_variant
from .predicates import RobotSafetyMonitor, is_blocking_collision


class PotLidEvaluator:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        close_pot: bool,
        expected_kind: str | None = None,
        stable_frames: int = 5,
    ) -> None:
        self.model = model
        self.close_pot = bool(close_pot)
        self.stable_frames = int(stable_frames)
        kind, pot_name, lid_name = active_pot_lid_variant(model)
        if expected_kind is not None and kind != expected_kind:
            raise RuntimeError(
                f"Pot-lid task requires {expected_kind!r}, active variant is {kind!r}",
            )
        self.lid_kind = kind
        self.pot = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pot_name))
        self.lid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lid_name))
        self.table = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface"))
        self.safety = RobotSafetyMonitor(model)
        self._initial_lid_z: float | None = None
        self._stable_count = 0
        self._success = False
        self._subgoals = {"placed": False}
        self._last_conditions: dict[str, Any] = {}
        if data is not None:
            self._initial_lid_z = float(data.xpos[self.lid, 2])

    @property
    def success(self) -> bool:
        return self._success and not is_blocking_collision(self.safety.collision)

    @property
    def collision(self):
        return self.safety.collision

    @property
    def should_abort(self) -> bool:
        return is_blocking_collision(self.safety.collision)

    def observe_contacts(self, data: mujoco.MjData, frame_index: int, time_s: float) -> None:
        self.safety.observe(data, frame_index, time_s)

    def _contact(self, data: mujoco.MjData, first: int, second: int) -> bool:
        for index in range(data.ncon):
            contact = data.contact[index]
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            if first in bodies and second in bodies and float(contact.dist) <= 1e-6:
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
        if self._initial_lid_z is None:
            self._initial_lid_z = float(data.xpos[self.lid, 2])
        lid_position = data.xpos[self.lid]
        pot_position = data.xpos[self.pot]
        separation = float(np.linalg.norm(lid_position[:2] - pot_position[:2]))
        upright = bool(
            float(data.xmat[self.lid].reshape(3, 3)[2, 2])
            >= np.cos(np.deg2rad(15.0))
        )
        table_contact = self._contact(data, self.lid, self.table)
        pot_contact = self._contact(data, self.lid, self.pot)
        placed = bool(
            separation <= 0.045 and pot_contact and upright
            if self.close_pot
            else separation >= 0.19 and table_contact and upright
        )
        self._success = bool(
            placed and not is_blocking_collision(self.safety.collision)
        )
        self._subgoals["placed"] = self._success
        self._last_conditions = {
            "lid_pot_separation_m": separation,
            "upright": upright,
            "table_contact": table_contact,
            "pot_contact": pot_contact,
        }

    def result(self, executor_finished: bool, timed_out: bool = False) -> dict[str, Any]:
        success = bool(self.success and executor_finished)
        if is_blocking_collision(self.safety.collision):
            reason = self.safety.collision.category
        elif success:
            reason = "success"
        elif timed_out:
            reason = "timeout"
        elif not executor_finished:
            reason = "planner_incomplete"
        else:
            reason = "lid_not_on_pot" if self.close_pot else "lid_not_beside_pot"
        return {
            "success": success,
            "termination_reason": reason,
            "subgoals": dict(self._subgoals),
            "final_conditions": dict(self._last_conditions),
            "collision": (
                None
                if self.safety.collision is None
                else self.safety.collision.as_dict()
            ),
            "knocked_over": None,
        }


__all__ = ("PotLidEvaluator",)
