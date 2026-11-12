"""Privileged evaluation for ordered three-cube stacking."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .predicates import RobotSafetyMonitor, is_blocking_collision
from .types import CollisionEvent
from ..catalog import COLOR_BLOCK_STACK_TASK_SPECS


_MAXIMUM_LAYER_CENTER_OFFSET_M = 0.012


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Color-stack evaluator object is missing: {name}")
    return int(object_id)


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(model.nbody):
        current = candidate
        while current > root:
            current = int(model.body_parentid[current])
        if current == root:
            result.add(candidate)
    return result


class ColorBlockStackEvaluator:
    """Require the selected bottom-to-top order, support and stability."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
        task_name: str,
        *,
        stable_frames: int = 5,
    ) -> None:
        self.model = model
        try:
            self.order = COLOR_BLOCK_STACK_TASK_SPECS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown color-stack task: {task_name}") from error
        self.block_names = {
            color: f"stack_cube_{color}" for color in self.order
        }
        self.block_bodies = {
            color: _required_id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.block_names[color],
            )
            for color in self.block_names
        }
        self.block_geoms = {
            color: next(
                int(geom_id)
                for geom_id in np.flatnonzero(
                    model.geom_bodyid == self.block_bodies[color]
                )
                if model.geom_contype[geom_id] != 0
            )
            for color in self.block_names
        }
        self.edge_m = 2.0 * float(
            model.geom_size[self.block_geoms[self.order[0]], 0]
        )
        table_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        self.table_bodies = _subtree(model, table_body)
        self.table_geom = int(model.body_geomadr[table_body])
        self.stable_frames = max(int(stable_frames), 1)
        self._stable_count = 0
        self._success = False
        self.safety = RobotSafetyMonitor(model)
        self._last_conditions: dict[str, Any] = {}
        del data

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

    def _body_contact(self, data: mujoco.MjData, first: int, second: int) -> bool:
        for index in range(data.ncon):
            contact = data.contact[index]
            if float(contact.dist) <= 1e-6 and {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            } == {first, second}:
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
        del gripper_closed
        centers = {
            color: data.geom_xpos[geom].copy()
            for color, geom in self.block_geoms.items()
        }
        base = self.order[0]
        table_top = float(
            data.geom_xpos[self.table_geom, 2]
            + self.model.geom_size[self.table_geom, 2]
        )
        expected_z = {
            color: table_top + (level + 0.5) * self.edge_m
            for level, color in enumerate(self.order)
        }
        height_errors = {
            color: abs(float(centers[color][2] - expected_z[color]))
            for color in self.order
        }
        xy_offsets = {
            color: centers[color][:2] - centers[base][:2]
            for color in self.order
        }
        xy_aligned = all(
            float(np.linalg.norm(xy_offsets[color]))
            <= _MAXIMUM_LAYER_CENTER_OFFSET_M
            for color in self.order[1:]
        )
        upright_angles = {
            color: float(np.arccos(np.clip(
                data.geom_xmat[geom].reshape(3, 3)[:, 2]
                @ np.array((0.0, 0.0, 1.0)),
                -1.0,
                1.0,
            )))
            for color, geom in self.block_geoms.items()
        }
        support_contacts = {
            f"{moving}_on_{support}": self._body_contact(
                data, self.block_bodies[moving], self.block_bodies[support],
            )
            for moving, support in zip(
                self.order[1:], self.order[:-1], strict=True,
            )
        }
        base_on_table = any(
            self._body_contact(data, self.block_bodies[base], table_body)
            for table_body in self.table_bodies
        )
        valid = bool(
            base_on_table
            and xy_aligned
            and all(error <= 0.008 for error in height_errors.values())
            and all(angle <= np.deg2rad(12.0) for angle in upright_angles.values())
            and all(support_contacts.values())
            and centers[self.order[0]][2]
            < centers[self.order[1]][2]
            < centers[self.order[2]][2]
            and not is_blocking_collision(self.collision)
        )
        self._stable_count = self._stable_count + 1 if valid else 0
        self._success = self._stable_count >= self.stable_frames
        self._last_conditions = {
            "cube_edge_m": self.edge_m,
            "bottom_to_top_order": list(self.order),
            "base_on_table": base_on_table,
            "xy_aligned": xy_aligned,
            "xy_offsets_m": {
                color: offset.tolist() for color, offset in xy_offsets.items()
            },
            "correct_heights": all(
                error <= 0.008 for error in height_errors.values()
            ),
            "height_errors_m": height_errors,
            "upright_angles_rad": upright_angles,
            "support_contacts": support_contacts,
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
            reason = "invalid_stack"
        return {
            "success": task_success,
            "termination_reason": reason,
            "subgoals": {"stacked": self._success},
            "final_conditions": dict(self._last_conditions),
            "collision": (
                None if self.collision is None else self.collision.as_dict()
            ),
            "knocked_over": None,
        }


__all__ = ("ColorBlockStackEvaluator",)
