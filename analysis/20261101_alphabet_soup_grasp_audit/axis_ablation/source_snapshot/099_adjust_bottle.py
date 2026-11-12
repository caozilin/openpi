"""Privileged-state evaluator matching RoboTwin's Adjust Bottle task."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .predicates import PickPlaceContactMonitor, is_blocking_collision


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Adjust Bottle evaluator object is missing: {name}")
    return int(object_id)


def _robot_contact_names(
    model: mujoco.MjModel,
) -> tuple[str, tuple[str, str]]:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0") >= 0:
        return "link0", ("left_finger", "right_finger")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_base") >= 0:
        return "link_base", ("left_finger", "right_finger")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base") >= 0:
        return "base", ("rb_left_pad", "rb_right_pad")
    raise RuntimeError("Cannot identify the robot gripper for Adjust Bottle")


class AdjustBottleEvaluator:
    """Judge only the bottle's terminal upright-on-table state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        upright_angle_rad: float = np.deg2rad(10.0),
        stable_frames: int = 5,
        linear_speed_limit_m_s: float = 0.03,
        angular_speed_limit_rad_s: float = 0.30,
    ) -> None:
        del stable_frames, linear_speed_limit_m_s, angular_speed_limit_rad_s
        self.model = model
        candidate_bodies = tuple(
            body_id for body_id in range(model.nbody)
            if (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                ) or ""
            ).startswith("adjust_bottle_")
        )
        if not candidate_bodies:
            raise RuntimeError("Adjust Bottle evaluator found no bottle variants")
        if data is None:
            self.target_body = candidate_bodies[0]
        else:
            active = [
                body_id for body_id in candidate_bodies
                if float(data.xpos[body_id, 2]) > 0.0
            ]
            if len(active) != 1:
                raise RuntimeError(
                    "Adjust Bottle evaluator requires one active bottle model",
                )
            self.target_body = active[0]
        target_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, self.target_body,
        )
        assert target_name is not None
        self.table_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        robot_root, fingers = _robot_contact_names(model)
        self.contacts = PickPlaceContactMonitor(
            model,
            target_body_name=target_name,
            receptacle_body_name="work_surface",
            distractor_body_names=(),
            robot_root_name=robot_root,
            finger_body_names=fingers,
        )
        self.upright_angle_rad = float(upright_angle_rad)
        self._success = False
        self._subgoals = {"upright_on_table": False}
        self._last_conditions: dict[str, Any] = {}

    @property
    def collision(self):
        return self.contacts.collision

    def _table_geometry(self) -> int:
        geom_ids = np.flatnonzero(self.model.geom_bodyid == self.table_body)
        if not geom_ids.size:
            raise RuntimeError("Work surface has no geometry")
        return int(geom_ids[np.argmax(
            np.prod(self.model.geom_size[geom_ids, :2], axis=1),
        )])

    def observe_contacts(
        self, data: mujoco.MjData, frame_index: int, time_s: float,
    ) -> None:
        self.contacts.observe(data, frame_index, time_s)

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        del gripper_closed, frame_index, time_s

        axis = data.xmat[self.target_body].reshape(3, 3)[:, 2]
        upright_angle = float(np.arccos(np.clip(axis[2], -1.0, 1.0)))
        # Bottle is "upright" (i.e. satisfying the task) when its long
        # axis is roughly aligned with the world Z, i.e. the bottle is
        # standing up. ``upright_angle`` here is the angle between the
        # bottle's long axis and the world Z; the bottle is standing
        # when this is near 0 deg, lying flat when this is near 90 deg.
        # ``upright_angle_rad`` is therefore a tolerance around the
        # standing target (default 20 deg).
        upright = upright_angle <= self.upright_angle_rad

        table_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > 1e-6:
                continue
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            if (
                self.target_body in bodies
                and bool(bodies & self.contacts.table_bodies)
            ):
                table_contact = True
                break

        self._success = bool(
            upright and table_contact and not is_blocking_collision(self.collision)
        )
        self._subgoals["upright_on_table"] = self._success
        self._last_conditions = {
            "upright": upright,
            "upright_angle_rad": upright_angle,
            "upright_threshold_rad": self.upright_angle_rad,
            "table_contact": table_contact,
        }

    @property
    def success(self) -> bool:
        return self._success and not is_blocking_collision(self.collision)

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
        elif not bool(self._last_conditions.get("upright", False)):
            reason = "bottle_not_upright"
        elif not bool(self._last_conditions.get("table_contact", False)):
            reason = "bottle_not_on_table"
        else:
            reason = "terminal_pose_failed"
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


__all__ = ("AdjustBottleEvaluator",)
