"""Privileged-state evaluators for pick-and-place tasks."""

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
        raise RuntimeError(f"Task evaluator object is missing: {name}")
    return int(object_id)


class BlockToPlateEvaluator:
    """Evaluate blue-block placement independently of the executing policy.

    The same evaluator is consumed by demonstration collection and benchmark
    runners, ensuring that train-data filtering and policy scoring use exactly
    the same task semantics.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        *,
        target_body_name: str = "teleop_tall_block",
        target_geom_name: str = "teleop_tall_block_geom",
        distractor_body_names: tuple[str, ...] = (
            "teleop_cube", "teleop_ball", "teleop_cylinder",
        ),
        topple_body_names: tuple[str, ...] = ("teleop_cylinder",),
        stable_frames: int = 5,
        linear_speed_limit_m_s: float = 0.03,
        angular_speed_limit_rad_s: float = 0.30,
        plate_edge_margin_m: float = 0.003,
        topple_angle_rad: float = np.deg2rad(60.0),
        target_upright_angle_rad: float = np.deg2rad(15.0),
        require_target_upright: bool = True,
        require_stability: bool = True,
        allow_lift_inferred_grasp: bool = True,
        plate_body_name: str = "teleop_plate",
        plate_geom_name: str = "teleop_plate_top",
        interior_site_name: str | None = None,
        receptacle_label: str = "plate",
    ) -> None:
        if stable_frames <= 0:
            raise ValueError("stable_frames must be positive")
        self.model = model
        self.stable_frames = int(stable_frames)
        self.linear_speed_limit_m_s = float(linear_speed_limit_m_s)
        self.angular_speed_limit_rad_s = float(angular_speed_limit_rad_s)
        self.plate_edge_margin_m = float(plate_edge_margin_m)
        self.topple_angle_rad = float(topple_angle_rad)
        self.target_upright_angle_rad = float(target_upright_angle_rad)
        self.require_target_upright = bool(require_target_upright)
        self.require_stability = bool(require_stability)
        self.allow_lift_inferred_grasp = bool(allow_lift_inferred_grasp)
        if not 0.0 < self.target_upright_angle_rad < np.pi / 2.0:
            raise ValueError("target_upright_angle_rad must be between 0 and pi/2")
        self.target_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, target_body_name,
        )
        self.target_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name,
        )
        self.plate_geom = _required_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, plate_geom_name,
        )
        self.interior_site = (
            None
            if interior_site_name is None
            else _required_id(
                model, mujoco.mjtObj.mjOBJ_SITE, interior_site_name,
            )
        )
        self.receptacle_label = receptacle_label
        self.table_body = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0") >= 0:
            robot_root, fingers = "link0", ("left_finger", "right_finger")
        elif mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_base") >= 0:
            robot_root, fingers = "link_base", ("left_finger", "right_finger")
        elif mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rb_base") >= 0:
            robot_root, fingers = "base", ("rb_left_pad", "rb_right_pad")
        else:
            raise RuntimeError("Cannot identify the robot gripper for task evaluation")
        self.contacts = PickPlaceContactMonitor(
            model,
            target_body_name=target_body_name,
            receptacle_body_name=plate_body_name,
            distractor_body_names=distractor_body_names,
            robot_root_name=robot_root,
            finger_body_names=fingers,
        )
        # Only elongated upright distractors can meaningfully be knocked over.
        # The sphere may roll and the cube is orientation-symmetric, so neither
        # should turn an otherwise valid demonstration into a failure.
        self._topple_body_ids = {
            _required_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in topple_body_names
        }
        self._initial_up_local: dict[int, np.ndarray] = {}
        self._target_up_local: np.ndarray | None = None
        self._knocked_over: dict[str, Any] | None = None
        if data is not None:
            self.capture_initial_state(data)
        self._stable_count = 0
        self._success = False
        self._subgoals = {"placed": False}
        self._last_conditions: dict[str, Any] = {}

    @property
    def collision(self):
        return self.contacts.collision

    @property
    def knocked_over(self) -> dict[str, Any] | None:
        return self._knocked_over

    def capture_initial_state(self, data: mujoco.MjData) -> None:
        """Capture each monitored object's initial upright direction."""
        world_up = np.array((0.0, 0.0, 1.0))
        self._target_up_local = (
            data.xmat[self.target_body].reshape(3, 3).T @ world_up
        )
        self._initial_up_local = {
            body_id: data.xmat[body_id].reshape(3, 3).T @ world_up
            for body_id in self._topple_body_ids
        }

    def _observe_toppling(
        self, data: mujoco.MjData, frame_index: int, time_s: float,
    ) -> None:
        if self._knocked_over is not None:
            return
        if not self._initial_up_local:
            self.capture_initial_state(data)
            return
        world_up = np.array((0.0, 0.0, 1.0))
        for body_id, initial_up_local in self._initial_up_local.items():
            current_up = data.xmat[body_id].reshape(3, 3) @ initial_up_local
            cosine = float(np.clip(current_up @ world_up, -1.0, 1.0))
            tilt = float(np.arccos(cosine))
            if tilt < self.topple_angle_rad:
                continue
            self._knocked_over = {
                "frame_index": int(frame_index),
                "time_s": float(time_s),
                "body": mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                ),
                "tilt_angle_rad": tilt,
                "threshold_rad": self.topple_angle_rad,
            }
            return

    def observe_contacts(
        self, data: mujoco.MjData, frame_index: int, time_s: float,
    ) -> None:
        """Inspect contacts after every physics step so brief impacts are kept."""
        self.contacts.observe(data, frame_index, time_s)

    def observe_frame(
        self,
        data: mujoco.MjData,
        gripper_closed: bool,
        frame_index: int = 0,
        time_s: float = 0.0,
    ) -> None:
        del gripper_closed, frame_index, time_s
        plate_center = data.geom_xpos[self.plate_geom]
        center_error = float(np.linalg.norm(
            data.xipos[self.target_body, :2] - plate_center[:2],
        ))
        if self.interior_site is None:
            plate_radius = float(self.model.geom_size[self.plate_geom, 0])
            inside_xy = center_error <= plate_radius
            local_xy = None
            interior_half_xy = None
        else:
            site_rotation = data.site_xmat[self.interior_site].reshape(3, 3)
            local_xy = (
                (data.xipos[self.target_body] - data.site_xpos[self.interior_site])
                @ site_rotation
            )[:2]
            interior_half_xy = self.model.site_size[self.interior_site, :2]
            plate_radius = float(np.min(interior_half_xy))
            inside_xy = bool(np.all(
                np.abs(local_xy) <= interior_half_xy - self.plate_edge_margin_m
            ))

        plate_body = self.contacts.receptacle_body
        support_contact = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if float(contact.dist) > 1e-6:
                continue
            bodies = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            if self.target_body in bodies and plate_body in bodies:
                support_contact = True
                break

        valid_final = bool(
            inside_xy
            and support_contact
            and (
                not self.require_target_upright
                or self._target_up_local is not None
                and float(np.clip(
                    (
                        data.xmat[self.target_body].reshape(3, 3)
                        @ self._target_up_local
                    ) @ np.array((0.0, 0.0, 1.0)),
                    -1.0,
                    1.0,
                )) >= np.cos(self.target_upright_angle_rad)
            )
            and not is_blocking_collision(self.collision)
        )
        target_tilt_rad = (
            None
            if self._target_up_local is None
            else float(np.arccos(np.clip(
                (
                    data.xmat[self.target_body].reshape(3, 3)
                    @ self._target_up_local
                ) @ np.array((0.0, 0.0, 1.0)),
                -1.0,
                1.0,
            )))
        )
        self._success = valid_final
        self._subgoals["placed"] = valid_final
        self._last_conditions = {
            "inside_plate": inside_xy,
            "center_inside_plate": inside_xy,
            "center_error_m": center_error,
            "support_contact": support_contact,
            "plate_radius_m": plate_radius,
            "inside_receptacle": inside_xy,
            "receptacle": self.receptacle_label,
            "target_tilt_rad": target_tilt_rad,
            "target_upright_required": self.require_target_upright,
            **(
                {}
                if local_xy is None or interior_half_xy is None
                else {
                    "center_local_xy_m": local_xy.tolist(),
                    "interior_half_xy_m": interior_half_xy.tolist(),
                }
            ),
        }

    @property
    def success(self) -> bool:
        return self._success and not is_blocking_collision(self.collision)

    @property
    def should_abort(self) -> bool:
        return is_blocking_collision(self.collision)

    def result(self, executor_finished: bool, timed_out: bool = False) -> dict[str, Any]:
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
            "collision": None if self.collision is None else self.collision.as_dict(),
            "knocked_over": self.knocked_over,
        }


class GeometryToPlateEvaluator(BlockToPlateEvaluator):
    """Evaluate the sole active geometry object in the shared tray."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        **kwargs: Any,
    ) -> None:
        active: list[tuple[str, str, int]] = []
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if not name.startswith("paper_object_"):
                continue
            geom_name = f"{name}_geom"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0 and int(model.geom_contype[geom_id]) != 0:
                active.append((name, geom_name, int(geom_id)))
        if len(active) != 1:
            raise RuntimeError(f"Expected one active geometry object, found {active}")
        body_name, geom_name, geom_id = active[0]
        sphere = int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        require_target_upright = bool(kwargs.pop(
            "require_target_upright", False,
        ))
        require_stability = bool(kwargs.pop("require_stability", not sphere))
        super().__init__(
            model,
            data,
            target_body_name=body_name,
            target_geom_name=geom_name,
            distractor_body_names=(),
            topple_body_names=(),
            require_target_upright=require_target_upright,
            require_stability=require_stability,
            plate_body_name="tray_khaki_medium",
            plate_geom_name="tray_khaki_medium_bottom",
            interior_site_name="tray_khaki_medium_interior_site",
            receptacle_label="tray",
            **kwargs,
        )


class CubeToPlateEvaluator(BlockToPlateEvaluator):
    """Evaluate the orange cube being placed on the Blocks scene plate."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            data,
            target_body_name="teleop_cube",
            target_geom_name="teleop_cube_geom",
            distractor_body_names=(
                "teleop_tall_block", "teleop_ball", "teleop_cylinder",
            ),
            **kwargs,
        )


class BallToPlateEvaluator(BlockToPlateEvaluator):
    """Evaluate orientation-invariant ball placement on the Blocks plate."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            data,
            target_body_name="teleop_ball",
            target_geom_name="teleop_ball_geom",
            distractor_body_names=(
                "teleop_tall_block", "teleop_cube", "teleop_cylinder",
            ),
            require_target_upright=False,
            require_stability=False,
            allow_lift_inferred_grasp=True,
            **kwargs,
        )
