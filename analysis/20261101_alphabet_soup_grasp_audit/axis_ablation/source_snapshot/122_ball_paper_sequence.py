"""Long task: place the ball on the plate, then the cube on paper."""

from __future__ import annotations

import mujoco
import numpy as np

from ...simulation.robot import RobotHandles
from ...simulation.scenes import SceneManager
from .paper_placement import PlaceBlockOnPaper
from .pick_place import (
    PickSphere,
    PlaceOnPlate,
    append_blended_motion,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame


SCENE_NAME = "Ball + Paper Placement"


class PlaceBallThenBlockOnPaper:
    """Compose the two tuned skills with an open-gripper safe transition."""

    def __init__(self, model: mujoco.MjModel, handles: RobotHandles) -> None:
        self.model = model
        self.handles = handles

    def plan_first_subtask(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        pick = PickSphere(
            self.model,
            body_name="teleop_ball",
            geom_name="teleop_ball_geom",
            lift_clearance_m=0.040,
            gripper_envelope=gripper_vertical_envelope(
                self.model, self.handles,
            ),
        ).plan(
            data, current_position, current_rotation, limits,
        )
        ball_frames = list(pick.frames)
        ball_frames.extend(PlaceOnPlate(
            self.model,
            body_name="teleop_plate",
            bottom_geom_name="teleop_plate_bottom",
            top_geom_name="teleop_plate_top",
            obstacle_body_names=SceneManager.body_names(SCENE_NAME),
            approach_clearance_m=0.040,
        ).plan(data, pick, limits))

        # The plate skill ends at its release pose.  Leave the completed ball
        # undisturbed and begin the second skill from a high, open pose.
        released = ball_frames[-1]
        retreat = released.position.copy()
        retreat[2] += 0.100
        retreat_limits = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.090),
            angular_speed_rad_s=limits.angular_speed_rad_s,
            command_rate_hz=limits.command_rate_hz,
        )
        append_blended_motion(
            ball_frames,
            released.position,
            released.rotation,
            retreat,
            released.rotation,
            False,
            "Retract after ball placement",
            retreat_limits,
            start_speed_m_s=0.0,
            end_speed_m_s=0.0,
        )
        return tuple(ball_frames)

    def plan_second_subtask(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        return PlaceBlockOnPaper(
            self.model,
            self.handles,
            object_label="cube",
        ).plan(
            data,
            current_position,
            current_rotation,
            limits,
        )

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        first = self.plan_first_subtask(
            data, current_position, current_rotation, limits,
        )
        second = self.plan_second_subtask(
            data, first[-1].position, first[-1].rotation, limits,
        )
        return (*first, *second)


__all__ = ("PlaceBallThenBlockOnPaper", "SCENE_NAME")
