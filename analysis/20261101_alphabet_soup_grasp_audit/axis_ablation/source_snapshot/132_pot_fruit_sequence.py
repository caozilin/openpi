"""Three-stage round-handle pot task: open, add pear, and close."""

from __future__ import annotations

import mujoco
import numpy as np

from ...simulation.robot import RobotHandles
from .fruit_receptacles import PlaceFruit
from .pot_lid import ManipulatePotLid, lid_table_destination
from .types import MotionLimits, TaskFrame


class RoundHandlePotPearSequence:
    """Compose the existing lid and fruit skills around one live pot scene."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
    ) -> None:
        self.model = model
        self.handles = handles

    def plan_first_subtask(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        return ManipulatePotLid(
            self.model,
            self.handles,
            close_pot=False,
            expected_kind="cylindrical",
        ).plan(data, current_position, current_rotation, limits)

    def plan_pear_subtask(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *, candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        return PlaceFruit(
            self.model,
            self.handles,
            fruit="pear",
            receptacle="pot",
            receptacle_body_name="teleop_pot",
        ).plan(
            data,
            current_position,
            current_rotation,
            limits,
            retract_after_release=True,
            dwell_after_retract=False,
            candidate_index=candidate_index,
        )

    def plan_close_subtask(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        close_frames = ManipulatePotLid(
            self.model,
            self.handles,
            close_pot=True,
            expected_kind="cylindrical",
        ).plan(
            data,
            current_position,
            current_rotation,
            limits,
        )
        return tuple(close_frames)

    def plan_remaining_subtasks(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        """Compatibility helper for callers that explicitly request a preview."""
        pear_frames = self.plan_pear_subtask(
            data, current_position, current_rotation, limits,
        )
        close_frames = self.plan_close_subtask(
            data, pear_frames[-1].position, pear_frames[-1].rotation, limits,
        )
        return (*pear_frames, *close_frames)

    def preview_data_after_lid_release(
        self,
        data: mujoco.MjData,
    ) -> mujoco.MjData:
        """Clone the scene and place the lid at the open skill's destination."""
        preview = mujoco.MjData(self.model)
        mujoco.mj_copyData(preview, self.model, data)
        lid_body = int(mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "teleop_pot_lid",
        ))
        if lid_body < 0:
            raise RuntimeError("Round-handle pot scene is missing its lid")
        joint = int(self.model.body_jntadr[lid_body])
        qpos_address = int(self.model.jnt_qposadr[joint])
        dof_address = int(self.model.jnt_dofadr[joint])
        preview.qpos[qpos_address:qpos_address + 3] = lid_table_destination(
            self.model,
            data,
            self.handles,
            expected_kind="cylindrical",
        )
        preview.qpos[qpos_address + 3:qpos_address + 7] = data.xquat[lid_body]
        preview.qvel[dof_address:dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, preview)
        return preview


__all__ = ("RoundHandlePotPearSequence",)
