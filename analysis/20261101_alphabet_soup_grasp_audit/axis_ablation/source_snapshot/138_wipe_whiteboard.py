"""Pick up the cloth-down eraser and sweep the marked board."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ...geometry import axis_angle_matrix, rotation_vector
from ...simulation.robot import RobotHandles
from .pick_place import (
    append_blended_motion, append_dwell, gripper_close_dwell_ticks,
    gripper_preclose_dwell_ticks, gripper_vertical_envelope,
)
from .types import PREGRASP_CLEARANCE_M, MotionLimits, TaskFrame

WIPE_COMPRESSION_DEPTH_M = 0.004
WIPE_COMPRESSION_JITTER_M = 0.001
WIPE_LINEAR_SPEED_M_S = 0.045


def _append_horizontal_turn(
    frames: list[TaskFrame], start: np.ndarray, goal: np.ndarray,
    start_rotation: np.ndarray, goal_rotation: np.ndarray,
    closed: bool, phase: str, align_phase: str, limits: MotionLimits,
) -> None:
    """Turn uniformly during lateral travel, allowing a coordinated descent."""
    moving: list[TaskFrame] = []
    append_blended_motion(
        moving, start, start_rotation, goal, start_rotation,
        closed, phase, limits, 0.0, 0.0,
    )
    relative = rotation_vector(start_rotation.T @ goal_rotation)
    angle = float(np.linalg.norm(relative))
    duration = len(moving) / limits.command_rate_hz
    turn_angle = min(angle, duration * limits.angular_speed_rad_s)
    final_rotation = start_rotation
    for index, frame in enumerate(moving, start=1):
        final_rotation = (
            start_rotation @ axis_angle_matrix(
                relative / angle, turn_angle * index / len(moving),
            ) if angle > 1e-12 else start_rotation.copy()
        )
        frames.append(TaskFrame(frame.position, final_rotation, closed, phase))
    if angle - turn_angle > 1e-9:
        append_blended_motion(
            frames, goal, final_rotation, goal, goal_rotation,
            closed, align_phase, limits, 0.0, 0.0,
        )


def _choose_wipe_rotation(
    initial_rotation: np.ndarray, wipe_rotation: np.ndarray,
) -> np.ndarray:
    """Keep the wipe target near the initial arm heading, independent of pickup."""
    candidates = (wipe_rotation, wipe_rotation @ np.diag((-1.0, -1.0, 1.0)))
    return max(candidates, key=lambda candidate: float(np.trace(initial_rotation.T @ candidate)))


class WipeWhiteboard:
    def __init__(self, model: mujoco.MjModel, handles: RobotHandles) -> None:
        self.model = model
        self.handles = handles

    def plan(
        self, data: mujoco.MjData, current_position: np.ndarray,
        current_rotation: np.ndarray, limits: MotionLimits,
        *, candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        eraser = data.body("board_eraser").xpos.copy()
        plastic = data.geom("board_eraser_plastic").xpos.copy()
        board = data.body("wipe_board").xpos.copy()
        board_top = float(data.geom("wipe_board_surface").xpos[2] + 0.002)
        table_geom = int(self.model.body_geomadr[int(self.model.body("work_surface").id)])
        table_top = float(data.geom_xpos[table_geom, 2] + self.model.geom_size[table_geom, 2])
        # The cloth supports the thin plastic above the table. Treating the
        # plastic bottom as the support plane raises the fingers to its top,
        # leaving no nominal side contact. Use the actual table and require
        # contact overlap through the shared grasp-envelope solver.
        envelope = gripper_vertical_envelope(self.model, self.handles)
        plastic_half_height = float(self.model.geom("board_eraser_plastic").size[2])
        panda = self.handles.spec.uid == "panda"
        # Panda's 17 mm contact band can cover the 15 mm plastic side with
        # their upper edges flush. The cloth leaves room below the plastic.
        offset = envelope.safe_tabletop_center_offset(
            plastic_half_height,
            support_plane_from_center_m=table_top - float(plastic[2]),
            preferred_center_offset_m=(
                plastic_half_height - envelope.contact_upper_above_ee_m if panda else None
            ),
            finger_support_clearance_m=0.001 if panda else 0.013,
            height_bias_m=0.0,
        )
        grasp = plastic.copy()
        grasp[2] += offset
        cloth_bottom = min(
            float(data.geom(name).xpos[2] - np.abs(
                data.geom(name).xmat.reshape(3, 3)[2]
            ) @ self.model.geom(name).size)
            for name in ("board_eraser_cloth_center", "board_eraser_cloth_left",
                         "board_eraser_cloth_right")
        )
        grasp_to_cloth_bottom = grasp[2] - cloth_bottom
        # Grip the eraser's local 4 cm side at its sampled tabletop yaw.
        # The fingers are symmetric: the two grasp headings differ by 180
        # degrees around tool Z. Compare them with the actual current pose.
        # Maximizing the relative rotation trace minimizes its SO(3) angle.
        tool_z = np.array((0.0, 0.0, -1.0))
        closing = data.body("board_eraser").xmat.reshape(3, 3)[:, 0].copy()
        rotation = np.column_stack((np.cross(closing, tool_z), closing, tool_z))
        flipped_rotation = rotation @ np.diag((-1.0, -1.0, 1.0))
        if candidate_index is not None:
            if type(candidate_index) is not int or not 0 <= candidate_index < 4:
                raise ValueError("Invalid whiteboard candidate index")
            if candidate_index // 2:
                rotation = flipped_rotation
        elif np.trace(current_rotation.T @ flipped_rotation) > np.trace(
            current_rotation.T @ rotation
        ):
            rotation = flipped_rotation
        canonical_closing = np.array((1.0, 0.0, 0.0))
        canonical_rotation = np.column_stack((
            np.cross(canonical_closing, tool_z), canonical_closing, tool_z,
        ))
        travel = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.10),
            angular_speed_rad_s=limits.angular_speed_rad_s,
            command_rate_hz=limits.command_rate_hz,
        )
        wipe = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, WIPE_LINEAR_SPEED_M_S),
            angular_speed_rad_s=limits.angular_speed_rad_s,
            command_rate_hz=limits.command_rate_hz,
        )
        frames: list[TaskFrame] = []
        motion_rotation = current_rotation
        def move(start: np.ndarray, goal: np.ndarray, closed: bool,
                 phase: str, speed: MotionLimits = travel) -> None:
            append_blended_motion(
                frames, start, motion_rotation, goal, motion_rotation,
                closed, phase,
                speed, 0.0, 0.0,
            )
        approach = grasp.copy()
        approach[2] += PREGRASP_CLEARANCE_M
        departure = current_position.copy()
        departure[2] = max(current_position[2], approach[2])
        move(current_position, departure, False, "Raise before approaching eraser")
        horizontal_goal = approach.copy()
        _append_horizontal_turn(
            frames, departure, horizontal_goal, current_rotation, rotation,
            False, "Approach eraser", "Finish pickup rotation", travel,
        )
        motion_rotation = rotation
        move(horizontal_goal, grasp, False, "Lower onto eraser")
        append_dwell(frames, grasp, rotation, False, "Align with eraser",
                     gripper_preclose_dwell_ticks(limits))
        append_dwell(frames, grasp, rotation, True, "Grip eraser",
                     gripper_close_dwell_ticks(limits))
        lifted = grasp.copy()
        lifted[2] += 0.09
        move(grasp, lifted, True, "Lift eraser")
        board_rotation = data.body("wipe_board").xmat.reshape(3, 3)
        first = self.model.geom("wipe_mark_1_00").pos[:2]
        last = self.model.geom("wipe_mark_1_60").pos[:2]
        direction = last - first
        angle = math.atan2(-direction[0], direction[1])
        cosine, sine = math.cos(angle), math.sin(angle)
        turn = np.array(((cosine, -sine, 0.0),
                         (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
        # Put the 10 cm cloth edge across the 6–8 cm ink band, then travel
        # once along the band's long axis. The eraser is initially long in Y.
        quarter_turn = np.array(((0.0, 1.0, 0.0),
                                 (-1.0, 0.0, 0.0),
                                 (0.0, 0.0, 1.0)))
        wipe_rotation = board_rotation @ turn @ quarter_turn @ canonical_rotation
        # Wipe heading depends on the board and ink, not on eraser pickup yaw.
        # Choose its symmetric candidate nearest the initial arm heading and
        # turn directly to it; forcing a reverse sign can select a 180-degree
        # flip even when only a small heading correction is needed.
        wipe_rotation = (_choose_wipe_rotation(current_rotation, wipe_rotation)
                         if candidate_index is None else
                         wipe_rotation @ np.diag((-1., -1., 1.)) if candidate_index % 2 else wipe_rotation)
        # Contact is defined by the cloth's bottom face. The eraser body origin
        # is at the plastic base, 7 mm above it; using that origin would drive
        # the deeper grip through the board/table. Request 4 mm of compression.
        target_z = board_top + grasp_to_cloth_bottom - WIPE_COMPRESSION_DEPTH_M
        mark_positions = np.array([
            self.model.geom(f"wipe_mark_{row}_{point:02d}").pos[:2]
            for row in range(3) for point in range(61)
        ])
        # Project ink into its own across/along frame. One centered pass
        # covers the full band even when its size and angle are randomized.
        local_marks = mark_positions @ turn[:2, :2]
        across = 0.5 * (
            float(np.min(local_marks[:, 0]))
            + float(np.max(local_marks[:, 0]))
        )
        along_min = float(np.min(local_marks[:, 1]))
        along_max = float(np.max(local_marks[:, 1]))
        def board_xy(along: float) -> np.ndarray:
            return board[:2] + board_rotation[:2, :2] @ (
                turn[:2, :2] @ np.array((across, along))
            )
        # Leave a longer runout for cloth lag inside the fingertips under
        # sustained board contact; both endpoints remain inside the board.
        start_xy = board_xy(along_min - 0.085)
        end_xy = board_xy(along_max + 0.085)
        # Begin at the end of the ink nearest the eraser's pickup location.
        # Compare the ink itself rather than the padded motion endpoints.
        if np.linalg.norm(board_xy(along_max) - eraser[:2]) < np.linalg.norm(
            board_xy(along_min) - eraser[:2]
        ):
            start_xy, end_xy = end_xy, start_xy
        high = np.array((start_xy[0], start_xy[1], target_z + PREGRASP_CLEARANCE_M))
        contact = high.copy()
        contact[2] = target_z
        finish = np.array((end_xy[0], end_xy[1], target_z))
        # Rotate while moving toward the ink start and descend to the shared
        # 5 cm stand-off. A short transfer finishes its angle before contact.
        prealign = high.copy()
        _append_horizontal_turn(
            frames, lifted, prealign, rotation, wipe_rotation, True,
            "Move near scribbles", "Align eraser with scribbles", travel,
        )
        motion_rotation = wipe_rotation
        move(prealign, high, True, "Move to the start of the scribbles")
        move(high, contact, True, "Press cloth onto whiteboard", wipe)
        move(contact, finish, True, "Wipe whiteboard", wipe)
        return tuple(frames)


__all__ = ("WipeWhiteboard",)
