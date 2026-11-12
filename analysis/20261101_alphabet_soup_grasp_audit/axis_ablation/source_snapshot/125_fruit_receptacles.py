"""Simple top-pick plans for placing one of three fruits in a YCB receptacle."""

from __future__ import annotations

import mujoco
import numpy as np

from .pick_place import (
    _nearest_downward_rotation,
    _geom_world_vertical_half_extent,
    _plan_top_pick,
    append_blended_motion,
    append_dwell,
    gripper_vertical_envelope,
)
from .types import MotionLimits, TaskFrame
from ..catalog import (
    FRUIT_TRAY_SEQUENCE_TASK_SPECS,
    BANANA_TO_PLATE_TASK,
    BANANA_TO_TRAY_TASK,
    PEAR_TO_BOWL_TASK,
    PEAR_TO_PLATE_TASK,
    PEAR_TO_TRAY_TASK,
    STRAWBERRY_TO_TRAY_TASK,
)
from ...simulation.robot import RobotHandles


_TASK_SPECS = {
    BANANA_TO_TRAY_TASK: ("banana", "tray"),
    STRAWBERRY_TO_TRAY_TASK: ("strawberry", "tray"),
    PEAR_TO_TRAY_TASK: ("pear", "tray"),
    BANANA_TO_PLATE_TASK: ("banana", "plate"),
    PEAR_TO_PLATE_TASK: ("pear", "plate"),
    PEAR_TO_BOWL_TASK: ("pear", "bowl"),
}
_RECEPTACLE_PREFIXES = {
    "plate": "fruit_task_plate_",
    "bowl": "fruit_task_bowl_",
    "tray": "tray_khaki_",
    "pot": "teleop_pot",
}
_FRUIT_GEOMETRY = {
    # Geometric-centre offset in the local body frame, vertical half extent,
    # and planar swept radius in the authored task pose.  The pear starts on
    # its side, so its long local-Z axis contributes to the planar radius.
    "banana": ((0.0, 0.0, 0.017), 0.017, 0.095),
    "strawberry": ((0.0, 0.0, 0.023), 0.023, 0.022),
    "apple": ((0.0, 0.0, 0.036), 0.036, 0.037),
    "pear": ((0.0, 0.0, 0.036), 0.0335, 0.052),
}
_PEAR_SYMMETRY_AXIS_LOCAL = np.array((0.0, 0.0, 1.0))
_PEAR_GRASP_AXIS_DISTANCE_M = 0.036
# PCA of the segmented green silhouette in the zero-Z-rotation wrist image
# gives a 20.12 degree major-axis tilt.  Positive local Y rotates the gripper
# longitudinal axis onto that visible pear axis while leaving its grasp point
# on the round bulb unchanged.
_PEAR_MODEL_AXIS_YAW_BIAS_DEG = 20.1
# Put the middle of the closed fingertip contact band near the lying pear's
# centre-of-mass height.  The small positive margin leaves the Panda palm
# about 10 mm above the top of the 67 mm-tall bulb at the nominal pose.
_PEAR_GRASP_CONTACT_CENTER_ABOVE_COM_M = 0.0065


def _pear_model_axis_local() -> np.ndarray:
    angle = np.radians(_PEAR_MODEL_AXIS_YAW_BIAS_DEG)
    return np.array((0.0, np.sin(angle), np.cos(angle)))


def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Fruit task object is missing: {name}")
    return int(object_id)


def _active_body(model: mujoco.MjModel, prefix: str) -> tuple[str, int]:
    active: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or ""
        )
        if not name.startswith(prefix):
            continue
        geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        if np.any(model.geom_contype[geom_ids] != 0):
            active.append((name, body_id))
    if len(active) != 1:
        raise RuntimeError(
            f"Expected one active body with prefix {prefix!r}, "
            f"found {[name for name, _ in active]}",
        )
    return active[0]


def _limited(limits: MotionLimits, linear_speed_m_s: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, linear_speed_m_s),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


def _oriented_fruit_grasp_rotation(
    object_rotation: np.ndarray,
    current_rotation: np.ndarray,
    sign_index: int | None = None,
) -> np.ndarray:
    closing_axis = object_rotation[:, 1].copy()
    closing_axis[2] = 0.0
    closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-12)
    if (sign_index == 1 if sign_index is not None else closing_axis @ current_rotation[:, 1] < 0.0):
        closing_axis *= -1.0
    tool_z = np.array((0.0, 0.0, -1.0))
    tool_x = np.cross(closing_axis, tool_z)
    tool_x /= max(float(np.linalg.norm(tool_x)), 1e-12)
    return np.column_stack((tool_x, closing_axis, tool_z))


def _axisymmetric_pear_grasp_rotation(
    object_rotation: np.ndarray,
    current_rotation: np.ndarray,
    sign_index: int | None = None,
) -> np.ndarray:
    """Align tool X with the lying pear axis and close exactly across it."""
    symmetry_axis = (
        np.asarray(object_rotation, dtype=float)
        @ _pear_model_axis_local()
    )
    horizontal_axis = symmetry_axis.copy()
    horizontal_axis[2] = 0.0
    norm = float(np.linalg.norm(horizontal_axis))
    if norm < 1e-9:
        # The authored pear is a side-lying object. If physics ever tips it
        # vertical, a downward grasp has no unique horizontal axial heading.
        horizontal_axis = np.asarray(current_rotation, dtype=float)[:, 0].copy()
        horizontal_axis[2] = 0.0
        norm = float(np.linalg.norm(horizontal_axis))
        if norm < 1e-9:
            horizontal_axis = np.array((1.0, 0.0, 0.0))
            norm = 1.0
    tool_x = horizontal_axis / norm
    # An axisymmetric pear has no directed head/tail requirement for the
    # gripper. Pick the equivalent sign requiring the smaller wrist turn.
    if (sign_index == 1 if sign_index is not None else tool_x @ np.asarray(current_rotation, dtype=float)[:, 0] < 0.0):
        tool_x *= -1.0
    tool_z = np.array((0.0, 0.0, -1.0))
    closing_axis = np.cross(tool_z, tool_x)
    closing_axis /= float(np.linalg.norm(closing_axis))
    return np.column_stack((tool_x, closing_axis, tool_z))


class PlaceFruit:
    """Pick the requested fruit and release it over the requested container."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        task_name: str | None = None,
        *,
        fruit: str | None = None,
        receptacle: str | None = None,
        receptacle_body_name: str | None = None,
    ) -> None:
        self.model = model
        self.handles = handles
        self.receptacle_body_name = receptacle_body_name
        if fruit is not None or receptacle is not None:
            if fruit not in _FRUIT_GEOMETRY or receptacle not in (
                "plate", "bowl", "tray", "pot",
            ):
                raise ValueError(
                    f"Unsupported fruit destination: {fruit!r} -> {receptacle!r}",
                )
            self.fruit, self.receptacle = str(fruit), str(receptacle)
        else:
            try:
                self.fruit, self.receptacle = _TASK_SPECS[task_name]
            except KeyError as error:
                raise ValueError(f"Unknown fruit task: {task_name}") from error

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *,
        retract_after_release: bool = True,
        dwell_after_retract: bool = True,
        destination_offset_local_xy: tuple[float, float] = (0.0, 0.0),
        candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        fruit_name = f"fruit_task_{self.fruit}"
        fruit_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, fruit_name,
        )
        if self.receptacle_body_name is None:
            receptacle_name, receptacle_body = _active_body(
                self.model, _RECEPTACLE_PREFIXES[self.receptacle],
            )
        else:
            receptacle_name = self.receptacle_body_name
            receptacle_body = _required_id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, receptacle_name,
            )
        center_offset, vertical_half, horizontal_radius = _FRUIT_GEOMETRY[
            self.fruit
        ]
        object_rotation = data.xmat[fruit_body].reshape(3, 3).copy()
        if self.fruit == "banana":
            # The banana body origin is the scan bounding-box centre, not the
            # useful pinch point.  Read the named middle capsule so seeded yaw
            # and position changes cannot move the grasp away from the middle.
            grasp_geom = _required_id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "fruit_task_banana_grasp_geom",
            )
            center = data.geom_xpos[grasp_geom].copy()
        elif self.fruit == "pear":
            # Treat the pear as a surface of revolution around body local Z.
            # The grasp section is a point on that exact axis: no mesh-centroid
            # or stem asymmetry is allowed to introduce a lateral offset.
            symmetry_axis = object_rotation @ _PEAR_SYMMETRY_AXIS_LOCAL
            center = (
                data.xpos[fruit_body].copy()
                + _PEAR_GRASP_AXIS_DISTANCE_M * symmetry_axis
            )
            # The long-axis point determines XY.  Its vertical component must
            # follow the live centre of mass so a slightly rolled pear is not
            # pinched high on its upper side.
            center[2] = data.xipos[fruit_body, 2]
        else:
            center = data.xpos[fruit_body].copy()
            center += object_rotation @ np.asarray(center_offset, dtype=float)
        grasp_rotation = (
            _axisymmetric_pear_grasp_rotation(
                object_rotation,
                current_rotation,
                None if candidate_index is None else candidate_index // (2 if self.receptacle == "tray" else 1),
            )
            if self.fruit == "pear"
            else _oriented_fruit_grasp_rotation(
                object_rotation,
                current_rotation,
                None if candidate_index is None else candidate_index // (2 if self.receptacle == "tray" else 1),
            )
            if self.fruit == "banana"
            else np.asarray(current_rotation, dtype=float).copy()
            if self.fruit == "apple"
            else _nearest_downward_rotation(current_rotation)
        )
        if candidate_index is not None and (self.fruit not in ("banana", "pear")
                or type(candidate_index) is not int
                or not 0 <= candidate_index < (4 if self.receptacle == "tray" else 2)):
            raise ValueError("Invalid fruit candidate index")
        gripper_envelope = gripper_vertical_envelope(
            self.model, self.handles,
        )
        pear_gripper_offset_z = (
            _PEAR_GRASP_CONTACT_CENTER_ABOVE_COM_M
            - 0.5 * (
                gripper_envelope.contact_upper_above_ee_m
                - gripper_envelope.closed_finger_bottom_z_offset_m
            )
        )
        pick = _plan_top_pick(
            center,
            vertical_half,
            horizontal_radius,
            gripper_envelope,
            grasp_rotation,
            current_position,
            current_rotation,
            limits,
            0.090,
            0.085,
            self.fruit,
            fruit_body,
            descent_speed_m_s=0.075,
            lift_speed_m_s=0.090,
            gripper_offset_z_m=(
                pear_gripper_offset_z
                if self.fruit == "pear"
                else gripper_envelope.safe_tabletop_center_offset(
                    vertical_half,
                )
            ),
        )

        bottom_geom = _required_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{receptacle_name}_bottom",
        )
        bottom_top = float(
            data.geom_xpos[bottom_geom, 2]
            + _geom_world_vertical_half_extent(
                self.model, data, bottom_geom,
            )
        )
        placement_rotation = pick.grasp_rotation
        receptacle_rotation = data.xmat[receptacle_body].reshape(3, 3).copy()
        if self.receptacle == "tray" and self.fruit != "apple":
            # Keep the long axis of a lying fruit parallel to the tray's long
            # axis.  The two equivalent axis directions let the wrist choose
            # the shorter turn from the pickup pose for any randomized tray yaw.
            candidates = (
                np.column_stack((
                    receptacle_rotation[:, 0],
                    -receptacle_rotation[:, 1],
                    -receptacle_rotation[:, 2],
                )),
                np.column_stack((
                    -receptacle_rotation[:, 0],
                    receptacle_rotation[:, 1],
                    -receptacle_rotation[:, 2],
                )),
            )
            placement_rotation = min(
                candidates,
                key=lambda candidate: float(np.arccos(np.clip(
                    0.5 * (np.trace(pick.grasp_rotation.T @ candidate) - 1.0),
                    -1.0,
                    1.0,
                ))),
            )
            if candidate_index is not None:
                placement_rotation = candidates[candidate_index % 2]
        if self.receptacle in ("bowl", "pot"):
            geom_ids = np.flatnonzero(
                self.model.geom_bodyid == receptacle_body,
            )
            collision_ids = geom_ids[
                self.model.geom_contype[geom_ids] != 0
            ]
            rim_top = max(
                float(
                    data.geom_xpos[geom_id, 2]
                    + _geom_world_vertical_half_extent(
                        self.model, data, int(geom_id),
                    )
                )
                for geom_id in collision_ids
            )
            release_center_z = rim_top + vertical_half + 0.008
        else:
            release_center_z = bottom_top + vertical_half + 0.003

        destination = data.xpos[receptacle_body].copy()
        destination += receptacle_rotation @ np.array((
            float(destination_offset_local_xy[0]),
            float(destination_offset_local_xy[1]),
            0.0,
        ))
        desired_object_center = np.array((
            destination[0],
            destination[1],
            release_center_z,
        ))
        object_in_gripper = (
            pick.grasp_rotation.T @ (center - pick.grasp_position)
        )
        place_position = (
            desired_object_center
            - placement_rotation @ object_in_gripper
        )
        table_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        table_geom = int(self.model.body_geomadr[table_body])
        table_top = float(
            data.geom_xpos[table_geom, 2]
            + self.model.geom_size[table_geom, 2]
        )
        # Release from above the receptacle rather than driving the fingertips
        # down beside it.  The short drop is safe for fruit and keeps the
        # entire Panda finger collision envelope clear of the tabletop.
        place_position[2] = max(
            float(place_position[2]),
            table_top + 0.075,
        )
        transfer_height = max(
            float(pick.lifted_position[2]),
            float(pick.lifted_position[2]),
            0.525,
            float(place_position[2] + 0.080),
        )
        raised = pick.lifted_position.copy()
        raised[2] = transfer_height
        above = place_position.copy()
        above[2] = transfer_height
        lift_limits = _limited(limits, 0.090)
        transfer_limits = _limited(limits, 0.130)
        lower_limits = _limited(
            limits,
            0.060,
        )

        frames = list(pick.frames)
        if transfer_height > pick.lifted_position[2] + 1e-9:
            append_blended_motion(
                frames,
                pick.lifted_position,
                pick.grasp_rotation,
                raised,
                pick.grasp_rotation,
                True,
                "Raise fruit to transfer height",
                lift_limits,
                start_speed_m_s=lift_limits.linear_speed_m_s,
                end_speed_m_s=lift_limits.linear_speed_m_s,
            )
        append_blended_motion(
            frames,
            raised,
            pick.grasp_rotation,
            above,
            placement_rotation,
            True,
            f"Move {self.fruit} above {self.receptacle}",
            transfer_limits,
            start_speed_m_s=lift_limits.linear_speed_m_s,
            end_speed_m_s=lower_limits.linear_speed_m_s,
        )
        append_blended_motion(
            frames,
            above,
            placement_rotation,
            place_position,
            placement_rotation,
            True,
            f"Lower {self.fruit} into {self.receptacle}",
            lower_limits,
            start_speed_m_s=lower_limits.linear_speed_m_s,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            place_position,
            placement_rotation,
            False,
            f"Release {self.fruit}",
            25,
        )
        if retract_after_release:
            retreat = place_position.copy()
            retreat[2] += 0.100
            append_blended_motion(
                frames,
                place_position,
                placement_rotation,
                retreat,
                placement_rotation,
                False,
                "Retract after release",
                lift_limits,
                start_speed_m_s=0.0,
                end_speed_m_s=0.0,
            )
            if dwell_after_retract:
                append_dwell(
                    frames,
                    retreat,
                    placement_rotation,
                    False,
                    f"{self.fruit} placed",
                    25,
                )
        return tuple(frames)


class PlaceFruitSequence:
    """Place the registered pair of fruits into one tray in fixed order."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        task_name: str,
    ) -> None:
        self.model = model
        self.handles = handles
        try:
            self.order = FRUIT_TRAY_SEQUENCE_TASK_SPECS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown fruit sequence task: {task_name}") from error

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        frames: list[TaskFrame] = []
        position = np.asarray(current_position, dtype=float).copy()
        rotation = np.asarray(current_rotation, dtype=float).copy()
        for index, fruit in enumerate(self.order):
            subtask_frames = PlaceFruit(
                self.model,
                self.handles,
                fruit=fruit,
                receptacle="tray",
            ).plan(
                data,
                position,
                rotation,
                limits,
                retract_after_release=index == 0,
                dwell_after_retract=False,
                destination_offset_local_xy=(
                    0.0, -0.045 if index == 0 else 0.045,
                ),
            )
            frames.extend(subtask_frames)
            position = subtask_frames[-1].position.copy()
            rotation = subtask_frames[-1].rotation.copy()
        return tuple(frames)

    def plan_subtask(
        self,
        index: int,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *, candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        """Plan one fruit placement from the live subtask-boundary pose."""
        if not 0 <= int(index) < len(self.order):
            raise IndexError("Fruit-sequence subtask index is out of range")
        fruit = self.order[int(index)]
        return PlaceFruit(
            self.model,
            self.handles,
            fruit=fruit,
            receptacle="tray",
        ).plan(
            data,
            current_position,
            current_rotation,
            limits,
            retract_after_release=int(index) < len(self.order) - 1,
            dwell_after_retract=False,
            destination_offset_local_xy=(
                0.0, -0.045 if int(index) == 0 else 0.045,
            ),
            candidate_index=candidate_index,
        )


__all__ = ("PlaceFruit", "PlaceFruitSequence")
