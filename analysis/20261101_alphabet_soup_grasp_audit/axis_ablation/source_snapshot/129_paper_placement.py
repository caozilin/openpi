"""Top-pick one object and align it with its matching paper outline."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ...geometry import axis_angle_matrix, rotation_vector
from .adjust_bottle import _CpuMotionPlanner, _IKSolution
from .pick_place import (
    _NARROW_FINGERTIP_CONTACT_HEIGHT_THRESHOLD_M,
    PickBox,
    _nearest_downward_rotation,
    _plan_top_pick,
    append_blended_motion,
    append_dwell,
    gripper_vertical_envelope,
)
from .types import MotionLimits, PlanningInfeasibleError, TaskFrame
from ...simulation.robot import RobotHandles


_SMALLEST_CUBE_GRASP_Z_ADJUSTMENT_M = -0.010
def _required_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Paper placement object is missing: {name}")
    return int(object_id)


def _limited(limits: MotionLimits, speed: float) -> MotionLimits:
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, speed),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


class PlaceBlockOnPaper:
    """Use one direct, axis-aligned pick-transfer-place trajectory."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
        *,
        object_label: str = "block",
    ) -> None:
        self.model = model
        self.handles = handles
        self.object_label = str(object_label)

    def _active_object(self) -> str:
        active: list[str] = []
        for body_id in range(self.model.nbody):
            name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                )
                or ""
            )
            if not name.startswith("paper_object_"):
                continue
            geom_ids = np.flatnonzero(self.model.geom_bodyid == body_id)
            if np.any(self.model.geom_contype[geom_ids] != 0):
                active.append(name)
        if len(active) != 1:
            raise RuntimeError(
                f"Expected one active paper-placement object, found {active}",
            )
        return active[0]

    def _pick_cylinder(
        self,
        data: mujoco.MjData,
        body_name: str,
        geom_name: str,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *,
        upright: bool,
        grasp_rotation: np.ndarray | None = None,
    ):
        body_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name,
        )
        geom_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name,
        )
        object_rotation = data.xmat[body_id].reshape(3, 3)
        if grasp_rotation is None and upright:
            grasp_rotation = _nearest_downward_rotation(current_rotation)
        elif grasp_rotation is None:
            closing_axis = object_rotation[:, 1].copy()
            closing_axis[2] = 0.0
            closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-12)
            if closing_axis @ current_rotation[:, 1] < 0.0:
                closing_axis *= -1.0
            tool_z = np.array((0.0, 0.0, -1.0))
            tool_x = np.cross(closing_axis, tool_z)
            tool_x /= max(float(np.linalg.norm(tool_x)), 1e-12)
            grasp_rotation = np.column_stack((tool_x, closing_axis, tool_z))
        assert grasp_rotation is not None
        radius = float(self.model.geom_size[geom_id, 0])
        half_length = float(self.model.geom_size[geom_id, 1])
        envelope = gripper_vertical_envelope(self.model, self.handles)
        grasp_height_bias = (
            0.0 if upright else 0.006 if radius <= 0.0151 else 0.0
        )
        # The xArm7 and UR5e finger envelopes extend substantially farther
        # below the EE site than Panda's.  A midpoint grasp on a lying
        # cylinder can therefore put the open/closing pad through the table.
        if not upright and self.handles.spec.uid in ("xarm7", "ur5e"):
            gripper_offset_z = envelope.safe_tabletop_center_offset(
                radius,
                height_bias_m=grasp_height_bias,
            )
        else:
            gripper_offset_z = None
        return _plan_top_pick(
            data.geom_xpos[geom_id].copy(),
            half_length if upright else radius,
            radius if upright else float(np.hypot(radius, half_length)),
            envelope,
            grasp_rotation,
            current_position,
            current_rotation,
            limits,
            0.085,
            0.095,
            "cylinder",
            body_id,
            grasp_height_bias_m=grasp_height_bias,
            gripper_offset_z_m=(
                envelope.upper_box_center_offset(
                    half_length,
                    height_bias_m=(
                        0.006
                        if (
                            upright
                            and envelope.finger_contact_height_m
                            >= _NARROW_FINGERTIP_CONTACT_HEIGHT_THRESHOLD_M
                        )
                        else 0.0
                    ),
                )
                if upright else gripper_offset_z
            ),
        )

    @staticmethod
    def _signed_downward_rotations(axis: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the two equivalent top-grasp wrist orientations."""
        closing_axis = np.asarray(axis, dtype=float).copy()
        closing_axis[2] = 0.0
        norm = float(np.linalg.norm(closing_axis))
        if norm <= 1e-9:
            raise RuntimeError("Object grasp axis is not horizontal")
        closing_axis /= norm
        tool_z = np.array((0.0, 0.0, -1.0))
        rotations: list[np.ndarray] = []
        for sign in (1.0, -1.0):
            tool_y = sign * closing_axis
            tool_x = np.cross(tool_y, tool_z)
            tool_x /= max(float(np.linalg.norm(tool_x)), 1e-12)
            rotations.append(np.column_stack((tool_x, tool_y, tool_z)))
        return tuple(rotations)

    def _lying_pick_candidates(
        self,
        data: mujoco.MjData,
        block_name: str,
        block_geom_name: str,
        object_kind: str,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ):
        """Build all equivalent lying-object grasps before choosing an IK branch."""
        block_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, block_name,
        )
        block_geom = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, block_geom_name,
        )
        object_rotation = data.xmat[block_body].reshape(3, 3).copy()
        if object_kind == "cylinder":
            rotations = self._signed_downward_rotations(object_rotation[:, 1])
            return tuple(
                self._pick_cylinder(
                    data,
                    block_name,
                    block_geom_name,
                    current_position,
                    current_rotation,
                    limits,
                    upright=False,
                    grasp_rotation=rotation,
                )
                for rotation in rotations
            )

        half_size = self.model.geom_size[block_geom, :3].copy()
        # Match PickBox's physical rule (pinch the narrower horizontal side),
        # but retain both wrist-equivalent signs for the joint-space search.
        horizontal_axes = [
            (index, object_rotation[:, index].copy())
            for index in range(3)
            if abs(float(object_rotation[2, index])) < 0.25
        ]
        if len(horizontal_axes) != 2:
            raise RuntimeError(
                "Lying box does not expose exactly two horizontal local axes"
            )
        narrow_extent = min(float(half_size[index]) for index, _ in horizontal_axes)
        pinch_axes = [
            axis for index, axis in horizontal_axes
            if float(half_size[index]) <= narrow_extent + 1e-6
        ]
        rotations = tuple(
            rotation
            for axis in pinch_axes
            for rotation in self._signed_downward_rotations(axis)
        )
        vertical_half_extent = float(
            np.sum(np.abs(object_rotation[2]) * half_size)
        )
        envelope = gripper_vertical_envelope(self.model, self.handles)
        gripper_offset_z = envelope.upper_box_center_offset(
            vertical_half_extent,
        )
        return tuple(
            _plan_top_pick(
                data.geom_xpos[block_geom].copy(),
                vertical_half_extent,
                float(np.linalg.norm(half_size[:2])),
                envelope,
                rotation,
                current_position,
                current_rotation,
                limits,
                0.085,
                0.095,
                self.object_label,
                block_body,
                gripper_offset_z_m=gripper_offset_z,
            )
            for rotation in rotations
        )

    def _ik_seeds(self, data: mujoco.MjData) -> tuple[np.ndarray, ...]:
        current = data.qpos[self.handles.qpos_ids].copy()
        home = np.asarray(self.handles.spec.home_q, dtype=float).copy()
        seeds = [home, current]
        if home.size >= 7:
            for offset in (-0.5 * math.pi, 0.5 * math.pi):
                seed = home.copy()
                seed[6] = np.clip(
                    seed[6] + offset,
                    self.handles.joint_lower[6],
                    self.handles.joint_upper[6],
                )
                seeds.append(seed)
        return tuple(seeds)

    def _solve_all_ik(
        self,
        planner: _CpuMotionPlanner,
        position: np.ndarray,
        rotation: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        **kwargs,
    ) -> tuple[_IKSolution, ...]:
        solutions: list[_IKSolution] = []
        for seed in seeds:
            solution = planner.solve_ik(
                position, rotation, (seed,), **kwargs,
            )
            if solution is None:
                continue
            if any(np.linalg.norm(solution.q - found.q) < 1e-4 for found in solutions):
                continue
            solutions.append(solution)
        return tuple(solutions)

    def _trace_pick_trajectory(
        self,
        data: mujoco.MjData,
        planner: _CpuMotionPlanner,
        pick,
        object_rotation: np.ndarray,
    ) -> tuple[_IKSolution, _IKSolution, float] | None:
        """Follow every commanded pick frame on one continuous IK branch."""
        current_q = data.qpos[self.handles.qpos_ids].copy()
        previous = _IKSolution(current_q, 0.0)
        grasp_solution: _IKSolution | None = None
        minimum_manipulability = math.inf
        body_rotation_in_gripper = pick.grasp_rotation.T @ object_rotation
        body_offset_in_gripper = pick.grasp_rotation.T @ (
            data.xpos[pick.object_body_id] - pick.grasp_position
        )
        planner.set_carried_transform(
            body_rotation_in_gripper, body_offset_in_gripper,
        )
        previous_frame: TaskFrame | None = None
        for frame in pick.frames:
            carried = bool(frame.gripper_closed)
            # Dwell frames repeat an already validated Cartesian pose.
            if (
                previous_frame is not None
                and np.allclose(frame.position, previous_frame.position, atol=1e-10)
                and np.allclose(frame.rotation, previous_frame.rotation, atol=1e-10)
            ):
                if not carried:
                    grasp_solution = previous
                previous_frame = frame
                continue
            solution = planner.solve_ik(
                frame.position,
                frame.rotation,
                (previous.q,),
                ignore_bottle=True,
                carried=carried,
                allow_low_carried_pose=carried,
            )
            if solution is None:
                return None
            if not planner.edge_valid(
                previous.q,
                solution.q,
                ignore_bottle=True,
                carried=carried,
                allow_low_carried_pose=carried,
            ):
                return None
            minimum_manipulability = min(
                minimum_manipulability, solution.manipulability,
            )
            if not carried:
                grasp_solution = solution
            previous = solution
            previous_frame = frame
        if grasp_solution is None:
            return None
        return grasp_solution, previous, minimum_manipulability

    def select_lying_pick_for_suffix(
        self,
        data: mujoco.MjData,
        picks: tuple,
        suffix_factory,
    ) -> tuple[object, tuple[TaskFrame, ...]]:
        """Choose a grasp whose complete pick/place command is feasible."""
        table_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        table_geom = int(self.model.body_geomadr[table_body])
        table_top = float(
            data.geom_xpos[table_geom, 2]
            + self.model.geom_size[table_geom, 2]
        )
        home = np.asarray(self.handles.spec.home_q, dtype=float)
        candidates: list[
            tuple[float, float, int, object, tuple[TaskFrame, ...]]
        ] = []
        for serial, pick in enumerate(picks):
            planner = _CpuMotionPlanner(
                self.model,
                data,
                self.handles,
                pick.object_body_id,
                table_geom,
                table_top,
            )
            object_rotation = data.xmat[pick.object_body_id].reshape(3, 3).copy()
            trace = self._trace_pick_trajectory(
                data, planner, pick, object_rotation,
            )
            if trace is None:
                continue
            grasp_ik, previous, minimum_manipulability = trace
            suffix = tuple(suffix_factory(pick))
            carried = True
            previous_frame = pick.frames[-1] if pick.frames else None
            valid = True
            for frame in suffix:
                if carried and not frame.gripper_closed:
                    carried = False
                if (
                    previous_frame is not None
                    and np.allclose(frame.position, previous_frame.position, atol=1e-10)
                    and np.allclose(frame.rotation, previous_frame.rotation, atol=1e-10)
                ):
                    previous_frame = frame
                    continue
                solution = planner.solve_ik(
                    frame.position,
                    frame.rotation,
                    (previous.q,),
                    ignore_bottle=True,
                    carried=carried,
                    allow_bottle_support=carried,
                    allow_low_carried_pose=carried,
                )
                if solution is None:
                    valid = False
                    break
                if not planner.edge_valid(
                    previous.q,
                    solution.q,
                    ignore_bottle=True,
                    carried=carried,
                    allow_bottle_support=carried,
                    allow_low_carried_pose=carried,
                ):
                    valid = False
                    break
                minimum_manipulability = min(
                    minimum_manipulability, solution.manipulability,
                )
                previous = solution
                previous_frame = frame
            if not valid:
                continue
            home_distance = float(np.linalg.norm(grasp_ik.q - home))
            candidates.append((
                home_distance, -minimum_manipulability, serial, pick, suffix,
            ))
        if not candidates:
            raise RuntimeError(
                "No lying-object grasp keeps the complete trajectory feasible"
            )
        selected = min(candidates, key=lambda item: item[:3])
        return selected[3], selected[4]

    def _select_lying_plan(
        self,
        data: mujoco.MjData,
        planner: _CpuMotionPlanner,
        picks: tuple,
        object_rotation: np.ndarray,
        reference_axis: np.ndarray,
        decal_axis: np.ndarray,
        desired_center: np.ndarray,
        object_center: np.ndarray,
        transfer_height: float,
    ) -> tuple[object, np.ndarray, np.ndarray]:
        """Prefer a Home-near grasp among feasible placement branches."""
        home = np.asarray(self.handles.spec.home_q, dtype=float)
        candidates: list[
            tuple[
                float, float, float, float, int,
                object, np.ndarray, np.ndarray,
            ]
        ] = []
        serial = 0
        for pick in picks:
            pick_trace = self._trace_pick_trajectory(
                data, planner, pick, object_rotation,
            )
            if pick_trace is None:
                continue
            grasp_ik, lifted_ik, minimum_manipulability = pick_trace
            home_distance = float(np.linalg.norm(grasp_ik.q - home))
            body_rotation_in_gripper = pick.grasp_rotation.T @ object_rotation
            body_offset_in_gripper = pick.grasp_rotation.T @ (
                data.xpos[pick.object_body_id] - pick.grasp_position
            )
            object_offset_in_gripper = pick.grasp_rotation.T @ (
                object_center - pick.grasp_position
            )
            planner.set_carried_transform(
                body_rotation_in_gripper, body_offset_in_gripper,
            )
            raised_position = pick.lifted_position.copy()
            raised_position[2] = transfer_height
            raised_solutions = self._solve_all_ik(
                planner,
                raised_position,
                pick.grasp_rotation,
                (lifted_ik.q,),
                ignore_bottle=True,
                carried=True,
            )
            if not raised_solutions:
                continue
            raised_ik = raised_solutions[0]
            raised_minimum_manipulability = min(
                minimum_manipulability, raised_ik.manipulability,
            )
            if not planner.edge_valid(
                lifted_ik.q,
                raised_ik.q,
                ignore_bottle=True,
                carried=True,
                allow_low_carried_pose=True,
            ):
                continue
            base_delta = self._alignment_delta(reference_axis, decal_axis, np.pi)
            for extra_yaw in (0.0, np.pi):
                yaw = base_delta + extra_yaw
                cosine, sine = np.cos(yaw), np.sin(yaw)
                yaw_rotation = np.array((
                    (cosine, -sine, 0.0),
                    (sine, cosine, 0.0),
                    (0.0, 0.0, 1.0),
                ))
                placement_rotation = yaw_rotation @ pick.grasp_rotation
                rotation_travel = abs(float(
                    (yaw + np.pi) % (2.0 * np.pi) - np.pi
                ))
                place_position = (
                    desired_center - placement_rotation @ object_offset_in_gripper
                )
                above_position = place_position.copy()
                above_position[2] = transfer_height
                relative_vector = rotation_vector(
                    pick.grasp_rotation.T @ placement_rotation
                )
                rotation_angle = float(np.linalg.norm(relative_vector))
                rotation_axis = (
                    relative_vector / rotation_angle
                    if rotation_angle > 1e-12 else np.array((0.0, 0.0, 1.0))
                )
                previous_ik = raised_ik
                transfer_valid = True
                trajectory_minimum_manipulability = (
                    raised_minimum_manipulability
                )
                # Validate the whole simultaneous translation/rotation, not
                # merely its endpoints, for IK and collision feasibility.
                for step in range(1, 13):
                    fraction = step / 12.0
                    sample_position = raised_position + fraction * (
                        above_position - raised_position
                    )
                    sample_rotation = (
                        pick.grasp_rotation
                        @ axis_angle_matrix(
                            rotation_axis, fraction * rotation_angle,
                        )
                    )
                    sample_solutions = self._solve_all_ik(
                        planner,
                        sample_position,
                        sample_rotation,
                        (previous_ik.q,),
                        ignore_bottle=True,
                        carried=True,
                    )
                    if not sample_solutions:
                        transfer_valid = False
                        break
                    sample_ik = sample_solutions[0]
                    if not planner.edge_valid(
                        previous_ik.q,
                        sample_ik.q,
                        ignore_bottle=True,
                        carried=True,
                        allow_low_carried_pose=True,
                    ):
                        transfer_valid = False
                        break
                    previous_ik = sample_ik
                    trajectory_minimum_manipulability = min(
                        trajectory_minimum_manipulability,
                        sample_ik.manipulability,
                    )
                if not transfer_valid:
                    continue
                above_ik = previous_ik
                if transfer_valid:
                    place_solutions = self._solve_all_ik(
                        planner,
                        place_position,
                        placement_rotation,
                        (above_ik.q,),
                        ignore_bottle=True,
                        carried=True,
                        allow_bottle_support=True,
                        allow_low_carried_pose=True,
                    )
                    for place_ik in place_solutions:
                        if not planner.edge_valid(
                            above_ik.q,
                            place_ik.q,
                            ignore_bottle=True,
                            carried=True,
                            allow_bottle_support=True,
                            allow_low_carried_pose=True,
                        ):
                            continue
                        retreat_position = place_position.copy()
                        retreat_position[2] += 0.100
                        retreat_solutions = self._solve_all_ik(
                            planner,
                            retreat_position,
                            placement_rotation,
                            (place_ik.q,),
                            ignore_bottle=True,
                            carried=False,
                        )
                        if not retreat_solutions:
                            continue
                        retreat_ik = retreat_solutions[0]
                        if not planner.edge_valid(
                            place_ik.q,
                            retreat_ik.q,
                            ignore_bottle=True,
                            carried=False,
                        ):
                            continue
                        placement_motion = float(np.linalg.norm(
                            place_ik.q - grasp_ik.q,
                        ))
                        branch_minimum_manipulability = min(
                            trajectory_minimum_manipulability,
                            place_ik.manipulability,
                            retreat_ik.manipulability,
                        )
                        candidates.append((
                            home_distance,
                            rotation_travel,
                            -branch_minimum_manipulability,
                            placement_motion,
                            serial,
                            pick,
                            placement_rotation,
                            place_position,
                        ))
                        serial += 1
        if not candidates:
            raise PlanningInfeasibleError(
                "No feasible lying-object grasp/placement branch"
            )
        selected = min(
            candidates,
            key=lambda item: (item[2], item[0], item[1], item[3], item[4]),
        )
        return selected[5], selected[6], selected[7]

    @staticmethod
    def _alignment_delta(
        object_axis: np.ndarray,
        decal_axis: np.ndarray,
        symmetry_rad: float,
    ) -> float:
        source = float(np.arctan2(object_axis[1], object_axis[0]))
        target = float(np.arctan2(decal_axis[1], decal_axis[0]))
        delta = target - source
        return float(
            (delta + 0.5 * symmetry_rad) % symmetry_rad
            - 0.5 * symmetry_rad
        )

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *,
        candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        block_name = self._active_object()
        variant_name = block_name.removeprefix("paper_object_")
        object_kind, size_name, *pose_parts = variant_name.split("_")
        upright = pose_parts == ["upright"]
        if pose_parts not in ([], ["upright"]):
            raise RuntimeError(f"Unknown paper-placement pose: {variant_name}")
        block_geom_name = f"{block_name}_geom"
        region_name = f"paper_decal_{variant_name}"
        block_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, block_name,
        )
        block_geom = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, block_geom_name,
        )
        region_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, region_name,
        )

        lying_with_wrist_candidates = (
            not upright and object_kind in ("box", "cylinder")
        )
        lying_picks = None
        placement_count = 2 if lying_with_wrist_candidates else 4
        if lying_with_wrist_candidates:
            lying_picks = self._lying_pick_candidates(
                data,
                block_name,
                block_geom_name,
                object_kind,
                current_position,
                current_rotation,
                limits,
            )
            pick = lying_picks[0]
            if candidate_index is not None:
                if type(candidate_index) is not int or not 0 <= candidate_index < 2 * len(lying_picks):
                    raise ValueError("Invalid paper grasp/placement candidate index")
                pick = lying_picks[candidate_index // 2]
        elif object_kind == "cylinder":
            pick = self._pick_cylinder(
                data,
                block_name,
                block_geom_name,
                current_position,
                current_rotation,
                limits,
                upright=upright,
            )
        else:
            pick = PickBox(
                self.model,
                body_name=block_name,
                geom_name=block_geom_name,
                gripper_envelope=gripper_vertical_envelope(
                    self.model, self.handles,
                ),
                approach_clearance_m=0.085,
                lift_clearance_m=0.095,
                gripper_offset_z_adjustment_m=(
                    _SMALLEST_CUBE_GRASP_Z_ADJUSTMENT_M
                    if (
                        object_kind == "cube"
                        and size_name == "2"
                        and self.handles.spec.uid == "panda"
                    )
                    else 0.0
                ),
                object_label=self.object_label,
            ).plan(
                data,
                current_position,
                current_rotation,
                limits,
                grasp_index=None if candidate_index is None else candidate_index // placement_count,
            )

        block_center = data.geom_xpos[block_geom].copy()
        object_in_gripper = (
            pick.grasp_rotation.T @ (block_center - pick.grasp_position)
        )
        object_rotation = data.xmat[block_body].reshape(3, 3)
        decal_rotation = data.xmat[region_body].reshape(3, 3)
        reference_axis = (
            object_rotation[:, 0]
            if object_kind == "cube" or upright
            else object_rotation[:, 2]
        )
        symmetry = (
            0.5 * np.pi
            if object_kind == "cube" or (upright and object_kind == "box")
            else np.pi
        )
        yaw_delta = (
            0.0
            if upright and object_kind == "cylinder"
            else self._alignment_delta(
                reference_axis,
                decal_rotation[:, 0],
                symmetry,
            )
        )
        if candidate_index is not None:
            if lying_picks is None and object_kind not in ("box", "cube"):
                raise ValueError("Explicit candidates require a box or lying cylinder")
            yaw_delta += (candidate_index % placement_count) * (2 * np.pi / placement_count)
        cosine = np.cos(yaw_delta)
        sine = np.sin(yaw_delta)
        yaw_rotation = np.array((
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ))
        placement_rotation = yaw_rotation @ pick.grasp_rotation
        table_body = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        table_geom = int(self.model.body_geomadr[table_body])
        table_top = float(
            data.geom_xpos[table_geom, 2]
            + self.model.geom_size[table_geom, 2]
        )
        desired_center = data.xpos[region_body].copy()
        if (
            upright
            and object_kind == "box"
            and self.handles.spec.uid == "xarm7"
            and float(self.model.geom_size[block_geom, 0]) <= 0.0151
        ):
            # The narrow upright box settles consistently away from the xArm
            # tool axis after closure. Compensate the measured world-frame
            # residual; wider variants do not exhibit this slip.
            desired_center[:2] += (-0.004, -0.005)
        object_height = 2.0 * pick.object_vertical_half_extent
        release_clearance = (
            0.012
            if object_height <= 0.0201
            else 0.009
            if object_height <= 0.0301
            else 0.001
        )
        desired_center[2] = (
            table_top
            + pick.object_vertical_half_extent
            + release_clearance
        )
        place_position = (
            desired_center - placement_rotation @ object_in_gripper
        )
        transfer_height = max(
            float(pick.lifted_position[2]),
            float(place_position[2] + 0.100),
            0.510,
        )
        if lying_picks is not None and candidate_index is None:
            planner = _CpuMotionPlanner(
                self.model,
                data,
                self.handles,
                block_body,
                table_geom,
                table_top,
            )
            pick, placement_rotation, place_position = self._select_lying_plan(
                data,
                planner,
                lying_picks,
                object_rotation,
                reference_axis,
                decal_rotation[:, 0],
                desired_center,
                block_center,
                transfer_height,
            )
            # Equivalent grasp signs share geometry but not their gripper-frame
            # object offset. Recompute the height in case that assumption ever
            # changes in an asset revision.
            object_in_gripper = (
                pick.grasp_rotation.T @ (block_center - pick.grasp_position)
            )
            transfer_height = max(
                float(pick.lifted_position[2]),
                float(place_position[2] + 0.100),
                0.510,
            )
        raised = pick.lifted_position.copy()
        raised[2] = transfer_height
        above = place_position.copy()
        above[2] = transfer_height

        lift_limits = _limited(limits, 0.105)
        transfer_limits = MotionLimits(
            linear_speed_m_s=min(
                limits.linear_speed_m_s,
                0.080 if object_kind != "cube" else 0.110,
            ),
            angular_speed_rad_s=min(
                limits.angular_speed_rad_s,
                math.radians(
                    30.0 if object_kind != "cube" else 60.0,
                ),
            ),
            command_rate_hz=limits.command_rate_hz,
        )
        lower_limits = _limited(limits, 0.070)
        frames = list(pick.frames)
        if transfer_height > pick.lifted_position[2] + 1e-9:
            append_blended_motion(
                frames,
                pick.lifted_position,
                pick.grasp_rotation,
                raised,
                pick.grasp_rotation,
                True,
                "Raise block to transfer height",
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
            f"Move {self.object_label} above matching paper",
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
            f"Lower {self.object_label} onto matching paper",
            lower_limits,
            start_speed_m_s=lower_limits.linear_speed_m_s,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            place_position,
            placement_rotation,
            True,
            "Hold aligned above matching paper",
            12,
        )
        append_dwell(
            frames,
            place_position,
            placement_rotation,
            False,
            f"Release {self.object_label}",
            20,
        )
        retreat = place_position.copy()
        retreat[2] += 0.100
        append_blended_motion(
            frames,
            place_position,
            placement_rotation,
            retreat,
            placement_rotation,
            False,
            "Retract after placement",
            lift_limits,
            start_speed_m_s=0.0,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            retreat,
            placement_rotation,
            False,
            f"{self.object_label.capitalize()} placed on paper",
            20,
        )
        return tuple(frames)
