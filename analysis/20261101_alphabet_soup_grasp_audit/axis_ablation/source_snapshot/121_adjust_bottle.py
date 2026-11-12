"""CPU candidate search and RRTConnect planning for Adjust Bottle."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import mujoco
import numpy as np

from ...config import GRIPPER_TRAVEL_TIME
from ...geometry import axis_angle_matrix, rotation_matrix, rotation_vector
from ...simulation.robot import RobotHandles
from .pick_place import (
    GripperVerticalEnvelope,
    append_blended_motion,
    append_dwell,
    gripper_close_dwell_ticks,
    gripper_preclose_dwell_ticks,
    gripper_vertical_envelope,
)
from .types import PREGRASP_CLEARANCE_M, MotionLimits, TaskFrame, PlanningInfeasibleError


_BOTTLE_BODY_PREFIX = "adjust_bottle_"
_FIXED_GRASP_AXIAL_FRACTION = 0.52
_GRASP_AXIAL_FRACTIONS = (_FIXED_GRASP_AXIAL_FRACTION,)
# Offline Franka calibration over the held-out Adjust Bottle SQP seeds found
# this single table-relative point feasible for every tested bottle pose.
_UNIFIED_RELEASE_OFFSET_M = np.array((-0.100, 0.0))
_FINAL_BASE_YAWS_RAD = (-0.5 * math.pi, 0.5 * math.pi)
_PREGRASP_CLEARANCE_M = PREGRASP_CLEARANCE_M
_LIFT_CLEARANCE_M = 0.115
_TURN_CLEARANCE_M = 0.140
_TABLE_EDGE_MARGIN_M = 0.035
_RRT_TOP_K = 24
_RRT_STEP_RAD = 0.12
_RRT_EDGE_RESOLUTION_RAD = 0.045
_RRT_MAX_ITERATIONS = 320
_TRAJECTORY_MAX_JOINT_STEP_RAD = 0.035
_GRASP_EE_ROTATION_MAX_RAD = math.radians(100.0)
_BOTTLE_TRANSFER_SPEED_M_S = 0.100
_BOTTLE_LOWER_HANDOFF_SPEED_M_S = 0.045
_BOTTLE_ACCELERATION_TIME_S = 0.500
_BOTTLE_TRANSFER_DECELERATION_TIME_S = 1.000
_BOTTLE_LOWER_DECELERATION_TIME_S = 2.000
# Candidate ranking is deliberately conservative; execution is owned by the
# preset runtime shared with the UI.


def _is_allowed_final_body_rotation(
    body_rotation: np.ndarray,
    *,
    atol: float = 1e-8,
) -> bool:
    """Return whether the bottle is at either absolute base-frame 90-degree yaw."""
    base_up = np.array((0.0, 0.0, 1.0))
    return any(
        np.allclose(
            body_rotation,
            axis_angle_matrix(base_up, yaw),
            atol=atol,
            rtol=0.0,
        )
        for yaw in _FINAL_BASE_YAWS_RAD
    )


@dataclass(frozen=True)
class _IKSolution:
    q: np.ndarray
    manipulability: float


@dataclass(frozen=True)
class _GraspBranch:
    index: int
    grasp_position: np.ndarray
    grasp_rotation: np.ndarray
    pregrasp_position: np.ndarray
    lifted_position: np.ndarray
    body_rotation_in_gripper: np.ndarray
    body_offset_in_gripper: np.ndarray
    pregrasp_ik: _IKSolution
    grasp_ik: _IKSolution
    lifted_ik: _IKSolution


def _select_rotation_efficient_grasp_branches(
    branches: list[_GraspBranch],
    current_rotation: np.ndarray,
) -> list[_GraspBranch]:
    """Prefer grasp branches requiring at most 100 deg of EE rotation."""
    if len(branches) < 2:
        return branches
    safe = [
        branch for branch in branches
        if float(np.linalg.norm(rotation_vector(
            current_rotation.T @ branch.grasp_rotation,
        ))) <= _GRASP_EE_ROTATION_MAX_RAD
    ]
    # Return the preferred subset when one exists.  The caller retains the
    # other branches as a feasibility fallback until planning has completed.
    return safe if safe else branches


@dataclass(frozen=True)
class _PlacementCandidate:
    branch: _GraspBranch
    placement_xy: np.ndarray
    final_base_yaw: float
    body_position: np.ndarray
    body_rotation: np.ndarray
    transfer_position: np.ndarray
    transfer_rotation: np.ndarray
    transfer_path: tuple[np.ndarray, ...]
    rotated_position: np.ndarray
    rotated_rotation: np.ndarray
    rotation_path: tuple[np.ndarray, ...]
    place_position: np.ndarray
    place_rotation: np.ndarray
    place_ik: _IKSolution
    prescore: float
    serial: int


@dataclass(frozen=True)
class _PlannedCandidate:
    candidate: _PlacementCandidate
    approach_path: tuple[np.ndarray, ...]
    pregrasp_cost: float
    postgrasp_cost: float
    cost: float


def _required_body(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Bottle task body is missing: {name}")
    return int(body_id)


def _mesh_vertices_in_body(
    model: mujoco.MjModel,
    geom_id: int,
) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise RuntimeError("Bottle collision geometry must be a mesh")
    first = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[first:first + count]
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, model.geom_quat[geom_id])
    return (
        vertices @ rotation_flat.reshape(3, 3).T
        + model.geom_pos[geom_id]
    )


def _bottle_local_geometry(
    model: mujoco.MjModel,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the active bottle's physical local bounds and vertices."""
    geom_ids = np.flatnonzero(
        (model.geom_bodyid == body_id)
        & (model.geom_contype != 0)
    )
    vertices: list[np.ndarray] = []
    for raw_geom_id in geom_ids:
        geom_id = int(raw_geom_id)
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            vertices.append(_mesh_vertices_in_body(model, geom_id))
            continue
        rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(rotation_flat, model.geom_quat[geom_id])
        rotation = rotation_flat.reshape(3, 3)
        size = model.geom_size[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis = rotation[:, 2]
            extent = (
                np.abs(axis) * size[1]
                + size[0] * np.sqrt(np.maximum(0.0, 1.0 - axis * axis))
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            axis = rotation[:, 2]
            extent = np.abs(axis) * size[1] + size[0]
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            extent = np.full(3, size[0])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = np.abs(rotation) @ size
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            extent = np.sqrt((rotation * rotation) @ (size * size))
        else:
            continue
        center = model.geom_pos[geom_id]
        vertices.append(np.vstack((center - extent, center + extent)))
    if not vertices:
        raise RuntimeError("Bottle has no supported collision geometry")
    combined = np.vstack(vertices)
    return np.min(combined, axis=0), np.max(combined, axis=0), combined


def _table_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[int, float, np.ndarray]:
    table_id = _required_body(model, "work_surface")
    geom_ids = np.flatnonzero(model.geom_bodyid == table_id)
    if not geom_ids.size:
        raise RuntimeError("Work surface has no geometry")
    geom_id = int(geom_ids[np.argmax(
        np.prod(model.geom_size[geom_ids, :2], axis=1),
    )])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    center = data.geom_xpos[geom_id]
    extent = np.abs(rotation) @ model.geom_size[geom_id]
    top = float(center[2] + extent[2])
    bounds = np.column_stack((center[:2] - extent[:2], center[:2] + extent[:2]))
    return geom_id, top, bounds


class _CpuMotionPlanner:
    """Side-effect-free IK, collision checks, and bidirectional RRTConnect."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        handles: RobotHandles,
        bottle_body: int,
        table_geom: int,
        table_top: float,
    ) -> None:
        self.model = model
        self.handles = handles
        self.bottle_body = int(bottle_body)
        self.table_body = int(model.geom_bodyid[int(table_geom)])
        self.table_top = float(table_top)
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = data.qpos
        self.robot_root = int(
            model.body_rootid[int(handles.link_body_ids[0])]
        )
        self.robot_bodies = {
            body_id for body_id in range(model.nbody)
            if int(model.body_rootid[body_id]) == self.robot_root
        }
        self.finger_bodies = set()
        for name in handles.spec.finger_joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name,
            )
            if joint_id >= 0:
                self.finger_bodies.add(int(model.jnt_bodyid[joint_id]))
        bottle_joint = int(model.body_jntadr[self.bottle_body])
        if (
            bottle_joint < 0
            or int(model.jnt_type[bottle_joint])
            != int(mujoco.mjtJoint.mjJNT_FREE)
        ):
            raise RuntimeError("Adjust Bottle body must use a free joint")
        bottle_qpos = int(model.jnt_qposadr[bottle_joint])
        self.bottle_qpos_slice = slice(bottle_qpos, bottle_qpos + 7)
        self.initial_bottle_qpos = data.qpos[
            self.bottle_qpos_slice
        ].copy()
        self.carried_body_rotation_in_gripper: np.ndarray | None = None
        self.carried_body_offset_in_gripper: np.ndarray | None = None

    def set_carried_transform(
        self,
        body_rotation_in_gripper: np.ndarray,
        body_offset_in_gripper: np.ndarray,
    ) -> None:
        self.carried_body_rotation_in_gripper = np.asarray(
            body_rotation_in_gripper,
            dtype=float,
        ).copy()
        self.carried_body_offset_in_gripper = np.asarray(
            body_offset_in_gripper,
            dtype=float,
        ).copy()

    def _set_q(self, q: np.ndarray, *, carried: bool = False) -> None:
        self.data.qpos[self.bottle_qpos_slice] = self.initial_bottle_qpos
        self.data.qpos[self.handles.qpos_ids] = q
        mujoco.mj_forward(self.model, self.data)
        if carried:
            if (
                self.carried_body_rotation_in_gripper is None
                or self.carried_body_offset_in_gripper is None
            ):
                return
            ee_position = self.data.site_xpos[
                self.handles.ee_site_id
            ].copy()
            ee_rotation = self.data.site_xmat[
                self.handles.ee_site_id
            ].reshape(3, 3).copy()
            body_position = (
                ee_position
                + ee_rotation @ self.carried_body_offset_in_gripper
            )
            body_rotation = (
                ee_rotation @ self.carried_body_rotation_in_gripper
            )
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, body_rotation.reshape(9))
            self.data.qpos[self.bottle_qpos_slice.start:
                           self.bottle_qpos_slice.start + 3] = body_position
            self.data.qpos[self.bottle_qpos_slice.start + 3:
                           self.bottle_qpos_slice.stop] = quaternion
            mujoco.mj_forward(self.model, self.data)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._set_q(q)
        return (
            self.data.site_xpos[self.handles.ee_site_id].copy(),
            self.data.site_xmat[self.handles.ee_site_id]
            .reshape(3, 3).copy(),
        )

    def configuration_valid(
        self,
        q: np.ndarray,
        *,
        ignore_bottle: bool = False,
        carried: bool = False,
        allow_bottle_support: bool = False,
        allow_low_carried_pose: bool = False,
    ) -> bool:
        attached = bool(
            carried
            and self.carried_body_rotation_in_gripper is not None
            and self.carried_body_offset_in_gripper is not None
        )
        self._set_q(q, carried=attached)
        if carried and not allow_low_carried_pose and float(
            self.data.site_xpos[self.handles.ee_site_id, 2]
        ) < self.table_top + 0.105:
            return False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if float(contact.dist) > 1e-6:
                continue
            body_a = int(self.model.geom_bodyid[int(contact.geom1)])
            body_b = int(self.model.geom_bodyid[int(contact.geom2)])
            if body_a == body_b:
                continue
            robot_a = body_a in self.robot_bodies
            robot_b = body_b in self.robot_bodies
            if robot_a and robot_b:
                if {body_a, body_b} <= self.finger_bodies:
                    continue
                return False
            if not (robot_a or robot_b):
                if attached and self.bottle_body in (body_a, body_b):
                    other_body = (
                        body_b if body_a == self.bottle_body else body_a
                    )
                    if (
                        allow_bottle_support
                        and other_body == self.table_body
                    ):
                        continue
                    return False
                continue
            other = body_b if robot_a else body_a
            robot_body = body_a if robot_a else body_b
            if other == self.bottle_body and ignore_bottle:
                if attached and robot_body not in self.finger_bodies:
                    return False
                continue
            if (
                other == self.bottle_body
                and robot_body in self.finger_bodies
            ):
                continue
            return False
        return True

    def solve_ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        *,
        ignore_bottle: bool = False,
        carried: bool = False,
        allow_bottle_support: bool = False,
        allow_low_carried_pose: bool = False,
    ) -> _IKSolution | None:
        """Solve strict key poses with the maintained IPOPT backend."""
        from ...control.pose import TargetPose
        from ...control.ipopt import IpoptIK, IpoptSettings

        best: _IKSolution | None = None
        for seed in seeds:
            initial = np.clip(
                np.asarray(seed, dtype=float),
                self.handles.joint_lower,
                self.handles.joint_upper,
            ).copy()
            self._set_q(initial)
            controller = IpoptIK(
                self.model,
                self.handles,
                IpoptSettings(
                    max_iterations=100,
                    max_cpu_time_s=1.0e6,
                ),
            )
            controller.set_warm_start(initial)
            q, diagnostics = controller.solve(
                self.data,
                TargetPose(
                    np.asarray(position, dtype=float).copy(),
                    np.asarray(rotation, dtype=float).copy(),
                ),
                np.ones(6, dtype=bool),
            )
            if not diagnostics.feasible or not self.configuration_valid(
                q,
                ignore_bottle=ignore_bottle,
                carried=carried,
                allow_bottle_support=allow_bottle_support,
                allow_low_carried_pose=allow_low_carried_pose,
            ):
                continue
            manipulability = float(diagnostics.manipulability or 0.0)
            solution = _IKSolution(q.copy(), manipulability)
            # Seeds are ordered by continuity preference. The optimizer
            # already enforces physical bounds, so accept the first feasible
            # branch instead of re-ranking solutions by limit clearance.
            if best is None:
                best = solution
        return best

    def edge_valid(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        ignore_bottle: bool,
        carried: bool,
        allow_bottle_support: bool = False,
        allow_bottle_support_at_goal: bool = False,
        allow_low_carried_pose: bool = False,
    ) -> bool:
        count = max(1, int(math.ceil(
            float(np.max(np.abs(goal - start)))
            / _RRT_EDGE_RESOLUTION_RAD
        )))
        return all(
            self.configuration_valid(
                start + (goal - start) * (index / count),
                ignore_bottle=ignore_bottle,
                carried=carried,
                allow_bottle_support=(
                    allow_bottle_support
                    or (allow_bottle_support_at_goal and index == count)
                ),
                allow_low_carried_pose=allow_low_carried_pose,
            )
            for index in range(1, count + 1)
        )

    @staticmethod
    def _trace(
        nodes: list[np.ndarray],
        parents: list[int],
        index: int,
    ) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        path.reverse()
        return path

    def rrt_connect(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        rng: np.random.Generator,
        *,
        ignore_bottle: bool,
        carried: bool,
    ) -> tuple[np.ndarray, ...] | None:
        if self.edge_valid(
            start,
            goal,
            ignore_bottle=ignore_bottle,
            carried=carried,
        ):
            return tuple(self._resample_path((start, goal)))

        nodes_a = [start.copy()]
        parents_a = [-1]
        nodes_b = [goal.copy()]
        parents_b = [-1]
        a_from_start = True

        def extend(
            nodes: list[np.ndarray],
            parents: list[int],
            target: np.ndarray,
        ) -> tuple[int | None, bool]:
            distances = [float(np.linalg.norm(node - target)) for node in nodes]
            nearest_index = int(np.argmin(distances))
            nearest = nodes[nearest_index]
            distance = distances[nearest_index]
            if distance <= 1e-12:
                return nearest_index, True
            candidate = nearest + (target - nearest) * min(
                1.0, _RRT_STEP_RAD / distance,
            )
            if not self.edge_valid(
                nearest,
                candidate,
                ignore_bottle=ignore_bottle,
                carried=carried,
            ):
                return None, False
            nodes.append(candidate)
            parents.append(nearest_index)
            return len(nodes) - 1, distance <= _RRT_STEP_RAD

        for iteration in range(_RRT_MAX_ITERATIONS):
            if iteration % 5 == 0:
                sample = nodes_b[0]
            else:
                sample = rng.uniform(
                    self.handles.joint_lower,
                    self.handles.joint_upper,
                )
            index_a, _ = extend(nodes_a, parents_a, sample)
            if index_a is not None:
                target = nodes_a[index_a]
                index_b: int | None = None
                reached = False
                while True:
                    index_b, reached = extend(nodes_b, parents_b, target)
                    if index_b is None or reached:
                        break
                if reached and index_b is not None:
                    path_a = self._trace(nodes_a, parents_a, index_a)
                    path_b = self._trace(nodes_b, parents_b, index_b)
                    if a_from_start:
                        joined = path_a + list(reversed(path_b[:-1]))
                    else:
                        joined = path_b + list(reversed(path_a[:-1]))
                    return tuple(self._shortcut_and_resample(
                        joined,
                        rng,
                        ignore_bottle=ignore_bottle,
                        carried=carried,
                    ))
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            a_from_start = not a_from_start
        return None

    @staticmethod
    def _resample_path(path: tuple[np.ndarray, ...] | list[np.ndarray]) -> list[np.ndarray]:
        result = [np.asarray(path[0], dtype=float).copy()]
        for raw_goal in path[1:]:
            start = result[-1]
            goal = np.asarray(raw_goal, dtype=float)
            count = max(1, int(math.ceil(
                float(np.max(np.abs(goal - start)))
                / _TRAJECTORY_MAX_JOINT_STEP_RAD
            )))
            result.extend(
                start + (goal - start) * (index / count)
                for index in range(1, count + 1)
            )
        return result

    def _shortcut_and_resample(
        self,
        path: list[np.ndarray],
        rng: np.random.Generator,
        *,
        ignore_bottle: bool,
        carried: bool,
    ) -> list[np.ndarray]:
        shortened = [node.copy() for node in path]
        for _ in range(40):
            if len(shortened) <= 2:
                break
            first, second = sorted(rng.integers(0, len(shortened), 2))
            if second <= first + 1:
                continue
            if self.edge_valid(
                shortened[first],
                shortened[second],
                ignore_bottle=ignore_bottle,
                carried=carried,
            ):
                shortened[first + 1:second] = []
        return self._resample_path(shortened)


class StandFallenBottle:
    """Select a fixed grasp with safe joints and small post-grasp excursion."""

    def __init__(
        self,
        model: mujoco.MjModel,
        handles: RobotHandles,
    ) -> None:
        self.model = model
        self.handles = handles
        self.body_ids = tuple(
            body_id for body_id in range(model.nbody)
            if (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                ) or ""
            ).startswith(_BOTTLE_BODY_PREFIX)
        )
        if not self.body_ids:
            raise RuntimeError("Adjust Bottle scene contains no bottle variants")
        self.gripper_envelope: GripperVerticalEnvelope = (
            gripper_vertical_envelope(model, handles)
        )
        self.last_planning_stats: dict[str, object] = {}

    def _active_body(self, data: mujoco.MjData) -> int:
        active = [
            body_id for body_id in self.body_ids
            if float(data.xpos[body_id, 2]) > 0.0
        ]
        if len(active) != 1:
            raise RuntimeError(
                f"Adjust Bottle requires exactly one active model, got {len(active)}",
            )
        return active[0]

    @staticmethod
    def _path_cost(path: tuple[np.ndarray, ...]) -> float:
        if len(path) < 2:
            return 0.0
        deltas = np.diff(np.asarray(path), axis=0)
        length = float(np.sum(np.linalg.norm(deltas, axis=1)))
        smoothness = (
            float(np.sum(np.linalg.norm(np.diff(deltas, axis=0), axis=1)))
            if len(deltas) > 1 else 0.0
        )
        return length + 0.25 * smoothness

    def _redundant_ik_seeds(
        self,
        *references: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Explore Panda elbow/wrist redundancy for the two terminal turns."""
        seeds: list[np.ndarray] = []
        for reference in references:
            base = np.asarray(reference, dtype=float).copy()
            seeds.append(base)
            if base.size != 7:
                continue
            for elbow_delta, wrist_delta in (
                (0.0, -0.5 * math.pi),
                (0.0, 0.5 * math.pi),
                (-0.45, 0.0),
                (0.45, 0.0),
                (-0.45, -0.5 * math.pi),
                (0.45, 0.5 * math.pi),
            ):
                seed = base.copy()
                seed[2] += elbow_delta
                seed[-1] += wrist_delta
                seeds.append(np.clip(
                    seed,
                    self.handles.joint_lower,
                    self.handles.joint_upper,
                ))
        return tuple(seeds)

    def _grasp_positions(
        self,
        data: mujoco.MjData,
        body_id: int,
        local_min: np.ndarray,
        local_max: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return the fixed grasp templates independently of IK feasibility."""
        radius = 0.5 * float(np.max(local_max[:2] - local_min[:2]))
        deepest, shallowest = self.gripper_envelope.object_center_offset_range(radius)
        offset_z = 0.5 * (deepest + shallowest)
        if self.handles.spec.uid in ("xarm7", "ur5e"):
            offset_z += 0.010
        center = 0.5 * (local_min[:2] + local_max[:2])
        positions = []
        for fraction in _GRASP_AXIAL_FRACTIONS:
            local_point = np.array((
                center[0], center[1],
                local_min[2] + fraction * (local_max[2] - local_min[2]),
            ))
            point = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_point
            point[2] += offset_z
            positions.append(point)
        return tuple(positions)

    def grasp_position(self, data: mujoco.MjData) -> np.ndarray:
        """Expose the fixed grasp target for scene placement, even beyond reach."""
        mujoco.mj_forward(self.model, data)
        body_id = self._active_body(data)
        local_min, local_max, _ = _bottle_local_geometry(self.model, body_id)
        return self._grasp_positions(data, body_id, local_min, local_max)[0].copy()

    @staticmethod
    def _grasp_rotation(body_rotation: np.ndarray, sign: float) -> np.ndarray:
        axis = body_rotation[:, 2].copy()
        axis[2] = 0.0
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        up = np.array((0.0, 0.0, 1.0))
        closing = np.cross(up, axis)
        closing = sign * closing / np.linalg.norm(closing)
        tool_x = np.cross(closing, -up)
        tool_x /= np.linalg.norm(tool_x)
        return np.column_stack((tool_x, closing, -up))

    @staticmethod
    def _release_body_position(local_min, local_max, table_top, table_bounds):
        xy = np.mean(table_bounds, axis=1) + _UNIFIED_RELEASE_OFFSET_M
        margin = 0.5 * float(np.max(local_max[:2] - local_min[:2])) + _TABLE_EDGE_MARGIN_M
        if np.any(xy < table_bounds[:, 0] + margin) or np.any(xy > table_bounds[:, 1] - margin):
            raise RuntimeError("Unified Adjust Bottle release point is off table")
        return np.array((xy[0], xy[1], table_top - local_min[2]))

    def _fixed_grasp_branches(
        self,
        data: mujoco.MjData,
        planner: _CpuMotionPlanner,
        body_id: int,
        local_min: np.ndarray,
        local_max: np.ndarray,
        current_q: np.ndarray,
        *, grasp_index: int | None = None,
    ) -> list[_GraspBranch]:
        body_position = data.xpos[body_id].copy()
        body_rotation = data.xmat[body_id].reshape(3, 3).copy()
        branches: list[_GraspBranch] = []
        serial = 0
        for point in self._grasp_positions(data, body_id, local_min, local_max):
            for sign in (1.0, -1.0):
                if grasp_index is not None and serial != grasp_index:
                    serial += 1
                    continue
                grasp_rotation = self._grasp_rotation(body_rotation, sign)
                grasp_position = point.copy()
                pregrasp_position = grasp_position.copy()
                pregrasp_position[2] += _PREGRASP_CLEARANCE_M
                lifted_position = grasp_position.copy()
                lifted_position[2] += _LIFT_CLEARANCE_M
                wrist_seed = current_q.copy()
                wrist_seed[-1] = np.clip(
                    wrist_seed[-1] + sign * 0.5 * math.pi,
                    self.handles.joint_lower[-1],
                    self.handles.joint_upper[-1],
                )
                seeds = (current_q, wrist_seed)
                pregrasp_ik = planner.solve_ik(
                    pregrasp_position,
                    grasp_rotation,
                    seeds,
                )
                if pregrasp_ik is None:
                    serial += 1
                    continue
                grasp_ik = planner.solve_ik(
                    grasp_position,
                    grasp_rotation,
                    (pregrasp_ik.q, current_q),
                    ignore_bottle=True,
                )
                if grasp_ik is None:
                    serial += 1
                    continue
                body_rotation_in_gripper = grasp_rotation.T @ body_rotation
                body_offset_in_gripper = grasp_rotation.T @ (
                    body_position - grasp_position
                )
                planner.set_carried_transform(
                    body_rotation_in_gripper,
                    body_offset_in_gripper,
                )
                lifted_ik = planner.solve_ik(
                    lifted_position,
                    grasp_rotation,
                    (grasp_ik.q, pregrasp_ik.q),
                    ignore_bottle=True,
                    carried=True,
                )
                if lifted_ik is None:
                    serial += 1
                    continue
                if not planner.edge_valid(
                    grasp_ik.q,
                    lifted_ik.q,
                    ignore_bottle=True,
                    carried=True,
                    allow_bottle_support=True,
                    allow_low_carried_pose=True,
                ):
                    serial += 1
                    continue
                branches.append(_GraspBranch(
                    serial,
                    grasp_position,
                    grasp_rotation,
                    pregrasp_position,
                    lifted_position,
                    body_rotation_in_gripper,
                    body_offset_in_gripper,
                    pregrasp_ik,
                    grasp_ik,
                    lifted_ik,
                ))
                serial += 1
        return branches

    def _placement_candidates(
        self,
        data: mujoco.MjData,
        planner: _CpuMotionPlanner,
        branches: list[_GraspBranch],
        local_min: np.ndarray,
        local_max: np.ndarray,
        table_top: float,
        table_bounds: np.ndarray,
        *, placement_index: int | None = None,
    ) -> tuple[list[_PlacementCandidate], int]:
        del data
        body_position = self._release_body_position(local_min, local_max, table_top, table_bounds)
        placement_xy = body_position[:2].copy()

        candidates: list[_PlacementCandidate] = []
        sampled = 0
        serial = 0
        base_up = np.array((0.0, 0.0, 1.0))
        body_above = body_position + np.array((0.0, 0.0, _TURN_CLEARANCE_M))
        for branch in branches:
            planner.set_carried_transform(
                branch.body_rotation_in_gripper,
                branch.body_offset_in_gripper,
            )
            transfer_rotation = branch.grasp_rotation
            transfer_position = (
                body_above
                - transfer_rotation @ branch.body_offset_in_gripper
            )
            for yaw_index, final_base_yaw in enumerate(_FINAL_BASE_YAWS_RAD):
                if placement_index is not None:
                    if yaw_index != placement_index:
                        continue
                    serial = branch.index * 2 + yaw_index
                sampled += 1
                body_rotation = axis_angle_matrix(
                    base_up,
                    final_base_yaw,
                )
                rotated_rotation = (
                    body_rotation
                    @ branch.body_rotation_in_gripper.T
                )
                if not _is_allowed_final_body_rotation(
                    body_rotation,
                ):
                    raise RuntimeError(
                        "Adjust Bottle final body pose is not base-frame left/right 90 degrees",
                    )
                rotated_position = (
                    body_above
                    - rotated_rotation @ branch.body_offset_in_gripper
                )
                rotated_ik = planner.solve_ik(
                    rotated_position,
                    rotated_rotation,
                    self._redundant_ik_seeds(
                        branch.lifted_ik.q,
                        branch.pregrasp_ik.q,
                        branch.grasp_ik.q,
                    ),
                    ignore_bottle=True,
                    carried=True,
                )
                if rotated_ik is None:
                    if os.environ.get("FRANKA_DEBUG_ADJUST_BOTTLE_REJECTIONS"):
                        print(
                            "Adjust Bottle candidate rejected: "
                            f"branch={branch.index} serial={serial} "
                            f"yaw_deg={math.degrees(final_base_yaw):.1f} "
                            "stage=rotated_ik"
                        )
                    serial += 1
                    continue
                translation_path = (branch.lifted_ik.q, rotated_ik.q)
                rotation_path = translation_path
                place_rotation = rotated_rotation
                place_position = (
                    body_position
                    - place_rotation @ branch.body_offset_in_gripper
                )
                place_ik = planner.solve_ik(
                    place_position,
                    place_rotation,
                    self._redundant_ik_seeds(
                        rotated_ik.q,
                        branch.lifted_ik.q,
                        branch.pregrasp_ik.q,
                    ),
                    ignore_bottle=True,
                    carried=True,
                    allow_bottle_support=True,
                    allow_low_carried_pose=True,
                )
                if place_ik is None:
                    if os.environ.get("FRANKA_DEBUG_ADJUST_BOTTLE_REJECTIONS"):
                        print(
                            "Adjust Bottle candidate rejected: "
                            f"branch={branch.index} serial={serial} "
                            f"yaw_deg={math.degrees(final_base_yaw):.1f} "
                            "stage=place_ik"
                        )
                    serial += 1
                    continue
                translation_motion = self._path_cost(tuple(translation_path))
                rotation_motion = self._path_cost(tuple(rotation_path))
                descent_motion = float(np.linalg.norm(
                    place_ik.q - rotated_ik.q,
                ))
                manipulability = min(
                    rotated_ik.manipulability,
                    place_ik.manipulability,
                )
                prescore = (
                    1.2 * translation_motion
                    + 0.8 * rotation_motion
                    + 0.35 * descent_motion
                    + 0.08 / max(manipulability, 1e-4)
                )
                candidates.append(_PlacementCandidate(
                    branch,
                    placement_xy.copy(),
                    final_base_yaw,
                    body_position.copy(),
                    body_rotation,
                    transfer_position.copy(),
                    transfer_rotation.copy(),
                    tuple(q.copy() for q in translation_path),
                    rotated_position.copy(),
                    rotated_rotation.copy(),
                    tuple(q.copy() for q in rotation_path),
                    place_position,
                    place_rotation,
                    place_ik,
                    prescore,
                    serial,
                ))
                serial += 1
        candidates.sort(key=lambda item: (item.prescore, item.serial))
        return candidates, sampled

    def _select_rrt_candidate(
        self,
        planner: _CpuMotionPlanner,
        current_q: np.ndarray,
        candidates: list[_PlacementCandidate],
    ) -> tuple[tuple[_PlannedCandidate, ...], int]:
        solved: list[_PlannedCandidate] = []
        attempted = min(_RRT_TOP_K, len(candidates))
        for candidate in candidates[:_RRT_TOP_K]:
            planner.set_carried_transform(
                candidate.branch.body_rotation_in_gripper,
                candidate.branch.body_offset_in_gripper,
            )
            rng = np.random.Generator(np.random.PCG64(
                0xAD0B071E + candidate.serial,
            ))
            approach = planner.rrt_connect(
                current_q,
                candidate.branch.pregrasp_ik.q,
                rng,
                ignore_bottle=False,
                carried=False,
            )
            if approach is None:
                continue
            pregrasp_path = (*approach, candidate.branch.grasp_ik.q)
            postgrasp_path = (
                candidate.branch.grasp_ik.q,
                candidate.branch.lifted_ik.q,
                candidate.rotation_path[-1],
                candidate.place_ik.q,
            )
            pregrasp_cost = self._path_cost(pregrasp_path)
            postgrasp_cost = self._path_cost(postgrasp_path)
            cost = pregrasp_cost + postgrasp_cost
            solved.append(_PlannedCandidate(
                candidate,
                approach,
                pregrasp_cost,
                postgrasp_cost,
                cost,
            ))
        if not solved:
            raise RuntimeError(
                "CPU RRTConnect found no collision-free Adjust Bottle path",
            )
        solved.sort(key=lambda item: (
            item.pregrasp_cost,
            item.postgrasp_cost,
            item.candidate.serial,
        ))
        if os.environ.get("FRANKA_DEBUG_ADJUST_BOTTLE_SELECTION"):
            print(
                "Adjust Bottle stage cost debug: "
                f"candidates={[(int(item.candidate.serial), int(item.candidate.branch.index), float(item.pregrasp_cost), float(item.postgrasp_cost)) for item in solved]} "
                f"stage_ranked_serial={int(solved[0].candidate.serial)} "
                f"start_q_deg={np.degrees(current_q).round(3).tolist()}"
            )
        return tuple(solved), attempted

    def _build_frames(
        self,
        selected: _PlannedCandidate,
        planner: _CpuMotionPlanner,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        """Expand one CPU candidate into the exact Cartesian frame sequence."""
        candidate = selected.candidate
        frames: list[TaskFrame] = []
        self._append_joint_path(
            frames,
            planner,
            selected.approach_path,
            False,
            "CPU RRTConnect approach",
        )
        precision_limits = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.045),
            angular_speed_rad_s=min(
                limits.angular_speed_rad_s,
                math.radians(18.0),
            ),
            command_rate_hz=limits.command_rate_hz,
        )
        transfer_limits = MotionLimits(
            linear_speed_m_s=min(limits.linear_speed_m_s, 0.100),
            angular_speed_rad_s=min(
                limits.angular_speed_rad_s,
                math.radians(30.0),
            ),
            command_rate_hz=limits.command_rate_hz,
        )
        append_blended_motion(
            frames,
            candidate.branch.pregrasp_position,
            candidate.branch.grasp_rotation,
            candidate.branch.grasp_position,
            candidate.branch.grasp_rotation,
            False,
            "Descend to fixed bottle grasp",
            precision_limits,
            start_speed_m_s=0.0,
            end_speed_m_s=0.0,
        )
        append_dwell(
            frames,
            candidate.branch.grasp_position,
            candidate.branch.grasp_rotation,
            False,
            "Align at bottle",
            gripper_preclose_dwell_ticks(limits),
        )
        append_dwell(
            frames,
            candidate.branch.grasp_position,
            candidate.branch.grasp_rotation,
            True,
            "Close gripper on bottle",
            gripper_close_dwell_ticks(limits),
        )
        append_blended_motion(
            frames,
            candidate.branch.grasp_position,
            candidate.branch.grasp_rotation,
            candidate.branch.lifted_position,
            candidate.branch.grasp_rotation,
            True,
            "Lift bottle without rotation",
            limits,
            start_speed_m_s=0.0,
            end_speed_m_s=_BOTTLE_TRANSFER_SPEED_M_S,
        )
        self._append_transfer_and_turn(
            frames,
            candidate,
            transfer_limits,
            start_speed_m_s=_BOTTLE_TRANSFER_SPEED_M_S,
            end_speed_m_s=_BOTTLE_LOWER_HANDOFF_SPEED_M_S,
        )
        append_blended_motion(
            frames,
            candidate.rotated_position,
            candidate.rotated_rotation,
            candidate.place_position,
            candidate.place_rotation,
            True,
            "Lower bottle at unified release point",
            precision_limits,
            start_speed_m_s=_BOTTLE_LOWER_HANDOFF_SPEED_M_S,
            end_speed_m_s=0.0,
            deceleration_time_s=_BOTTLE_LOWER_DECELERATION_TIME_S,
        )
        append_dwell(
            frames,
            candidate.place_position,
            candidate.place_rotation,
            True,
            "Hold bottle upright",
            10,
        )
        gripper_release_frames = max(
            1,
            int(math.ceil(
                GRIPPER_TRAVEL_TIME * limits.command_rate_hz,
            )),
        )
        append_dwell(
            frames,
            candidate.place_position,
            candidate.place_rotation,
            False,
            "Release bottle",
            gripper_release_frames,
        )
        return tuple(frames)

    @staticmethod
    def _copy_kinematic_data(
        model: mujoco.MjModel,
        source: mujoco.MjData,
    ) -> mujoco.MjData:
        """Copy state for an isolated optimizer rollout without physics."""
        data = mujoco.MjData(model)
        data.qpos[:] = source.qpos
        data.qvel[:] = source.qvel
        if data.act.size:
            data.act[:] = source.act
        if data.ctrl.size:
            data.ctrl[:] = source.ctrl
        mujoco.mj_forward(model, data)
        return data

    def _kinematic_closed_loop_rollout(
        self,
        data: mujoco.MjData,
        frames: tuple[TaskFrame, ...],
    ) -> tuple[bool, str, int]:
        """Validate nominal frames through the maintained one-step optimizer.

        Each accepted optimizer command becomes the measured joint state for
        the next 10 Hz frame.  This intentionally performs no actuator,
        contact, gripper, or object-dynamics simulation: it answers only
        whether the selected nominal branch remains kinematically executable
        under the same closed-loop one-step solver used by the runtime.
        """
        from ...counterfactual.runner import CounterfactualTrialSpec, PreparedNominalTrial, run_counterfactual_trial
        from ..catalog import ADJUST_BOTTLE_TASK
        if not frames:
            return False, "empty_trajectory", 0
        result = run_counterfactual_trial(CounterfactualTrialSpec(
            scene="Adjust Bottle", task=ADJUST_BOTTLE_TASK, robot_uid=self.handles.spec.uid,
            maximum_frames=max(600, len(frames)), execution_mode="kinematic", angle_deg=0,
            nominal_plan_selection_enabled=False, sqp_loss_profile="delta_q_squared_only",
            sqp_loss_weights=(1., 0., 0.)), prepared_nominal=PreparedNominalTrial(
                self.model, data, self.handles.spec.uid, frames))
        return (bool(result.task_success and result.evaluation.get("trajectory_complete")),
                result.termination_reason, int(result.evaluation.get("executed_frames", 0)))

    def _select_kinematically_feasible_candidate(
        self,
        data: mujoco.MjData,
        planner: _CpuMotionPlanner,
        planned: tuple[_PlannedCandidate, ...],
        limits: MotionLimits,
    ) -> tuple[
        _PlannedCandidate,
        tuple[TaskFrame, ...],
        list[dict[str, object]],
    ]:
        """Try stage-cost-ranked candidates until optimizer rollout passes."""
        attempts: list[dict[str, object]] = []
        for item in planned:
            frames = self._build_frames(item, planner, limits)
            passed, reason, frame_count = self._kinematic_closed_loop_rollout(
                data,
                frames,
            )
            attempts.append({
                "serial": int(item.candidate.serial),
                "pregrasp_cost": float(item.pregrasp_cost),
                "postgrasp_cost": float(item.postgrasp_cost),
                "passed": bool(passed),
                "reason": reason,
                "validated_frames": int(frame_count),
            })
            if passed:
                return item, frames, attempts
        raise RuntimeError(
            "All Adjust Bottle candidates failed optimizer kinematic rollout: "
            f"{attempts}",
        )

    @staticmethod
    def _append_joint_path(
        frames: list[TaskFrame],
        planner: _CpuMotionPlanner,
        path: tuple[np.ndarray, ...],
        gripper_closed: bool,
        phase: str,
    ) -> None:
        for q in path[1:]:
            position, rotation = planner.fk(q)
            frames.append(TaskFrame(
                position,
                rotation,
                gripper_closed,
                phase,
            ))

    @staticmethod
    def _append_transfer_and_turn(
        frames: list[TaskFrame],
        candidate: _PlacementCandidate,
        limits: MotionLimits,
        start_speed_m_s: float,
        end_speed_m_s: float,
    ) -> None:
        """Translate/turn the bottle with nonzero smooth boundary speeds."""
        goal_body_position = (
            candidate.transfer_position
            + candidate.transfer_rotation
            @ candidate.branch.body_offset_in_gripper
        )
        start_body_position = (
            candidate.branch.lifted_position
            + candidate.branch.grasp_rotation
            @ candidate.branch.body_offset_in_gripper
        )
        body_travel = float(np.linalg.norm(
            goal_body_position - start_body_position,
        ))
        relative_rotation_vector = rotation_vector(
            candidate.branch.grasp_rotation.T
            @ candidate.rotated_rotation
        )
        rotation_angle = float(np.linalg.norm(relative_rotation_vector))
        arc_radius = float(np.linalg.norm(
            candidate.branch.body_offset_in_gripper,
        ))
        effective_distance = body_travel + arc_radius * rotation_angle
        speed_limit = min(
            _BOTTLE_TRANSFER_SPEED_M_S,
            limits.linear_speed_m_s,
        )
        start_speed = min(max(float(start_speed_m_s), 0.0), speed_limit)
        end_speed = min(max(float(end_speed_m_s), 0.0), speed_limit)
        peak_speed = speed_limit
        acceleration_time = (
            _BOTTLE_ACCELERATION_TIME_S
            if peak_speed > start_speed + 1e-12 else 0.0
        )
        deceleration_time = (
            _BOTTLE_TRANSFER_DECELERATION_TIME_S
            if peak_speed > end_speed + 1e-12 else 0.0
        )
        acceleration_distance = (
            0.5 * (start_speed + peak_speed) * acceleration_time
        )
        deceleration_distance = (
            0.5 * (peak_speed + end_speed) * deceleration_time
        )
        if acceleration_distance + deceleration_distance > effective_distance:
            scale = effective_distance / max(
                acceleration_distance + deceleration_distance, 1e-12,
            )
            acceleration_time *= scale
            deceleration_time *= scale
            acceleration_distance *= scale
            deceleration_distance *= scale
        cruise_distance = max(
            effective_distance
            - acceleration_distance
            - deceleration_distance,
            0.0,
        )
        cruise_time = cruise_distance / peak_speed
        translation_duration = (
            acceleration_time + cruise_time + deceleration_time
        )
        duration = max(
            translation_duration,
            rotation_angle / limits.angular_speed_rad_s,
        )
        steps = max(1, int(math.ceil(duration * limits.command_rate_hz)))
        sampled_duration = steps / limits.command_rate_hz
        time_scale = translation_duration / sampled_duration
        cruise_end_time = acceleration_time + cruise_time

        def ramp_integral(elapsed: float, ramp_duration: float) -> float:
            if ramp_duration <= 0.0:
                return elapsed
            return (
                0.5 * elapsed
                - ramp_duration / (2.0 * math.pi)
                * math.sin(math.pi * elapsed / ramp_duration)
        )

        direction = "left" if candidate.final_base_yaw > 0.0 else "right"
        for step in range(1, steps + 1):
            elapsed = min(
                step / limits.command_rate_hz * time_scale,
                translation_duration,
            )
            if elapsed <= acceleration_time:
                travelled = (
                    start_speed * elapsed
                    + (peak_speed - start_speed)
                    * ramp_integral(elapsed, acceleration_time)
                )
            elif elapsed <= cruise_end_time:
                travelled = (
                    acceleration_distance
                    + peak_speed * (elapsed - acceleration_time)
                )
            else:
                deceleration_elapsed = elapsed - cruise_end_time
                travelled = (
                    acceleration_distance
                    + cruise_distance
                    + peak_speed * deceleration_elapsed
                    + (end_speed - peak_speed)
                    * ramp_integral(
                        deceleration_elapsed, deceleration_time,
                    )
                )
            fraction = float(np.clip(
                travelled / max(effective_distance, 1e-12), 0.0, 1.0,
            ))
            rotation = (
                candidate.branch.grasp_rotation
                @ rotation_matrix(relative_rotation_vector * fraction)
            )
            body_position = (
                start_body_position
                + fraction * (goal_body_position - start_body_position)
            )
            position = (
                body_position
                - rotation @ candidate.branch.body_offset_in_gripper
            )
            frames.append(TaskFrame(
                position,
                rotation,
                True,
                f"Move bottle to base-{direction} 90-degree pose",
            ))

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *, candidate_index: int | None = None,
    ) -> tuple[TaskFrame, ...]:
        if candidate_index is not None and (type(candidate_index) is not int
                or not 0 <= candidate_index < 4 * len(_GRASP_AXIAL_FRACTIONS)):
            raise ValueError("Invalid bottle candidate index")
        mujoco.mj_forward(self.model, data)
        body_id = self._active_body(data)
        local_min, local_max, _ = _bottle_local_geometry(
            self.model,
            body_id,
        )
        bottle_axis = data.xmat[body_id].reshape(3, 3)[:, 2]
        if abs(float(bottle_axis[2])) > 0.30:
            raise RuntimeError("Bottle must start fallen with a near-horizontal axis")

        def infeasible(message):
            error = PlanningInfeasibleError(message)
            if candidate_index is not None:
                body_rotation = data.xmat[body_id].reshape(3, 3)
                grasp = self._grasp_rotation(body_rotation,
                    1. if (candidate_index // 2) % 2 == 0 else -1.)
                body_in_gripper = grasp.T @ body_rotation
                place = axis_angle_matrix(np.array((0., 0., 1.)),
                    _FINAL_BASE_YAWS_RAD[candidate_index % 2]) @ body_in_gripper.T
                error.context["nominal_reference_rotations"] = (grasp, place)
            return error

        table_geom, table_top, table_bounds = _table_geometry(
            self.model,
            data,
        )
        planner = _CpuMotionPlanner(
            self.model,
            data,
            self.handles,
            body_id,
            table_geom,
            table_top,
        )
        current_q = data.qpos[self.handles.qpos_ids].copy()
        branches = self._fixed_grasp_branches(
            data,
            planner,
            body_id,
            local_min,
            local_max,
            current_q,
            grasp_index=None if candidate_index is None else candidate_index // 2,
        )
        if not branches:
            raise infeasible("No fixed top-down bottle grasp passed CPU IK")
        reachable_branches_before_rotation_rule = len(branches)
        rotation_preferred_branches = _select_rotation_efficient_grasp_branches(
            branches,
            current_rotation,
        )
        rotation_preferred_branch_indices = {
            branch.index for branch in rotation_preferred_branches
        }
        candidates, sampled = self._placement_candidates(
            data,
            planner,
            branches,
            local_min,
            local_max,
            table_top,
            table_bounds,
            placement_index=None if candidate_index is None else candidate_index % 2,
        )
        if not candidates:
            raise infeasible("No placement candidate passed CPU filters")
        try:
            planned, rrt_attempted = self._select_rrt_candidate(planner, current_q, candidates)
        except RuntimeError as error:
            if str(error) != "CPU RRTConnect found no collision-free Adjust Bottle path":
                raise
            raise infeasible(str(error)) from error
        forced_branch_env = os.environ.get("FRANKA_DEBUG_ADJUST_BOTTLE_BRANCH")
        if forced_branch_env is not None:
            try:
                forced_branch = int(forced_branch_env)
            except ValueError as exc:
                raise ValueError("FRANKA_DEBUG_ADJUST_BOTTLE_BRANCH must be 0 or 1") from exc
            if forced_branch not in (0, 1):
                raise ValueError("FRANKA_DEBUG_ADJUST_BOTTLE_BRANCH must be 0 or 1")
            planned = tuple(item for item in planned if int(item.candidate.branch.index) == forced_branch)
            if not planned:
                raise RuntimeError(f"forced grasp branch {forced_branch} has no planned candidate")
        forced_serial_env = os.environ.get("FRANKA_DEBUG_ADJUST_BOTTLE_SERIAL")
        if forced_serial_env is not None:
            try:
                forced_serial = int(forced_serial_env)
            except ValueError as exc:
                raise ValueError("FRANKA_DEBUG_ADJUST_BOTTLE_SERIAL must be an integer") from exc
            planned = tuple(
                item for item in planned
                if int(item.candidate.serial) == forced_serial
            )
            if not planned:
                raise RuntimeError(
                    f"forced candidate serial {forced_serial} has no planned candidate"
                )
        # Rank by the path cost of each planning stage, rather than by one
        # whole-trajectory score or an extreme single-joint excursion.  The
        # pre-grasp stage selects the grasp branch; the post-grasp stage then
        # orders its equivalent left/right placement choices.
        planned = tuple(sorted(
            planned,
            key=lambda item: (
                item.pregrasp_cost,
                item.postgrasp_cost,
                item.candidate.serial,
            ),
        ))
        if os.environ.get("FRANKA_DEBUG_ADJUST_BOTTLE_SELECTION"):
            print(
                "Adjust Bottle configuration selection debug: "
                f"candidates={[(int(item.candidate.serial), float(item.pregrasp_cost), float(item.postgrasp_cost)) for item in planned]}"
            )
        if candidate_index is not None:
            # Shared nominal table screens the exact perturbed reference.
            # No task-local rollout or redundant validation of this template.
            selected = planned[0]
            frames_tuple = self._build_frames(selected, planner, limits)
            kinematic_attempts = []
        else:
            selected, frames_tuple, kinematic_attempts = self._select_kinematically_feasible_candidate(
                data,
                planner,
                planned,
                limits,
            )
        candidate = selected.candidate
        used_rotation_fallback = bool(
            candidate.branch.index not in rotation_preferred_branch_indices
        )
        if not _is_allowed_final_body_rotation(
            candidate.body_rotation,
        ):
            raise RuntimeError(
                "Selected Adjust Bottle final body pose is not base-frame left/right 90 degrees",
            )
        self.last_planning_stats = {
            "fixed_grasp_templates": 2 * len(_GRASP_AXIAL_FRACTIONS),
            "reachable_grasp_branches_before_rotation_rule": (
                reachable_branches_before_rotation_rule
            ),
            "reachable_grasp_branches": len(branches),
            "rotation_preferred_grasp_branches": len(
                rotation_preferred_branches
            ),
            "used_rotation_fallback": used_rotation_fallback,
            "sampled_candidates": sampled,
            "filtered_candidates": len(candidates),
            "candidate_final_base_yaw_deg": [
                math.degrees(item.final_base_yaw) for item in candidates
            ],
            "candidate_grasp_wrist_deg": [
                math.degrees(float(item.branch.grasp_ik.q[-1]))
                for item in candidates
            ],
            "candidate_grasp_ee_rotation_deg": [
                math.degrees(float(np.linalg.norm(rotation_vector(
                    current_rotation.T @ item.branch.grasp_rotation,
                ))))
                for item in candidates
            ],
            "candidate_final_wrist_deg": [
                math.degrees(float(item.place_ik.q[-1]))
                for item in candidates
            ],
            "candidate_prescore": [item.prescore for item in candidates],
            "rrt_candidates_attempted": rrt_attempted,
            "planned_candidate_serials": [
                int(item.candidate.serial) for item in planned
            ],
            "planned_candidate_costs": [
                float(item.cost) for item in planned
            ],
            "planned_pregrasp_costs": [
                float(item.pregrasp_cost) for item in planned
            ],
            "planned_postgrasp_costs": [
                float(item.postgrasp_cost) for item in planned
            ],
            "optimizer_kinematic_candidate_attempts": kinematic_attempts,
            "optimizer_kinematic_candidates_tested": len(kinematic_attempts),
            "selected_candidate_serial": int(candidate.serial),
            "selected_cost": selected.cost,
            "selected_pregrasp_cost": selected.pregrasp_cost,
            "selected_postgrasp_cost": selected.postgrasp_cost,
            "selected_grasp_wrist_deg": math.degrees(float(
                candidate.branch.grasp_ik.q[-1]
            )),
            "selected_grasp_ee_rotation_deg": math.degrees(float(
                np.linalg.norm(rotation_vector(
                    current_rotation.T @ candidate.branch.grasp_rotation,
                )),
            )),
            "selected_final_wrist_deg": math.degrees(float(
                candidate.place_ik.q[-1]
            )),
            "selected_placement_x": float(candidate.placement_xy[0]),
            "selected_placement_y": float(candidate.placement_xy[1]),
            "selected_final_base_yaw_deg": math.degrees(
                candidate.final_base_yaw,
            ),
            "selected_ee_rotation_deg": math.degrees(float(np.linalg.norm(
                rotation_vector(
                    candidate.branch.grasp_rotation.T
                    @ candidate.place_rotation,
                ),
            ))),
        }

        return frames_tuple


__all__ = ("StandFallenBottle", "_bottle_local_geometry")
