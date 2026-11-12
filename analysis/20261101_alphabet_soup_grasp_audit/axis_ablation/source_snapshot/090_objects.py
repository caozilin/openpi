"""Deterministic tabletop object layouts inside a conservative Panda workspace."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import zlib

import mujoco
import numpy as np

from ...config import WORKCELL_ELEVATION_M
from ..scenes import (
    BANANA_APPLE_TRAY_SCENE,
    OPEN_CYLINDRICAL_POT_LID_SCENE,
    OPEN_HANDLE_POT_LID_SCENE,
    ROUND_HANDLE_POT_PEAR_SCENE,
    TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
    UNSEEN_BOX_REGION_SCENE,
    UNSEEN_FRUIT_RECEPTACLES_SCENE,
    WHITEBOARD_SCENE,
    LayoutObject,
    SceneManager,
)


_COLOR_STACK_MINIMUM_CLEARANCE_M = 0.120
# Every scene, including seed=-1 authored layouts and specialized samplers,
# receives this final geometry-only guard. Task samplers may demand the larger
# configurable clearance below; this universal margin is deliberately smaller
# so it rejects contact/near-contact without invalidating safe authored poses.
_UNIVERSAL_INITIAL_CLEARANCE_M = 0.010


@dataclass(frozen=True)
class PandaWorkspace:
    """Conservative shared base-frame envelope for tabletop manipulation."""

    shoulder_height_m: float = WORKCELL_ELEVATION_M + 0.333
    # Stay away from nominal kinematic reach limits: SQP targets remain
    # feasible there, but physical position tracking and object transport
    # degrade noticeably, especially for UR5e's six-axis arm.
    maximum_reach_m: float = 0.75
    minimum_planar_reach_m: float = 0.18
    nominal_grasp_height_m: float = 0.10

    def planar_outer_radius(self, table_height_m: float) -> float:
        vertical = table_height_m + self.nominal_grasp_height_m - self.shoulder_height_m
        return math.sqrt(max(self.maximum_reach_m**2 - vertical**2, 0.0))


@dataclass(frozen=True)
class ObjectLayoutLimits:
    minimum_clearance_m: float = 0.05
    table_edge_margin_m: float = 0.015
    maximum_layout_attempts: int = 96
    maximum_group_attempts: int = 2500

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_clearance_m <= 0.15:
            raise ValueError("Object clearance must be between 0 and 0.15 m")
        if not 0.0 <= self.table_edge_margin_m <= 0.10:
            raise ValueError("Table edge margin must be between 0 and 0.10 m")
        if self.maximum_layout_attempts <= 0 or self.maximum_group_attempts <= 0:
            raise ValueError("Object placement attempt limits must be positive")


@dataclass(frozen=True)
class LayoutPlacement:
    names: tuple[str, ...]
    anchor_xy: np.ndarray
    yaw_rad: float
    footprint: np.ndarray


@dataclass(frozen=True)
class RandomizedObjectLayout:
    seed: int
    table_bounds_xy: np.ndarray
    planar_reach_m: tuple[float, float]
    sampling_center_xy: np.ndarray
    sampling_half_span_xy: np.ndarray
    placements: tuple[LayoutPlacement, ...]
    minimum_clearance_m: float

    @property
    def preferred_center_xy(self) -> np.ndarray:
        """Backward-compatible alias for the former Gaussian metadata."""
        return self.sampling_center_xy

    @property
    def position_std_xy(self) -> np.ndarray:
        """Backward-compatible alias; now denotes the uniform half-span."""
        return self.sampling_half_span_xy


@dataclass(frozen=True)
class ObjectPlacementRegion:
    """Seeded-layout anchor envelope before object-footprint clearance."""

    table_bounds_xy: np.ndarray
    planar_reach_m: tuple[float, float]
    table_height_m: float
    table_edge_margin_m: float


@dataclass(frozen=True)
class _GroupTemplate:
    members: tuple[LayoutObject, ...]
    anchor_xy: np.ndarray
    footprint_local: np.ndarray


_BOUND_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "Pot + lid": (
        ("teleop_pot", "teleop_pot_lid"),
        ("teleop_pot_lift", "teleop_pot_lift_lid"),
    ),
    "Open Pot Lid": (
        ("teleop_pot", "teleop_pot_lid"),
        ("teleop_pot_lift", "teleop_pot_lift_lid"),
    ),
    OPEN_CYLINDRICAL_POT_LID_SCENE: (("teleop_pot", "teleop_pot_lid"),),
    OPEN_HANDLE_POT_LID_SCENE: (("teleop_pot_lift", "teleop_pot_lift_lid"),),
    ROUND_HANDLE_POT_PEAR_SCENE: (("teleop_pot", "teleop_pot_lid"),),
    "RoboCasa saucepans": tuple(
        (f"robocasa_saucepan_{suffix}", f"robocasa_lid_{suffix}")
        for suffix in ("002", "003", "004", "006", "008", "013")
    ),
}


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if points.shape[0] <= 2:
        return points
    ordered = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        first_delta = first - origin
        second_delta = second - origin
        return float(first_delta[0] * second_delta[1] - first_delta[1] * second_delta[0])

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _circle_points(radius: float, count: int = 24) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(count)))


def _sphere_points(radius: float) -> np.ndarray:
    points = []
    for z in (-1.0, -0.5, 0.0, 0.5, 1.0):
        ring_radius = math.sqrt(max(1.0 - z * z, 0.0))
        for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            points.append((radius * ring_radius * math.cos(angle), radius * ring_radius * math.sin(angle), radius * z))
    return np.asarray(points)


def _geom_world_points(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        local = np.array([
            (sx * size[0], sy * size[1], sz * size[2])
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
        ])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        # A finite visual plane has rbound=0 in MuJoCo. Using that bound
        # collapses a white target to its centre and lets its corners leave
        # the table. Its visible rectangle is defined by the first two sizes.
        if size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError("An infinite plane has no finite layout footprint")
        local = np.array([
            (sx * size[0], sy * size[1], 0.0)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
        ])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        local = _sphere_points(float(size[0]))
    elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        rings = []
        for z in (-float(size[1]), float(size[1])):
            ring = _circle_points(float(size[0]))
            ring[:, 2] = z
            rings.append(ring)
        local = np.vstack(rings)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            sphere = _sphere_points(float(size[0]))
            local = np.vstack((local, sphere + (0.0, 0.0, -float(size[1])), sphere + (0.0, 0.0, float(size[1]))))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        local = _sphere_points(1.0) * size
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        address = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = model.mesh_vert[address:address + count].copy()
    else:
        local = _circle_points(float(model.geom_rbound[geom_id]))
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return local @ rotation.T + data.geom_xpos[geom_id]


def body_planar_footprint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    *,
    include_visual_geoms: bool = False,
) -> np.ndarray:
    """Return the convex XY footprint of a body subtree.

    Normal layout clearance uses collision geometry.  Strict tabletop/region
    containment can request every visual geom as well, so no visible point is
    allowed to cross an edge.
    """
    descendants = []
    for candidate in range(model.nbody):
        ancestor = candidate
        while ancestor > body_id:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == body_id:
            descendants.append(candidate)
    geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, descendants))
    contact_geoms = (
        geom_ids
        if include_visual_geoms
        else geom_ids[
            (model.geom_contype[geom_ids] != 0)
            | (model.geom_conaffinity[geom_ids] != 0)
        ]
    )
    if contact_geoms.size == 0:
        contact_geoms = geom_ids
    if contact_geoms.size == 0:
        raise RuntimeError(f"Body {body_id} has no geometry for a planar footprint")
    return _convex_hull(np.vstack([
        _geom_world_points(model, data, int(geom_id))[:, :2]
        for geom_id in contact_geoms
    ]))


def _polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    edges = np.vstack((np.roll(first, -1, axis=0) - first, np.roll(second, -1, axis=0) - second))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
    first_projection = first @ axes.T
    second_projection = second @ axes.T
    separated = (first_projection.max(axis=0) < second_projection.min(axis=0)) | (second_projection.max(axis=0) < first_projection.min(axis=0))
    return not bool(np.any(separated))


def _points_to_polygon_edges(points: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    deltas = np.roll(polygon, -1, axis=0) - polygon
    denominator = np.einsum("ij,ij->i", deltas, deltas)
    relative = points[:, None, :] - starts[None, :, :]
    numerator = np.einsum("pij,ij->pi", relative, deltas)
    fractions = np.divide(numerator, denominator[None, :], out=np.zeros_like(numerator), where=denominator[None, :] > 1e-16)
    fractions = np.clip(fractions, 0.0, 1.0)
    nearest = starts[None, :, :] + fractions[:, :, None] * deltas[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - nearest, axis=2)
    return float(np.min(distances))


def polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    if _polygons_overlap(first, second):
        return 0.0
    return min(_points_to_polygon_edges(first, second), _points_to_polygon_edges(second, first))


class ObjectLayoutRandomizer:
    """Randomize all active object groups in XY and world-Z yaw only."""

    _RNG_STREAM_ID = 0x4F424A454354  # "OBJECT", independent of other domains.

    def __init__(
        self,
        model: mujoco.MjModel,
        scene_manager: SceneManager,
        workspace: PandaWorkspace | None = None,
        limits: ObjectLayoutLimits | None = None,
    ) -> None:
        self.model = model
        self.scene_manager = scene_manager
        self.workspace = workspace or PandaWorkspace()
        self.limits = limits or ObjectLayoutLimits()
        self._table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
        if self._table_body_id < 0:
            raise RuntimeError("Object randomization requires work_surface")
        table_geoms = np.flatnonzero(model.geom_bodyid == self._table_body_id)
        if table_geoms.size == 0:
            raise RuntimeError("work_surface has no geometry")
        self._table_geom_id = int(table_geoms[np.argmax(np.prod(model.geom_size[table_geoms, :2], axis=1))])

    def _table_bounds(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        geom_id = self._table_geom_id
        center = data.geom_xpos[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        half = np.abs(rotation[:2]) @ self.model.geom_size[geom_id]
        bounds = np.array(((center[0] - half[0], center[0] + half[0]), (center[1] - half[1], center[1] + half[1])))
        table_height = float(center[2] + np.abs(rotation[2]) @ self.model.geom_size[geom_id])
        return bounds, table_height

    def placement_region(self, data: mujoco.MjData) -> ObjectPlacementRegion:
        """Return the current seeded initializer's base XY anchor envelope."""
        table_bounds, table_height = self._table_bounds(data)
        return ObjectPlacementRegion(
            table_bounds_xy=table_bounds,
            planar_reach_m=(
                self.workspace.minimum_planar_reach_m,
                self.workspace.planar_outer_radius(table_height),
            ),
            table_height_m=table_height,
            table_edge_margin_m=self.limits.table_edge_margin_m,
        )

    def _groups(self, data: mujoco.MjData) -> tuple[_GroupTemplate, ...]:
        objects = {item.body_name: item for item in self.scene_manager.layout_objects(data)}
        configured = _BOUND_GROUPS.get(self.scene_manager.current_name, ())
        member_groups: list[tuple[str, ...]] = [group for group in configured if all(name in objects for name in group)]
        already_grouped = {name for group in member_groups for name in group}
        member_groups.extend((name,) for name in objects if name not in already_grouped)

        templates = []
        for names in member_groups:
            members = tuple(objects[name] for name in names)
            anchor = np.mean([member.position[:2] for member in members], axis=0)
            points = []
            for member in members:
                contact_geoms = [
                    int(geom_id) for geom_id in member.geom_ids
                    if self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id]
                ]
                if not contact_geoms:
                    contact_geoms = [int(geom_id) for geom_id in member.geom_ids]
                for geom_id in contact_geoms:
                    points.append(_geom_world_points(self.model, data, geom_id)[:, :2])
            footprint = _convex_hull(np.vstack(points))
            if footprint.shape[0] < 3:
                centre = np.mean(footprint, axis=0)
                footprint = centre + np.array(((-0.01, -0.01), (0.01, -0.01), (0.01, 0.01), (-0.01, 0.01)))
            templates.append(_GroupTemplate(members, anchor, footprint - anchor))
        # Largest footprints are placed first to make dense scenes reliable.
        templates.sort(key=lambda group: (-float(np.ptp(group.footprint_local[:, 0]) * np.ptp(group.footprint_local[:, 1])), group.members[0].body_name))
        return tuple(templates)

    @staticmethod
    def _rotate(points: np.ndarray, yaw: float) -> np.ndarray:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        return points @ rotation.T

    def _valid_footprint(
        self,
        footprint: np.ndarray,
        table_bounds: np.ndarray,
        outer_reach: float,
        accepted: list[np.ndarray],
    ) -> bool:
        margin = self.limits.table_edge_margin_m
        if np.any(footprint[:, 0] < table_bounds[0, 0] + margin) or np.any(footprint[:, 0] > table_bounds[0, 1] - margin):
            return False
        if np.any(footprint[:, 1] < table_bounds[1, 0] + margin) or np.any(footprint[:, 1] > table_bounds[1, 1] - margin):
            return False
        radii = np.linalg.norm(footprint, axis=1)
        if np.any(radii > outer_reach) or np.any(radii < self.workspace.minimum_planar_reach_m):
            return False
        return all(polygon_distance(footprint, other) >= self.limits.minimum_clearance_m for other in accepted)

    def _uniform_distribution(
        self,
        table_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return center/half-span for uniform sampling inside table margins."""
        margin = self.limits.table_edge_margin_m
        lower = table_bounds[:, 0] + margin
        upper = table_bounds[:, 1] - margin
        return 0.5 * (lower + upper), 0.5 * (upper - lower)

    @staticmethod
    def _sample_uniform_anchor(
        rng: np.random.Generator,
        center: np.ndarray,
        half_span: np.ndarray,
    ) -> np.ndarray:
        return rng.uniform(center - half_span, center + half_span)

    def _apply_group(self, data: mujoco.MjData, group: _GroupTemplate, anchor: np.ndarray, yaw: float) -> None:
        yaw_quaternion = np.array((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
        for member in group.members:
            position = member.position.copy()
            position[:2] = anchor + self._rotate((member.position[:2] - group.anchor_xy)[None, :], yaw)[0]
            quaternion = np.empty(4)
            mujoco.mju_mulQuat(quaternion, yaw_quaternion, member.quaternion)
            self.scene_manager.set_layout_pose(data, member.body_name, position, quaternion)

    def _validate_initial_clearance(
        self,
        data: mujoco.MjData,
        layout: RandomizedObjectLayout,
    ) -> RandomizedObjectLayout:
        """Enforce inflated planar clearance for every generated scene."""
        mujoco.mj_forward(self.model, data)
        groups = self._groups(data)
        accepted: list[np.ndarray] = []
        margin = self.limits.table_edge_margin_m
        for item in self.scene_manager.layout_objects(data):
            if not (
                item.body_name.startswith("paper_decal_")
                or item.body_name == "unseen_box_white_region"
            ):
                continue
            footprint = body_planar_footprint(
                self.model, data, self.model.body(item.body_name).id,
                include_visual_geoms=True,
            )
            if (
                np.any(footprint < layout.table_bounds_xy[:, 0] + margin - 1e-9)
                or np.any(footprint > layout.table_bounds_xy[:, 1] - margin + 1e-9)
            ):
                raise RuntimeError(
                    f"White region {item.body_name!r} extends beyond the tabletop margin",
                )
        for group in groups:
            footprint = group.footprint_local + group.anchor_xy
            if any(
                polygon_distance(footprint, other)
                < _UNIVERSAL_INITIAL_CLEARANCE_M
                for other in accepted
            ):
                names = ", ".join(member.body_name for member in group.members)
                raise RuntimeError(
                    "Initial object clearance validation failed for "
                    f"{names!r} in {self.scene_manager.current_name!r}",
                )
            accepted.append(footprint)
        minimum = min(
            (
                polygon_distance(first, second)
                for index, first in enumerate(accepted)
                for second in accepted[index + 1:]
            ),
            default=float("inf"),
        )
        return replace(layout, minimum_clearance_m=minimum)

    def _apply_insert_object_layout(
        self,
        data: mujoco.MjData,
        seed: int,
        table_bounds: np.ndarray,
        outer_reach: float,
        preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Apply the task's hole-up or constrained front-facing layout."""
        objects = {
            item.body_name: item
            for item in self.scene_manager.layout_objects(data)
        }
        object_name = next(
            name for name in objects if name.startswith("insert_object_")
        )
        container_name = next(
            name for name in objects if name.startswith("insert_container_")
        )
        inserted_object = objects[object_name]
        container = objects[container_name]
        scene_stream = zlib.crc32(b"Insert Object layout")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed), self._RNG_STREAM_ID, scene_stream,
        ))))

        object_position = inserted_object.position.copy()
        object_position[:2] = self.scene_manager.workspace_xy(np.array((
            rng.uniform(0.37, 0.45),
            rng.uniform(-0.10, 0.10),
        )))
        object_yaw = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        object_yaw_quat = np.array((
            math.cos(0.5 * object_yaw),
            0.0,
            0.0,
            math.sin(0.5 * object_yaw),
        ))

        container_position = container.position.copy()
        if container_name.endswith("_up"):
            object_geom = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{object_name}_geom",
            )
            if object_geom < 0:
                raise RuntimeError(
                    f"Missing insertion object geom for {object_name!r}",
                )
            if int(self.model.geom_type[object_geom]) == int(
                mujoco.mjtGeom.mjGEOM_BOX,
            ):
                half_length = float(
                    self.model.geom_size[object_geom, 2],
                )
                lying_half_height = float(
                    self.model.geom_size[object_geom, 0],
                )
            else:
                half_length = float(
                    self.model.geom_size[object_geom, 1],
                )
                lying_half_height = float(
                    self.model.geom_size[object_geom, 0],
                )
            table_height = (
                float(inserted_object.position[2])
                - lying_half_height
            )
            object_position[2] = table_height + half_length
            # With an upward-facing hole the object starts upright.  Its
            # local +Z insertion axis is world +Z; yaw remains randomized.
            object_quaternion = object_yaw_quat
            container_position[:2] = self.scene_manager.workspace_xy(
                np.array((
                    rng.uniform(0.61, 0.66),
                    rng.uniform(-0.20, 0.20),
                )),
            )
            container_yaw = float(rng.uniform(-np.pi, np.pi))
            container_quaternion = np.array((
                math.cos(0.5 * container_yaw),
                0.0,
                0.0,
                math.sin(0.5 * container_yaw),
            ))
        else:
            object_quaternion = np.empty(4)
            mujoco.mju_mulQuat(
                object_quaternion,
                object_yaw_quat,
                inserted_object.quaternion,
            )
            left_side = container_name.endswith("_front_left")
            container_position[:2] = self.scene_manager.workspace_xy(
                np.array((
                    rng.uniform(0.49, 0.54),
                    rng.uniform(0.19, 0.23)
                    if left_side
                    else rng.uniform(-0.23, -0.19)
                )),
            )
            # Face the table centre along +/-Y.  The only allowed planar
            # error rotates the opening toward world +X by 0..15 degrees.
            centre_sign_y = -1.0 if left_side else 1.0
            yaw_offset = float(rng.uniform(
                0.0,
                np.deg2rad(15.0),
            ))
            cosine = math.cos(yaw_offset)
            sine = math.sin(yaw_offset)
            z_axis = np.array((
                sine,
                centre_sign_y * cosine,
                0.0,
            ))
            y_axis = np.array((0.0, 0.0, 1.0))
            x_axis = np.cross(y_axis, z_axis)
            rotation = np.column_stack((x_axis, y_axis, z_axis))
            container_quaternion = np.empty(4)
            mujoco.mju_mat2Quat(
                container_quaternion,
                rotation.reshape(-1),
            )

        self.scene_manager.set_layout_pose(
            data,
            object_name,
            object_position,
            object_quaternion,
        )
        self.scene_manager.set_layout_pose(
            data,
            container_name,
            container_position,
            container_quaternion,
        )
        mujoco.mj_forward(self.model, data)

        def rectangle(
            center: np.ndarray,
            half_x: float,
            half_y: float,
            yaw: float,
        ) -> np.ndarray:
            local = np.array((
                (-half_x, -half_y),
                (half_x, -half_y),
                (half_x, half_y),
                (-half_x, half_y),
            ))
            return self._rotate(local, yaw) + center[:2]

        object_points = []
        for geom_id in inserted_object.geom_ids:
            if self.model.geom_contype[geom_id]:
                object_points.append(
                    _geom_world_points(
                        self.model, data, int(geom_id),
                    )[:, :2],
                )
        object_footprint = _convex_hull(np.vstack(object_points))
        container_footprint = rectangle(
            container_position,
            0.07,
            0.07,
            0.0,
        )
        minimum = polygon_distance(
            object_footprint,
            container_footprint,
        )
        placements = (
            LayoutPlacement(
                (object_name,),
                object_position[:2].copy(),
                object_yaw,
                object_footprint,
            ),
            LayoutPlacement(
                (container_name,),
                container_position[:2].copy(),
                0.0,
                container_footprint,
            ),
        )
        return RandomizedObjectLayout(
            seed,
            table_bounds,
            (self.workspace.minimum_planar_reach_m, outer_reach),
            preferred_center,
            position_std,
            placements,
            minimum,
        )

    def _apply_paper_placement_layout(
        self,
        data: mujoco.MjData,
        seed: int,
        table_bounds: np.ndarray,
        outer_reach: float,
        preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Randomize source and decal yaw independently for matching."""
        objects = {
            item.body_name: item
            for item in self.scene_manager.layout_objects(data)
        }
        if self.scene_manager.current_name == UNSEEN_BOX_REGION_SCENE:
            object_name = "ycb_sugar_box"
            decal_name = "unseen_box_white_region"
        else:
            object_name = next(
                name for name in objects
                if name.startswith("paper_object_")
            )
            decal_name = next(
                name for name in objects
                if name.startswith("paper_decal_")
            )
        manipulated = objects[object_name]
        decal = objects[decal_name]
        scene_stream = zlib.crc32(b"Paper Placement layout")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed), self._RNG_STREAM_ID, scene_stream,
        ))))
        is_ball_cube_sequence = (
            self.scene_manager.current_name == "Ball + Paper Placement"
        )
        is_tolerance_discrimination = (
            self.scene_manager.current_name
            == TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE
        )
        object_yaw_limit = np.deg2rad(15.0) if is_ball_cube_sequence else np.pi
        object_yaw = float(rng.uniform(-object_yaw_limit, object_yaw_limit))
        decal_yaw = (
            object_yaw if is_ball_cube_sequence
            else float(rng.uniform(-np.pi, np.pi))
        )
        decal_local = body_planar_footprint(
            self.model, data, self.model.body(decal_name).id,
            include_visual_geoms=True,
        ) - decal.position[:2]

        def valid_decal_anchor(anchor: np.ndarray) -> bool:
            footprint = self._rotate(decal_local, decal_yaw) + anchor
            margin = self.limits.table_edge_margin_m
            return bool(
                np.all(footprint >= table_bounds[:, 0] + margin)
                and np.all(footprint <= table_bounds[:, 1] - margin)
            )

        object_position = manipulated.position.copy()
        decal_position = decal.position.copy()
        ball = objects.get("teleop_ball")
        plate = objects.get("teleop_plate")
        tray = objects.get("tray_khaki_medium")
        ball_position: np.ndarray | None = None
        plate_position: np.ndarray | None = None
        plate_yaw = 0.0
        # Special multi-object scenes use fixed lanes. Their centre-distance
        # checks must also account for the finite white rectangle, now that
        # the final clearance validator no longer sees a zero-radius point.
        special_local = {
            item.body_name: body_planar_footprint(
                self.model, data, self.model.body(item.body_name).id,
            ) - item.position[:2]
            for item in objects.values()
        } if is_ball_cube_sequence or is_tolerance_discrimination else {}

        def special_clearance(candidates: list[tuple[str, np.ndarray, float]]) -> bool:
            footprints = [
                self._rotate(special_local[name], yaw) + anchor
                for name, anchor, yaw in candidates
            ]
            return all(
                polygon_distance(first, second) >= _UNIVERSAL_INITIAL_CLEARANCE_M
                for index, first in enumerate(footprints)
                for second in footprints[index + 1:]
            )

        if is_ball_cube_sequence:
            if ball is None or plate is None:
                raise RuntimeError(
                    "Ball/cube sequence requires a ball and a plate",
                )
            target_slots = (
                plate.position[:2].copy(),
                decal.position[:2].copy(),
            )
            target_order = rng.permutation(2)
            ball_position = ball.position.copy()
            plate_position = plate.position.copy()
            plate_position[:2] = target_slots[int(target_order[0])]
            decal_position[:2] = target_slots[int(target_order[1])]
            # The nominal ball and cube sit symmetrically on opposite sides
            # of the table.  Seeded layouts do not preserve those lanes: each
            # source keeps its own X centre with an 8 cm total span, while Y
            # is sampled freely across the shared source workspace.
            source_x_jitter = 0.040
            source_y_bounds = self.scene_manager.workspace_xy(np.array((
                (0.49, -0.200),
                (0.49, 0.200),
            )))[:, 1]
            target_xy_jitter = np.array((0.040, 0.060))
            for _ in range(self.limits.maximum_group_attempts):
                ball_xy = np.array((
                    ball_position[0] + rng.uniform(
                        -source_x_jitter, source_x_jitter,
                    ),
                    rng.uniform(*source_y_bounds),
                ))
                object_xy = np.array((
                    object_position[0] + rng.uniform(
                        -source_x_jitter, source_x_jitter,
                    ),
                    rng.uniform(*source_y_bounds),
                ))
                plate_xy = plate_position[:2] + rng.uniform(
                    -target_xy_jitter, target_xy_jitter,
                )
                decal_xy = decal_position[:2] + rng.uniform(
                    -target_xy_jitter, target_xy_jitter,
                )
                sources_are_separated = (
                    np.linalg.norm(ball_xy - object_xy) >= 0.085
                )
                sources_clear_targets = all(
                    np.linalg.norm(source_xy - target_xy) >= 0.115
                    for source_xy in (ball_xy, object_xy)
                    for target_xy in (plate_xy, decal_xy)
                )
                targets_are_separated = (
                    np.linalg.norm(plate_xy - decal_xy) >= 0.120
                )
                if (
                    sources_are_separated
                    and sources_clear_targets
                    and targets_are_separated
                    and valid_decal_anchor(decal_xy)
                    and special_clearance([
                        (ball.body_name, ball_xy, 0.0),
                        (object_name, object_xy, object_yaw),
                        (plate.body_name, plate_xy, 0.0),
                        (decal_name, decal_xy, decal_yaw),
                    ])
                ):
                    ball_position[:2] = ball_xy
                    object_position[:2] = object_xy
                    plate_position[:2] = plate_xy
                    decal_position[:2] = decal_xy
                    break
            else:
                raise RuntimeError(
                    "Unable to sample a separated ball/cube task layout",
                )
        elif is_tolerance_discrimination:
            plate = tray
            if plate is None:
                raise RuntimeError(
                    "Tolerance-discrimination layout requires a tray",
                )
            target_slots = (
                plate.position[:2].copy(),
                decal.position[:2].copy(),
            )
            # The tray is substantially wider than the former round plate;
            # keep it in the outer target lane so it cannot crowd the source
            # pickup zone. The two task instructions still share this exact
            # same randomized scene.
            target_order = np.arange(2)
            plate_position = plate.position.copy()
            plate_yaw = float(rng.uniform(-np.pi, np.pi))
            plate_position[:2] = target_slots[int(target_order[0])]
            decal_position[:2] = target_slots[int(target_order[1])]
            source_x_jitter = 0.035
            source_y_bounds = self.scene_manager.workspace_xy(np.array((
                (0.49, -0.120),
                (0.49, 0.120),
            )))[:, 1]
            target_xy_jitter = np.array((0.025, 0.030))
            for _ in range(self.limits.maximum_group_attempts):
                object_xy = np.array((
                    object_position[0] + rng.uniform(
                        -source_x_jitter, source_x_jitter,
                    ),
                    rng.uniform(*source_y_bounds),
                ))
                plate_xy = plate_position[:2] + rng.uniform(
                    -target_xy_jitter, target_xy_jitter,
                )
                decal_xy = decal_position[:2] + rng.uniform(
                    -target_xy_jitter, target_xy_jitter,
                )
                if (
                    np.linalg.norm(object_xy - plate_xy) >= 0.250
                    and np.linalg.norm(object_xy - decal_xy) >= 0.135
                    and np.linalg.norm(plate_xy - decal_xy) >= 0.255
                    and valid_decal_anchor(decal_xy)
                    and special_clearance([
                        (object_name, object_xy, object_yaw),
                        (plate.body_name, plate_xy, plate_yaw),
                        (decal_name, decal_xy, decal_yaw),
                    ])
                ):
                    object_position[:2] = object_xy
                    plate_position[:2] = plate_xy
                    decal_position[:2] = decal_xy
                    break
            else:
                raise RuntimeError(
                    "Unable to sample a separated tolerance-discrimination layout",
                )
        if not is_ball_cube_sequence and not is_tolerance_discrimination:
            groups = {
                member.body_name: group
                for group in self._groups(data)
                for member in group.members
            }
            object_group = groups[object_name]
            decal_group = groups[decal_name]
            for _ in range(self.limits.maximum_layout_attempts):
                object_anchor = self._sample_uniform_anchor(
                    rng, preferred_center, position_std,
                )
                object_footprint = (
                    self._rotate(object_group.footprint_local, object_yaw)
                    + object_anchor
                )
                if not self._valid_footprint(
                    object_footprint, table_bounds, outer_reach, [],
                ):
                    continue
                decal_anchor = self._sample_uniform_anchor(
                    rng, preferred_center, position_std,
                )
                decal_footprint = (
                    self._rotate(decal_group.footprint_local, decal_yaw)
                    + decal_anchor
                )
                if not self._valid_footprint(
                    decal_footprint,
                    table_bounds,
                    outer_reach,
                    [object_footprint],
                ):
                    continue
                # Mesh-derived footprints can change by a few millimetres
                # after MuJoCo recomputes the transformed scene. Keep a small
                # sampling buffer so the final geometry check still preserves
                # the configured source-to-target clearance.
                if polygon_distance(
                    object_footprint, decal_footprint,
                ) < self.limits.minimum_clearance_m + 0.010:
                    continue
                object_position[:2] = object_anchor
                decal_position[:2] = decal_anchor
                break
            else:
                raise RuntimeError(
                    "Unable to sample separated object and white-region poses",
                )

        def yaw_quaternion(yaw: float) -> np.ndarray:
            return np.array((
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            ))

        object_quaternion = np.empty(4)
        mujoco.mju_mulQuat(
            object_quaternion,
            yaw_quaternion(object_yaw),
            manipulated.quaternion,
        )
        decal_quaternion = np.empty(4)
        mujoco.mju_mulQuat(
            decal_quaternion,
            yaw_quaternion(decal_yaw),
            decal.quaternion,
        )
        self.scene_manager.set_layout_pose(
            data,
            object_name,
            object_position,
            object_quaternion,
        )
        self.scene_manager.set_layout_pose(
            data,
            decal_name,
            decal_position,
            decal_quaternion,
        )
        if is_ball_cube_sequence or is_tolerance_discrimination:
            assert plate is not None and plate_position is not None
            if is_ball_cube_sequence:
                assert ball is not None and ball_position is not None
                self.scene_manager.set_layout_pose(
                    data, "teleop_ball", ball_position, ball.quaternion,
                )
            plate_quaternion = plate.quaternion.copy()
            if is_tolerance_discrimination:
                mujoco.mju_mulQuat(
                    plate_quaternion,
                    yaw_quaternion(plate_yaw),
                    plate.quaternion,
                )
            self.scene_manager.set_layout_pose(
                data, plate.body_name, plate_position, plate_quaternion,
            )

        color = self.scene_manager.active_geometry_color_rgba
        if color is None:
            colors = np.array((
                (0.88, 0.12, 0.10, 1.0),
                (0.12, 0.62, 0.20, 1.0),
                (0.10, 0.32, 0.88, 1.0),
            ))
            color = colors[int(rng.integers(0, len(colors)))]
        collision_geoms = manipulated.geom_ids[
            self.model.geom_contype[manipulated.geom_ids] != 0
        ]
        self.model.geom_rgba[collision_geoms] = color
        mujoco.mj_forward(self.model, data)

        object_points = np.vstack([
            _geom_world_points(self.model, data, int(geom_id))[:, :2]
            for geom_id in collision_geoms
        ])
        object_footprint = _convex_hull(object_points)
        decal_footprint = body_planar_footprint(
            self.model, data, self.model.body(decal_name).id,
            include_visual_geoms=True,
        )
        extra_placements: tuple[LayoutPlacement, ...] = ()
        if is_ball_cube_sequence or is_tolerance_discrimination:
            assert plate is not None and plate_position is not None
            plate_collision = plate.geom_ids[
                self.model.geom_contype[plate.geom_ids] != 0
            ]
            plate_footprint = _convex_hull(np.vstack([
                _geom_world_points(self.model, data, int(geom_id))[:, :2]
                for geom_id in plate_collision
            ]))
            plate_placement = LayoutPlacement(
                (plate.body_name,), plate_position[:2].copy(), plate_yaw,
                plate_footprint,
            )
            if is_ball_cube_sequence:
                assert ball is not None and ball_position is not None
                ball_collision = ball.geom_ids[
                    self.model.geom_contype[ball.geom_ids] != 0
                ]
                ball_footprint = _convex_hull(np.vstack([
                    _geom_world_points(self.model, data, int(geom_id))[:, :2]
                    for geom_id in ball_collision
                ]))
                extra_placements = (
                    LayoutPlacement(
                        ("teleop_ball",), ball_position[:2].copy(), 0.0,
                        ball_footprint,
                    ),
                    plate_placement,
                )
            else:
                extra_placements = (plate_placement,)
        return RandomizedObjectLayout(
            seed,
            table_bounds,
            (
                self.workspace.minimum_planar_reach_m,
                outer_reach,
            ),
            preferred_center,
            position_std,
            (
                LayoutPlacement(
                    (object_name,),
                    object_position[:2].copy(),
                    object_yaw,
                    object_footprint,
                ),
                LayoutPlacement(
                    (decal_name,),
                    decal_position[:2].copy(),
                    decal_yaw,
                    decal_footprint,
                ),
                *extra_placements,
            ),
            self.limits.minimum_clearance_m,
        )

    def _apply_fruit_receptacle_layout(
        self,
        data: mujoco.MjData,
        seed: int,
        table_bounds: np.ndarray,
        outer_reach: float,
        preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Randomize the active fruits and receptacles in safe pickup zones."""
        templates = {
            group.members[0].body_name: group
            for group in self._groups(data)
            if len(group.members) == 1
        }
        training_regions = {
            "fruit_task_banana": np.array(((0.30, 0.35), (-0.23, -0.17))),
            "fruit_task_strawberry": np.array(((0.27, 0.34), (0.15, 0.22))),
        }
        training_yaw_limits = {
            "fruit_task_banana": np.pi,
            "fruit_task_strawberry": np.pi,
        }
        if self.scene_manager.current_name == UNSEEN_FRUIT_RECEPTACLES_SCENE:
            # Reuse the two outside pickup lanes from the training task; only
            # the body names differ for the held-out fruit models.
            fruit_regions = {
                "fruit_apple": training_regions["fruit_task_banana"],
                "fruit_lemon": training_regions["fruit_task_strawberry"],
            }
            yaw_limits = {name: np.pi for name in fruit_regions}
            receptacle_names = (
                next(name for name in templates if name.startswith("ycb_plate_")),
                next(name for name in templates if name.startswith("ycb_bowl_")),
            )
        elif self.scene_manager.current_name == BANANA_APPLE_TRAY_SCENE:
            fruit_regions = {
                "fruit_task_banana": training_regions["fruit_task_banana"],
                "fruit_task_apple": training_regions["fruit_task_strawberry"],
            }
            yaw_limits = {
                "fruit_task_banana": training_yaw_limits[
                    "fruit_task_banana"
                ],
                "fruit_task_apple": np.pi,
            }
            receptacle_names = ("tray_khaki_medium",)
        else:
            fruit_regions = training_regions
            yaw_limits = training_yaw_limits
            receptacle_names = ("tray_khaki_medium",)
        missing = set(fruit_regions) - templates.keys()
        if missing:
            raise RuntimeError(
                f"Fruit layout is missing templates: {sorted(missing)}",
            )

        scene_stream = zlib.crc32(b"Fruit Receptacles layout")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed), self._RNG_STREAM_ID, scene_stream,
        ))))

        receptacle_slots = (
            tuple(
                templates[name].anchor_xy.copy()
                for name in receptacle_names
            )
            if self.scene_manager.current_name
            == UNSEEN_FRUIT_RECEPTACLES_SCENE
            else (self.scene_manager.workspace_xy(np.array((0.64, 0.0))),)
        )
        sampled: list[
            tuple[_GroupTemplate, np.ndarray, float, np.ndarray]
        ] = []
        accepted: list[np.ndarray] = []
        for name, slot_index in zip(
            receptacle_names,
            rng.permutation(len(receptacle_slots)),
            strict=True,
        ):
            group = templates[name]
            slot = receptacle_slots[int(slot_index)]
            for _ in range(self.limits.maximum_group_attempts):
                anchor = slot + rng.uniform(-0.020, 0.020, size=2)
                yaw = (
                    float(rng.uniform(-np.pi, np.pi))
                    if name.startswith("tray_khaki_") else 0.0
                )
                footprint = self._rotate(group.footprint_local, yaw) + anchor
                if all(
                    polygon_distance(footprint, other)
                    >= self.limits.minimum_clearance_m
                    for other in accepted
                ):
                    sampled.append((group, anchor, yaw, footprint))
                    accepted.append(footprint)
                    break
            else:
                raise RuntimeError(
                    f"Unable to sample a collision-free pose for {name!r}",
                )

        fruit_names = tuple(fruit_regions)
        shuffled_regions = {
            name: fruit_regions[fruit_names[int(slot_index)]]
            for name, slot_index in zip(
                fruit_names,
                rng.permutation(len(fruit_names)),
                strict=True,
            )
        }
        for name in fruit_names:
            group = templates[name]
            region = shuffled_regions[name]
            slot_center = self.scene_manager.workspace_xy(
                np.mean(region, axis=1),
            )
            # Long fruit/receptacle footprints need more fore-aft separation
            # than compact blocks.  Keep fruit pickups on the base-side of
            # the compact workspace while the two receptacles move inward.
            # Preserve the authored lateral lanes: contracting those lanes
            # makes the banana footprint crowd the neighbouring fruit.
            workspace_center = self.scene_manager.workspace_xy(
                np.array((0.49, 0.0)),
            )
            slot_center[0] -= 0.050
            slot_center[1] = float(np.mean(region[1])) + workspace_center[1]
            for _ in range(self.limits.maximum_group_attempts):
                anchor = slot_center + rng.uniform(
                    -0.020, 0.020, size=2,
                )
                yaw_limit = float(yaw_limits[name])
                yaw = float(rng.uniform(-yaw_limit, yaw_limit))
                footprint = (
                    self._rotate(group.footprint_local, yaw) + anchor
                )
                if self._valid_footprint(
                    footprint,
                    table_bounds,
                    outer_reach,
                    accepted,
                ):
                    sampled.append((group, anchor, yaw, footprint))
                    accepted.append(footprint)
                    break
            else:
                raise RuntimeError(
                    f"Unable to sample a valid seeded pose for {name!r}",
                )

        for group, anchor, yaw, _ in sampled:
            # _apply_group left-multiplies world yaw onto the nominal pose.
            # Consequently any authored non-upright pose stays non-upright.
            self._apply_group(data, group, anchor, yaw)
        mujoco.mj_forward(self.model, data)
        return RandomizedObjectLayout(
            seed,
            table_bounds,
            (self.workspace.minimum_planar_reach_m, outer_reach),
            preferred_center,
            position_std,
            tuple(
                LayoutPlacement(
                    tuple(member.body_name for member in group.members),
                    anchor.copy(),
                    yaw,
                    footprint.copy(),
                )
                for group, anchor, yaw, footprint in sampled
            ),
            self.limits.minimum_clearance_m,
        )

    def _apply_whiteboard_layout(
        self, data: mujoco.MjData, seed: int, table_bounds: np.ndarray,
        outer_reach: float, preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Sample the board, its local ink, then an independent tabletop eraser."""
        rng = np.random.default_rng(np.random.SeedSequence((
            int(seed), self._RNG_STREAM_ID, zlib.crc32(b"Whiteboard layout"),
        )))
        groups = {
            group.members[0].body_name: group for group in self._groups(data)
        }
        board_group = groups["wipe_board"]
        eraser_group = groups["board_eraser"]
        for attempt in range(2 * self.limits.maximum_group_attempts):
            # Preserve the authored neighbourhood when it fits. Some robot
            # workspaces (notably UR5e) cannot contain the board there, so
            # continue with the shared tabletop sampler and the same checks.
            board_anchor = (
                board_group.anchor_xy + rng.uniform(
                    (-0.060, -0.120), (0.060, 0.120), size=2,
                )
                if attempt < self.limits.maximum_group_attempts
                else self._sample_uniform_anchor(rng, preferred_center, position_std)
            )
            board_yaw = float(rng.uniform(-math.pi / 12.0, math.pi / 12.0))
            board_footprint = (
                self._rotate(board_group.footprint_local, board_yaw)
                + board_anchor
            )
            if self._valid_footprint(
                board_footprint, table_bounds, outer_reach, [],
            ):
                break
        else:
            raise RuntimeError("Unable to sample a valid whiteboard pose")
        self._apply_group(data, board_group, board_anchor, board_yaw)

        # Ink is authored in board-local coordinates, so the board pose is
        # sampled first and every mark retains its margin as the board turns.
        half_length = float(rng.uniform(0.075, 0.115))
        width = float(rng.uniform(0.060, 0.080))
        phase = rng.uniform(-0.6, 0.6, size=3)
        angle = float(rng.uniform(-math.pi / 6.0, math.pi / 6.0))
        cosine, sine = math.cos(angle), math.sin(angle)
        for row in range(3):
            for point in range(61):
                geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM,
                    f"wipe_mark_{row}_{point:02d}",
                )
                t = point / 60.0
                across = ((row - 1) * (width - 0.010) / 2.0
                          + 0.005 * math.sin(
                              6.0 * math.pi * t + row * 1.2 + phase[row]
                          ))
                along = (2.0 * t - 1.0) * half_length
                self.model.geom_pos[geom_id, :2] = (
                    cosine * across - sine * along,
                    sine * across + cosine * along,
                )

        # The cloth stays down on the table for a usable grasp, while XY and
        # the complete tabletop yaw range are independent of the board pose.
        for _ in range(self.limits.maximum_group_attempts):
            eraser_anchor = self._sample_uniform_anchor(
                rng, preferred_center, position_std,
            )
            eraser_yaw = float(rng.uniform(-math.pi, math.pi))
            eraser_footprint = (
                self._rotate(eraser_group.footprint_local, eraser_yaw)
                + eraser_anchor
            )
            if self._valid_footprint(
                eraser_footprint, table_bounds, outer_reach,
                [board_footprint],
            ):
                break
        else:
            raise RuntimeError("Unable to sample a separated board eraser pose")
        self._apply_group(data, eraser_group, eraser_anchor, eraser_yaw)
        mujoco.mj_forward(self.model, data)
        return RandomizedObjectLayout(
            seed, table_bounds,
            (self.workspace.minimum_planar_reach_m, outer_reach),
            preferred_center, position_std,
            (
                LayoutPlacement(
                    ("wipe_board",), board_anchor.copy(), board_yaw,
                    board_footprint.copy(),
                ),
                LayoutPlacement(
                    ("board_eraser",), eraser_anchor.copy(), eraser_yaw,
                    eraser_footprint.copy(),
                ),
            ),
            polygon_distance(board_footprint, eraser_footprint),
        )

    def _apply_rotate_knob_layout(
        self,
        data: mujoco.MjData,
        seed: int,
        table_bounds: np.ndarray,
        outer_reach: float,
        preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Randomize the knob variant, base XY position and color.

        The root yaw and articulated paddle angle always use their authored
        nominal values so every seed presents the same initial orientation.
        """
        root_name = next(
            name for name in self.scene_manager.active_body_names
            if name.startswith("rotate_knob_")
        )
        scene_stream = zlib.crc32(b"Rotate Knob layout")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed), self._RNG_STREAM_ID, scene_stream,
        ))))
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{root_name}_joint",
        )
        paddle_visual_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{root_name}_paddle_visual_geom",
        )
        if joint_id < 0 or paddle_visual_id < 0:
            raise RuntimeError(
                f"Rotate Knob is missing joint or geom for {root_name}",
            )
        knob_group = next(
            group for group in self._groups(data)
            if tuple(member.body_name for member in group.members)
            == (root_name,)
        )
        for _ in range(self.limits.maximum_group_attempts):
            anchor = self._sample_uniform_anchor(
                rng, preferred_center, position_std,
            )
            base_yaw = 0.0
            footprint = (
                self._rotate(knob_group.footprint_local, base_yaw) + anchor
            )
            if self._valid_footprint(
                footprint, table_bounds, outer_reach, [],
            ):
                break
        else:
            raise RuntimeError("Unable to sample a valid Rotate Knob base pose")
        self._apply_group(data, knob_group, anchor, base_yaw)
        qpos_id = int(self.model.jnt_qposadr[joint_id])
        dof_id = int(self.model.jnt_dofadr[joint_id])
        data.qpos[qpos_id] = self.model.qpos0[qpos_id]
        data.qvel[dof_id] = 0.0
        colors = np.array((
            (0.88, 0.12, 0.10, 1.0),
            (0.12, 0.62, 0.20, 1.0),
            (0.10, 0.32, 0.88, 1.0),
        ))
        self.model.geom_rgba[paddle_visual_id] = colors[
            int(rng.integers(0, len(colors)))
        ]
        mujoco.mj_forward(self.model, data)
        return RandomizedObjectLayout(
            seed,
            table_bounds,
            (
                self.workspace.minimum_planar_reach_m,
                outer_reach,
            ),
            preferred_center,
            position_std,
            (LayoutPlacement(
                (root_name,), anchor.copy(), base_yaw, footprint.copy(),
            ),),
            self.limits.minimum_clearance_m,
        )

    def _set_nominal_insert_object_upright(
        self,
        data: mujoco.MjData,
    ) -> None:
        objects = {
            item.body_name: item
            for item in self.scene_manager.layout_objects(data)
        }
        container_name = next(
            name for name in objects
            if name.startswith("insert_container_")
        )
        if not container_name.endswith("_up"):
            return
        object_name = next(
            name for name in objects
            if name.startswith("insert_object_")
        )
        inserted_object = objects[object_name]
        geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{object_name}_geom",
        )
        if int(self.model.geom_type[geom_id]) == int(
            mujoco.mjtGeom.mjGEOM_BOX,
        ):
            half_length = float(self.model.geom_size[geom_id, 2])
            lying_half_height = float(self.model.geom_size[geom_id, 0])
        else:
            half_length = float(self.model.geom_size[geom_id, 1])
            lying_half_height = float(self.model.geom_size[geom_id, 0])
        position = inserted_object.position.copy()
        table_height = float(position[2]) - lying_half_height
        position[2] = table_height + half_length
        self.scene_manager.set_layout_pose(
            data,
            object_name,
            position,
            np.array((1.0, 0.0, 0.0, 0.0)),
        )
        mujoco.mj_forward(self.model, data)

    def _apply_color_block_stack_layout(
        self,
        data: mujoco.MjData,
        seed: int,
        table_bounds: np.ndarray,
        outer_reach: float,
        preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Randomize three active cubes with at least 12 cm clear space."""
        templates = {
            group.members[0].body_name: group
            for group in self._groups(data)
            if len(group.members) == 1
            and group.members[0].body_name.startswith("stack_cube_")
        }
        scene_stream = zlib.crc32(b"Color Block Stack layout")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed), self._RNG_STREAM_ID, scene_stream,
        ))))
        edge_m = 0.01 * (3 + self.scene_manager.selected_variant_index)
        half_edge_m = 0.5 * edge_m
        table_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        )
        table_geom = int(self.model.body_geomadr[table_body])
        table_top = float(
            data.geom_xpos[table_geom, 2]
            + self.model.geom_size[table_geom, 2]
        )
        for group in templates.values():
            item = group.members[0]
            collision_ids = item.geom_ids[
                self.model.geom_contype[item.geom_ids] != 0
            ]
            self.model.geom_size[collision_ids, :3] = half_edge_m
            qpos_address, _ = self.scene_manager._joint_addresses[item.body_name]
            data.qpos[qpos_address + 2] = table_top + half_edge_m
        mujoco.mj_forward(self.model, data)
        templates = {
            group.members[0].body_name: group
            for group in self._groups(data)
            if len(group.members) == 1
            and group.members[0].body_name.startswith("stack_cube_")
        }
        cube_names = tuple(sorted(templates))
        if len(cube_names) != 3:
            raise RuntimeError(
                f"Color-stack scene requires three active cubes, found {cube_names}",
            )
        source_region = np.array((
            (0.27, 0.70),
            (-0.30, 0.30),
        ))
        sampled: list[
            tuple[_GroupTemplate, np.ndarray, float, np.ndarray]
        ] = []
        accepted: list[np.ndarray] = []
        # A valid first pair can occasionally leave no 12 cm-clear slot for
        # the third cube. Restart the whole group instead of making dataset
        # initialization depend on that unlucky partial layout.
        for _ in range(self.limits.maximum_layout_attempts):
            trial_sampled: list[
                tuple[_GroupTemplate, np.ndarray, float, np.ndarray]
            ] = []
            trial_accepted: list[np.ndarray] = []
            for name in cube_names:
                group = templates[name]
                for _ in range(self.limits.maximum_group_attempts):
                    anchor = self.scene_manager.workspace_xy(rng.uniform(
                        source_region[:, 0], source_region[:, 1],
                    ))
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    footprint = self._rotate(
                        group.footprint_local, yaw,
                    ) + anchor
                    if (
                        self._valid_footprint(
                            footprint, table_bounds, outer_reach, (),
                        )
                        and all(
                            polygon_distance(footprint, other)
                            >= _COLOR_STACK_MINIMUM_CLEARANCE_M
                            for other in trial_accepted
                        )
                    ):
                        trial_sampled.append((
                            group, anchor, yaw, footprint,
                        ))
                        trial_accepted.append(footprint)
                        break
                else:
                    break
            else:
                sampled = trial_sampled
                accepted = trial_accepted
                break
        else:
            raise RuntimeError(
                "Unable to sample a 12 cm-clear color-stack layout",
            )
        placements: list[LayoutPlacement] = []
        for group, anchor, yaw, footprint in sampled:
            self._apply_group(data, group, anchor, yaw)
            placements.append(LayoutPlacement(
                tuple(member.body_name for member in group.members),
                anchor.copy(), yaw, footprint.copy(),
            ))
        mujoco.mj_forward(self.model, data)
        return RandomizedObjectLayout(
            seed,
            table_bounds,
            (self.workspace.minimum_planar_reach_m, outer_reach),
            preferred_center,
            position_std,
            tuple(placements),
            min(
                polygon_distance(first, second)
                for index, first in enumerate(accepted)
                for second in accepted[index + 1:]
            ),
        )

    def _apply_round_handle_pot_pear_layout(
        self,
        data: mujoco.MjData,
        seed: int,
        table_bounds: np.ndarray,
        outer_reach: float,
        preferred_center: np.ndarray,
        position_std: np.ndarray,
    ) -> RandomizedObjectLayout:
        """Keep the large pot and pickup pear in separate reliable zones."""
        groups = self._groups(data)
        pot_group = next(
            group for group in groups
            if {member.body_name for member in group.members}
            == {"teleop_pot", "teleop_pot_lid"}
        )
        pear_group = next(
            group for group in groups
            if tuple(member.body_name for member in group.members)
            == ("fruit_task_pear",)
        )
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(seed),
            self._RNG_STREAM_ID,
            zlib.crc32(b"Round Handle Pot with Pear layout"),
        ))))
        slots = (
            (
                pot_group,
                self.scene_manager.workspace_xy(np.array((0.53, -0.05))),
                np.array((0.012, 0.018)),
                np.deg2rad(20.0),
            ),
            (
                pear_group,
                self.scene_manager.workspace_xy(np.array((0.28, 0.18))),
                np.array((0.015, 0.018)),
                np.pi,
            ),
        )
        for _ in range(self.limits.maximum_layout_attempts):
            accepted: list[np.ndarray] = []
            sampled: list[
                tuple[_GroupTemplate, np.ndarray, float, np.ndarray]
            ] = []
            for group, center, jitter, yaw_limit in slots:
                anchor = center + rng.uniform(-jitter, jitter)
                yaw = float(rng.uniform(-yaw_limit, yaw_limit))
                footprint = self._rotate(group.footprint_local, yaw) + anchor
                if not self._valid_footprint(
                    footprint, table_bounds, outer_reach, accepted,
                ):
                    break
                accepted.append(footprint)
                sampled.append((group, anchor, yaw, footprint))
            else:
                for group, anchor, yaw, _ in sampled:
                    self._apply_group(data, group, anchor, yaw)
                mujoco.mj_forward(self.model, data)
                return RandomizedObjectLayout(
                    seed,
                    table_bounds,
                    (self.workspace.minimum_planar_reach_m, outer_reach),
                    preferred_center,
                    position_std,
                    tuple(LayoutPlacement(
                        tuple(member.body_name for member in group.members),
                        anchor.copy(),
                        yaw,
                        footprint.copy(),
                    ) for group, anchor, yaw, footprint in sampled),
                    polygon_distance(accepted[0], accepted[1]),
                )
        raise RuntimeError(
            "Unable to sample the round-handle pot and pear safe layout",
        )

    def apply(
        self,
        data: mujoco.MjData,
        seed: int,
        *,
        variant_index: int | None = None,
        variant_indices: tuple[int, ...] | None = None,
    ) -> RandomizedObjectLayout:
        if seed < -1:
            raise ValueError("Domain seed must be -1 (disabled) or a non-negative integer")
        # Always start from the scene's nominal pose at the current table offset.
        # Scenes with interchangeable object models select exactly one variant
        # through a separate deterministic stream before layout randomization.
        if variant_index is not None and variant_indices is not None:
            raise ValueError("variant_index and variant_indices are mutually exclusive")
        if variant_index is None:
            self.scene_manager.select_seeded_variant(seed, variant_indices)
        else:
            self.scene_manager.select_variant_index(variant_index)
        self.scene_manager.activate(data, self.scene_manager.current_name)
        table_bounds, table_height = self._table_bounds(data)
        outer_reach = self.workspace.planar_outer_radius(table_height)
        sampling_center, sampling_half_span = self._uniform_distribution(table_bounds)
        if seed == -1:
            if self.scene_manager.current_name == WHITEBOARD_SCENE:
                for row in range(3):
                    for point in range(61):
                        geom_id = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_GEOM,
                            f"wipe_mark_{row}_{point:02d}",
                        )
                        t = point / 60.0
                        self.model.geom_pos[geom_id, :2] = (
                            (row - 1) * 0.025
                            + 0.005 * math.sin(6.0 * math.pi * t + row * 1.2),
                            -0.105 + 0.210 * t,
                        )
                mujoco.mj_forward(self.model, data)
            if self.scene_manager.current_name == "Insert Object":
                self._set_nominal_insert_object_upright(data)
            # Authored asset-gallery scenes can intentionally be denser than
            # task layouts. Random generation is universally validated below;
            # task-specific nominal layouts are covered by their regressions.
            return RandomizedObjectLayout(
                seed, table_bounds,
                (self.workspace.minimum_planar_reach_m, outer_reach),
                sampling_center, sampling_half_span, (),
                self.limits.minimum_clearance_m,
            )
        if self.scene_manager.current_name == "Insert Object":
            return self._validate_initial_clearance(
                data,
                self._apply_insert_object_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )
        if self.scene_manager.current_name in (
            "Fruit Receptacles",
            BANANA_APPLE_TRAY_SCENE,
            UNSEEN_FRUIT_RECEPTACLES_SCENE,
        ):
            return self._validate_initial_clearance(
                data,
                self._apply_fruit_receptacle_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )
        if self.scene_manager.current_name in (
            "Paper Placement", "Ball + Paper Placement",
            TOLERANCE_DISCRIMINATION_BOX_LYING_SCENE,
            UNSEEN_BOX_REGION_SCENE,
        ) or self.scene_manager.current_name.startswith("Geometry Region -"):
            return self._validate_initial_clearance(
                data,
                self._apply_paper_placement_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )
        if self.scene_manager.current_name == "Color Block Stack":
            return self._validate_initial_clearance(
                data,
                self._apply_color_block_stack_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )
        if self.scene_manager.current_name == ROUND_HANDLE_POT_PEAR_SCENE:
            return self._validate_initial_clearance(
                data,
                self._apply_round_handle_pot_pear_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )
        if self.scene_manager.current_name == WHITEBOARD_SCENE:
            return self._validate_initial_clearance(
                data, self._apply_whiteboard_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )
        if self.scene_manager.current_name == "Rotate Knob":
            return self._validate_initial_clearance(
                data,
                self._apply_rotate_knob_layout(
                    data, seed, table_bounds, outer_reach,
                    sampling_center, sampling_half_span,
                ),
            )

        scene_stream = zlib.crc32(self.scene_manager.current_name.encode("utf-8"))
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((int(seed), self._RNG_STREAM_ID, scene_stream))))
        groups = self._groups(data)
        for _ in range(self.limits.maximum_layout_attempts):
            accepted: list[np.ndarray] = []
            placements: list[tuple[_GroupTemplate, np.ndarray, float, np.ndarray]] = []
            successful = True
            for group in groups:
                for _ in range(self.limits.maximum_group_attempts):
                    anchor = self._sample_uniform_anchor(
                        rng, sampling_center, sampling_half_span,
                    )
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    footprint = self._rotate(group.footprint_local, yaw) + anchor
                    if self._valid_footprint(footprint, table_bounds, outer_reach, accepted):
                        accepted.append(footprint)
                        placements.append((group, anchor, yaw, footprint))
                        break
                else:
                    successful = False
                    break
            if successful:
                for group, anchor, yaw, _ in placements:
                    self._apply_group(data, group, anchor, yaw)
                mujoco.mj_forward(self.model, data)
                minimum = min(
                    (polygon_distance(first, second) for index, first in enumerate(accepted) for second in accepted[index + 1:]),
                    default=float("inf"),
                )
                return self._validate_initial_clearance(
                    data,
                    RandomizedObjectLayout(
                        seed, table_bounds,
                        (self.workspace.minimum_planar_reach_m, outer_reach),
                        sampling_center, sampling_half_span,
                        tuple(LayoutPlacement(
                            tuple(member.body_name for member in group.members),
                            anchor.copy(), yaw, footprint.copy(),
                        ) for group, anchor, yaw, footprint in placements),
                        minimum,
                    ),
                )
        raise RuntimeError(
            f"Could not place {len(groups)} object groups in {self.scene_manager.current_name!r} "
            f"with {self.limits.minimum_clearance_m:.3f} m clearance"
        )
