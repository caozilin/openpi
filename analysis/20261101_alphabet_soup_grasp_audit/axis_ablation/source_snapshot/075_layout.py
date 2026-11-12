"""Translate Blocks objects using one global tabletop reachability map."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shlex
import zlib

import mujoco
import numpy as np

from ..config import PROJECT_ROOT
from ..data.task_sets import MULTI_SUBTASK_SUBCATEGORY, training_tasks
from ..simulation.robot import RobotHandles, ee_pose
from ..simulation.randomization.objects import (
    body_planar_footprint,
    polygon_distance,
)
from ..simulation.scenes import SceneManager
from ..tasks.catalog import (
    ADJUST_BOTTLE_TASK,
    BALL_TO_PLATE_TASK,
    BLOCK_TO_PLATE_TASK,
    CUBE_TO_PLATE_TASK,
    CYLINDER_TO_PLATE_TASK,
    CLOSE_CYLINDRICAL_POT_LID_TASK,
    CLOSE_HANDLE_POT_LID_TASK,
    PRESS_BUTTON_TASK,
    ROTATE_KNOB_TASK,
    WIPE_WHITEBOARD_TASK,
)
from ..tasks.planning.adjust_bottle import StandFallenBottle
from ..tasks.planning.nominal_candidates import nominal_candidate_definitions
from ..tasks.planning.presets import PresetTaskExecutor
from ..tasks.subtasks import identify_pick_place_subtasks
from .map import ReachabilityMap
from .boundary_surface import (
    WorkspaceBoundarySurface,
    validate_distance_factor_range,
)


_TASK_SOURCE = {
    BLOCK_TO_PLATE_TASK: "teleop_tall_block",
    CUBE_TO_PLATE_TASK: "teleop_cube",
    BALL_TO_PLATE_TASK: "teleop_ball",
    CYLINDER_TO_PLATE_TASK: "teleop_cylinder",
}

_BOUNDARY_CACHE: dict[
    tuple[str, int], tuple[np.ndarray, np.ndarray]
] = {}

WORKSPACE_BOUNDARY_DISTANCE_MIN = 1.0
WORKSPACE_BOUNDARY_DISTANCE_MAX = 1.0
WORKSPACE_BOUNDARY_MODES = (
    "pre_boundary",
)


def default_map_path(robot_uid: str = "panda") -> Path:
    return (
        PROJECT_ROOT / "assets" / "reachability"
        / f"{robot_uid}_table_home_v1.npz"
    )


def default_boundary_surface_path(robot_uid: str = "panda") -> Path:
    return (
        PROJECT_ROOT / "assets" / "reachability"
        / f"{robot_uid}_workspace_boundary_v2.npz"
    )


def _stage_endpoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    task_name: str,
    scene_name: str = "Blocks",
) -> dict[str, np.ndarray]:
    position, rotation = ee_pose(data, handles)
    executor = PresetTaskExecutor(model, handles)
    executor.start(task_name, scene_name, data, position, rotation)
    endpoints: dict[str, np.ndarray] = {}
    previous_closed = False
    for frame in executor.frames:
        if frame.gripper_closed and not previous_closed and "grasp" not in endpoints:
            endpoints["grasp"] = frame.position.copy()
        if not frame.gripper_closed and previous_closed and "place" not in endpoints:
            endpoints["place"] = frame.position.copy()
        previous_closed = frame.gripper_closed
    return endpoints


def _pre_post_endpoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    task_name: str,
    scene_name: str,
    *,
    nominal_selection_enabled: bool = True,
    candidate_index: int | None = None,
) -> dict[str, np.ndarray]:
    """Return the first scripted Pre/Post endpoint for every training task."""
    position, rotation = ee_pose(data, handles)
    executor = PresetTaskExecutor(model, handles)
    executor.start(
        task_name, scene_name, data, position, rotation,
        nominal_selection_enabled=nominal_selection_enabled,
        _nominal_candidate_index=candidate_index,
    )
    spans = identify_pick_place_subtasks(executor.frames)
    if not spans:
        raise RuntimeError(
            f"Training task {task_name!r} has no Pre/Post endpoint pair",
        )
    first = spans[0]
    return {
        "pregrasp": executor.frames[first.pregrasp_index].position.copy(),
        "postgrasp": executor.frames[first.postgrasp_index].position.copy(),
    }


def training_workspace_boundary_anchor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
    task_name: str,
    scene_name: str,
) -> tuple[np.ndarray, str]:
    """Resolve a task's geometric grasp/operation target before feasibility checks.

    Layout sampling may intentionally go beyond the IK boundary. Bottle grasp
    geometry therefore comes from the same template used by its CPU planner,
    without requiring the complete stand-up path to be feasible beforehand.
    Other tasks use the first declared template without nominal FK screening;
    normal candidate selection remains part of execution planning.
    """
    if task_name == ADJUST_BOTTLE_TASK:
        return StandFallenBottle(model, handles).grasp_position(data), "grasp"
    if task_name == PRESS_BUTTON_TASK:
        position, rotation = ee_pose(data, handles)
        executor = PresetTaskExecutor(model, handles)
        executor.start(
            task_name, scene_name, data, position, rotation,
            nominal_selection_enabled=False,
        )
        return executor.frames[-1].position.copy(), "press"
    return _pre_post_endpoints(
        model, data, handles, task_name, scene_name,
        nominal_selection_enabled=False,
        candidate_index=(
            0 if nominal_candidate_definitions(model, data, scene_name, task_name)
            else None
        ),
    )["pregrasp"], "grasp"


def _one_active_name(
    names: tuple[str, ...],
    *,
    prefix: str | None = None,
    exact: str | None = None,
) -> str:
    matches = tuple(
        name for name in names
        if (exact is not None and name == exact)
        or (prefix is not None and name.startswith(prefix))
    )
    if len(matches) != 1:
        selector = exact if exact is not None else f"{prefix}*"
        raise RuntimeError(
            f"Expected one active workspace-boundary body {selector!r}, "
            f"found {matches}",
        )
    return matches[0]


def _training_boundary_roles(
    scene_name: str,
    task_name: str,
    active_names: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Resolve source/device roots for short tasks; keep targets stationary."""
    if any(
        scene.runtime_scene == scene_name and task.runtime_task == task_name
        and scene.subcategory == MULTI_SUBTASK_SUBCATEGORY
        for scene, task in training_tasks()
    ):
        raise ValueError(
            f"Workspace-boundary placement requires a short training task: "
            f"scene={scene_name!r}, task={task_name!r}",
        )
    contextual_sources = {
        ADJUST_BOTTLE_TASK: ("adjust_bottle_", None),
        PRESS_BUTTON_TASK: ("press_button_", None),
        ROTATE_KNOB_TASK: ("rotate_knob_", None),
        WIPE_WHITEBOARD_TASK: (None, "board_eraser"),
        CLOSE_CYLINDRICAL_POT_LID_TASK: (None, "teleop_pot_lid"),
        CLOSE_HANDLE_POT_LID_TASK: (None, "teleop_pot_lift_lid"),
    }
    if task_name in contextual_sources:
        prefix, exact = contextual_sources[task_name]
        return (
            (_one_active_name(active_names, prefix=prefix, exact=exact),),
            (), "source_only",
        )
    paper_sources = tuple(
        name for name in active_names if name.startswith("paper_object_")
    )
    if paper_sources:
        source = (_one_active_name(active_names, prefix="paper_object_"),)
        if "tray_khaki_medium" in active_names:
            _one_active_name(active_names, exact="tray_khaki_medium")
        elif "teleop_plate" in active_names:
            _one_active_name(active_names, exact="teleop_plate")
        else:
            _one_active_name(active_names, prefix="paper_decal_")
        # Boundary evaluation always moves the source object. The Post-grasp
        # target (tray or white region) deliberately keeps its normal pose.
        return source, (), "source_only"

    for source_prefix in ("narrow_box_cylinder_", "narrow_box_block_"):
        sources = tuple(
            name for name in active_names if name.startswith(source_prefix)
        )
        if sources:
            return (
                (_one_active_name(active_names, prefix=source_prefix),),
                (),
                "source_only",
            )

    raise ValueError(
        f"Workspace-boundary placement is not registered for "
        f"scene={scene_name!r}, task={task_name!r}",
    )


def _set_group_xy_delta(
    scene_manager: SceneManager,
    data: mujoco.MjData,
    originals: dict[str, tuple[np.ndarray, np.ndarray]],
    names: tuple[str, ...],
    delta_xy: np.ndarray,
) -> None:
    for name in names:
        position, quaternion = originals[name]
        moved = position.copy()
        moved[:2] += np.asarray(delta_xy, dtype=float)
        scene_manager.set_layout_pose(data, name, moved, quaternion)
    mujoco.mj_forward(scene_manager.model, data)


def _group_inside_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...],
) -> bool:
    return all(
        _inside_table(
            model, data,
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)),
        )
        for name in names
    )


def _group_has_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...],
    active_names: tuple[str, ...],
    minimum_clearance_m: float = 0.01,
) -> bool:
    moved = set(names)
    for name in names:
        body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, name,
        ))
        footprint = body_planar_footprint(model, data, body_id)
        for other_name in active_names:
            if other_name in moved:
                continue
            other_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, other_name,
            ))
            other = body_planar_footprint(model, data, other_id)
            if polygon_distance(footprint, other) < minimum_clearance_m:
                return False
    return True


def apply_training_workspace_boundary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_manager: SceneManager,
    handles: RobotHandles,
    task_name: str,
    mode: str,
    seed: int,
    *,
    distance_factor_lower: float = WORKSPACE_BOUNDARY_DISTANCE_MIN,
    distance_factor_upper: float = WORKSPACE_BOUNDARY_DISTANCE_MAX,
    map_directory: Path | None = None,
) -> dict[str, object]:
    """Place the source object on a signed offset of the dense IK boundary.

    The Post-grasp target remains at its normal pose. Buttons use the press
    endpoint and move the complete device root. Complete footprints are constrained to
    the far 50 cm x central 90% rectangle; equality at its edge is legal.
    """
    if mode not in WORKSPACE_BOUNDARY_MODES:
        raise ValueError(f"Unknown workspace-boundary mode: {mode!r}")
    factor_lower, factor_upper = validate_distance_factor_range(
        distance_factor_lower, distance_factor_upper,
    )
    objects = {
        item.body_name: item for item in scene_manager.layout_objects(data)
    }
    active_names = tuple(objects)
    pre_names, _, role_kind = _training_boundary_roles(
        scene_manager.current_name, task_name, active_names,
    )
    originals = {
        name: (item.position.copy(), item.quaternion.copy())
        for name, item in objects.items()
    }
    directory = (
        PROJECT_ROOT / "assets" / "reachability"
        if map_directory is None else Path(map_directory)
    )
    surface_path = directory / f"{handles.spec.uid}_workspace_boundary_v2.npz"
    if not surface_path.exists():
        raise FileNotFoundError(
            f"Missing workspace-boundary surface: {surface_path}. Run "
            f"`uv run python tools/build_workspace_boundary_surface.py "
            f"--robot {handles.spec.uid}` first.",
        )
    surface = _load_boundary_surface(str(surface_path.resolve()))
    if surface.robot_uid != handles.spec.uid:
        raise ValueError(
            f"Workspace-boundary surface {surface_path} belongs to another robot",
        )

    effective_stages = ("pregrasp",)
    stage_groups = {"pregrasp": pre_names}
    effective_mode = "pre_boundary"

    root_seed = 0 if seed == -1 else int(seed)
    factor_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
        root_seed, zlib.crc32(b"workspace boundary distance factor"),
    ))))
    requested_factor = float(factor_rng.uniform(factor_lower, factor_upper))
    if factor_lower == factor_upper:
        requested_factor = factor_lower
    placements: list[dict[str, object]] = []
    accepted_endpoints: list[np.ndarray] = []
    boundary_moved_names: list[str] = []
    role_names = {
        name
        for stage in effective_stages
        for name in stage_groups[stage]
    }
    static_names = tuple(
        name for name in active_names if name not in role_names
    )
    for stage in effective_stages:
        names = stage_groups[stage]
        if not names:
            continue
        endpoint, anchor_kind = training_workspace_boundary_anchor(
            model, data, handles, task_name, scene_manager.current_name,
        )
        if not surface.z_m[0] - 1e-9 <= endpoint[2] <= surface.z_m[-1] + 1e-9:
            raise ValueError(
                f"Workspace-boundary {anchor_kind} height {endpoint[2]:.4f} m "
                f"is outside surface range [{surface.z_m[0]:.4f}, "
                f"{surface.z_m[-1]:.4f}] m. Extend the surface with "
                "tools/build_workspace_boundary_surface.py "
                f"--robot {handles.spec.uid} "
                f"--output-dir {shlex.quote(str(directory))} "
                f"--z-min {min(surface.z_m[0], endpoint[2] - 0.03):.3f} "
                f"--z-max {max(surface.z_m[-1], endpoint[2] + 0.03):.3f}",
            )
        rectangle = surface.search_bounds_xy_m.copy()
        table_body = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
        ))
        table_geoms = np.flatnonzero(model.geom_bodyid == table_body)
        table_geom = int(table_geoms[
            np.argmax(np.prod(model.geom_size[table_geoms, :2], axis=1))
        ])
        table_center = data.geom_xpos[table_geom]
        table_rotation = data.geom_xmat[table_geom].reshape(3, 3)
        table_half_xy = np.abs(table_rotation[:2]) @ model.geom_size[table_geom]
        table_bounds = np.array((
            (table_center[0] - table_half_xy[0], table_center[0] + table_half_xy[0]),
            (table_center[1] - table_half_xy[1], table_center[1] + table_half_xy[1]),
        ))
        # Intersect with the live tabletop as well as the authored rectangle.
        # Equality is intentionally retained: the object's outermost point may
        # touch either boundary but may never cross it.
        rectangle[:, 0] = np.maximum(rectangle[:, 0], table_bounds[:, 0])
        rectangle[:, 1] = np.minimum(rectangle[:, 1], table_bounds[:, 1])
        footprints = []
        for name in names:
            body_id = int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name,
            ))
            footprints.append(body_planar_footprint(
                model, data, body_id, include_visual_geoms=True,
            ))
        footprint = np.vstack(footprints)
        delta_bounds = np.array((
            (
                rectangle[0, 0] - float(np.min(footprint[:, 0])),
                rectangle[0, 1] - float(np.max(footprint[:, 0])),
            ),
            (
                rectangle[1, 0] - float(np.min(footprint[:, 1])),
                rectangle[1, 1] - float(np.max(footprint[:, 1])),
            ),
        ))
        legal_endpoint_bounds = delta_bounds + endpoint[:2, None]
        if np.any(legal_endpoint_bounds[:, 0] > legal_endpoint_bounds[:, 1]):
            raise RuntimeError(
                "The active object's complete footprint cannot fit inside "
                "the workspace-boundary rectangle",
            )
        valid_y = surface.y_m[
            (surface.y_m >= legal_endpoint_bounds[1, 0] - 1e-12)
            & (surface.y_m <= legal_endpoint_bounds[1, 1] + 1e-12)
        ]
        if valid_y.size == 0:
            raise RuntimeError("No boundary-curve Y sample fits the active object")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            root_seed,
            zlib.crc32(f"workspace boundary {stage}".encode("utf-8")),
        ))))
        order = rng.permutation(len(valid_y))
        accepted = None
        for order_index in order:
            curve_y = float(valid_y[int(order_index)])
            try:
                offset = surface.offset_sample(
                    y_m=curve_y,
                    z_m=float(endpoint[2]),
                    factor=requested_factor,
                    legal_center_bounds_xy_m=legal_endpoint_bounds,
                )
            except ValueError:
                continue
            requested = offset.clipped_point_m
            if any(
                np.linalg.norm(requested[:2] - previous[:2]) < 0.08
                for previous in accepted_endpoints
            ):
                continue
            delta_xy = requested[:2] - endpoint[:2]
            _set_group_xy_delta(
                scene_manager, data, originals, names, delta_xy,
            )
            # Recompute the complete visual/collision footprint after translation. The
            # comparison is inclusive so an outer point may touch an edge.
            inside_rectangle = True
            for name in names:
                body_id = int(mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, name,
                ))
                moved_footprint = body_planar_footprint(
                    model, data, body_id, include_visual_geoms=True,
                )
                if (
                    np.any(moved_footprint[:, 0] < rectangle[0, 0] - 1e-9)
                    or np.any(moved_footprint[:, 0] > rectangle[0, 1] + 1e-9)
                    or np.any(moved_footprint[:, 1] < rectangle[1, 0] - 1e-9)
                    or np.any(moved_footprint[:, 1] > rectangle[1, 1] + 1e-9)
                ):
                    inside_rectangle = False
                    break
            if (
                not inside_rectangle
                or not _group_has_clearance(
                    model, data, names,
                    tuple((*names, *static_names, *boundary_moved_names)),
                )
            ):
                _set_group_xy_delta(
                    scene_manager, data, originals, names, np.zeros(2),
                )
                continue
            try:
                actual, _ = training_workspace_boundary_anchor(
                    model, data, handles, task_name,
                    scene_manager.current_name,
                )
            except (RuntimeError, ValueError):
                _set_group_xy_delta(
                    scene_manager, data, originals, names, np.zeros(2),
                )
                continue
            if not np.allclose(actual, requested, atol=1e-6, rtol=0.0):
                _set_group_xy_delta(
                    scene_manager, data, originals, names, np.zeros(2),
                )
                continue
            accepted = (int(order_index), offset, actual.copy(), delta_xy.copy())
            if accepted is not None:
                break
        if accepted is None:
            raise RuntimeError(
                f"No surface-guided {stage} placement fits the legal rectangle",
            )
        boundary_index, offset, actual, delta_xy = accepted
        accepted_endpoints.append(actual)
        boundary_moved_names.extend(names)
        placements.append({
            "stage": stage,
            "anchor_kind": anchor_kind,
            "body_names": list(names),
            "requested_endpoint_m": offset.clipped_point_m.tolist(),
            "actual_endpoint_m": actual.tolist(),
            "translation_xy_m": delta_xy.tolist(),
            "boundary_index": boundary_index,
            "curve_parameter_y_m": float(offset.base_point_m[1]),
            "boundary_point_m": offset.base_point_m.tolist(),
            "boundary_outward_normal_xy": offset.outward_normal_xy.tolist(),
            "requested_distance_factor": offset.requested_factor,
            "requested_signed_offset_m": offset.requested_offset_m,
            "unclipped_requested_endpoint_m": offset.requested_point_m.tolist(),
            "effective_signed_offset_m": offset.effective_offset_m,
            "effective_distance_factor": offset.effective_factor,
            "rectangle_saturated": offset.saturated,
            "legal_endpoint_center_bounds_xy_m": legal_endpoint_bounds.tolist(),
            "complete_object_rectangle_bounds_xy_m": rectangle.tolist(),
            "boundary_distance_m": abs(offset.effective_offset_m),
        })

    return {
        "mode": "workspace_boundary",
        "requested_strategy": mode,
        "effective_strategy": effective_mode,
        "role_kind": role_kind,
        "distance_factor_lower": factor_lower,
        "distance_factor_upper": factor_upper,
        "sampled_distance_factor": requested_factor,
        "distance_factor_scale_m": 0.50,
        "surface_path": str(surface_path),
        "boundary_reference": "fixed_home_rotation_ik",
        "path_collision_dynamics_checked": False,
        "placements": placements,
    }


@lru_cache(maxsize=8)
def _load_boundary_surface(path: str) -> WorkspaceBoundarySurface:
    return WorkspaceBoundarySurface.load(path)


@lru_cache(maxsize=8)
def _load_reachability_map(path: str) -> ReachabilityMap:
    return ReachabilityMap.load(path)


def _boundary_pairs(
    reachability_map: ReachabilityMap,
    map_path: Path,
    fixed_z_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearby 3-D boundary midpoints and horizontal outward normals."""
    cache_key = (str(map_path.resolve()), int(round(fixed_z_m * 1000.0)))
    cached = _BOUNDARY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    z_distance = np.abs(reachability_map.positions[:, 2] - fixed_z_m)
    nearest_z = float(np.min(z_distance))
    indices = np.flatnonzero(z_distance <= max(0.025, nearest_z + 1e-9))
    positions = reachability_map.positions[indices]
    reachable = reachability_map.reachable[indices]
    if not np.any(reachable) or np.all(reachable):
        raise RuntimeError("Map slice has no reachable/unreachable boundary")

    bucket_width = 0.02
    buckets: dict[tuple[int, int, int, bool], list[int]] = {}
    bucket_xyz = np.floor(positions / bucket_width).astype(int)
    for local_index, (bx, by, bz) in enumerate(bucket_xyz):
        key = (int(bx), int(by), int(bz), bool(reachable[local_index]))
        buckets.setdefault(key, []).append(local_index)

    # Keep the closest opposite-label pair per centimetre of boundary. The
    # adaptive map resolves mixed cells at roughly 1 cm, so 3 cm is a
    # deliberately conservative pairing ceiling.
    best: dict[tuple[int, int, int], tuple[float, np.ndarray, np.ndarray]] = {}
    source_label = bool(np.sum(reachable) <= np.sum(~reachable))
    for local_index in np.flatnonzero(reachable == source_label):
        bx, by, bz = bucket_xyz[local_index]
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    opposite = buckets.get((
                        int(bx + dx), int(by + dy), int(bz + dz),
                        not source_label,
                    ), ())
                    if not opposite:
                        continue
                    other_positions = positions[
                        np.asarray(opposite, dtype=int)
                    ]
                    distances = np.linalg.norm(
                        other_positions - positions[local_index], axis=1,
                    )
                    nearest_index = int(np.argmin(distances))
                    distance = float(distances[nearest_index])
                    if not 1e-9 < distance <= 0.03:
                        continue
                    other_local = int(opposite[nearest_index])
                    reachable_position = (
                        positions[local_index]
                        if source_label else positions[other_local]
                    )
                    unreachable_position = (
                        positions[other_local]
                        if source_label else positions[local_index]
                    )
                    midpoint = 0.5 * (
                        reachable_position + unreachable_position
                    )
                    if abs(float(midpoint[2] - fixed_z_m)) > 0.02:
                        continue
                    horizontal = (
                        unreachable_position[:2] - reachable_position[:2]
                    )
                    horizontal_norm = float(np.linalg.norm(horizontal))
                    if horizontal_norm <= 1e-6:
                        continue
                    normal = horizontal / horizontal_norm
                    bin_key = tuple(
                        np.round(midpoint / 0.01).astype(int)
                    )
                    previous = best.get(bin_key)
                    if previous is None or distance < previous[0]:
                        best[bin_key] = (distance, midpoint, normal)
    if not best:
        raise RuntimeError("No refined reachability boundary found on map slice")
    midpoints = np.asarray([item[1] for item in best.values()])
    normals = np.asarray([item[2] for item in best.values()])
    result = (midpoints, normals)
    _BOUNDARY_CACHE[cache_key] = result
    return result


def _inside_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body_id: int,
    margin_m: float = 0.005,
) -> bool:
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
    if table_body < 0:
        return True
    table_geoms = np.flatnonzero(model.geom_bodyid == table_body)
    table_geom = int(table_geoms[np.argmax(np.prod(model.geom_size[table_geoms, :2], axis=1))])
    center = data.geom_xpos[table_geom]
    half = np.abs(data.geom_xmat[table_geom].reshape(3, 3)[:2]) @ model.geom_size[table_geom]
    descendants = []
    for candidate in range(model.nbody):
        ancestor = candidate
        while ancestor > object_body_id:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == object_body_id:
            descendants.append(candidate)
    for geom_id in np.flatnonzero(np.isin(model.geom_bodyid, descendants)):
        radius = float(model.geom_rbound[geom_id])
        xy = data.geom_xpos[geom_id, :2]
        if np.any(xy - radius < center[:2] - half + margin_m):
            return False
        if np.any(xy + radius > center[:2] + half - margin_m):
            return False
    return True


def apply_blocks_difficulty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_manager: SceneManager,
    handles: RobotHandles,
    task_name: str,
    difficulty: float,
    seed: int,
    *,
    map_directory: Path | None = None,
) -> dict[str, object] | None:
    """Apply difficulty after the legacy layout; zero is an exact no-op."""
    difficulty = float(difficulty)
    if difficulty == 0.0:
        return None
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("Placement difficulty must be in [0, 1]")
    if scene_manager.current_name != "Blocks" or task_name not in _TASK_SOURCE:
        raise ValueError("Reachability-guided placement currently supports the two Blocks pick/place tasks")
    source_name = _TASK_SOURCE[task_name]
    objects = {item.body_name: item for item in scene_manager.layout_objects(data)}
    source = objects[source_name]
    plate = objects["teleop_plate"]
    # The map is fixed at Home orientation. Reset the selected top grasp to the
    # seed=-1 yaw while retaining its randomized position.
    scene_manager.set_layout_pose(
        data, source_name, source.position, np.array((1.0, 0.0, 0.0, 0.0)),
    )
    mujoco.mj_forward(model, data)
    endpoints = _stage_endpoints(model, data, handles, task_name)
    root_seed = 0 if seed == -1 else int(seed)
    stage_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
        root_seed, zlib.crc32(b"placement difficulty stage"),
    ))))
    stage = "grasp" if int(stage_rng.integers(0, 2)) == 0 else "place"
    directory = PROJECT_ROOT / "assets" / "reachability" if map_directory is None else Path(map_directory)
    map_path = directory / f"{handles.spec.uid}_table_home_v1.npz"
    if not map_path.exists():
        raise FileNotFoundError(
            f"Missing tabletop reachability map: {map_path}. "
            "Run tools/build_reachability_map.py first."
        )
    reachability_map = _load_reachability_map(str(map_path.resolve()))
    if reachability_map.metadata.get("robot_uid") != handles.spec.uid:
        raise ValueError(
            f"Reachability map {map_path} is not a {handles.spec.uid} map"
        )
    target_name = source_name if stage == "grasp" else "teleop_plate"
    target = source if stage == "grasp" else plate
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
    original_position = target.position.copy()
    original_quaternion = (
        np.array((1.0, 0.0, 0.0, 0.0))
        if stage == "grasp" else target.quaternion.copy()
    )
    chosen = None
    for attempt in range(64):
        sample = reachability_map.sample(
            difficulty, root_seed + attempt * 104729,
            fixed_z_m=float(endpoints[stage][2]),
        )
        delta = sample.position - endpoints[stage]
        candidate = original_position.copy()
        candidate[:2] += delta[:2]
        # Tabletop objects retain their physical support height; Z is queried
        # from the 3-D map only as the nearest terminal-pose slice.
        scene_manager.set_layout_pose(
            data, target_name, candidate, original_quaternion,
        )
        mujoco.mj_forward(model, data)
        if _inside_table(model, data, body_id):
            chosen = sample
            break
    if chosen is None:
        scene_manager.set_layout_pose(
            data, target_name, original_position, original_quaternion,
        )
        mujoco.mj_forward(model, data)
        raise RuntimeError("No map-guided placement fit on the physical table")
    actual = _stage_endpoints(model, data, handles, task_name)[stage]
    return {
        "placement_difficulty": difficulty,
        "stage": stage,
        "map_path": str(map_path),
        "map_index": chosen.index,
        "requested_endpoint_m": chosen.position.tolist(),
        "actual_endpoint_m": actual.tolist(),
        "badness": chosen.badness,
        "reachable": chosen.reachable,
        "quantile": chosen.quantile,
        "resolution_m": chosen.resolution_m,
    }


def apply_task_boundary_placement(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_manager: SceneManager,
    handles: RobotHandles,
    task_name: str,
    seed: int,
    *,
    map_directory: Path | None = None,
    boundary_half_width_m: float = 0.02,
) -> dict[str, object] | None:
    """Place only the task's pickup object within +/-2 cm of the map boundary."""
    if boundary_half_width_m <= 0.0:
        raise ValueError("Boundary half-width must be positive")
    if scene_manager.current_name != "Blocks" or task_name not in _TASK_SOURCE:
        return None
    source_name = _TASK_SOURCE[task_name]
    objects = {
        item.body_name: item for item in scene_manager.layout_objects(data)
    }
    source = objects[source_name]
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, source_name,
    )
    original_position = source.position.copy()
    home_yaw = np.array((1.0, 0.0, 0.0, 0.0))
    scene_manager.set_layout_pose(
        data, source_name, original_position, home_yaw,
    )
    mujoco.mj_forward(model, data)
    endpoints = _stage_endpoints(
        model, data, handles, task_name, scene_manager.current_name,
    )
    if "grasp" not in endpoints:
        raise RuntimeError("Selected task has no grasp endpoint")
    grasp_endpoint = endpoints["grasp"]
    directory = (
        PROJECT_ROOT / "assets" / "reachability"
        if map_directory is None else Path(map_directory)
    )
    map_path = directory / f"{handles.spec.uid}_table_home_v1.npz"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing tabletop reachability map: {map_path}")
    reachability_map = _load_reachability_map(str(map_path.resolve()))
    if reachability_map.metadata.get("robot_uid") != handles.spec.uid:
        raise ValueError(f"Reachability map {map_path} belongs to another robot")
    midpoints, normals = _boundary_pairs(
        reachability_map, map_path, float(grasp_endpoint[2]),
    )
    root_seed = 0 if seed == -1 else int(seed)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
        root_seed, zlib.crc32(b"task boundary unreachable placement"),
    ))))
    order = rng.permutation(len(midpoints))
    for boundary_index in order[:min(len(order), 512)]:
        vertical_gap = abs(float(
            midpoints[boundary_index, 2] - grasp_endpoint[2]
        ))
        horizontal_limit = float(np.sqrt(max(
            boundary_half_width_m**2 - vertical_gap**2, 0.0,
        )))
        signed_offset = float(rng.uniform(
            -horizontal_limit, horizontal_limit,
        ))
        target_xy = (
            midpoints[boundary_index, :2]
            + signed_offset * normals[boundary_index]
        )
        candidate = original_position.copy()
        candidate[:2] += target_xy - grasp_endpoint[:2]
        scene_manager.set_layout_pose(
            data, source_name, candidate, home_yaw,
        )
        mujoco.mj_forward(model, data)
        if not _inside_table(model, data, body_id):
            continue
        actual = _stage_endpoints(
            model, data, handles, task_name, scene_manager.current_name,
        )["grasp"]
        boundary_distance = float(np.linalg.norm(
            actual - midpoints[boundary_index],
        ))
        if boundary_distance > boundary_half_width_m + 1e-6:
            continue
        return {
            "mode": "boundary_unreachable",
            "task": task_name,
            "source_body": source_name,
            "map_path": str(map_path),
            "boundary_midpoint_m": midpoints[boundary_index].tolist(),
            "signed_boundary_offset_m": signed_offset,
            "boundary_distance_m": boundary_distance,
            "actual_grasp_endpoint_m": actual.tolist(),
        }
    scene_manager.set_layout_pose(
        data, source_name, original_position, source.quaternion,
    )
    mujoco.mj_forward(model, data)
    raise RuntimeError(
        "No reachability-boundary placement fit completely on the fixed table"
    )
