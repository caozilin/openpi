"""Build dense tabletop reachability boundary surfaces from fixed-pose IK."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import math
from typing import Callable

import mujoco
import numpy as np

from ..data.task_sets import GEOMETRIC_BASIS_SUBCATEGORY, training_tasks
from ..simulation.loader import load_scene_model
from ..simulation.randomization.objects import ObjectLayoutRandomizer
from ..simulation.randomization.table import TableRandomizer
from ..simulation.robot import ee_pose, find_handles, set_home
from ..simulation.robots import get_robot_spec
from ..simulation.scenes import SceneManager
from ..tasks.planning.presets import PresetTaskExecutor
from ..tasks.subtasks import identify_pick_place_subtasks
from .boundary_surface import (
    BOUNDARY_FAR_DEPTH_M,
    BOUNDARY_GRID_SPACING_M,
    BOUNDARY_SURFACE_VERSION,
    BOUNDARY_TABLE_Y_FRACTION,
    WorkspaceBoundarySurface,
)
from .metric import BadnessSettings, ReachabilityProbe


def _inclusive_axis(lower: float, upper: float, spacing: float) -> np.ndarray:
    start = math.ceil((lower - 1e-12) / spacing) * spacing
    stop = math.floor((upper + 1e-12) / spacing) * spacing
    if stop < start:
        raise ValueError("Requested boundary axis contains no grid point")
    count = int(round((stop - start) / spacing))
    return start + spacing * np.arange(count + 1, dtype=float)


def tabletop_search_bounds(
    robot_uid: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return XY rectangle, table height and full tabletop XY bounds."""
    # The IK probe uses the empty robot model, but that deliberately omits the
    # work surface.  Read the compiled robot-specific table from a lightweight
    # reference task scene instead.
    model = load_scene_model("Blocks", robot_uid)
    data = mujoco.MjData(model)
    handles = find_handles(model, get_robot_spec(robot_uid))
    set_home(model, data, handles)
    table_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
    )
    if table_body < 0:
        raise RuntimeError("Reference Blocks scene has no work_surface")
    table_geoms = np.flatnonzero(model.geom_bodyid == table_body)
    table_geom = int(table_geoms[
        np.argmax(np.prod(model.geom_size[table_geoms, :2], axis=1))
    ])
    center = data.geom_xpos[table_geom]
    rotation = data.geom_xmat[table_geom].reshape(3, 3)
    half_xy = np.abs(rotation[:2]) @ model.geom_size[table_geom]
    tabletop_bounds = np.array((
        (center[0] - half_xy[0], center[0] + half_xy[0]),
        (center[1] - half_xy[1], center[1] + half_xy[1]),
    ))
    table_height = float(
        center[2] + np.abs(rotation[2]) @ model.geom_size[table_geom]
    )
    far_x = float(tabletop_bounds[0, 1])
    half_y = 0.5 * float(tabletop_bounds[1, 1] - tabletop_bounds[1, 0])
    center_y = 0.5 * float(tabletop_bounds[1, 1] + tabletop_bounds[1, 0])
    search_bounds = np.array((
        (max(float(tabletop_bounds[0, 0]), far_x - BOUNDARY_FAR_DEPTH_M), far_x),
        (
            center_y - BOUNDARY_TABLE_Y_FRACTION * half_y,
            center_y + BOUNDARY_TABLE_Y_FRACTION * half_y,
        ),
    ))
    return search_bounds, table_height, tabletop_bounds


def geometry_grasp_height_range(
    robot_uid: str,
) -> tuple[float, float, list[dict[str, object]]]:
    """Scan all 15 Geometric Basis task variants for nominal Pre-grasp Z."""
    records: list[dict[str, object]] = []
    for scene_spec, task_spec in training_tasks():
        if scene_spec.subcategory != GEOMETRIC_BASIS_SUBCATEGORY:
            continue
        scene_name = scene_spec.runtime_scene
        model = load_scene_model(scene_name, robot_uid)
        handles = find_handles(model, get_robot_spec(robot_uid))
        manager = SceneManager(model, loaded_scene=scene_name)
        variants = SceneManager.variant_indices_for_bodies(scene_name, ())
        for variant_index in variants:
            data = mujoco.MjData(model)
            set_home(model, data, handles)
            manager.select_variant_index(variant_index)
            manager.activate(data, scene_name)
            ObjectLayoutRandomizer(model, manager).apply(
                data, -1, variant_index=variant_index,
            )
            position, rotation = ee_pose(data, handles)
            executor = PresetTaskExecutor(model, handles)
            executor.start(
                task_spec.runtime_task, scene_name, data, position, rotation,
            )
            spans = identify_pick_place_subtasks(executor.frames)
            if not spans:
                raise RuntimeError(f"No Pre-grasp endpoint for {scene_name}")
            endpoint = executor.frames[spans[0].pregrasp_index].position
            records.append({
                "scene": scene_name,
                "task": task_spec.runtime_task,
                "variant_index": variant_index,
                "pregrasp_z_m": float(endpoint[2]),
            })
    if not records:
        raise RuntimeError("No geometry task variants provide nominal grasp heights")
    heights = np.asarray([record["pregrasp_z_m"] for record in records], dtype=float)
    return float(np.min(heights)), float(np.max(heights)), records


def build_workspace_boundary_surface(
    *,
    robot_uid: str,
    spacing_m: float = BOUNDARY_GRID_SPACING_M,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
    settings: BadnessSettings = BadnessSettings(),
    progress: Callable[[int, int, int], None] | None = None,
) -> WorkspaceBoundarySurface:
    """Search the farthest Home-seeded IK point for every 2 mm Y/Z cell."""
    if spacing_m <= 0.0:
        raise ValueError("Boundary spacing must be positive")
    spec = get_robot_spec(robot_uid)
    search_bounds, table_height, tabletop_bounds = tabletop_search_bounds(robot_uid)
    measured_min, measured_max, height_records = geometry_grasp_height_range(robot_uid)
    table_z_padding = float(TableRandomizer.MAX_OFFSET_M[2])
    requested_z_min = (
        measured_min - table_z_padding
        if z_min_m is None else float(z_min_m)
    )
    requested_z_max = (
        measured_max + table_z_padding
        if z_max_m is None else float(z_max_m)
    )
    if requested_z_min > requested_z_max:
        raise ValueError("Boundary Z minimum must not exceed maximum")
    y_axis = _inclusive_axis(search_bounds[1, 0], search_bounds[1, 1], spacing_m)
    z_axis = _inclusive_axis(requested_z_min, requested_z_max, spacing_m)
    x_axis = _inclusive_axis(search_bounds[0, 0], search_bounds[0, 1], spacing_m)

    model = load_scene_model(SceneManager.EMPTY_SCENE, robot_uid)
    data = mujoco.MjData(model)
    handles = find_handles(model, spec)
    set_home(model, data, handles)
    _, home_rotation = ee_pose(data, handles)
    probe = ReachabilityProbe(
        model, handles, home_rotation,
        replace(settings, restart_count=1),
    )
    cache: dict[tuple[int, int, int], bool] = {}
    evaluations = 0

    def reachable(ix: int, iy: int, iz: int) -> bool:
        nonlocal evaluations
        key = (ix, iy, iz)
        if key not in cache:
            target = np.array((x_axis[ix], y_axis[iy], z_axis[iz]))
            cache[key] = bool(probe.probe(target, use_restarts=False).reachable)
            evaluations += 1
        return cache[key]

    def boundary_index(iy: int, iz: int, start: int) -> int:
        index = int(np.clip(start, 0, len(x_axis) - 1))
        if reachable(index, iy, iz):
            while index + 1 < len(x_axis) and reachable(index + 1, iy, iz):
                index += 1
            return index
        while index > 0 and not reachable(index, iy, iz):
            index -= 1
        return index if reachable(index, iy, iz) else -1

    boundary = np.full((len(z_axis), len(y_axis)), np.nan, dtype=float)
    center_y_index = int(np.argmin(np.abs(y_axis)))
    previous_center_x = 0
    for iz in range(len(z_axis)):
        center_x = boundary_index(center_y_index, iz, previous_center_x)
        if center_x >= 0:
            boundary[iz, center_y_index] = x_axis[center_x]
            previous_center_x = center_x
        for direction in (-1, 1):
            previous_x = center_x
            iy = center_y_index + direction
            while 0 <= iy < len(y_axis):
                start_x = max(previous_x, 0)
                current_x = boundary_index(iy, iz, start_x)
                if current_x >= 0:
                    boundary[iz, iy] = x_axis[current_x]
                    previous_x = current_x
                else:
                    previous_x = 0
                iy += direction
        if progress is not None:
            progress(iz + 1, len(z_axis), evaluations)

    metadata = {
        "format_version": BOUNDARY_SURFACE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot_uid": robot_uid,
        "robot_label": spec.label,
        "reference_scene": SceneManager.EMPTY_SCENE,
        "ik_seed": "robot_home_q_only",
        "home_q": spec.home_q.tolist(),
        "target_rotation_matrix": home_rotation.tolist(),
        "reachability_definition": "pose IK residual within configured position/rotation tolerances",
        "path_collision_dynamics_checked": False,
        "grid_spacing_m": spacing_m,
        "search_bounds_xy_m": search_bounds.tolist(),
        "tabletop_bounds_xy_m": tabletop_bounds.tolist(),
        "tabletop_height_m": table_height,
        "far_edge_depth_m": BOUNDARY_FAR_DEPTH_M,
        "table_y_fraction": BOUNDARY_TABLE_Y_FRACTION,
        "z_requested_range_m": [requested_z_min, requested_z_max],
        "z_measured_geometry_pregrasp_range_m": [measured_min, measured_max],
        "table_domain_randomization_z_padding_m": table_z_padding,
        "geometry_pregrasp_heights": height_records,
        "position_tolerance_m": settings.position_tolerance_m,
        "rotation_tolerance_rad": settings.rotation_tolerance_rad,
        "maximum_iterations": settings.maximum_iterations,
        "ik_evaluations": evaluations,
        "finite_boundary_points": int(np.sum(np.isfinite(boundary))),
    }
    return WorkspaceBoundarySurface(y_axis, z_axis, boundary, metadata)


def extend_workspace_boundary_surface_for_table_randomization(
    existing: WorkspaceBoundarySurface,
    *,
    padding_m: float | None = None,
    settings: BadnessSettings = BadnessSettings(),
    progress: Callable[[int, int, int], None] | None = None,
) -> WorkspaceBoundarySurface:
    """Add only missing Z slices needed by vertical table randomization."""
    padding = (
        float(TableRandomizer.MAX_OFFSET_M[2])
        if padding_m is None else float(padding_m)
    )
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("Table-height padding must be a finite non-negative value")
    spacing = float(existing.metadata.get("grid_spacing_m", 0.0))
    if spacing <= 0.0:
        raise ValueError("Existing boundary surface has no valid grid spacing")
    measured_range = np.asarray(
        existing.metadata.get("z_measured_geometry_pregrasp_range_m"),
        dtype=float,
    )
    if measured_range.shape != (2,) or not np.all(np.isfinite(measured_range)):
        raise ValueError("Existing boundary surface has no measured grasp-height range")

    requested_min = float(measured_range[0] - padding)
    requested_max = float(measured_range[1] + padding)
    target_z = _inclusive_axis(requested_min, requested_max, spacing)
    tolerance = spacing * 1e-6
    lower_z = target_z[target_z < existing.z_m[0] - tolerance]
    upper_z = target_z[target_z > existing.z_m[-1] + tolerance]
    total_new_slices = int(lower_z.size + upper_z.size)

    new_parts: list[WorkspaceBoundarySurface] = []
    completed_before = 0
    evaluations_before = 0
    for missing_z in (lower_z, upper_z):
        if missing_z.size == 0:
            continue

        def part_progress(
            completed: int,
            _total: int,
            evaluations: int,
            *,
            completed_offset: int = completed_before,
            evaluation_offset: int = evaluations_before,
        ) -> None:
            if progress is not None:
                progress(
                    completed_offset + completed,
                    total_new_slices,
                    evaluation_offset + evaluations,
                )

        part = build_workspace_boundary_surface(
            robot_uid=existing.robot_uid,
            spacing_m=spacing,
            z_min_m=float(missing_z[0]),
            z_max_m=float(missing_z[-1]),
            settings=settings,
            progress=part_progress,
        )
        if not np.allclose(part.y_m, existing.y_m, atol=tolerance, rtol=0.0):
            raise ValueError("Incremental surface Y grid differs from the existing asset")
        if not np.allclose(part.z_m, missing_z, atol=tolerance, rtol=0.0):
            raise ValueError("Incremental surface Z grid differs from the requested slices")
        new_parts.append(part)
        completed_before += int(missing_z.size)
        evaluations_before += int(part.metadata["ik_evaluations"])

    lower_part = new_parts[0] if lower_z.size else None
    upper_part = new_parts[-1] if upper_z.size else None
    z_parts = [
        *(part.z_m for part in (lower_part,) if part is not None),
        existing.z_m,
        *(part.z_m for part in (upper_part,) if part is not None),
    ]
    boundary_parts = [
        *(part.x_boundary_m for part in (lower_part,) if part is not None),
        existing.x_boundary_m,
        *(part.x_boundary_m for part in (upper_part,) if part is not None),
    ]
    combined_z = np.concatenate(z_parts)
    combined_boundary = np.concatenate(boundary_parts, axis=0)
    if np.any(np.abs(np.diff(combined_z) - spacing) > tolerance):
        raise ValueError("Incremental surface would produce a discontinuous Z grid")

    metadata = dict(existing.metadata)
    previous_range = [float(existing.z_m[0]), float(existing.z_m[-1])]
    metadata.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "z_requested_range_m": [requested_min, requested_max],
        "z_axis_range_m": [float(combined_z[0]), float(combined_z[-1])],
        "table_domain_randomization_z_padding_m": padding,
        "incremental_extension_previous_z_axis_range_m": previous_range,
        "incremental_extension_added_z_slices": total_new_slices,
        "incremental_extension_ik_evaluations": evaluations_before,
        "ik_evaluations": (
            int(existing.metadata.get("ik_evaluations", 0)) + evaluations_before
        ),
        "finite_boundary_points": int(np.sum(np.isfinite(combined_boundary))),
    })
    return WorkspaceBoundarySurface(
        existing.y_m,
        combined_z,
        combined_boundary,
        metadata,
    )


__all__ = [
    "build_workspace_boundary_surface",
    "extend_workspace_boundary_surface_for_table_randomization",
    "geometry_grasp_height_range",
    "tabletop_search_bounds",
]
