"""MuJoCo user-scene point-cloud rendering for reachability maps."""

from __future__ import annotations

import mujoco
import numpy as np

from .map import ReachabilityMap
from .boundary_surface import (
    BOUNDARY_FACTOR_DISTANCE_M,
    WorkspaceBoundarySurface,
    validate_distance_factor_range,
)


def reachability_color(
    badness: float,
    reachable: bool,
    finite_scale: float,
) -> np.ndarray:
    if not reachable:
        return np.array((0.72, 0.12, 0.90, 0.72), dtype=np.float32)
    normalized = (
        1.0
        if not np.isfinite(badness)
        else float(np.clip(badness / max(finite_scale, 1e-9), 0.0, 1.0))
    )
    if normalized <= 0.5:
        alpha = 2.0 * normalized
        return np.array((
            0.10 + 0.85 * alpha,
            0.78 - 0.20 * alpha,
            0.18,
            0.68,
        ), dtype=np.float32)
    alpha = 2.0 * (normalized - 0.5)
    return np.array((
        0.95,
        0.58 * (1.0 - alpha) + 0.10 * alpha,
        0.18 * (1.0 - alpha) + 0.10 * alpha,
        0.72,
    ), dtype=np.float32)


def spatial_map_indices(
    reachability_map: ReachabilityMap,
    maximum_points: int,
) -> np.ndarray:
    if maximum_points <= 0:
        raise ValueError("maximum_points must be positive")
    if reachability_map.count <= maximum_points:
        return np.arange(reachability_map.count)
    # The map itself is adaptive, so sampling by score would heavily favour
    # dense danger cells and visually erase safe parts of the requested box.
    # Select at most one point per regular spatial bin instead.
    bins_per_axis = max(2, int(np.floor(maximum_points ** (1.0 / 3.0))))
    lower = np.asarray(
        reachability_map.metadata.get(
            "lower_m", reachability_map.positions.min(axis=0),
        ),
        dtype=float,
    )
    upper = np.asarray(
        reachability_map.metadata.get(
            "upper_m", reachability_map.positions.max(axis=0),
        ),
        dtype=float,
    )
    span = np.maximum(upper - lower, 1e-9)
    normalized = np.clip(
        (reachability_map.positions - lower) / span,
        0.0,
        np.nextafter(1.0, 0.0),
    )
    bins = np.floor(normalized * bins_per_axis).astype(int)
    linear = np.ravel_multi_index(
        (bins[:, 0], bins[:, 1], bins[:, 2]),
        (bins_per_axis, bins_per_axis, bins_per_axis),
    )
    bin_centers = lower + (
        bins.astype(float) + 0.5
    ) / bins_per_axis * span
    # Make the outer bin representatives lie on the requested faces. This
    # keeps the displayed cloud itself, not only the wireframe, visibly at the
    # exact +/-20 cm XY and -10/+30 cm Z limits.
    for axis in range(3):
        bin_centers[bins[:, axis] == 0, axis] = lower[axis]
        bin_centers[bins[:, axis] == bins_per_axis - 1, axis] = upper[axis]
    distance = np.linalg.norm(
        (reachability_map.positions - bin_centers) / span,
        axis=1,
    )
    ordered = np.lexsort((distance, linear))
    first_in_bin = np.concatenate((
        np.array((True,)),
        linear[ordered][1:] != linear[ordered][:-1],
    ))
    return ordered[first_in_bin][:maximum_points]


def _map_bounds(
    reachability_map: ReachabilityMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(
        reachability_map.metadata.get(
            "lower_m", reachability_map.positions.min(axis=0),
        ),
        dtype=float,
    )
    upper = np.asarray(
        reachability_map.metadata.get(
            "upper_m", reachability_map.positions.max(axis=0),
        ),
        dtype=float,
    )
    center = np.asarray(
        reachability_map.metadata.get(
            "center_m", 0.5 * (lower + upper),
        ),
        dtype=float,
    )
    return lower, upper, center


def _add_line(
    scene: mujoco.MjvScene,
    index: int,
    start: np.ndarray,
    end: np.ndarray,
    rgba: np.ndarray,
    width: float = 0.002,
) -> None:
    mujoco.mjv_initGeom(
        scene.geoms[index],
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), rgba,
    )
    mujoco.mjv_connector(
        scene.geoms[index], mujoco.mjtGeom.mjGEOM_LINE,
        width, start, end,
    )
    scene.geoms[index].category = mujoco.mjtCatBit.mjCAT_DECOR


def populate_reachability_scene(
    scene: mujoco.MjvScene,
    reachability_map: ReachabilityMap,
    *,
    maximum_points: int = 3000,
    point_size_m: float = 0.0035,
) -> int:
    """Replace user geoms with a score-coloured reachability point cloud."""
    if point_size_m <= 0.0:
        raise ValueError("point_size_m must be positive")
    decoration_count = 16  # 12 box edges, centre marker, and XYZ axes.
    if int(scene.maxgeom) <= decoration_count:
        raise ValueError("MuJoCo user scene has no capacity for map decorations")
    point_capacity = min(
        maximum_points,
        max(1, int(scene.maxgeom) - decoration_count),
    )
    indices = spatial_map_indices(reachability_map, point_capacity)
    finite = reachability_map.badness[np.isfinite(reachability_map.badness)]
    finite_scale = float(np.quantile(finite, 0.95)) if finite.size else 1.0
    identity = np.eye(3).reshape(-1)
    sphere_size = np.array((point_size_m, 0.0, 0.0))
    for output_index, map_index in enumerate(indices):
        mujoco.mjv_initGeom(
            scene.geoms[output_index],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            sphere_size,
            reachability_map.positions[map_index],
            identity,
            reachability_color(
                float(reachability_map.badness[map_index]),
                bool(reachability_map.reachable[map_index]),
                finite_scale,
            ),
        )
        scene.geoms[output_index].category = mujoco.mjtCatBit.mjCAT_DECOR
    output_index = len(indices)
    lower, upper, center = _map_bounds(reachability_map)
    corners = np.array([
        (x, y, z)
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ])
    box_color = np.array((0.15, 0.85, 1.0, 0.95), dtype=np.float32)
    for first in range(8):
        for axis in range(3):
            second = first ^ (1 << (2 - axis))
            if first >= second:
                continue
            _add_line(
                scene, output_index, corners[first], corners[second], box_color,
            )
            output_index += 1
    mujoco.mjv_initGeom(
        scene.geoms[output_index], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array((0.009, 0.0, 0.0)), center,
        identity, np.array((1.0, 0.70, 0.05, 1.0), dtype=np.float32),
    )
    scene.geoms[output_index].category = mujoco.mjtCatBit.mjCAT_DECOR
    output_index += 1
    axis_colors = (
        np.array((1.0, 0.18, 0.12, 1.0), dtype=np.float32),
        np.array((0.18, 1.0, 0.20, 1.0), dtype=np.float32),
        np.array((0.18, 0.45, 1.0, 1.0), dtype=np.float32),
    )
    for axis, color in enumerate(axis_colors):
        end = center.copy()
        end[axis] += 0.08
        _add_line(scene, output_index, center, end, color, width=0.004)
        output_index += 1
    scene.ngeom = output_index
    return len(indices)


def clear_reachability_scene(scene: mujoco.MjvScene) -> None:
    scene.ngeom = 0


def _offset_curve_xy(
    x: np.ndarray,
    y: np.ndarray,
    factor: float,
    bounds: np.ndarray,
) -> np.ndarray:
    derivative = np.gradient(x, y)
    normals = np.column_stack((np.ones_like(x), -derivative))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    result = np.column_stack((x, y))
    result += (float(factor) - 1.0) * BOUNDARY_FACTOR_DISTANCE_M * normals
    result[:, 0] = np.clip(result[:, 0], bounds[0, 0], bounds[0, 1])
    result[:, 1] = np.clip(result[:, 1], bounds[1, 0], bounds[1, 1])
    return result


def populate_workspace_boundary_surface_scene(
    scene: mujoco.MjvScene,
    surface: WorkspaceBoundarySurface,
    *,
    factor_lower: float = 1.0,
    factor_upper: float = 1.0,
    maximum_lines: int = 3000,
    append: bool = False,
) -> int:
    """Render Z-stacked boundary arcs and selected signed-offset limits.

    ``append=True`` preserves an existing rendered model/user scene, which is
    required by the main UI's offscreen renderer. The default retains the
    historical replace-user-scene behavior.
    """
    lower_factor, upper_factor = validate_distance_factor_range(
        factor_lower, factor_upper,
    )
    start_index = int(scene.ngeom) if append else 0
    remaining_capacity = int(scene.maxgeom) - start_index
    if maximum_lines <= 24 or remaining_capacity <= 24:
        raise ValueError("MuJoCo user scene has insufficient boundary-line capacity")
    capacity = min(remaining_capacity, maximum_lines)
    factor_specs = [(1.0, np.array((0.10, 0.82, 1.0, 0.95), dtype=np.float32))]
    if abs(lower_factor - 1.0) > 1e-12:
        factor_specs.append((
            lower_factor,
            np.array((0.20, 0.92, 0.35, 0.72), dtype=np.float32),
        ))
    if (
        abs(upper_factor - 1.0) > 1e-12
        and abs(upper_factor - lower_factor) > 1e-12
    ):
        factor_specs.append((
            upper_factor,
            np.array((1.0, 0.48, 0.12, 0.72), dtype=np.float32),
        ))
    decoration_lines = 12
    available = max(1, capacity - decoration_lines)
    z_stride = max(1, int(np.ceil(
        len(surface.z_m) * max(1, len(surface.y_m) - 1)
        * len(factor_specs) / available,
    )))
    selected_z = np.arange(0, len(surface.z_m), z_stride, dtype=int)
    if selected_z[-1] != len(surface.z_m) - 1:
        selected_z = np.append(selected_z, len(surface.z_m) - 1)
    y_stride = max(1, int(np.ceil(
        len(selected_z) * max(1, len(surface.y_m) - 1)
        * len(factor_specs) / available,
    )))
    selected_y = np.arange(0, len(surface.y_m), y_stride, dtype=int)
    if selected_y[-1] != len(surface.y_m) - 1:
        selected_y = np.append(selected_y, len(surface.y_m) - 1)

    bounds = surface.search_bounds_xy_m
    output_index = 0
    for factor, color in factor_specs:
        for iz in selected_z:
            x_full = surface.x_boundary_m[int(iz)]
            valid = np.isfinite(x_full)
            if np.sum(valid) < 2:
                continue
            xy_full = _offset_curve_xy(
                x_full[valid], surface.y_m[valid], factor, bounds,
            )
            xyz_full = np.column_stack((
                xy_full, np.full(len(xy_full), surface.z_m[int(iz)]),
            ))
            local_indices = np.flatnonzero(valid)
            keep = np.isin(local_indices, selected_y)
            if np.sum(keep) < 2:
                keep[[0, -1]] = True
            xyz = xyz_full[keep]
            for first, second in zip(xyz[:-1], xyz[1:]):
                if output_index >= capacity - decoration_lines:
                    break
                _add_line(
                    scene, start_index + output_index,
                    first, second, color, width=0.0022,
                )
                output_index += 1

    lower_z = float(surface.z_m[0])
    upper_z = float(surface.z_m[-1])
    corners = np.array([
        (x, y, z)
        for x in bounds[0]
        for y in bounds[1]
        for z in (lower_z, upper_z)
    ])
    box_color = np.array((0.88, 0.88, 0.92, 0.72), dtype=np.float32)
    for first in range(8):
        for axis in range(3):
            second = first ^ (1 << (2 - axis))
            if first >= second or output_index >= capacity:
                continue
            _add_line(
                scene, start_index + output_index,
                corners[first], corners[second],
                box_color, width=0.0015,
            )
            output_index += 1
    scene.ngeom = start_index + output_index
    return output_index
