"""Creation-mode visualization of the seeded object placement envelope."""

from __future__ import annotations

import mujoco
import numpy as np

from ..simulation.randomization.objects import ObjectPlacementRegion


def placement_region_points(
    region: ObjectPlacementRegion,
    *,
    samples_per_axis: int = 31,
    height_offset_m: float = 0.004,
) -> np.ndarray:
    """Sample visible points in the table-margin/reach-annulus intersection."""
    if samples_per_axis < 2:
        raise ValueError("Placement-region visualization needs at least 2 samples")
    bounds = np.asarray(region.table_bounds_xy, dtype=float)
    margin = float(region.table_edge_margin_m)
    x = np.linspace(bounds[0, 0] + margin, bounds[0, 1] - margin, samples_per_axis)
    y = np.linspace(bounds[1, 0] + margin, bounds[1, 1] - margin, samples_per_axis)
    xy = np.stack(np.meshgrid(x, y, indexing="xy"), axis=-1).reshape(-1, 2)
    radius = np.linalg.norm(xy, axis=1)
    inner, outer = region.planar_reach_m
    xy = xy[(radius >= inner) & (radius <= outer)]
    return np.column_stack((
        xy,
        np.full(len(xy), region.table_height_m + height_offset_m),
    ))


def append_placement_region(
    scene: mujoco.MjvScene,
    region: ObjectPlacementRegion,
    *,
    maximum_points: int = 700,
) -> int:
    """Append translucent reference dots without replacing rendered geoms."""
    points = placement_region_points(region)
    capacity = max(0, int(scene.maxgeom) - int(scene.ngeom))
    count = min(len(points), maximum_points, capacity)
    if count <= 0:
        return 0
    if len(points) > count:
        indices = np.linspace(0, len(points) - 1, count).astype(int)
        points = points[indices]
    start = int(scene.ngeom)
    identity = np.eye(3).reshape(-1)
    size = np.array((0.0035, 0.0, 0.0))
    color = np.array((0.05, 0.85, 0.72, 0.34), dtype=np.float32)
    for offset, point in enumerate(points):
        mujoco.mjv_initGeom(
            scene.geoms[start + offset],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            point,
            identity,
            color,
        )
        scene.geoms[start + offset].category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom = start + len(points)
    return len(points)
