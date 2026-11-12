"""Geometry predicates independent of a controller or trajectory planner."""

from __future__ import annotations

import mujoco
import numpy as np


def box_world_corners(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    """Return the eight world-space corners of a box geometry."""
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("box_world_corners requires a box geometry")
    half = model.geom_size[geom_id, :3]
    local = np.array([
        (x, y, z)
        for x in (-half[0], half[0])
        for y in (-half[1], half[1])
        for z in (-half[2], half[2])
    ])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return data.geom_xpos[geom_id] + local @ rotation.T


def contained_by_circle(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    radius_m: float,
    edge_margin_m: float = 0.0,
) -> tuple[bool, float]:
    """Test whether every supplied point lies inside a circular usable area."""
    radii = np.linalg.norm(
        np.asarray(points_xy, dtype=float) - np.asarray(center_xy, dtype=float),
        axis=1,
    )
    maximum_radius = float(np.max(radii))
    return maximum_radius <= float(radius_m) - float(edge_margin_m), maximum_radius

