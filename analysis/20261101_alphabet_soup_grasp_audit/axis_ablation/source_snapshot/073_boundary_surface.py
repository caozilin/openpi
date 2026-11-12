"""Dense fixed-Home-rotation IK boundary surfaces for tabletop placement."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


BOUNDARY_GRID_SPACING_M = 0.002
BOUNDARY_FAR_DEPTH_M = 0.50
BOUNDARY_TABLE_Y_FRACTION = 0.90
BOUNDARY_FACTOR_MIN = 0.0
BOUNDARY_FACTOR_MAX = 2.0
BOUNDARY_FACTOR_DISTANCE_M = 0.50
BOUNDARY_SURFACE_VERSION = 2


def validate_distance_factor_range(lower: float, upper: float) -> tuple[float, float]:
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Workspace-boundary distance factors must be finite")
    if not BOUNDARY_FACTOR_MIN <= lower <= upper <= BOUNDARY_FACTOR_MAX:
        raise ValueError("Workspace-boundary distance factors must satisfy 0 <= lower <= upper <= 2")
    return lower, upper


@dataclass(frozen=True)
class BoundaryOffsetSample:
    base_point_m: np.ndarray
    outward_normal_xy: np.ndarray
    requested_factor: float
    requested_offset_m: float
    requested_point_m: np.ndarray
    clipped_point_m: np.ndarray
    effective_offset_m: float
    effective_factor: float
    saturated: bool


@dataclass(frozen=True)
class WorkspaceBoundarySurface:
    """One robot's boundary ``x = f(y, z)`` at 2 mm horizontal/Z spacing."""

    y_m: np.ndarray
    z_m: np.ndarray
    x_boundary_m: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        y = np.asarray(self.y_m, dtype=float)
        z = np.asarray(self.z_m, dtype=float)
        x = np.asarray(self.x_boundary_m, dtype=float)
        if y.ndim != 1 or z.ndim != 1 or y.size < 2 or z.size < 1:
            raise ValueError("Boundary surface requires non-empty 1-D Y/Z axes")
        if x.shape != (z.size, y.size):
            raise ValueError("Boundary X must have shape [len(z), len(y)]")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
            raise ValueError("Boundary axes must be finite")
        if np.any(np.diff(y) <= 0.0) or np.any(np.diff(z) <= 0.0):
            raise ValueError("Boundary axes must be strictly increasing")
        if not np.any(np.isfinite(x)):
            raise ValueError("Boundary surface contains no finite curve points")
        object.__setattr__(self, "y_m", y.copy())
        object.__setattr__(self, "z_m", z.copy())
        object.__setattr__(self, "x_boundary_m", x.copy())

    @property
    def robot_uid(self) -> str:
        return str(self.metadata["robot_uid"])

    @property
    def search_bounds_xy_m(self) -> np.ndarray:
        bounds = np.asarray(self.metadata["search_bounds_xy_m"], dtype=float)
        if bounds.shape != (2, 2):
            raise ValueError("Boundary metadata search bounds must be [[xmin,xmax],[ymin,ymax]]")
        return bounds

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            y_m=self.y_m,
            z_m=self.z_m,
            x_boundary_m=self.x_boundary_m,
            metadata_json=np.array(json.dumps(self.metadata, sort_keys=True)),
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> "WorkspaceBoundarySurface":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            surface = cls(
                y_m=archive["y_m"].copy(),
                z_m=archive["z_m"].copy(),
                x_boundary_m=archive["x_boundary_m"].copy(),
                metadata=metadata,
            )
        if int(surface.metadata.get("format_version", -1)) != BOUNDARY_SURFACE_VERSION:
            raise ValueError(f"Unsupported workspace-boundary surface: {path}")
        return surface

    def _z_curve(self, z_m: float) -> np.ndarray:
        z_value = float(np.clip(z_m, self.z_m[0], self.z_m[-1]))
        upper = int(np.searchsorted(self.z_m, z_value, side="right"))
        if upper <= 0:
            return self.x_boundary_m[0].copy()
        if upper >= self.z_m.size:
            return self.x_boundary_m[-1].copy()
        lower = upper - 1
        fraction = (z_value - self.z_m[lower]) / (self.z_m[upper] - self.z_m[lower])
        first = self.x_boundary_m[lower]
        second = self.x_boundary_m[upper]
        result = (1.0 - fraction) * first + fraction * second
        only_first = np.isfinite(first) & ~np.isfinite(second)
        only_second = ~np.isfinite(first) & np.isfinite(second)
        result[only_first] = first[only_first]
        result[only_second] = second[only_second]
        return result

    def curve_at_z(self, z_m: float) -> np.ndarray:
        """Return finite ordered XYZ samples at the nearest/interpolated height."""
        curve = self._z_curve(z_m)
        valid = np.isfinite(curve)
        return np.column_stack((curve[valid], self.y_m[valid], np.full(np.sum(valid), float(z_m))))

    def boundary_x(self, y_m: float, z_m: float) -> float:
        curve = self._z_curve(z_m)
        valid = np.flatnonzero(np.isfinite(curve))
        if valid.size == 0:
            raise ValueError(f"Boundary surface has no curve at z={z_m:.4f} m")
        y_value = float(np.clip(y_m, self.y_m[valid[0]], self.y_m[valid[-1]]))
        return float(np.interp(y_value, self.y_m[valid], curve[valid]))

    def outward_normal(self, y_m: float, z_m: float) -> np.ndarray:
        curve = self._z_curve(z_m)
        valid = np.flatnonzero(np.isfinite(curve))
        if valid.size < 2:
            return np.array((1.0, 0.0))
        x = curve[valid]
        y = self.y_m[valid]
        derivative = np.gradient(x, y)
        slope = float(np.interp(float(y_m), y, derivative))
        normal = np.array((1.0, -slope), dtype=float)
        return normal / max(float(np.linalg.norm(normal)), 1e-12)

    def offset_sample(
        self,
        *,
        y_m: float,
        z_m: float,
        factor: float,
        legal_center_bounds_xy_m: np.ndarray,
    ) -> BoundaryOffsetSample:
        validate_distance_factor_range(factor, factor)
        bounds = np.asarray(legal_center_bounds_xy_m, dtype=float)
        if bounds.shape != (2, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Legal object-centre bounds must be [[xmin,xmax],[ymin,ymax]]")
        base = np.array((self.boundary_x(y_m, z_m), float(y_m), float(z_m)))
        normal = self.outward_normal(y_m, z_m)
        requested_offset = (float(factor) - 1.0) * BOUNDARY_FACTOR_DISTANCE_M
        requested = base.copy()
        requested[:2] += requested_offset * normal
        clipped = requested.copy()
        clipped[0] = np.clip(clipped[0], bounds[0, 0], bounds[0, 1])
        clipped[1] = np.clip(clipped[1], bounds[1, 0], bounds[1, 1])
        effective_offset = float((clipped[:2] - base[:2]) @ normal)
        effective_factor = float(np.clip(
            1.0 + effective_offset / BOUNDARY_FACTOR_DISTANCE_M,
            BOUNDARY_FACTOR_MIN,
            BOUNDARY_FACTOR_MAX,
        ))
        saturated = not np.allclose(clipped[:2], requested[:2], atol=1e-10, rtol=0.0)
        return BoundaryOffsetSample(
            base_point_m=base,
            outward_normal_xy=normal,
            requested_factor=float(factor),
            requested_offset_m=requested_offset,
            requested_point_m=requested,
            clipped_point_m=clipped,
            effective_offset_m=effective_offset,
            effective_factor=effective_factor,
            saturated=saturated,
        )


__all__ = [
    "BOUNDARY_FACTOR_DISTANCE_M",
    "BOUNDARY_FACTOR_MAX",
    "BOUNDARY_FACTOR_MIN",
    "BOUNDARY_FAR_DEPTH_M",
    "BOUNDARY_GRID_SPACING_M",
    "BOUNDARY_SURFACE_VERSION",
    "BOUNDARY_TABLE_Y_FRACTION",
    "BoundaryOffsetSample",
    "WorkspaceBoundarySurface",
    "validate_distance_factor_range",
]
