"""Offline fixed-orientation reachability maps and difficult-layout sampling."""

from .map import ReachabilityMap, ReachabilitySample
from .boundary_surface import WorkspaceBoundarySurface
from .metric import BadnessSettings, ReachabilityProbe, ReachabilityProbeResult
from .visualization import (
    populate_reachability_scene,
    populate_workspace_boundary_surface_scene,
)

__all__ = [
    "BadnessSettings",
    "WorkspaceBoundarySurface",
    "ReachabilityMap",
    "ReachabilityProbe",
    "ReachabilityProbeResult",
    "ReachabilitySample",
    "populate_reachability_scene",
    "populate_workspace_boundary_surface_scene",
]
