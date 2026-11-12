"""Build adaptive fixed-home-orientation reachability maps in an empty scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from ..simulation.loader import load_scene_model
from ..simulation.randomization.objects import ObjectLayoutRandomizer
from ..simulation.robot import ee_pose, find_handles, set_home
from ..simulation.robots import get_robot_spec
from ..simulation.scenes import SceneManager
from ..tasks.catalog import BLOCK_TO_PLATE_TASK
from ..tasks.planning.presets import PresetTaskExecutor
from .map import ReachabilityMap
from .metric import BadnessSettings, ReachabilityProbe, ReachabilityProbeResult


@dataclass(frozen=True)
class AnalyticReachBound:
    """Conservative fixed-orientation bound for the terminal joint anchor."""

    base_position: np.ndarray
    ee_from_terminal_anchor: np.ndarray
    maximum_anchor_reach_m: float
    safety_margin_m: float = 0.002

    def excess_distance(self, target_position: np.ndarray) -> float:
        anchor = np.asarray(target_position) - self.ee_from_terminal_anchor
        distance = float(np.linalg.norm(anchor - self.base_position))
        return distance - self.maximum_anchor_reach_m - self.safety_margin_m

    def cell_is_unreachable(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> bool:
        anchor_lower = np.asarray(lower) - self.ee_from_terminal_anchor
        anchor_upper = np.asarray(upper) - self.ee_from_terminal_anchor
        nearest = np.minimum(
            np.maximum(self.base_position, anchor_lower), anchor_upper,
        )
        return (
            np.linalg.norm(nearest - self.base_position)
            > self.maximum_anchor_reach_m + self.safety_margin_m
        )


def analytic_reach_bound(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles,
) -> AnalyticReachBound:
    """Build a no-false-negative serial-chain reach upper bound at Home."""
    base_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, handles.spec.base_body,
    )
    terminal_joint = int(handles.joint_ids[-1])
    terminal_body = int(model.jnt_bodyid[terminal_joint])
    maximum_reach = float(np.linalg.norm(model.jnt_pos[terminal_joint]))
    body = terminal_body
    while body != base_body:
        if body <= 0:
            raise RuntimeError("Terminal arm joint is not below the robot base")
        maximum_reach += float(np.linalg.norm(model.body_pos[body]))
        body = int(model.body_parentid[body])
    home_position, _ = ee_pose(data, handles)
    return AnalyticReachBound(
        base_position=data.xpos[base_body].copy(),
        ee_from_terminal_anchor=(
            home_position - data.xanchor[terminal_joint]
        ),
        maximum_anchor_reach_m=maximum_reach,
    )


def nominal_stage_endpoints(
    robot_uid: str = "panda",
    task_name: str = BLOCK_TO_PLATE_TASK,
) -> dict[str, np.ndarray]:
    """Return seed=-1 grasp/place terminal EE positions for Blocks."""
    model = load_scene_model("Blocks", robot_uid)
    data = mujoco.MjData(model)
    handles = find_handles(model, get_robot_spec(robot_uid))
    set_home(model, data, handles)
    manager = SceneManager(model, loaded_scene="Blocks")
    manager.activate(data, "Blocks")
    ObjectLayoutRandomizer(model, manager).apply(data, -1)
    current_position, current_rotation = ee_pose(data, handles)
    executor = PresetTaskExecutor(model, handles)
    executor.start(
        task_name, "Blocks", data, current_position, current_rotation,
    )
    result: dict[str, np.ndarray] = {}
    previous_closed = False
    for frame in executor.frames:
        if frame.gripper_closed and not previous_closed and "grasp" not in result:
            result["grasp"] = frame.position.copy()
        if not frame.gripper_closed and previous_closed and "place" not in result:
            result["place"] = frame.position.copy()
        previous_closed = frame.gripper_closed
    if set(result) != {"grasp", "place"}:
        raise RuntimeError("Preset did not expose both grasp and place endpoints")
    return result


def tabletop_reachability_volume(
    robot_uid: str = "panda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nominal table reference, lower bound, and upper bound.

    XY covers the complete tabletop AABB plus 20 cm on every side. Z is
    referenced to the tabletop surface and spans -10 cm through +30 cm.
    """
    model = load_scene_model("Blocks", robot_uid)
    data = mujoco.MjData(model)
    handles = find_handles(model, get_robot_spec(robot_uid))
    set_home(model, data, handles)
    table_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "work_surface",
    )
    if table_body < 0:
        raise RuntimeError("Reference scene has no work_surface")
    table_geoms = np.flatnonzero(model.geom_bodyid == table_body)
    table_geom = int(table_geoms[
        np.argmax(np.prod(model.geom_size[table_geoms, :2], axis=1))
    ])
    center = data.geom_xpos[table_geom]
    rotation = data.geom_xmat[table_geom].reshape(3, 3)
    half_xy = np.abs(rotation[:2]) @ model.geom_size[table_geom]
    table_top = float(
        center[2] + np.abs(rotation[2]) @ model.geom_size[table_geom]
    )
    reference = np.array((center[0], center[1], table_top))
    lower = np.array((
        center[0] - half_xy[0] - 0.20,
        center[1] - half_xy[1] - 0.20,
        table_top - 0.10,
    ))
    upper = np.array((
        center[0] + half_xy[0] + 0.20,
        center[1] + half_xy[1] + 0.20,
        table_top + 0.30,
    ))
    return reference, lower, upper


def _axis(lower: float, upper: float, spacing: float) -> np.ndarray:
    count = max(1, int(round((upper - lower) / spacing)))
    return np.linspace(lower, upper, count + 1)


def _point_key(point: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(round(value * 1_000_000.0)) for value in point)


def build_table_reachability_map(
    *,
    robot_uid: str = "panda",
    coarse_spacing_m: float = 0.05,
    transition_spacing_m: float = 0.025,
    minimum_spacing_m: float = 0.01,
    danger_badness: float = 1.0,
    variation_threshold: float = 0.75,
    settings: BadnessSettings = BadnessSettings(),
    progress: Callable[[int, int | None], None] | None = None,
) -> ReachabilityMap:
    if not 0 < minimum_spacing_m <= transition_spacing_m <= coarse_spacing_m <= 0.05:
        raise ValueError("Map spacings must satisfy 0 < min <= transition <= coarse <= 0.05 m")
    spec = get_robot_spec(robot_uid)
    model = load_scene_model(SceneManager.EMPTY_SCENE, robot_uid)
    data = mujoco.MjData(model)
    handles = find_handles(model, spec)
    set_home(model, data, handles)
    _, home_rotation = ee_pose(data, handles)
    center, lower, upper = tabletop_reachability_volume(robot_uid)
    probe = ReachabilityProbe(model, handles, home_rotation, settings)
    reach_bound = analytic_reach_bound(model, data, handles)
    cache: dict[tuple[int, int, int], ReachabilityProbeResult] = {}
    resolutions: dict[tuple[int, int, int], float] = {}
    analytic_keys: set[tuple[int, int, int]] = set()
    warm: np.ndarray | None = None

    def evaluate(point: np.ndarray, resolution: float) -> ReachabilityProbeResult:
        nonlocal warm
        key = _point_key(point)
        resolutions[key] = min(resolutions.get(key, resolution), resolution)
        if key not in cache:
            excess = reach_bound.excess_distance(point)
            if excess > 0.0:
                # A finite increasing score preserves ordering among points
                # that are geometrically outside the serial-chain envelope.
                result = ReachabilityProbeResult(
                    target_position=point.copy(),
                    q=handles.spec.home_q.copy(),
                    reachable=False,
                    badness=float(
                        max(danger_badness, 1.0)
                        + 0.5 * (excess / settings.position_tolerance_m) ** 2
                    ),
                    position_error_m=float(excess),
                    rotation_error_rad=0.0,
                    minimum_singular_value=0.0,
                    manipulability=0.0,
                    joint_margin_fraction=0.0,
                    self_collision_clearance_m=float("nan"),
                    iterations=0,
                )
                analytic_keys.add(key)
            else:
                result = probe.probe(
                    point,
                    warm_start=warm,
                    use_restarts=resolution >= coarse_spacing_m - 1e-12,
                )
            cache[key] = result
            if result.reachable:
                warm = result.q
            if progress is not None:
                progress(len(cache), None)
        return cache[key]

    coarse_axes = tuple(_axis(lower[axis], upper[axis], coarse_spacing_m) for axis in range(3))
    for z in coarse_axes[2]:
        for y_index, y in enumerate(coarse_axes[1]):
            xs = coarse_axes[0] if y_index % 2 == 0 else coarse_axes[0][::-1]
            for x in xs:
                evaluate(np.array((x, y, z)), coarse_spacing_m)

    coarse_cells = []
    for ix, iy, iz in itertools.product(
        range(len(coarse_axes[0]) - 1),
        range(len(coarse_axes[1]) - 1),
        range(len(coarse_axes[2]) - 1),
    ):
        coarse_cells.append((
            np.array((coarse_axes[0][ix], coarse_axes[1][iy], coarse_axes[2][iz])),
            np.array((coarse_axes[0][ix + 1], coarse_axes[1][iy + 1], coarse_axes[2][iz + 1])),
        ))

    def refine_cells(
        parent_cells: list[tuple[np.ndarray, np.ndarray]],
        parent_spacing: float,
        child_spacing: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        selected: list[tuple[np.ndarray, np.ndarray]] = []
        for cell_lower, cell_upper in parent_cells:
            if reach_bound.cell_is_unreachable(cell_lower, cell_upper):
                continue
            corner_results = []
            for offsets in itertools.product((0, 1), repeat=3):
                point = np.where(np.asarray(offsets, dtype=bool), cell_upper, cell_lower)
                corner_results.append(evaluate(point, parent_spacing))
            reachability = {item.reachable for item in corner_results}
            # A uniformly failed coarse cell carries no evidence of an
            # interior boundary. Keep it sparse; refine mixed cells instead.
            if reachability == {False}:
                continue
            finite_scores = [item.badness for item in corner_results if np.isfinite(item.badness)]
            score_range = (
                max(finite_scores) - min(finite_scores)
                if len(finite_scores) >= 2 else float("inf")
            )
            dangerous = (
                len(reachability) > 1
                or any(
                    item.reachable and item.badness >= danger_badness
                    for item in corner_results
                )
                or score_range >= variation_threshold
            )
            if dangerous:
                selected.append((cell_lower, cell_upper))
        child_cells: list[tuple[np.ndarray, np.ndarray]] = []
        for cell_lower, cell_upper in selected:
            axes = tuple(
                _axis(cell_lower[axis], cell_upper[axis], child_spacing)
                for axis in range(3)
            )
            for point_values in itertools.product(*axes):
                evaluate(np.asarray(point_values, dtype=float), child_spacing)
            for ix, iy, iz in itertools.product(
                range(len(axes[0]) - 1),
                range(len(axes[1]) - 1),
                range(len(axes[2]) - 1),
            ):
                child_cells.append((
                    np.array((axes[0][ix], axes[1][iy], axes[2][iz])),
                    np.array((axes[0][ix + 1], axes[1][iy + 1], axes[2][iz + 1])),
                ))
        return child_cells

    transition_cells = refine_cells(
        coarse_cells, coarse_spacing_m, transition_spacing_m,
    )
    if minimum_spacing_m < transition_spacing_m:
        refine_cells(
            transition_cells, transition_spacing_m, minimum_spacing_m,
        )

    keys = sorted(cache)
    results = [cache[key] for key in keys]
    metadata = {
        "format_version": 1,
        "metric_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot_uid": robot_uid,
        "robot_label": spec.label,
        "scene": SceneManager.EMPTY_SCENE,
        "reference_scene": "Blocks",
        "task": None,
        "stage": None,
        "domain": "tabletop_aabb_plus_20cm_xy_minus10_plus30cm_z",
        "home_q": handles.spec.home_q.tolist(),
        "home_rotation": home_rotation.tolist(),
        "center_m": center.tolist(),
        "lower_m": lower.tolist(),
        "upper_m": upper.tolist(),
        "coarse_spacing_m": coarse_spacing_m,
        "transition_spacing_m": transition_spacing_m,
        "minimum_spacing_m": minimum_spacing_m,
        "danger_badness": danger_badness,
        "variation_threshold": variation_threshold,
        "analytic_bound": {
            "base_position_m": reach_bound.base_position.tolist(),
            "ee_from_terminal_anchor_m": (
                reach_bound.ee_from_terminal_anchor.tolist()
            ),
            "maximum_anchor_reach_m": reach_bound.maximum_anchor_reach_m,
            "safety_margin_m": reach_bound.safety_margin_m,
            "direct_unreachable_count": len(analytic_keys),
            "uniform_unreachable_cells_are_not_refined": True,
        },
        "settings": asdict(settings),
    }
    return ReachabilityMap(
        positions=np.asarray([item.target_position for item in results]),
        badness=np.asarray([item.badness for item in results]),
        reachable=np.asarray([item.reachable for item in results], dtype=bool),
        resolution_m=np.asarray([resolutions[key] for key in keys]),
        q=np.asarray([item.q for item in results]),
        position_error_m=np.asarray([item.position_error_m for item in results]),
        rotation_error_rad=np.asarray([item.rotation_error_rad for item in results]),
        minimum_singular_value=np.asarray([item.minimum_singular_value for item in results]),
        manipulability=np.asarray([item.manipulability for item in results]),
        joint_margin_fraction=np.asarray([item.joint_margin_fraction for item in results]),
        self_collision_clearance_m=np.asarray([item.self_collision_clearance_m for item in results]),
        metadata=metadata,
    )
