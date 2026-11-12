"""Deterministic front-camera randomization with tabletop framing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from ...geometry import axis_angle_matrix


@dataclass(frozen=True)
class CameraRandomizationLimits:
    """Hard truncation bounds and Gaussian standard deviations."""

    lateral_m: float = 0.025
    vertical_m: float = 0.025
    depth_m: float = 0.025
    lateral_std_m: float = 0.01
    vertical_std_m: float = 0.01
    depth_std_m: float = 0.01
    tilt_x_deg: float = 2.0
    tilt_y_deg: float = 2.0
    roll_deg: float = 1.0
    tilt_x_std_deg: float = 1.5
    tilt_y_std_deg: float = 1.5
    roll_std_deg: float = 0.6

    def __post_init__(self) -> None:
        values = np.asarray(tuple(vars(self).values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Camera limits must be finite and non-negative")
        if min(self.lateral_m, self.vertical_m, self.depth_m) <= 0.0:
            raise ValueError("Camera translation bounds must be positive")
        for bound, std in (
            (self.tilt_x_deg, self.tilt_x_std_deg),
            (self.tilt_y_deg, self.tilt_y_std_deg),
            (self.roll_deg, self.roll_std_deg),
        ):
            if bound == 0.0 and std > 0.0:
                raise ValueError("A zero angular bound requires zero standard deviation")


@dataclass(frozen=True)
class RandomizedCameraPose:
    position: np.ndarray
    quaternion: np.ndarray
    local_translation: np.ndarray
    local_angles_deg: np.ndarray
    focus: np.ndarray
    aim_error_deg: float
    framing_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tabletop_visible_fraction_before: float | None = None
    tabletop_visible_fraction_after: float | None = None
    tabletop_visible_fraction_reference: float | None = None


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=float).reshape(9))
    return quaternion


def optical_forward(quaternion: np.ndarray) -> np.ndarray:
    """Return the MuJoCo camera optical axis in world coordinates."""
    return -quaternion_to_matrix(quaternion)[:, 2]


def tabletop_corners(model: mujoco.MjModel) -> np.ndarray | None:
    """Read the current fixed tabletop top polygon, in perimeter order.

    Model coordinates include layout changes even before the next mj_forward.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "work_surface")
    if body < 0:
        return None
    if model.body_parentid[body] != 0:
        raise ValueError("Tabletop framing requires a world-attached work_surface")
    boxes = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == body
             and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    if not boxes:
        raise ValueError("work_surface is missing its tabletop box")
    geom = max(boxes, key=lambda g: float(np.prod(model.geom_size[g, :2])))
    x, y, z = model.geom_size[geom]
    corners = np.array(((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)))
    return ((corners @ quaternion_to_matrix(model.geom_quat[geom]).T
             + model.geom_pos[geom]) @ quaternion_to_matrix(model.body_quat[body]).T
            + model.body_pos[body])


def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    relative = vertices - vertices[0]
    return 0.5 * float(np.linalg.norm(np.sum(
        np.cross(relative, np.roll(relative, -1, axis=0)), axis=0,
    )))


def _validate_tabletop(tabletop: np.ndarray) -> np.ndarray:
    points = np.asarray(tabletop, dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Tabletop must be four finite corners in perimeter order")
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    if np.linalg.norm(normal) <= 1e-12:
        raise ValueError("Tabletop must have positive area")
    normal /= np.linalg.norm(normal)
    edges = np.roll(points, -1, axis=0) - points
    turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
    if (abs(float((points[3] - points[0]) @ normal)) > 1e-8
            or np.any(turns <= 1e-12)):
        raise ValueError("Tabletop corners must form a planar convex perimeter")
    return points


def _visible_fraction(
    position: np.ndarray, rotation: np.ndarray, tabletop: np.ndarray,
    frustum_planes: np.ndarray, total_area: float,
) -> float:
    vertices = (tabletop - position) @ rotation
    # Clip in 3D before projecting: screen-polygon area would overvalue the
    # nearer portion of the table, and counting visible corners misses strips.
    for normal in frustum_planes:
        if len(vertices) < 3:
            return 0.0
        distances = vertices @ normal
        inside = distances >= 0.0
        if np.all(inside):
            continue
        if not np.any(inside):
            return 0.0
        clipped = []
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            if inside[i]:
                clipped.append(vertices[i])
            if inside[i] != inside[j]:
                weight = distances[i] / (distances[i] - distances[j])
                clipped.append(vertices[i] + weight * (vertices[j] - vertices[i]))
        vertices = np.asarray(clipped)
    return float(np.clip(_polygon_area(vertices) / total_area, 0.0, 1.0))


def _frustum_planes(vertical_fov_deg: float, aspect_ratio: float) -> np.ndarray:
    if (not np.isfinite(vertical_fov_deg) or not 0.0 < vertical_fov_deg < 180.0
            or not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0):
        raise ValueError("Tabletop framing requires valid perspective FOV and aspect ratio")
    y = math.tan(math.radians(vertical_fov_deg / 2.0))
    x = y * aspect_ratio
    return np.array(((0, 0, -1), (1, 0, -x), (-1, 0, -x), (0, 1, -y), (0, -1, -y)))


def tabletop_visible_fraction(
    position: np.ndarray, quaternion: np.ndarray, tabletop: np.ndarray,
    vertical_fov_deg: float, aspect_ratio: float = 4.0 / 3.0,
) -> float:
    """Fraction of physical tabletop area inside the perspective frustum.

    This measures cropping, independently of robot/object occlusion.
    """
    tabletop = _validate_tabletop(tabletop)
    return _visible_fraction(
        position, quaternion_to_matrix(quaternion), tabletop,
        _frustum_planes(vertical_fov_deg, aspect_ratio), _polygon_area(tabletop),
    )


def _tilt_rotation(angles_deg: np.ndarray) -> np.ndarray:
    x, y = np.radians(angles_deg)
    return (axis_angle_matrix(np.array((1.0, 0.0, 0.0)), x)
            @ axis_angle_matrix(np.array((0.0, 1.0, 0.0)), y))


# The quadratic angular cost uses 0.5 percentage points per square degree
# for framing and additional departure from the initial pose. Prefer nearby
# framing over spending many degrees on a small coverage increase.
_FRAMING_REFERENCE_PENALTY_PER_DEG2 = 0.005
_FRAMING_EDGE_SOFTNESS = 0.04
_FRAMING_COVERAGE_LOSS_WEIGHT = 10.0
_AREA_NODES, _AREA_WEIGHTS = np.polynomial.legendre.leggauss(40)


def _tabletop_area_samples(tabletop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate physical area with deterministic bilinear quadrature."""
    u, v = np.meshgrid((_AREA_NODES + 1.0) / 2.0, (_AREA_NODES + 1.0) / 2.0)
    u, v = u.reshape(-1, 1), v.reshape(-1, 1)
    p0, p1, p2, p3 = tabletop
    points = ((1 - u) * (1 - v) * p0 + u * (1 - v) * p1
              + u * v * p2 + (1 - u) * v * p3)
    du = (1 - v) * (p1 - p0) + v * (p2 - p3)
    dv = (1 - u) * (p3 - p0) + u * (p2 - p1)
    weights = (np.outer(_AREA_WEIGHTS, _AREA_WEIGHTS).ravel()
               * np.linalg.norm(np.cross(du, dv), axis=1))
    return points, weights / np.sum(weights)


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _smooth_tabletop_fraction(
    relative_samples: np.ndarray, sample_weights: np.ndarray,
    rotation: np.ndarray, plane_normals: np.ndarray, softness_m: float,
) -> float:
    signed_distances = relative_samples @ rotation @ plane_normals.T
    gates = 1.0 / (1.0 + np.exp(-np.clip(signed_distances / softness_m, -40.0, 40.0)))
    return float(sample_weights @ np.prod(gates, axis=1))


def _frame_tabletop(
    position: np.ndarray, look_rotation: np.ndarray, residual: np.ndarray,
    reference_position: np.ndarray, reference_rotation: np.ndarray,
    reference_tabletop: np.ndarray, focus: np.ndarray, tabletop: np.ndarray,
    vertical_fov_deg: float, aspect_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    planes = _frustum_planes(vertical_fov_deg, aspect_ratio)
    total_area = _polygon_area(tabletop)
    half_y = math.tan(math.radians(vertical_fov_deg / 2.0))
    half_x = half_y * aspect_ratio
    bounds = np.degrees(np.arctan(0.4 * np.array((half_y, half_x))))
    original_rotation = look_rotation @ residual
    before = _visible_fraction(position, original_rotation, tabletop, planes, total_area)
    original_reference_distance = _rotation_distance_deg(reference_rotation, look_rotation)
    samples, sample_weights = _tabletop_area_samples(tabletop)
    relative_samples = samples - position
    plane_normals = planes / np.linalg.norm(planes, axis=1, keepdims=True)
    softness_m = (_FRAMING_EDGE_SOFTNESS * half_y
                  * max(float(np.linalg.norm(np.mean(tabletop, axis=0) - position)), 0.1))
    reference_samples, reference_weights = _tabletop_area_samples(reference_tabletop)
    reference_softness_m = (_FRAMING_EDGE_SOFTNESS * half_y
                           * max(float(np.linalg.norm(np.mean(reference_tabletop, axis=0) - reference_position)), 0.1))
    reference_coverage = _smooth_tabletop_fraction(
        reference_samples - reference_position, reference_weights,
        reference_rotation, plane_normals, reference_softness_m,
    )
    initial_coverage = _smooth_tabletop_fraction(
        relative_samples, sample_weights, original_rotation, plane_normals, softness_m,
    )
    # Initial cropping is the reference, not an error to fix. Do not introduce
    # a deterministic reframing when randomization causes no coverage loss.
    if initial_coverage >= reference_coverage - 1e-10:
        return original_rotation, np.zeros(3), before, before

    def evaluate(angles: np.ndarray) -> tuple[float, float, np.ndarray, bool]:
        base_rotation = look_rotation @ _tilt_rotation(angles)
        rotation = base_rotation @ residual
        coverage = _visible_fraction(position, rotation, tabletop, planes, total_area)
        # Compare the automatic base orientation with the calibrated initial
        # orientation; sampled residual tilt/roll remains a separate factor.
        distance = _rotation_distance_deg(reference_rotation, base_rotation)
        additional_distance = max(0.0, distance - original_reference_distance)
        penalty = float(angles @ angles) + additional_distance ** 2
        # Exact clipping is continuous but has corner/edge kinks. Use smooth
        # frustum membership for optimization; metrics and the no-regression
        # check still use exact physical-area clipping above.
        smooth_coverage = _smooth_tabletop_fraction(
            relative_samples, sample_weights, rotation, plane_normals, softness_m,
        )
        loss = max(0.0, reference_coverage - smooth_coverage)
        score = (-_FRAMING_COVERAGE_LOSS_WEIGHT * loss ** 2
                 - _FRAMING_REFERENCE_PENALTY_PER_DEG2 * penalty)
        local = (focus - position) @ rotation
        focus_visible = (local[2] < -1e-8
                         and abs(local[0]) <= 0.5 * -local[2] * half_x + 1e-12
                         and abs(local[1]) <= 0.5 * -local[2] * half_y + 1e-12)
        return score, coverage, rotation, focus_visible

    # Follow a continuous objective from the original draw. Finite-difference
    # gradients and backtracking replace quantized grid angles and the former
    # 0/25/50/75/100-percent correction selection.
    angles = np.zeros(2)
    score, after, result, _ = evaluate(angles)
    gradient_step_deg = 0.002
    for _ in range(32):
        gradient = np.empty(2)
        curvature = np.empty((2, 2))
        for axis in range(2):
            delta = np.zeros(2)
            delta[axis] = gradient_step_deg
            plus, minus = evaluate(angles + delta)[0], evaluate(angles - delta)[0]
            gradient[axis] = (plus - minus) / (2.0 * gradient_step_deg)
            curvature[axis, axis] = -(plus - 2.0 * score + minus) / gradient_step_deg ** 2
        dx, dy = np.eye(2) * gradient_step_deg
        mixed = -(evaluate(angles + dx + dy)[0] - evaluate(angles + dx - dy)[0]
                  - evaluate(angles - dx + dy)[0] + evaluate(angles - dx - dy)[0]) / (4.0 * gradient_step_deg ** 2)
        curvature[0, 1] = curvature[1, 0] = mixed
        values, vectors = np.linalg.eigh(curvature)
        curvature = (vectors * np.maximum(values, 2.0 * _FRAMING_REFERENCE_PENALTY_PER_DEG2)) @ vectors.T
        direction = np.linalg.solve(curvature, gradient)
        size = float(np.linalg.norm(direction))
        if size < 1e-4:
            break
        direction *= min(1.0, 0.5 / size)
        accepted = False
        for _ in range(16):
            candidate = np.clip(angles + direction, -bounds, bounds)
            new_score, coverage, rotation, focus_visible = evaluate(candidate)
            if (focus_visible and coverage >= before - 1e-12
                    and new_score > score + 1e-12):
                angles, score, after, result = candidate, new_score, coverage, rotation
                accepted = True
                break
            direction *= 0.5
        if not accepted or np.linalg.norm(direction) < 1e-4:
            break
    return result, np.r_[angles, 0.0], before, after


def focus_on_world_x(position: np.ndarray, quaternion: np.ndarray, world_x: float) -> np.ndarray:
    """Intersect the original optical axis with a world-X focal plane."""
    forward = optical_forward(quaternion)
    if abs(float(forward[0])) < 1e-8:
        raise ValueError("Front camera optical axis is parallel to the focal plane")
    distance = (float(world_x) - float(position[0])) / float(forward[0])
    if distance <= 0.0:
        raise ValueError("Front camera focal plane must lie in front of the camera")
    return np.asarray(position, dtype=float) + distance * forward


def _sample_truncated_normal_ball(
    rng: np.random.Generator,
    normalized_stddevs: np.ndarray,
) -> np.ndarray:
    """Sample a centered Gaussian truncated by a unit ball."""
    while True:
        sample = rng.normal(0.0, normalized_stddevs)
        if float(sample @ sample) <= 1.0:
            return sample


def _sample_truncated_normal(
    rng: np.random.Generator,
    stddev: float,
    bound: float,
) -> float:
    """Sample one centered Gaussian variate within a symmetric hard bound."""
    while True:
        sample = float(rng.normal(0.0, stddev))
        if abs(sample) <= bound:
            return sample


def _look_at_rotation(position: np.ndarray, focus: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = np.asarray(focus, dtype=float) - np.asarray(position, dtype=float)
    distance = float(np.linalg.norm(forward))
    if not np.isfinite(distance) or distance <= 1e-8:
        raise ValueError("Camera position must be separated from its finite focus")
    forward /= distance
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-8:
        # Pick the least aligned axis, including when looking straight up/down.
        right = np.cross(forward, np.eye(3)[int(np.argmin(np.abs(forward)))])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    # MuJoCo cameras look along local -Z; the matrix columns are local axes.
    return np.column_stack((right, up, -forward))


class FrontCameraRandomizer:
    """Sample reproducible poses and improve framing around a retained focus."""

    def __init__(
        self,
        center_position: np.ndarray,
        center_quaternion: np.ndarray,
        focus: np.ndarray,
        limits: CameraRandomizationLimits | None = None,
    ) -> None:
        self.center_position = np.asarray(center_position, dtype=float).copy()
        self.center_quaternion = np.asarray(center_quaternion, dtype=float).copy()
        self.focus = np.asarray(focus, dtype=float).copy()
        self.limits = limits or CameraRandomizationLimits()

    def sample(
        self,
        seed: int,
        center_position: np.ndarray | None = None,
        center_quaternion: np.ndarray | None = None,
        *,
        tabletop: np.ndarray | None = None,
        reference_tabletop: np.ndarray | None = None,
        vertical_fov_deg: float = 42.0,
        aspect_ratio: float = 4.0 / 3.0,
    ) -> RandomizedCameraPose:
        """Repeat the pose for identical seed, calibration and table geometry."""
        position_center = self.center_position if center_position is None else np.asarray(center_position, dtype=float)
        quaternion_center = self.center_quaternion if center_quaternion is None else np.asarray(center_quaternion, dtype=float)
        if (
            position_center.shape != (3,) or self.focus.shape != (3,)
            or not np.all(np.isfinite(position_center))
            or not np.all(np.isfinite(self.focus))
            or np.linalg.norm(self.focus - position_center) <= 1e-8
        ):
            raise ValueError("Camera position and focus must be finite separated 3D points")
        if (
            quaternion_center.shape != (4,)
            or not np.all(np.isfinite(quaternion_center))
            or not np.isclose(np.linalg.norm(quaternion_center), 1.0, atol=1e-8)
        ):
            raise ValueError("Camera quaternion must be finite and unit length")
        rotation_center = quaternion_to_matrix(quaternion_center)
        # UI calibration defines a new reference optical axis. Reusing the
        # uncalibrated focus here would undo its pitch adjustment on every draw.
        focus = (
            self.focus if center_position is None and center_quaternion is None
            else focus_on_world_x(position_center, quaternion_center, float(self.focus[0]))
        )
        if tabletop is not None:
            tabletop = _validate_tabletop(tabletop)
            reference_tabletop = tabletop if reference_tabletop is None else _validate_tabletop(reference_tabletop)
            _frustum_planes(vertical_fov_deg, aspect_ratio)
        reference_coverage = (None if tabletop is None else tabletop_visible_fraction(
            position_center, quaternion_center, reference_tabletop, vertical_fov_deg, aspect_ratio,
        ))
        if seed == -1:
            ideal_forward = focus - position_center
            ideal_forward /= np.linalg.norm(ideal_forward)
            actual_forward = -rotation_center[:, 2]
            aim_error_deg = math.degrees(math.acos(float(np.clip(ideal_forward @ actual_forward, -1.0, 1.0))))
            coverage = (None if tabletop is None else tabletop_visible_fraction(
                position_center, quaternion_center, tabletop, vertical_fov_deg, aspect_ratio,
            ))
            return RandomizedCameraPose(
                position_center.copy(), quaternion_center.copy(), np.zeros(3), np.zeros(3),
                focus.copy(), aim_error_deg,
                tabletop_visible_fraction_before=coverage,
                tabletop_visible_fraction_after=coverage,
                tabletop_visible_fraction_reference=reference_coverage,
            )
        if seed < -1:
            raise ValueError("Camera seed must be -1 (disabled) or a non-negative integer")
        rng = np.random.Generator(np.random.PCG64(int(seed) % (1 << 64)))

        translation_bounds = np.array((
            self.limits.lateral_m,
            self.limits.vertical_m,
            self.limits.depth_m,
        ))
        translation_stddevs = np.array((
            self.limits.lateral_std_m,
            self.limits.vertical_std_m,
            self.limits.depth_std_m,
        ))
        normalized_translation = _sample_truncated_normal_ball(
            rng,
            translation_stddevs / translation_bounds,
        )
        local_translation = normalized_translation * translation_bounds
        right, up, backward = rotation_center[:, 0], rotation_center[:, 1], rotation_center[:, 2]
        forward = -backward
        position = position_center + right * local_translation[0] + up * local_translation[1] + forward * local_translation[2]

        local_angles_deg = np.array((
            _sample_truncated_normal(rng, self.limits.tilt_x_std_deg, self.limits.tilt_x_deg),
            _sample_truncated_normal(rng, self.limits.tilt_y_std_deg, self.limits.tilt_y_deg),
            _sample_truncated_normal(rng, self.limits.roll_std_deg, self.limits.roll_deg),
        ))
        local_angles = np.radians(local_angles_deg)

        # The angle limits below constrain only residual misalignment, never
        # the position-dependent look-at correction relative to the base pose.
        look_rotation = _look_at_rotation(position, focus, rotation_center[:, 1])
        residual = (
            axis_angle_matrix(np.array((1.0, 0.0, 0.0)), local_angles[0])
            @ axis_angle_matrix(np.array((0.0, 1.0, 0.0)), local_angles[1])
            @ axis_angle_matrix(np.array((0.0, 0.0, 1.0)), local_angles[2])
        )
        rotation = look_rotation @ residual
        framing_angles = np.zeros(3)
        before = after = None
        if tabletop is not None:
            rotation, framing_angles, before, after = _frame_tabletop(
                position, look_rotation, residual, position_center, rotation_center,
                reference_tabletop, focus, tabletop,
                vertical_fov_deg, aspect_ratio,
            )
        quaternion = matrix_to_quaternion(rotation)
        if (not np.any(local_translation) and not np.any(local_angles_deg)
                and not np.any(framing_angles)):
            quaternion = quaternion_center.copy()

        ideal_forward = focus - position
        ideal_forward /= np.linalg.norm(ideal_forward)
        actual_forward = -rotation[:, 2]
        aim_error_deg = math.degrees(math.acos(float(np.clip(ideal_forward @ actual_forward, -1.0, 1.0))))
        return RandomizedCameraPose(
            position.copy(), quaternion, local_translation, local_angles_deg, focus.copy(), aim_error_deg,
            framing_angles, before, after, reference_coverage,
        )
