"""Minimal pick skills and the shared plate-placement skill used by Blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from ...geometry import axis_angle_matrix, rotation_vector
from ...simulation.robot import RobotHandles
from .types import PREGRASP_CLEARANCE_M, MotionLimits, TaskFrame


_COARSE_LINEAR_SPEED_M_S = 0.150
# Final downward grasp approach.
_GRASP_DESCENT_LINEAR_SPEED_M_S = 0.0525
_PRECISION_LINEAR_SPEED_M_S = 0.0525
_LIFT_LINEAR_SPEED_M_S = 0.100
_TRANSFER_LINEAR_SPEED_M_S = 0.100
_CARTESIAN_AXIS_SPEED_LIMIT_M_S = 0.100
_CARTESIAN_ACCELERATION_TIME_S = 0.500
_CARTESIAN_DECELERATION_TIME_S = 0.500
_DESCENT_DECELERATION_TIME_S = 1.000
_PREGRASP_HEIGHT_M = PREGRASP_CLEARANCE_M
_PREGRASP_VERTICAL_RAMP_TIME_S = 1.000
_FINAL_GRASP_DESCENT_SPEED_M_S = 0.040
_FINAL_GRASP_DESCENT_DURATION_S = 2.000
GRIPPER_CLOSE_DWELL_SECONDS = 1.2
GRIPPER_PRECLOSE_DWELL_SECONDS = 0.5
_SCURVE_RAMP_FRACTION = 0.20
_SCURVE_DURATION_FACTOR = 1.0 / (1.0 - _SCURVE_RAMP_FRACTION)
_MINIMUM_GRIPPER_BASE_CLEARANCE_M = 0.005
# The specific-goal collector can overshoot 6--9 mm during the final descent.
# Reserve 10 mm in the nominal geometry so the physical clearance remains at
# least 5 mm instead of merely satisfying it in the planned pose.
_GRASP_TRACKING_RESERVE_M = 0.010
# Keep the lowest finger collision point this far above the object's support
# plane.  This common reference absorbs multi-robot Cartesian tracking error
# without relying on robot-specific task offsets.
_DEFAULT_FINGER_SUPPORT_CLEARANCE_M = (
    _MINIMUM_GRIPPER_BASE_CLEARANCE_M + _GRASP_TRACKING_RESERVE_M
)
_MINIMUM_CONTACT_OVERLAP_M = 0.002
_REFERENCE_FINGER_CONTACT_HEIGHT_M = 0.027
_REFERENCE_FINGER_BOTTOM_SUPPORT_CLEARANCE_M = 0.027
_FINGER_HEIGHT_COMPENSATION_GAIN = 0.5
# xArm/Robotiq fingers taper to a small lower end.  They can safely centre on
# round objects without the extra lift needed by Panda's broad fingertip pads.
_NARROW_FINGERTIP_CONTACT_HEIGHT_THRESHOLD_M = 0.030


def _quintic_ramp_integral(fraction: float) -> float:
    """Integral of 10u^3 - 15u^4 + 6u^5 on [0, fraction]."""
    u = float(np.clip(fraction, 0.0, 1.0))
    return u**4 * (2.5 - 3.0 * u + u**2)


def _raised_cosine_integral(elapsed: float, duration: float) -> float:
    """Integral of a raised-cosine 0-to-1 transition over ``duration``."""
    if duration <= 0.0:
        return elapsed
    clamped = float(np.clip(elapsed, 0.0, duration))
    return (
        0.5 * clamped
        - duration / (2.0 * math.pi)
        * math.sin(math.pi * clamped / duration)
    )


def _final_descent_distance(elapsed_s: float) -> float:
    """Distance for the 2 s, 4 cm/s-to-zero asymmetric smooth deceleration."""
    duration = _FINAL_GRASP_DESCENT_DURATION_S
    u = float(np.clip(elapsed_s / duration, 0.0, 1.0))
    normalized_integral = (
        u
        - 3.5 * u**6
        + 5.0 * u**7
        - 1.875 * u**8
    )
    return (
        _FINAL_GRASP_DESCENT_SPEED_M_S
        * duration
        * normalized_integral
    )


def _final_descent_velocity(elapsed_s: float) -> float:
    """Velocity paired with :func:`_final_descent_distance`."""
    u = float(np.clip(
        elapsed_s / _FINAL_GRASP_DESCENT_DURATION_S, 0.0, 1.0,
    ))
    return _FINAL_GRASP_DESCENT_SPEED_M_S * (
        1.0 - 21.0 * u**5 + 35.0 * u**6 - 15.0 * u**7
    )


def _s_curve_progress(fraction: float) -> float:
    """Fast rest-to-rest progress with smooth 20% ramps and 60% cruise."""
    u = float(np.clip(fraction, 0.0, 1.0))
    ramp = _SCURVE_RAMP_FRACTION
    raw_total = 1.0 - ramp
    if u < ramp:
        raw = ramp * _quintic_ramp_integral(u / ramp)
    elif u <= 1.0 - ramp:
        raw = 0.5 * ramp + (u - ramp)
    else:
        remaining = ramp * _quintic_ramp_integral((1.0 - u) / ramp)
        raw = raw_total - remaining
    return raw / raw_total


def _limited_linear_speed(limits: MotionLimits, speed_m_s: float) -> MotionLimits:
    episode_scale = limits.linear_speed_m_s / _COARSE_LINEAR_SPEED_M_S
    return MotionLimits(
        linear_speed_m_s=min(limits.linear_speed_m_s, speed_m_s * episode_scale),
        angular_speed_rad_s=limits.angular_speed_rad_s,
        command_rate_hz=limits.command_rate_hz,
    )


def gripper_close_dwell_ticks(limits: MotionLimits) -> int:
    """Convert the shared post-close 1.2 second wait to command frames."""
    return max(
        1,
        int(math.ceil(
            GRIPPER_CLOSE_DWELL_SECONDS * limits.command_rate_hz,
        )),
    )


def gripper_preclose_dwell_ticks(limits: MotionLimits) -> int:
    """Convert the shared 0.5 second grasp-ready wait to command frames."""
    return max(
        1,
        int(round(
            GRIPPER_PRECLOSE_DWELL_SECONDS * limits.command_rate_hz,
        )),
    )


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Task body not found: {name}")
    return body_id


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"Task geometry not found: {name}")
    return geom_id


def _site_id(model: mujoco.MjModel, name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Task site not found: {name}")
    return site_id


def _geom_world_vertical_half_extent(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    """Return a geom's exact/conservative half extent along world Z."""
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    world_z_in_geom = np.abs(
        data.geom_xmat[geom_id].reshape(3, 3)[2]
    )
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(world_z_in_geom @ size[:3])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        radial = float(np.linalg.norm(world_z_in_geom[:2]) * size[0])
        axial = float(world_z_in_geom[2] * size[1])
        return radial + axial
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return float(np.linalg.norm(world_z_in_geom * size[:3]))
    return float(model.geom_rbound[geom_id])


def _mesh_vertices_in_body(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return a mesh geom's vertices in its owning body frame."""
    mesh_id = int(model.geom_dataid[geom_id])
    first = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[first:first + count]
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, model.geom_quat[geom_id])
    rotation = rotation_flat.reshape(3, 3)
    return vertices @ rotation.T + model.geom_pos[geom_id]


@dataclass(frozen=True)
class GripperVerticalEnvelope:
    """Tool-axis dimensions used by every top/side pinch grasp."""

    base_above_ee_m: float
    contact_upper_above_ee_m: float
    open_finger_bottom_z_offset_m: float = 0.0
    closed_finger_bottom_z_offset_m: float = 0.0
    finger_contact_height_m: float = 0.0

    def closure_fingertip_drop_m(self) -> float:
        """Return how far the lowest fingertip descends when closing."""
        return max(
            0.0,
            float(
                self.closed_finger_bottom_z_offset_m
                - self.open_finger_bottom_z_offset_m
            ),
        )

    def _bottom_support_clearance(self) -> float:
        return float(
            _REFERENCE_FINGER_BOTTOM_SUPPORT_CLEARANCE_M
            + _FINGER_HEIGHT_COMPENSATION_GAIN
            * (
                self.finger_contact_height_m
                - _REFERENCE_FINGER_CONTACT_HEIGHT_M
            )
        )

    def object_center_offset_range(self, vertical_half_extent: float) -> tuple[float, float]:
        """Return deepest-to-shallowest legal EE heights from object centre."""
        top = float(vertical_half_extent)
        deepest = top - (
            self.base_above_ee_m
            - _MINIMUM_GRIPPER_BASE_CLEARANCE_M
            - _GRASP_TRACKING_RESERVE_M
        )
        shallowest = top - self.contact_upper_above_ee_m
        if deepest > shallowest:
            raise RuntimeError(
                "The gripper geometry has no legal vertical grasp interval "
                f"({deepest=:.4f}, {shallowest=:.4f})"
            )
        return deepest, shallowest

    def bottom_first_center_offset(self, vertical_half_extent: float) -> float:
        """Place the finger bottom using one height-aware multi-gripper rule."""
        half_extent = float(vertical_half_extent)
        if half_extent <= 0.0 or self.finger_contact_height_m <= 0.0:
            raise ValueError("Bottom-first grasp requires positive object and finger heights")
        # A fixed bottom height made the short Panda pads sit too high while the
        # taller xArm/Robotiq pads sat too low.  Compensate symmetrically around
        # a common reference height: short pads descend and tall pads rise.
        bottom_support_clearance = self._bottom_support_clearance()
        bottom_relative_to_center = (
            bottom_support_clearance - half_extent
        )
        contact_top_relative_to_center = (
            bottom_relative_to_center + self.finger_contact_height_m
        )
        if (
            bottom_relative_to_center >= half_extent
            or contact_top_relative_to_center <= -half_extent
        ):
            raise RuntimeError("The finger contact band does not overlap the object")
        return float(
            self.closed_finger_bottom_z_offset_m
            - half_extent
            + bottom_support_clearance
        )

    def upper_box_center_offset(
        self,
        vertical_half_extent: float,
        height_bias_m: float = 0.0,
    ) -> float:
        """Pinch a box's upper portion without sending long fingers to its base."""
        half_extent = float(vertical_half_extent)
        if half_extent <= 0.0 or self.finger_contact_height_m <= 0.0:
            raise ValueError("Upper-box grasp requires positive object and finger heights")
        deepest, shallowest = self.object_center_offset_range(half_extent)
        geometric_offset = 0.5 * (deepest + shallowest) + float(height_bias_m)
        finger_table_floor = (
            self.closed_finger_bottom_z_offset_m
            - half_extent
            + _MINIMUM_GRIPPER_BASE_CLEARANCE_M
            + _GRASP_TRACKING_RESERVE_M
        )
        chosen = max(geometric_offset, deepest, finger_table_floor)
        # Long tapered fingers already provide ample contact length.  On an
        # object at least as tall as their contact band, stop the closed finger
        # bottom in the object's upper half instead of descending through its
        # lower half.  Panda's short, broad pads retain their tuned upper pinch.
        if (
            self.finger_contact_height_m
            >= _NARROW_FINGERTIP_CONTACT_HEIGHT_THRESHOLD_M
            and 2.0 * half_extent >= 0.75 * self.finger_contact_height_m
        ):
            chosen = max(
                chosen,
                self.closed_finger_bottom_z_offset_m
                + 0.5 * half_extent
                + max(0.0, float(height_bias_m)),
            )
        contact_ceiling = (
            half_extent
            + self.closed_finger_bottom_z_offset_m
            + _GRASP_TRACKING_RESERVE_M
        )
        if chosen >= contact_ceiling:
            raise RuntimeError(
                "No collision-free box grasp keeps the finger band on the object"
            )
        return float(chosen)

    def safe_tabletop_center_offset(
        self,
        vertical_half_extent: float,
        *,
        support_plane_from_center_m: float | None = None,
        finger_support_clearance_m: float = _DEFAULT_FINGER_SUPPORT_CLEARANCE_M,
        height_bias_m: float = 0.0,
        minimum_contact_overlap_m: float = _MINIMUM_CONTACT_OVERLAP_M,
        preferred_center_offset_m: float | None = None,
    ) -> float:
        """Solve a collision-safe tabletop grasp without changing tuned skills.

        All values are expressed along world Z for a downward-facing tool.  The
        solution simultaneously keeps the worst of the open/closed fingertip
        bottoms above the support plane, keeps the object below the palm, and
        retains a non-zero closed-finger contact overlap with the object.
        """
        half_extent = float(vertical_half_extent)
        if half_extent <= 0.0 or self.finger_contact_height_m <= 0.0:
            raise ValueError("Safe grasp requires positive object and finger heights")
        support_plane = (
            -half_extent
            if support_plane_from_center_m is None
            else float(support_plane_from_center_m)
        )
        finger_clearance = float(finger_support_clearance_m)
        overlap = float(minimum_contact_overlap_m)
        if finger_clearance < 0.0 or overlap <= 0.0:
            raise ValueError("Grasp clearances must be non-negative and overlap positive")

        deepest, shallowest = self.object_center_offset_range(half_extent)
        preferred = (
            0.5 * (deepest + shallowest) + float(height_bias_m)
            if preferred_center_offset_m is None
            else float(preferred_center_offset_m) + float(height_bias_m)
        )
        worst_bottom_offset = max(
            self.open_finger_bottom_z_offset_m,
            self.closed_finger_bottom_z_offset_m,
        )
        # world fingertip bottom = EE_z - bottom_offset
        finger_floor = support_plane + worst_bottom_offset + finger_clearance
        # world palm bottom = EE_z + base_above_ee
        palm_floor = (
            half_extent - self.base_above_ee_m
            + _MINIMUM_GRIPPER_BASE_CLEARANCE_M
            + _GRASP_TRACKING_RESERVE_M
        )
        # Closed contact band in world Z is
        # [EE_z - closed_bottom, EE_z + contact_upper].
        contact_lower_bound = (
            -half_extent - self.contact_upper_above_ee_m + overlap
        )
        contact_upper_bound = (
            half_extent + self.closed_finger_bottom_z_offset_m - overlap
        )
        lower = max(finger_floor, palm_floor, contact_lower_bound)
        upper = contact_upper_bound
        if lower > upper:
            raise RuntimeError(
                "No safe grasp satisfies support, palm, and contact constraints "
                f"({lower=:.4f}, {upper=:.4f})"
            )
        return float(np.clip(preferred, lower, upper))

    def sphere_center_offset(self, radius_m: float) -> float:
        """Choose a sphere grasp from fingertip morphology, not robot name."""
        radius = float(radius_m)
        if (
            self.finger_contact_height_m
            < _NARROW_FINGERTIP_CONTACT_HEIGHT_THRESHOLD_M
        ):
            return self.bottom_first_center_offset(radius)
        return self.safe_tabletop_center_offset(
            radius,
            preferred_center_offset_m=0.0,
        )


def panda_gripper_vertical_envelope(model: mujoco.MjModel) -> GripperVerticalEnvelope:
    """Measure the Panda palm and fingertip-pad envelope from the MJCF model."""
    hand_id = _body_id(model, "hand")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_site_id < 0 or int(model.site_bodyid[ee_site_id]) != hand_id:
        raise RuntimeError("Panda ee_site must be attached directly to hand")
    ee_z = float(model.site_pos[ee_site_id, 2])

    hand_meshes = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == hand_id
        and int(model.geom_contype[geom_id]) != 0
        and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    ]
    if not hand_meshes:
        raise RuntimeError("Panda hand has no collision mesh")
    base_lowest_z = max(
        float(np.max(_mesh_vertices_in_body(model, geom_id)[:, 2]))
        for geom_id in hand_meshes
    )
    base_above_ee = ee_z - base_lowest_z

    pad_upper_z: list[float] = []
    pad_lower_z: list[float] = []
    for finger_name in ("left_finger", "right_finger"):
        finger_id = _body_id(model, finger_name)
        if int(model.body_parentid[finger_id]) != hand_id:
            raise RuntimeError("Panda fingers must be direct children of hand")
        # The current Panda model rotates fingers only around local Z, so their
        # local Z coordinates are identical in the hand frame.
        for geom_id in range(model.ngeom):
            if (
                int(model.geom_bodyid[geom_id]) == finger_id
                and int(model.geom_contype[geom_id]) != 0
                and int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
            ):
                half = model.geom_size[geom_id]
                rotation_flat = np.empty(9)
                mujoco.mju_quat2Mat(rotation_flat, model.geom_quat[geom_id])
                rotation = rotation_flat.reshape(3, 3)
                vertical_half = float(np.abs(rotation[2]) @ half)
                pad_upper_z.append(
                    float(model.body_pos[finger_id, 2] + model.geom_pos[geom_id, 2] - vertical_half)
                )
                pad_lower_z.append(
                    float(model.body_pos[finger_id, 2] + model.geom_pos[geom_id, 2] + vertical_half)
                )
    if not pad_upper_z:
        raise RuntimeError("Panda fingers have no fingertip pad collision boxes")
    contact_upper_above_ee = ee_z - min(pad_upper_z)
    finger_bottom_z_offset = max(pad_lower_z) - ee_z
    finger_contact_height = max(pad_lower_z) - min(pad_upper_z)
    return GripperVerticalEnvelope(
        base_above_ee,
        contact_upper_above_ee,
        finger_bottom_z_offset,
        finger_bottom_z_offset,
        finger_contact_height,
    )


def gripper_vertical_envelope(
    model: mujoco.MjModel,
    handles: RobotHandles | None = None,
) -> GripperVerticalEnvelope:
    """Return the calibrated top-grasp envelope for the selected robot."""
    if handles is None or handles.spec.uid == "panda":
        return panda_gripper_vertical_envelope(model)
    return GripperVerticalEnvelope(
        handles.spec.gripper_base_above_ee_m,
        handles.spec.gripper_contact_upper_above_ee_m,
        handles.spec.gripper_open_finger_bottom_z_offset_m,
        handles.spec.gripper_closed_finger_bottom_z_offset_m,
        handles.spec.gripper_finger_contact_height_m,
    )


def append_dwell(
    frames: list[TaskFrame],
    position: np.ndarray,
    rotation: np.ndarray,
    gripper_closed: bool,
    phase: str,
    ticks: int,
) -> None:
    frames.extend(
        TaskFrame(position.copy(), rotation.copy(), gripper_closed, phase)
        for _ in range(ticks)
    )


def append_blended_motion(
    frames: list[TaskFrame],
    start_position: np.ndarray,
    start_rotation: np.ndarray,
    goal_position: np.ndarray,
    goal_rotation: np.ndarray,
    gripper_closed: bool,
    phase: str,
    limits: MotionLimits,
    start_speed_m_s: float,
    end_speed_m_s: float,
    fixed_duration_s: float | None = None,
    use_smooth_speed_profile: bool = True,
    deceleration_time_s: float | None = None,
) -> None:
    """Append a straight segment with a capped, smoothly ramped speed profile.

    Each Cartesian axis is capped at 0.10 m/s.  The acceleration ramp lasts
    0.5 s; the deceleration ramp lasts 0.5 s normally and 1.0 s on downward
    segments.  Short moves lower their peak speed instead of shortening those
    ramps.  Raised-cosine velocity transitions replace piecewise-linear ramps
    so acceleration enters and leaves each transition continuously.

    ``start_speed_m_s`` and ``end_speed_m_s`` identify true stops versus
    pass-through waypoints.  Nonzero values retain motion through adjacent
    lift/transfer/lower segments; zero is reserved for an actual start, grasp,
    release or terminal target.
    """
    if fixed_duration_s is not None:
        fixed_duration_s = float(fixed_duration_s)
        if not np.isfinite(fixed_duration_s) or fixed_duration_s <= 0.0:
            raise ValueError("Fixed motion duration must be finite and positive")
    if deceleration_time_s is not None:
        deceleration_time_s = float(deceleration_time_s)
        if not np.isfinite(deceleration_time_s) or deceleration_time_s <= 0.0:
            raise ValueError(
                "Deceleration duration must be finite and positive",
            )

    delta = goal_position - start_position
    distance = float(np.linalg.norm(delta))
    start_speed = float(np.clip(start_speed_m_s, 0.0, limits.linear_speed_m_s))
    end_speed = float(np.clip(end_speed_m_s, 0.0, limits.linear_speed_m_s))
    relative_vector = rotation_vector(start_rotation.T @ goal_rotation)
    angle = float(np.linalg.norm(relative_vector))
    if distance <= 1e-12:
        if angle <= 1e-12:
            return
        duration = max(
            fixed_duration_s
            if fixed_duration_s is not None
            else _SCURVE_DURATION_FACTOR * angle / limits.angular_speed_rad_s,
            1.0 / limits.command_rate_hz,
        )
        steps = max(1, int(math.ceil(duration * limits.command_rate_hz)))
        for step in range(1, steps + 1):
            fraction = _s_curve_progress(step / steps)
            rotation = start_rotation @ axis_angle_matrix(
                relative_vector / angle, fraction * angle,
            )
            frames.append(TaskFrame(
                start_position.copy(), rotation, gripper_closed, phase,
            ))
        return

    if not use_smooth_speed_profile:
        duration = max(
            fixed_duration_s
            if fixed_duration_s is not None
            else max(
                distance / limits.linear_speed_m_s,
                _SCURVE_DURATION_FACTOR * angle / limits.angular_speed_rad_s,
            ),
            1.0 / limits.command_rate_hz,
        )
        samples = np.linspace(0.0, 1.0, 129)
        for _ in range(12):
            derivative = (
                distance * (-6.0 * samples**2 + 6.0 * samples)
                + duration * start_speed
                * (3.0 * samples**2 - 4.0 * samples + 1.0)
                + duration * end_speed
                * (3.0 * samples**2 - 2.0 * samples)
            ) / duration
            peak_speed = float(np.max(derivative))
            if peak_speed <= limits.linear_speed_m_s * (1.0 + 1e-9):
                break
            duration *= peak_speed / limits.linear_speed_m_s * 1.001
        steps = max(1, int(math.ceil(duration * limits.command_rate_hz)))
        while True:
            duration = steps / limits.command_rate_hz
            derivative = (
                distance * (-6.0 * samples**2 + 6.0 * samples)
                + duration * start_speed
                * (3.0 * samples**2 - 4.0 * samples + 1.0)
                + duration * end_speed
                * (3.0 * samples**2 - 2.0 * samples)
            ) / duration
            if float(np.max(derivative)) <= limits.linear_speed_m_s * (1.0 + 1e-9):
                break
            steps += 1
        for step in range(1, steps + 1):
            u = step / steps
            h01 = -2.0 * u**3 + 3.0 * u**2
            h10 = u**3 - 2.0 * u**2 + u
            h11 = u**3 - u**2
            travelled = (
                h01 * distance
                + h10 * duration * start_speed
                + h11 * duration * end_speed
            )
            fraction = float(np.clip(travelled / distance, 0.0, 1.0))
            position = start_position + fraction * delta
            rotation_fraction = _s_curve_progress(u)
            rotation = (
                start_rotation @ axis_angle_matrix(
                    relative_vector / angle, rotation_fraction * angle,
                )
                if angle > 1e-9 else start_rotation.copy()
            )
            frames.append(TaskFrame(
                position.copy(), rotation.copy(), gripper_closed, phase,
            ))
        return
    axis_distance = float(np.max(np.abs(delta)))
    axis_speed_limit = _CARTESIAN_AXIS_SPEED_LIMIT_M_S
    descending = bool(delta[2] < -1e-12)
    nominal_acceleration_time = _CARTESIAN_ACCELERATION_TIME_S
    nominal_deceleration_time = (
        deceleration_time_s
        if deceleration_time_s is not None
        else (
            _DESCENT_DECELERATION_TIME_S
            if descending else _CARTESIAN_DECELERATION_TIME_S
        )
    )
    # A nonzero boundary argument means that this waypoint is a pass-through,
    # not a stop.  Preserve its requested speed (capped by the new global
    # single-axis limit) so adjacent lift/transfer/lower segments join without
    # dropping their scalar speed to zero.
    initial_speed = min(max(start_speed, 0.0), axis_speed_limit)
    final_speed = min(max(end_speed, 0.0), axis_speed_limit)

    def ramp_distance(candidate_peak: float) -> float:
        result = 0.0
        if candidate_peak > initial_speed + 1e-12:
            result += 0.5 * (
                initial_speed + candidate_peak
            ) * nominal_acceleration_time
        if candidate_peak > final_speed + 1e-12:
            result += 0.5 * (
                candidate_peak + final_speed
            ) * nominal_deceleration_time
        return result

    peak_floor = max(initial_speed, final_speed)
    full_speed_ramp_distance = ramp_distance(axis_speed_limit)
    if axis_distance >= full_speed_ramp_distance - 1e-12:
        peak_speed = axis_speed_limit
    else:
        lower = peak_floor
        upper = axis_speed_limit
        if ramp_distance(lower) > axis_distance + 1e-12:
            # The segment is physically too short for the requested incoming
            # speed and nominal ramp duration.  Keep it continuous and shorten
            # only the necessary ramp rather than introducing a stop.
            peak_speed = peak_floor
        else:
            for _ in range(64):
                middle = 0.5 * (lower + upper)
                if ramp_distance(middle) <= axis_distance:
                    lower = middle
                else:
                    upper = middle
            peak_speed = lower

    acceleration_time = (
        nominal_acceleration_time
        if peak_speed > initial_speed + 1e-12 else 0.0
    )
    deceleration_time = (
        nominal_deceleration_time
        if peak_speed > final_speed + 1e-12 else 0.0
    )
    acceleration_distance = (
        0.5 * (initial_speed + peak_speed) * acceleration_time
    )
    deceleration_distance = (
        0.5 * (peak_speed + final_speed) * deceleration_time
    )
    if acceleration_distance + deceleration_distance > axis_distance + 1e-12:
        scale = axis_distance / (acceleration_distance + deceleration_distance)
        acceleration_time *= scale
        deceleration_time *= scale
        acceleration_distance *= scale
        deceleration_distance *= scale
    cruise_distance = max(
        axis_distance - acceleration_distance - deceleration_distance,
        0.0,
    )
    cruise_time = cruise_distance / max(peak_speed, 1e-12)
    duration = acceleration_time + cruise_time + deceleration_time
    rotation_duration = (
        _SCURVE_DURATION_FACTOR * angle / limits.angular_speed_rad_s
        if angle > 1e-12 else 0.0
    )
    scheduled_duration = max(
        duration,
        rotation_duration,
        fixed_duration_s or 0.0,
    )
    steps = max(1, int(math.ceil(
        scheduled_duration * limits.command_rate_hz,
    )))
    sampled_duration = steps / limits.command_rate_hz
    cruise_end_time = acceleration_time + cruise_time
    cruise_distance = peak_speed * cruise_time

    for step in range(1, steps + 1):
        elapsed = step / limits.command_rate_hz
        # Stretch the analytic profile by at most one command interval so its
        # endpoint lands exactly on the 10 Hz grid.  This avoids a tiny final
        # action followed by a speed jump at a pass-through waypoint.
        profile_elapsed = elapsed * duration / sampled_duration
        if profile_elapsed <= acceleration_time:
            axis_travelled = (
                initial_speed * profile_elapsed
                + (peak_speed - initial_speed)
                * _raised_cosine_integral(
                    profile_elapsed, acceleration_time,
                )
            )
        elif profile_elapsed <= cruise_end_time:
            axis_travelled = (
                acceleration_distance
                + peak_speed * (profile_elapsed - acceleration_time)
            )
        else:
            deceleration_elapsed = profile_elapsed - cruise_end_time
            axis_travelled = (
                acceleration_distance
                + cruise_distance
                + peak_speed * deceleration_elapsed
                + (final_speed - peak_speed)
                * _raised_cosine_integral(
                    deceleration_elapsed, deceleration_time,
                )
            )
        fraction = float(np.clip(axis_travelled / axis_distance, 0.0, 1.0))
        position = start_position + fraction * delta
        u = step / steps
        rotation_fraction = _s_curve_progress(u)
        if angle > 1e-9:
            rotation = start_rotation @ axis_angle_matrix(
                relative_vector / angle, rotation_fraction * angle,
            )
        else:
            rotation = start_rotation.copy()
        frames.append(TaskFrame(position.copy(), rotation.copy(), gripper_closed, phase))


def append_rest_to_rest_motion(
    frames: list[TaskFrame],
    start_position: np.ndarray,
    start_rotation: np.ndarray,
    goal_position: np.ndarray,
    goal_rotation: np.ndarray,
    gripper_closed: bool,
    phase: str,
    limits: MotionLimits,
) -> None:
    """Append one 20% acceleration, 60% cruise, 20% deceleration move."""
    delta = goal_position - start_position
    distance = float(np.linalg.norm(delta))
    relative_vector = rotation_vector(start_rotation.T @ goal_rotation)
    angle = float(np.linalg.norm(relative_vector))
    if distance <= 1e-12 and angle <= 1e-12:
        return
    duration = max(
        (
            _SCURVE_DURATION_FACTOR * distance / limits.linear_speed_m_s
            if distance > 1e-12 else 0.0
        ),
        (
            _SCURVE_DURATION_FACTOR * angle / limits.angular_speed_rad_s
            if angle > 1e-12 else 0.0
        ),
        1.0 / limits.command_rate_hz,
    )
    steps = max(1, int(math.ceil(duration * limits.command_rate_hz)))
    for step in range(1, steps + 1):
        fraction = _s_curve_progress(step / steps)
        position = start_position + fraction * delta
        rotation = (
            start_rotation @ axis_angle_matrix(
                relative_vector / angle, fraction * angle,
            )
            if angle > 1e-12 else start_rotation.copy()
        )
        frames.append(TaskFrame(
            position.copy(), rotation.copy(), gripper_closed, phase,
        ))


@dataclass(frozen=True)
class PickResult:
    frames: tuple[TaskFrame, ...]
    grasp_position: np.ndarray
    grasp_rotation: np.ndarray
    lifted_position: np.ndarray
    object_vertical_half_extent: float
    object_horizontal_radius: float
    gripper_offset_z: float
    object_label: str
    object_body_id: int


def _nearest_downward_rotation(current_rotation: np.ndarray) -> np.ndarray:
    """Keep the current horizontal finger direction while pointing tool Z down."""
    closing_axis = current_rotation[:, 1].copy()
    closing_axis[2] = 0.0
    if np.linalg.norm(closing_axis) < 1e-9:
        closing_axis = np.array((0.0, 1.0, 0.0))
    closing_axis /= np.linalg.norm(closing_axis)
    tool_z = np.array((0.0, 0.0, -1.0))
    tool_x = np.cross(closing_axis, tool_z)
    tool_x /= np.linalg.norm(tool_x)
    return np.column_stack((tool_x, closing_axis, tool_z))


def _pregrasp_vertical_profile(
    distance_m: float,
) -> tuple[float, float, float, float, float, float]:
    """Return duration, peak/end speeds and ramp/cruise durations."""
    distance = max(float(distance_m), 0.0)
    ramp = _PREGRASP_VERTICAL_RAMP_TIME_S
    handoff = _FINAL_GRASP_DESCENT_SPEED_M_S
    if distance <= 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if distance < 0.5 * handoff * ramp:
        end_speed = 2.0 * distance / ramp
        return ramp, end_speed, end_speed, ramp, 0.0, 0.0
    if distance < 0.060:
        cruise = (distance - 0.5 * handoff * ramp) / handoff
        return ramp + cruise, handoff, handoff, ramp, cruise, 0.0
    peak = min(0.100, distance - 0.020)
    ramp_distance = peak + 0.020
    cruise = max((distance - ramp_distance) / peak, 0.0)
    return 2.0 * ramp + cruise, peak, handoff, ramp, cruise, ramp


def _pregrasp_vertical_travel(
    elapsed_s: float,
    peak_speed_m_s: float,
    end_speed_m_s: float,
    acceleration_time_s: float,
    cruise_time_s: float,
    deceleration_time_s: float,
) -> float:
    elapsed = max(float(elapsed_s), 0.0)
    acceleration_distance = 0.5 * peak_speed_m_s * acceleration_time_s
    if elapsed <= acceleration_time_s:
        return peak_speed_m_s * _raised_cosine_integral(
            elapsed, acceleration_time_s,
        )
    elapsed -= acceleration_time_s
    if elapsed <= cruise_time_s:
        return acceleration_distance + peak_speed_m_s * elapsed
    elapsed -= cruise_time_s
    elapsed = min(elapsed, deceleration_time_s)
    return (
        acceleration_distance
        + peak_speed_m_s * cruise_time_s
        + peak_speed_m_s * elapsed
        + (end_speed_m_s - peak_speed_m_s)
        * _raised_cosine_integral(elapsed, deceleration_time_s)
    )


def _append_coordinated_pregrasp(
    frames: list[TaskFrame],
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    grasp_position: np.ndarray,
    grasp_rotation: np.ndarray,
    object_label: str,
    limits: MotionLimits,
) -> tuple[np.ndarray, float]:
    """Coordinate horizontal travel with a safe vertical pregrasp approach."""
    start_position = np.asarray(current_position, dtype=float).copy()
    start_rotation = np.asarray(current_rotation, dtype=float).copy()
    pregrasp_z = float(grasp_position[2] + _PREGRASP_HEIGHT_M)

    # A pose at or below the grasp height is an exceptional recovery case:
    # raise vertically before approaching horizontally rather than sweeping
    # the open gripper through the table plane.
    if start_position[2] <= grasp_position[2] + 1e-9:
        raised = start_position.copy()
        raised[2] = pregrasp_z
        append_blended_motion(
            frames,
            start_position,
            start_rotation,
            raised,
            start_rotation,
            False,
            f"Raise before approaching {object_label}",
            limits,
            start_speed_m_s=0.0,
            end_speed_m_s=0.0,
        )
        start_position = raised

    descend_during_translation = max(start_position[2] - pregrasp_z, 0.0)
    handoff_z = (
        pregrasp_z
        if descend_during_translation > 0.0
        else float(start_position[2])
    )
    (
        active_duration,
        peak_speed,
        handoff_speed,
        acceleration_time,
        cruise_time,
        deceleration_time,
    ) = _pregrasp_vertical_profile(descend_during_translation)

    horizontal_target = np.array((
        grasp_position[0], grasp_position[1], start_position[2],
    ))
    first_index = len(frames)
    append_blended_motion(
        frames,
        start_position,
        start_rotation,
        horizontal_target,
        grasp_rotation,
        False,
        f"Move above {object_label}",
        limits,
        start_speed_m_s=0.0,
        end_speed_m_s=0.0,
        fixed_duration_s=(active_duration if active_duration > 0.0 else None),
    )
    motion_frames = frames[first_index:]
    if not motion_frames and active_duration > 0.0:
        steps = max(1, int(math.ceil(
            active_duration * limits.command_rate_hz,
        )))
        frames.extend(TaskFrame(
            horizontal_target.copy(), grasp_rotation.copy(), False,
            f"Move above {object_label}",
        ) for _ in range(steps))
        motion_frames = frames[first_index:]

    if descend_during_translation > 0.0 and motion_frames:
        total_duration = len(motion_frames) / limits.command_rate_hz
        delay = max(total_duration - active_duration, 0.0)
        for offset, frame in enumerate(tuple(motion_frames)):
            elapsed = (offset + 1) / limits.command_rate_hz
            travelled = _pregrasp_vertical_travel(
                max(elapsed - delay, 0.0),
                peak_speed,
                handoff_speed,
                acceleration_time,
                cruise_time,
                deceleration_time,
            )
            position = frame.position.copy()
            position[2] = max(
                handoff_z,
                float(start_position[2] - travelled),
            )
            frames[first_index + offset] = TaskFrame(
                position, frame.rotation, frame.gripper_closed, frame.phase,
            )
        final_frame = frames[-1]
        final_position = final_frame.position.copy()
        final_position[2] = handoff_z
        frames[-1] = TaskFrame(
            final_position,
            final_frame.rotation,
            final_frame.gripper_closed,
            final_frame.phase,
        )

    handoff_position = np.array((
        grasp_position[0], grasp_position[1], handoff_z,
    ))
    return handoff_position, handoff_speed


def _append_final_grasp_descent(
    frames: list[TaskFrame],
    start_position: np.ndarray,
    grasp_position: np.ndarray,
    grasp_rotation: np.ndarray,
    object_label: str,
    limits: MotionLimits,
    incoming_speed_m_s: float,
) -> None:
    """Append the canonical two-second final descent into a grasp."""
    distance = float(start_position[2] - grasp_position[2])
    if distance <= 1e-12:
        return
    steps = max(1, int(round(
        _FINAL_GRASP_DESCENT_DURATION_S * limits.command_rate_hz,
    )))
    canonical = (
        abs(distance - _PREGRASP_HEIGHT_M) <= 1e-9
        and incoming_speed_m_s
        >= _FINAL_GRASP_DESCENT_SPEED_M_S - 1e-9
    )
    for step in range(1, steps + 1):
        elapsed = step / limits.command_rate_hz
        if canonical:
            travelled = _final_descent_distance(elapsed)
            fraction = travelled / _PREGRASP_HEIGHT_M
        else:
            fraction = _s_curve_progress(step / steps)
        position = start_position + float(np.clip(fraction, 0.0, 1.0)) * (
            grasp_position - start_position
        )
        frames.append(TaskFrame(
            position.copy(), grasp_rotation.copy(), False,
            f"Descend to {object_label}",
        ))


def _plan_top_pick(
    center: np.ndarray,
    vertical_half_extent: float,
    horizontal_radius: float,
    gripper_envelope: GripperVerticalEnvelope,
    grasp_rotation: np.ndarray,
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    limits: MotionLimits,
    approach_clearance_m: float,
    lift_clearance_m: float,
    object_label: str,
    object_body_id: int,
    descent_speed_m_s: float = _GRASP_DESCENT_LINEAR_SPEED_M_S,
    lift_speed_m_s: float = _LIFT_LINEAR_SPEED_M_S,
    grasp_height_bias_m: float = 0.0,
    gripper_offset_z_m: float | None = None,
) -> PickResult:
    deepest, shallowest = gripper_envelope.object_center_offset_range(
        vertical_half_extent,
    )
    gripper_offset_z = (
        float(0.5 * (deepest + shallowest) + grasp_height_bias_m)
        if gripper_offset_z_m is None
        else float(gripper_offset_z_m)
    )
    grasp_position = center.copy()
    grasp_position[2] += gripper_offset_z
    lifted_position = grasp_position.copy()
    lifted_position[2] += lift_clearance_m

    frames: list[TaskFrame] = []
    lift_limits = _limited_linear_speed(limits, lift_speed_m_s)
    _ = approach_clearance_m, descent_speed_m_s
    handoff_position, handoff_speed = _append_coordinated_pregrasp(
        frames,
        current_position,
        current_rotation,
        grasp_position,
        grasp_rotation,
        object_label,
        limits,
    )
    _append_final_grasp_descent(
        frames,
        handoff_position,
        grasp_position,
        grasp_rotation,
        object_label,
        limits,
        handoff_speed,
    )
    append_dwell(
        frames,
        grasp_position,
        grasp_rotation,
        False,
        f"Align at {object_label}",
        gripper_preclose_dwell_ticks(limits),
    )
    append_dwell(
        frames, grasp_position, grasp_rotation, True, "Close gripper",
        gripper_close_dwell_ticks(limits),
    )
    append_blended_motion(
        frames, grasp_position, grasp_rotation,
        lifted_position, grasp_rotation, True, f"Lift {object_label}", lift_limits,
        start_speed_m_s=0.0, end_speed_m_s=lift_limits.linear_speed_m_s,
    )
    return PickResult(
        tuple(frames), grasp_position, grasp_rotation, lifted_position,
        vertical_half_extent, horizontal_radius, gripper_offset_z, object_label,
        object_body_id,
    )


class PickBox:
    """Top-grasp the one box requested by the current Blocks task."""

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str,
        geom_name: str,
        approach_clearance_m: float = 0.100,
        lift_clearance_m: float = 0.040,
        gripper_envelope: GripperVerticalEnvelope | None = None,
        grasp_height_bias_m: float = 0.0,
        gripper_offset_z_adjustment_m: float = 0.0,
        object_label: str = "block",
    ) -> None:
        self.model = model
        self.body_id = _body_id(model, body_name)
        self.geom_id = _geom_id(model, geom_name)
        self.gripper_envelope = gripper_envelope or panda_gripper_vertical_envelope(model)
        self.approach_clearance_m = approach_clearance_m
        self.lift_clearance_m = lift_clearance_m
        self.grasp_height_bias_m = float(grasp_height_bias_m)
        self.gripper_offset_z_adjustment_m = float(
            gripper_offset_z_adjustment_m
        )
        self.object_label = str(object_label)

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
        *,
        grasp_index: int | None = None,
    ) -> PickResult:
        center = data.xpos[self.body_id].copy()
        object_rotation = data.xmat[self.body_id].reshape(3, 3).copy()
        half_size = self.model.geom_size[self.geom_id, :3].copy()
        vertical_half_extent = float(np.sum(np.abs(object_rotation[2]) * half_size))
        horizontal_radius = float(np.linalg.norm(half_size[:2]))
        grasp_rotation = self._downward_grasp_rotation(object_rotation, half_size, current_rotation)
        if grasp_index is not None:
            rotations = self.grasp_rotations(object_rotation, half_size)
            if type(grasp_index) is not int or not 0 <= grasp_index < len(rotations):
                raise ValueError("Invalid box grasp index")
            grasp_rotation = rotations[grasp_index]
        return _plan_top_pick(
            center, vertical_half_extent, horizontal_radius,
            self.gripper_envelope, grasp_rotation,
            current_position, current_rotation, limits,
            self.approach_clearance_m, self.lift_clearance_m, self.object_label, self.body_id,
            gripper_offset_z_m=(
                self.gripper_envelope.upper_box_center_offset(
                    vertical_half_extent,
                    self.grasp_height_bias_m,
                )
                + self.gripper_offset_z_adjustment_m
            ),
        )

    @staticmethod
    def grasp_rotations(object_rotation, half_size):
        """Both signs of each physically equivalent narrow pinch axis."""
        axes = (0, 1) if abs(half_size[0] - half_size[1]) < 1e-6 else (int(half_size[1] < half_size[0]),)
        rotations = []
        for axis in axes:
            closing = object_rotation[:, axis].copy()
            closing[2] = 0.
            closing /= max(np.linalg.norm(closing), 1e-12)
            z = np.array((0., 0., -1.))
            for sign in (1., -1.):
                y = sign * closing
                rotations.append(np.column_stack((np.cross(y, z), y, z)))
        return tuple(rotations)

    @staticmethod
    def _downward_grasp_rotation(
        object_rotation: np.ndarray,
        half_size: np.ndarray,
        current_rotation: np.ndarray,
    ) -> np.ndarray:
        # Panda fingers close along tool Y.  Align that axis with the narrower
        # horizontal box axis, choosing the smaller wrist turn for a square.
        candidates = (object_rotation[:, 0].copy(), object_rotation[:, 1].copy())
        if abs(half_size[0] - half_size[1]) < 1e-6:
            axis_index = int(abs(candidates[1] @ current_rotation[:, 1]) > abs(candidates[0] @ current_rotation[:, 1]))
        else:
            axis_index = int(half_size[1] < half_size[0])
        closing_axis = candidates[axis_index]
        closing_axis[2] = 0.0
        closing_axis /= max(np.linalg.norm(closing_axis), 1e-12)
        if closing_axis @ current_rotation[:, 1] < 0.0:
            closing_axis *= -1.0
        tool_z = np.array((0.0, 0.0, -1.0))
        tool_x = np.cross(closing_axis, tool_z)
        tool_x /= max(np.linalg.norm(tool_x), 1e-12)
        return np.column_stack((tool_x, closing_axis, tool_z))


class PickSphere:
    """Top-grasp a sphere through its centre with a symmetric finger pose."""

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str,
        geom_name: str,
        approach_clearance_m: float = 0.100,
        lift_clearance_m: float = 0.040,
        gripper_envelope: GripperVerticalEnvelope | None = None,
    ) -> None:
        self.model = model
        self.body_id = _body_id(model, body_name)
        self.geom_id = _geom_id(model, geom_name)
        self.gripper_envelope = gripper_envelope or panda_gripper_vertical_envelope(model)
        self.approach_clearance_m = approach_clearance_m
        self.lift_clearance_m = lift_clearance_m

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> PickResult:
        radius = float(self.model.geom_size[self.geom_id, 0])
        return _plan_top_pick(
            data.xpos[self.body_id].copy(), radius, radius,
            self.gripper_envelope, _nearest_downward_rotation(current_rotation),
            current_position, current_rotation, limits,
            self.approach_clearance_m, self.lift_clearance_m, "ball", self.body_id,
            gripper_offset_z_m=(
                self.gripper_envelope.sphere_center_offset(radius)
            ),
        )


class PickCylinder:
    """Top-grasp the currently required upright circular cylinder."""

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str,
        geom_name: str,
        approach_clearance_m: float = 0.100,
        lift_clearance_m: float = 0.040,
        gripper_envelope: GripperVerticalEnvelope | None = None,
    ) -> None:
        self.model = model
        self.body_id = _body_id(model, body_name)
        self.geom_id = _geom_id(model, geom_name)
        self.gripper_envelope = gripper_envelope or panda_gripper_vertical_envelope(model)
        self.approach_clearance_m = approach_clearance_m
        self.lift_clearance_m = lift_clearance_m

    def plan(
        self,
        data: mujoco.MjData,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        limits: MotionLimits,
    ) -> PickResult:
        object_rotation = data.xmat[self.body_id].reshape(3, 3)
        if abs(float(object_rotation[2, 2])) < 0.95:
            raise RuntimeError("The current cylinder skill supports upright cylinders only.")
        radius = float(self.model.geom_size[self.geom_id, 0])
        half_height = float(self.model.geom_size[self.geom_id, 1])
        return _plan_top_pick(
            data.xpos[self.body_id].copy(), half_height, radius,
            self.gripper_envelope, _nearest_downward_rotation(current_rotation),
            current_position, current_rotation, limits,
            self.approach_clearance_m, self.lift_clearance_m, "cylinder", self.body_id,
            gripper_offset_z_m=(
                self.gripper_envelope.upper_box_center_offset(
                    half_height,
                    height_bias_m=(
                        0.006
                        if self.gripper_envelope.finger_contact_height_m
                        >= _NARROW_FINGERTIP_CONTACT_HEIGHT_THRESHOLD_M
                        else 0.0
                    ),
                )
            ),
        )


class PlaceOnPlate:
    """Move the held object above a fixed shallow receptacle and release it."""

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str,
        bottom_geom_name: str,
        top_geom_name: str,
        obstacle_body_names: tuple[str, ...] = (),
        approach_clearance_m: float = 0.040,
        obstacle_vertical_margin_m: float = 0.010,
        corridor_margin_m: float = 0.020,
        interior_site_name: str | None = None,
        receptacle_label: str = "plate",
    ) -> None:
        self.model = model
        self.body_id = _body_id(model, body_name)
        self.bottom_geom_id = _geom_id(model, bottom_geom_name)
        self.top_geom_id = _geom_id(model, top_geom_name)
        self.interior_site_id = (
            None
            if interior_site_name is None
            else _site_id(model, interior_site_name)
        )
        self.receptacle_label = receptacle_label
        self.obstacle_body_ids = tuple(
            body_id for name in obstacle_body_names
            if (body_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
            and body_id != self.body_id
        )
        self.approach_clearance_m = approach_clearance_m
        self.obstacle_vertical_margin_m = obstacle_vertical_margin_m
        self.corridor_margin_m = corridor_margin_m

    def _body_geom_ids(self, root_body_id: int) -> np.ndarray:
        descendants = []
        for candidate in range(self.model.nbody):
            ancestor = candidate
            while ancestor > root_body_id:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == root_body_id:
                descendants.append(candidate)
        return np.flatnonzero(np.isin(self.model.geom_bodyid, descendants))

    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        delta = end - start
        denominator = float(delta @ delta)
        if denominator <= 1e-16:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip((point - start) @ delta / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + fraction * delta)))

    def _transfer_height(
        self,
        data: mujoco.MjData,
        pick: PickResult,
        destination_xy: np.ndarray,
        minimum_height: float,
    ) -> float:
        """Find a conservative straight-line height over active scene objects."""
        required_height = minimum_height
        # The closed gripper can extend below O_T_EE farther than a small ball;
        # retain a conservative 60 mm lower swept extent in that case.
        carried_below_ee = max(
            pick.object_vertical_half_extent + pick.gripper_offset_z,
            0.060,
        )
        carried_radius = max(pick.object_horizontal_radius, 0.045)
        start_xy = pick.grasp_position[:2]

        for body_id in self.obstacle_body_ids:
            if body_id == pick.object_body_id:
                continue
            geom_ids = self._body_geom_ids(body_id)
            geom_ids = geom_ids[
                (self.model.geom_contype[geom_ids] != 0)
                | (self.model.geom_conaffinity[geom_ids] != 0)
            ]
            if geom_ids.size == 0:
                continue
            intersects_corridor = False
            for geom_id in geom_ids:
                obstacle_radius = float(self.model.geom_rbound[geom_id])
                distance = self._point_segment_distance(
                    data.geom_xpos[geom_id, :2], start_xy, destination_xy,
                )
                if distance <= carried_radius + obstacle_radius + self.corridor_margin_m:
                    intersects_corridor = True
                    break
            if not intersects_corridor:
                continue
            obstacle_top = max(
                float(data.geom_xpos[geom_id, 2] + self.model.geom_rbound[geom_id])
                for geom_id in geom_ids
            )
            required_height = max(
                required_height,
                obstacle_top + carried_below_ee + self.obstacle_vertical_margin_m,
            )
        return required_height

    def plan(
        self,
        data: mujoco.MjData,
        pick: PickResult,
        limits: MotionLimits,
    ) -> tuple[TaskFrame, ...]:
        plate_center = data.xpos[self.body_id].copy()
        if self.interior_site_id is None:
            available_radius = float(
                self.model.geom_size[self.bottom_geom_id, 0]
            )
        else:
            available_radius = float(np.min(
                self.model.site_size[self.interior_site_id, :2]
            ))
        if pick.object_horizontal_radius + 0.010 >= available_radius:
            raise RuntimeError(
                f"The {pick.object_label} does not fit inside the "
                f"{self.receptacle_label} with a safe margin."
            )
        plate_top = float(
            data.geom_xpos[self.top_geom_id, 2]
            + _geom_world_vertical_half_extent(
                self.model, data, self.top_geom_id,
            )
        )
        release_surface_top = plate_top
        if self.interior_site_id is not None:
            # A tray must be approached with the held object above its rim;
            # after opening, gravity settles the object onto the bottom.
            # Driving the gripper to the bottom-placement height makes its
            # fingers intersect the raised walls for long lying objects.
            receptacle_geom_ids = self._body_geom_ids(self.body_id)
            receptacle_geom_ids = receptacle_geom_ids[
                (self.model.geom_contype[receptacle_geom_ids] != 0)
                | (self.model.geom_conaffinity[receptacle_geom_ids] != 0)
            ]
            release_surface_top = max(
                float(
                    data.geom_xpos[geom_id, 2]
                    + _geom_world_vertical_half_extent(
                        self.model, data, int(geom_id),
                    )
                )
                for geom_id in receptacle_geom_ids
            )
        place_position = np.array((
            plate_center[0],
            plate_center[1],
            release_surface_top + pick.object_vertical_half_extent + 0.002
            + pick.gripper_offset_z,
        ))
        minimum_transfer_height = max(
            float(pick.lifted_position[2]),
            float(place_position[2] + self.approach_clearance_m),
        )
        transfer_height = self._transfer_height(
            data, pick, plate_center[:2], minimum_transfer_height,
        )
        raised_position = pick.lifted_position.copy()
        raised_position[2] = transfer_height
        above_plate = np.array((plate_center[0], plate_center[1], transfer_height))

        # Once the object is grasped, all placement segments keep the exact
        # grasp orientation: in particular there is no additional tool-Z yaw.
        frames: list[TaskFrame] = []
        lift_limits = _limited_linear_speed(limits, _LIFT_LINEAR_SPEED_M_S)
        transfer_limits = _limited_linear_speed(limits, _TRANSFER_LINEAR_SPEED_M_S)
        precision_limits = _limited_linear_speed(limits, _PRECISION_LINEAR_SPEED_M_S)
        lift_speed = lift_limits.linear_speed_m_s
        precision_speed = precision_limits.linear_speed_m_s
        if transfer_height > pick.lifted_position[2] + 1e-9:
            append_blended_motion(
                frames, pick.lifted_position, pick.grasp_rotation,
                raised_position, pick.grasp_rotation, True,
                "Raise to transfer height", lift_limits,
                start_speed_m_s=lift_speed, end_speed_m_s=lift_speed,
            )
        append_blended_motion(
            frames, raised_position, pick.grasp_rotation,
            above_plate, pick.grasp_rotation, True,
            f"Move above {self.receptacle_label}", transfer_limits,
            start_speed_m_s=lift_speed, end_speed_m_s=precision_speed,
        )
        append_blended_motion(
            frames, above_plate, pick.grasp_rotation,
            place_position, pick.grasp_rotation, True,
            f"Lower into {self.receptacle_label}", precision_limits,
            start_speed_m_s=precision_speed, end_speed_m_s=0.0,
        )
        append_dwell(
            frames, place_position, pick.grasp_rotation,
            False, f"Release {pick.object_label}", 12,
        )
        return tuple(frames)
