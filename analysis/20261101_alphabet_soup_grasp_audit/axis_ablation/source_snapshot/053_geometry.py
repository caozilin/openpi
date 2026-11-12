from __future__ import annotations

import math

import mujoco
import numpy as np

from .config import ANGULAR_SPEED, LINEAR_SPEED


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return MuJoCo's principal SO(3) logarithm."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    result = np.empty(3)
    # For dt=1, quaternion angular velocity is the principal rotation
    # vector. MuJoCo's quaternion path handles both zero and pi without the
    # ill-conditioned angle/sin(angle) matrix-log quotient.
    mujoco.mju_quat2Vel(result, quaternion, 1.0)
    return result


def canonical_rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return one deterministic principal rotvec for a fixed orientation.

    Fixed labels such as a tolerance frame must never use lifted 2*pi
    branches.  At the otherwise ambiguous exact-pi boundary, choose the axis
    sign whose first non-zero component is positive.
    """
    result = rotation_vector(rotation)
    angle = float(np.linalg.norm(result))
    if abs(angle - math.pi) <= 1.0e-9:
        significant = np.flatnonzero(np.abs(result) > 1.0e-12)
        if significant.size and result[int(significant[0])] < 0.0:
            result = -result
    return result


def continuous_rotation_vector(
    rotation: np.ndarray,
    previous: np.ndarray,
    reference: np.ndarray | None = None,
    reference_weight: float = 0.0,
) -> np.ndarray:
    """Choose the equivalent rotvec branch consistent with a time series.

    ``previous`` lifts the principal SO(3) logarithm onto the closest 2*pi
    branch. ``reference`` can additionally keep two nearby pose series, such
    as actual and nominal EE orientation, on the same branch.
    """
    principal = rotation_vector(rotation)
    previous = np.asarray(previous, dtype=float)
    if previous.shape != (3,):
        raise ValueError("Previous rotation vector must have shape (3,)")
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (3,):
            raise ValueError("Reference rotation vector must have shape (3,)")
    if reference_weight < 0.0:
        raise ValueError("Reference weight must be non-negative")

    previous_norm = float(np.linalg.norm(previous))
    angle = float(np.linalg.norm(principal))
    if angle < 1e-12:
        if previous_norm < 1e-12:
            candidates = [np.zeros(3)]
        else:
            axis = previous / previous_norm
            center = int(round(previous_norm / (2.0 * math.pi)))
            candidates = [2.0 * math.pi * k * axis for k in range(center - 3, center + 4)]
            candidates.append(np.zeros(3))
    else:
        axis = principal / angle
        projected_previous = float(previous @ axis)
        center = int(round((projected_previous - angle) / (2.0 * math.pi)))
        candidates = [principal + 2.0 * math.pi * k * axis for k in range(center - 4, center + 5)]

    def score(candidate: np.ndarray) -> float:
        value = float(np.linalg.norm(candidate - previous))
        if reference is not None and reference_weight > 0.0:
            value += reference_weight * float(np.linalg.norm(candidate - reference))
        return value

    return min(candidates, key=score).copy()


def rotation_matrix(vector: np.ndarray) -> np.ndarray:
    """Return SO(3) exponential coordinates as a rotation matrix."""
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    return axis_angle_matrix(vector / angle, angle)


def fixed_xyz_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Return the fixed-axis XYZ/RPY rotation ``Rz(yaw) Ry(pitch) Rx(roll)``.

    The three angles are ordered ``[roll_x, pitch_y, yaw_z]``. They are
    active rotations about fixed axes: X is applied first, followed by Y and
    then Z. This bounded tolerance chart is deliberately separate from
    :func:`rotation_matrix`, whose input is one exponential-coordinate vector.
    """
    value = np.asarray(angles, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Fixed-axis XYZ angles must contain three finite values")
    roll, pitch, yaw = (float(item) for item in value)
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    return np.array((
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    ))


def fixed_xyz_rotation_angles(rotation: np.ndarray) -> np.ndarray:
    """Return principal fixed-axis XYZ/RPY angles for a rotation matrix.

    The result ``[roll_x, pitch_y, yaw_z]`` satisfies
    ``R = Rz(yaw_z) @ Ry(pitch_y) @ Rx(roll_x)``. The principal pitch lies
    in ``[-pi/2, pi/2]``. Authored TaskTol pitch bounds remain well inside
    this chart; the deterministic gimbal branch handles telemetry robustly.
    """
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Fixed-axis XYZ extraction requires a finite 3x3 matrix")
    horizontal = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    pitch = math.atan2(-float(matrix[2, 0]), horizontal)
    if horizontal > 1.0e-10:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        # At pitch +/- pi/2, roll and yaw are coupled. Choose yaw=0 and the
        # unique roll that reconstructs the same matrix on that branch.
        yaw = 0.0
        if matrix[2, 0] < 0.0:
            roll = math.atan2(float(matrix[0, 1]), float(matrix[0, 2]))
        else:
            roll = math.atan2(-float(matrix[0, 1]), -float(matrix[0, 2]))
    return np.array((roll, pitch, yaw))


def rotation_tolerance_coordinates(
    actual: np.ndarray,
    target: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Return actual-minus-target fixed-axis XYZ coordinates in ``frame``.

    ``frame`` is the current (outer-cycle frozen) tolerance frame in base/world coordinates::

        Delta_F = F.T @ actual @ target.T @ F
        Delta_F = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    actual_array = np.asarray(actual, dtype=float)
    target_array = np.asarray(target, dtype=float)
    frame_array = np.asarray(frame, dtype=float)
    if any(value.shape != (3, 3) for value in (
        actual_array, target_array, frame_array,
    )):
        raise ValueError("Tolerance coordinates require three 3x3 rotations")
    return fixed_xyz_rotation_angles(
        frame_array.T @ actual_array @ target_array.T @ frame_array,
    )


def nominal_rotation_coordinates(
    actual: np.ndarray,
    nominal: np.ndarray,
    tolerance_frame: np.ndarray,
) -> np.ndarray:
    """Measure nominal-relative release in the declared tolerance axes.

    R_nom defines the zero pose; F defines the axes. These are independent:
    e = Euler_XYZ(F.T @ R_actual @ R_nom.T @ F).
    Using R_nom.T @ R_actual instead silently measures in tool axes and can
    swap gravity-Z with tool-X when the gripper is sideways. Require F
    explicitly so constraints, gradients, bias and telemetry share this chart.
    """
    return rotation_tolerance_coordinates(actual, nominal, tolerance_frame)


def stage_reference_rotation(
    stage_handoff_rotation: np.ndarray,
    stage_handoff_nominal_rotation: np.ndarray,
    nominal_rotation: np.ndarray,
) -> np.ndarray:
    """Transport the physical stage handoff by nominal base-frame motion.

    The action integrator advances orientation by left-multiplying base-frame
    increments.  Apply the same accumulated increment to the accepted handoff
    pose so a rigidly grasped object follows the authored nominal motion:

    ``R_ref,k = (R_nom,k @ R_nom,s.T) @ R_opt,s``.
    """
    handoff = np.asarray(stage_handoff_rotation, dtype=float)
    handoff_nominal = np.asarray(
        stage_handoff_nominal_rotation, dtype=float,
    )
    nominal = np.asarray(nominal_rotation, dtype=float)
    if any(value.shape != (3, 3) for value in (
        handoff, handoff_nominal, nominal,
    )):
        raise ValueError("Stage reference requires three 3x3 rotations")
    nominal_motion = nominal @ handoff_nominal.T
    return nominal_motion @ handoff


def stage_relative_rotation_coordinates(
    actual_rotation: np.ndarray,
    nominal_rotation: np.ndarray,
    stage_handoff_rotation: np.ndarray,
    stage_handoff_nominal_rotation: np.ndarray,
    tolerance_frame: np.ndarray,
) -> np.ndarray:
    """Return the current stage-relative fixed-XYZ rotation coordinates."""
    reference = stage_reference_rotation(
        stage_handoff_rotation,
        stage_handoff_nominal_rotation,
        nominal_rotation,
    )
    return rotation_tolerance_coordinates(actual_rotation, reference, tolerance_frame)


def rotation_increment_coordinates(
    increment: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Express a base-frame exponential increment in a fixed XYZ frame."""
    value = np.asarray(increment, dtype=float)
    frame_array = np.asarray(frame, dtype=float)
    if value.shape != (3,) or frame_array.shape != (3, 3):
        raise ValueError("Rotation increment coordinates require a 3-vector and 3x3 frame")
    return fixed_xyz_rotation_angles(
        frame_array.T @ rotation_matrix(value) @ frame_array,
    )


def rotation_from_tolerance_coordinates(
    target: np.ndarray,
    frame: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Apply fixed-axis XYZ tolerance coordinates to an absolute target."""
    target_array = np.asarray(target, dtype=float)
    frame_array = np.asarray(frame, dtype=float)
    if target_array.shape != (3, 3) or frame_array.shape != (3, 3):
        raise ValueError("Tolerance reconstruction requires two 3x3 rotations")
    return (
        frame_array
        @ fixed_xyz_rotation_matrix(coordinates)
        @ frame_array.T
        @ target_array
    )


def rotation_tolerance_coordinate_jacobian(
    actual: np.ndarray,
    target: np.ndarray,
    frame: np.ndarray,
    angular_world_jacobian: np.ndarray,
) -> np.ndarray:
    """Map a world angular-velocity Jacobian to fixed-axis XYZ rates."""
    coordinates = rotation_tolerance_coordinates(actual, target, frame)
    _, pitch, yaw = (float(item) for item in coordinates)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    # omega_F = E(q) qdot for q = [roll_x, pitch_y, yaw_z].
    rate_to_spatial = np.array((
        (cz * cy, -sz, 0.0),
        (sz * cy, cz, 0.0),
        (-sy, 0.0, 1.0),
    ))
    angular = np.asarray(angular_world_jacobian, dtype=float)
    if angular.ndim != 2 or angular.shape[0] != 3:
        raise ValueError("Angular Jacobian must have shape (3, N)")
    # At the Euler chart's gimbal line the three rates are not uniquely
    # recoverable.  Keep the same Euler-coordinate residuals, but use the
    # minimum-norm damped solve instead of aborting the whole IK cycle.
    if abs(cy) <= 1.0e-8:
        damping = 1.0e-8
        gram = rate_to_spatial.T @ rate_to_spatial + damping * np.eye(3)
        return np.linalg.solve(
            gram, rate_to_spatial.T @ (np.asarray(frame).T @ angular),
        )
    return np.linalg.solve(rate_to_spatial, np.asarray(frame).T @ angular)


def nominal_rotation_coordinate_jacobian(
    actual: np.ndarray,
    nominal: np.ndarray,
    angular_world_jacobian: np.ndarray,
    tolerance_frame: np.ndarray,
) -> np.ndarray:
    """Jacobian for :func:`nominal_rotation_coordinates`."""
    actual_array = np.asarray(actual, dtype=float)
    nominal_array = np.asarray(nominal, dtype=float)
    if actual_array.shape != (3, 3) or nominal_array.shape != (3, 3):
        raise ValueError("Nominal rotation Jacobian requires two 3x3 rotations")
    return rotation_tolerance_coordinate_jacobian(
        actual_array,
        nominal_array,
        tolerance_frame,
        angular_world_jacobian,
    )


def rotation_6d_to_matrix(value: np.ndarray, name: str = "rotation 6D") -> np.ndarray:
    """Restore Zhou et al.'s two-column 6D representation to SO(3)."""
    value = np.asarray(value, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain six finite values")
    first, second = value[:3], value[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-8:
        raise ValueError(f"{name} first vector is degenerate")
    b1 = first / first_norm
    second_orthogonal = second - float(b1 @ second) * b1
    second_norm = float(np.linalg.norm(second_orthogonal))
    if second_norm <= 1e-8:
        raise ValueError(f"{name} second vector is degenerate")
    b2 = second_orthogonal / second_norm
    return np.column_stack((b1, b2, np.cross(b1, b2)))


def project_to_rotation_matrix(value: np.ndarray) -> np.ndarray:
    """Project a finite 3x3 matrix onto the nearest proper rotation."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Rotation projection requires a finite 3x3 matrix")
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def project_rotation_target(rotation: np.ndarray, tolerance_rotation: np.ndarray, active_rotations: np.ndarray) -> np.ndarray:
    """Remove released fixed-axis XYZ components from an absolute target."""
    relative = rotation_tolerance_coordinates(
        rotation, tolerance_rotation, tolerance_rotation,
    )
    relative *= np.asarray(active_rotations, dtype=float)
    return rotation_from_tolerance_coordinates(
        tolerance_rotation, tolerance_rotation, relative,
    )


def rotation_error_in_frame(actual: np.ndarray, target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Actual-minus-target fixed-axis XYZ tolerance coordinates.

    The compatibility name is retained for telemetry and controller setup.
    It no longer denotes SO(3)-log components when a tolerance frame is used.
    """
    return rotation_tolerance_coordinates(actual, target, reference)


def update_target(
    position: np.ndarray,
    rotation: np.ndarray,
    pressed: set[str],
    active_dofs: np.ndarray,
    dt: float,
    rotation_in_base: bool = False,
    tolerance_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    action = action_in_base_frame(
        action_from_pressed(pressed, dt), rotation, rotation_in_base,
    )
    return apply_action(
        position, rotation, action, active_dofs, tolerance_rotation,
    )


def action_in_base_frame(
    action: np.ndarray,
    reference_rotation: np.ndarray,
    already_in_base: bool,
) -> np.ndarray:
    """Convert a 6-D source action to the base-only runtime convention."""
    converted = np.asarray(action, dtype=float).copy()
    reference = np.asarray(reference_rotation, dtype=float)
    if converted.shape not in ((6,), (7,)) or reference.shape != (3, 3):
        raise ValueError("Action conversion requires 6/7 values and a 3x3 rotation")
    if not already_in_base:
        converted[3:6] = reference @ converted[3:6]
    return converted


def action_from_pressed(pressed: set[str], dt: float) -> np.ndarray:
    """Return the complete 6-D keyboard increment, without a DoF mask."""
    translation_signs = np.array((int("s" in pressed) - int("w" in pressed), int("d" in pressed) - int("a" in pressed), int("i" in pressed) - int("k" in pressed)), dtype=float)
    rotation_signs = np.array((int("q" in pressed) - int("e" in pressed), int("u" in pressed) - int("o" in pressed), int("j" in pressed) - int("l" in pressed)), dtype=float)
    return np.concatenate((LINEAR_SPEED * dt * translation_signs, ANGULAR_SPEED * dt * rotation_signs))


def apply_action(
    position: np.ndarray,
    rotation: np.ndarray,
    action: np.ndarray,
    active_dofs: np.ndarray,
    tolerance_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a base-frame 6-D action after masking active DoFs."""
    position = position.copy()
    action = np.asarray(action, dtype=float)
    position += action[:3] * active_dofs[:3]
    # Do not impose an artificial Cartesian workspace box here. The robot's
    # joint limits, collision model, and IK feasibility determine the actual
    # reachable region.

    action_rotation = action[3:6].copy()
    if tolerance_rotation is None:
        action_rotation *= active_dofs[3:]
        return position, rotation_matrix(action_rotation) @ rotation

    updated = rotation_matrix(action_rotation) @ rotation
    return position, project_rotation_target(updated, tolerance_rotation, active_dofs[3:])
