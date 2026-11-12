"""Robot descriptions shared by teleoperation and scene compilation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RobotSpec:
    uid: str
    label: str
    source_xml: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    gripper_actuator: str
    ee_site: str
    home_q: np.ndarray
    joint_velocity_limits: np.ndarray
    link_body_names: tuple[str, ...]
    finger_joint_names: tuple[str, str]
    gripper_open_ctrl: float
    gripper_closed_ctrl: float
    gripper_open_at_upper: bool
    gripper_max_aperture_m: float
    gripper_endpoint_threshold_m: float
    base_body: str
    base_position_m: np.ndarray
    table_offset_m: np.ndarray
    wrist_camera_mount: str
    wrist_camera_pos: np.ndarray
    wrist_camera_quat: np.ndarray
    wrist_camera_trim_pos: np.ndarray
    wrist_camera_trim_pitch_deg: float
    gripper_base_above_ee_m: float
    gripper_contact_upper_above_ee_m: float
    gripper_open_finger_bottom_z_offset_m: float
    gripper_closed_finger_bottom_z_offset_m: float
    gripper_finger_contact_height_m: float
    gripper_force_multiplier: float = 1.0
    finger_friction_multiplier: float = 1.0
    manipulability_display_offset: float = 0.0
    self_collision_display_offset: float = 0.0

    def __post_init__(self) -> None:
        limits = np.asarray(self.joint_velocity_limits, dtype=float)
        if limits.shape != (self.arm_dof,):
            raise ValueError(
                f"{self.label} joint velocity limits must have shape "
                f"({self.arm_dof},)",
            )
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError(
                f"{self.label} joint velocity limits must be finite and positive",
            )

    @property
    def arm_dof(self) -> int:
        return len(self.joint_names)


PANDA = RobotSpec(
    uid="panda",
    label="Franka Panda",
    source_xml="assets/mujoco_menagerie/franka_emika_panda/panda.xml",
    joint_names=tuple(f"joint{i}" for i in range(1, 8)),
    actuator_names=tuple(f"actuator{i}" for i in range(1, 8)),
    gripper_actuator="actuator8",
    ee_site="ee_site",
    home_q=np.array((0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0)),
    joint_velocity_limits=np.array((
        2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61,
    )),
    link_body_names=tuple(f"link{i}" for i in range(8)),
    finger_joint_names=("finger_joint1", "finger_joint2"),
    gripper_open_ctrl=255.0,
    gripper_closed_ctrl=0.0,
    gripper_open_at_upper=True,
    gripper_max_aperture_m=0.080,
    gripper_endpoint_threshold_m=0.0005,
    base_body="link0",
    base_position_m=np.array((0.0, 0.0, 0.5)),
    table_offset_m=np.zeros(3),
    wrist_camera_mount="hand",
    wrist_camera_pos=np.array((0.058, 0.0, 0.010)),
    wrist_camera_quat=np.array((0.110616, 0.698401, 0.698401, 0.110616)),
    wrist_camera_trim_pos=np.array((0.0, 0.0, 0.004)),
    wrist_camera_trim_pitch_deg=0.0,
    gripper_base_above_ee_m=0.03743784885673351,
    gripper_contact_upper_above_ee_m=0.009,
    gripper_open_finger_bottom_z_offset_m=0.008858,
    gripper_closed_finger_bottom_z_offset_m=0.008858,
    gripper_finger_contact_height_m=0.017,
    manipulability_display_offset=0.90,
    self_collision_display_offset=1.70,
)

XARM7 = RobotSpec(
    uid="xarm7",
    label="UFACTORY xArm7",
    source_xml="assets/mujoco_menagerie/ufactory_xarm7/xarm7.xml",
    joint_names=tuple(f"joint{i}" for i in range(1, 8)),
    actuator_names=tuple(f"act{i}" for i in range(1, 8)),
    gripper_actuator="gripper",
    ee_site="link_tcp",
    home_q=np.array((0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)),
    joint_velocity_limits=np.full(7, math.pi),
    link_body_names=("link_base", *(f"link{i}" for i in range(1, 8))),
    finger_joint_names=("left_driver_joint", "right_driver_joint"),
    gripper_open_ctrl=0.0,
    gripper_closed_ctrl=255.0,
    gripper_open_at_upper=False,
    gripper_max_aperture_m=0.086,
    gripper_endpoint_threshold_m=0.0030,
    base_body="link_base",
    # Calibrated against Panda's tabletop 6-D IK coverage, not its Home point.
    base_position_m=np.array((0.050, 0.0, 0.760)),
    # Y=0 downward-grasp workspace quality centre: X=0.532 m.
    table_offset_m=np.array((0.042, 0.0, 0.0)),
    wrist_camera_mount="xarm_gripper_base_link",
    # Calibrated below the palm: both fingertips remain in the lower corners.
    # Opposite-side flange bracket: match the Panda/UR5e world-facing view
    # while keeping both fingers visible at the bottom of the RGB image.
    wrist_camera_pos=np.array((-0.086, 0.0, 0.068)),
    wrist_camera_quat=np.array((
        0.17106486, 0.68610244, -0.68610281, -0.17106486,
    )),
    wrist_camera_trim_pos=np.array((0.014, 0.0, -0.006)),
    wrist_camera_trim_pitch_deg=1.0,
    gripper_base_above_ee_m=0.067,
    gripper_contact_upper_above_ee_m=0.0484,
    gripper_open_finger_bottom_z_offset_m=-0.011363,
    gripper_closed_finger_bottom_z_offset_m=0.000998,
    gripper_finger_contact_height_m=0.037,
    gripper_force_multiplier=1.5,
    finger_friction_multiplier=1.5,
)

UR5E = RobotSpec(
    uid="ur5e",
    label="UR5e + Robotiq 2F-85",
    source_xml="assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml",
    joint_names=(
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ),
    actuator_names=("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
    gripper_actuator="rb_fingers_actuator",
    ee_site="rb_pinch",
    # Rotate the stock Menagerie home about the base so the tool starts over
    # the centreline of this project's +X tabletop rather than at +Y.
    # Centreline Home: EE Y ~= 0 and rotation vector ~= (180, 0, 0) deg.
    home_q=np.array((
        2.87568460,
        -math.pi / 2.0,
        math.pi / 2.0,
        -math.pi / 2.0,
        -math.pi / 2.0,
        -0.26590805,
    )),
    joint_velocity_limits=np.full(6, math.pi),
    link_body_names=("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"),
    finger_joint_names=("rb_left_driver_joint", "rb_right_driver_joint"),
    gripper_open_ctrl=0.0,
    gripper_closed_ctrl=255.0,
    gripper_open_at_upper=False,
    gripper_max_aperture_m=0.085,
    gripper_endpoint_threshold_m=0.0015,
    base_body="base",
    # Calibrated against Panda's tabletop 6-D IK coverage, not its Home point.
    base_position_m=np.array((0.0, 0.0, 0.740)),
    # Y=0 downward-grasp workspace quality centre: X=0.595 m.
    table_offset_m=np.array((0.105, 0.0, 0.0)),
    wrist_camera_mount="rb_base",
    wrist_camera_pos=np.array((0.050, 0.0, 0.068)),
    wrist_camera_quat=np.array((0.183013, 0.683013, 0.683013, 0.183013)),
    wrist_camera_trim_pos=np.array((0.0, 0.0, -0.006)),
    wrist_camera_trim_pitch_deg=1.0,
    gripper_base_above_ee_m=0.055,
    gripper_contact_upper_above_ee_m=0.0331,
    gripper_open_finger_bottom_z_offset_m=0.00442,
    gripper_closed_finger_bottom_z_offset_m=0.017718,
    gripper_finger_contact_height_m=0.0375,
    gripper_force_multiplier=1.5,
    finger_friction_multiplier=1.5,
)


ROBOT_SPECS = {spec.uid: spec for spec in (PANDA, XARM7, UR5E)}
ROBOT_CHOICES = tuple(ROBOT_SPECS)
ROBOT_LABELS = tuple(spec.label for spec in ROBOT_SPECS.values())
ROBOT_UID_BY_LABEL = {spec.label: spec.uid for spec in ROBOT_SPECS.values()}
DATASET_JOINT_STATE_DIMENSION = 7


def get_robot_spec(uid: str) -> RobotSpec:
    try:
        return ROBOT_SPECS[uid]
    except KeyError as error:
        raise ValueError(f"Unknown robot {uid!r}; choose from {', '.join(ROBOT_CHOICES)}") from error


def dataset_joint_state(value: np.ndarray, spec: RobotSpec) -> np.ndarray:
    """Serialize every supported arm into the shared seven-joint contract."""
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != spec.arm_dof:
        raise ValueError(
            f"{spec.label} joint state must have {spec.arm_dof} values, "
            f"got {vector.size}"
        )
    if vector.size > DATASET_JOINT_STATE_DIMENSION:
        raise ValueError("Robot has more joints than the dataset state contract")
    if vector.size == DATASET_JOINT_STATE_DIMENSION:
        return vector.copy()
    return np.pad(
        vector,
        (0, DATASET_JOINT_STATE_DIMENSION - vector.size),
        mode="constant",
    )
