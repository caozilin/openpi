from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import (
    DEFAULT_FINGER_ROLLING_FRICTION,
    DEFAULT_FINGER_TORSIONAL_FRICTION,
    MAX_SAFE_GRIPPER_FORCE_SCALE,
)
from .robots import PANDA, RobotSpec


@dataclass(frozen=True)
class RobotHandles:
    spec: RobotSpec
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray
    gripper_id: int
    ee_site_id: int
    finger_joint_ids: np.ndarray
    finger_qpos_ids: np.ndarray
    link_body_ids: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    # Fixed rotation from a model's control site to the physical tool frame.
    # Identity preserves all existing robot definitions.
    ee_rotation_offset: np.ndarray = field(default_factory=lambda: np.eye(3), compare=False)


def find_handles(model: mujoco.MjModel, spec: RobotSpec = PANDA) -> RobotHandles:
    if np.asarray(spec.joint_velocity_limits).shape != (spec.arm_dof,):
        raise ValueError(
            f"{spec.label} joint velocity limits must have shape "
            f"({spec.arm_dof},)",
        )
    if (
        not np.all(np.isfinite(spec.joint_velocity_limits))
        or np.any(spec.joint_velocity_limits <= 0.0)
    ):
        raise ValueError(
            f"{spec.label} joint velocity limits must be finite and positive",
        )
    joint_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in spec.joint_names
    ])
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in spec.actuator_names]
    )
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.gripper_actuator)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site)
    finger_joint_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in spec.finger_joint_names
    ])
    link_body_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in spec.link_body_names
    ])
    if any(np.any(ids < 0) for ids in (joint_ids, actuator_ids, finger_joint_ids, link_body_ids)) or gripper_id < 0 or ee_site_id < 0:
        raise RuntimeError(f"The {spec.label} model is missing expected joints, actuators, links, gripper, or EE site.")
    return RobotHandles(
        spec=spec,
        joint_ids=joint_ids,
        qpos_ids=model.jnt_qposadr[joint_ids],
        dof_ids=model.jnt_dofadr[joint_ids],
        actuator_ids=actuator_ids,
        gripper_id=gripper_id,
        ee_site_id=ee_site_id,
        finger_joint_ids=finger_joint_ids,
        finger_qpos_ids=model.jnt_qposadr[finger_joint_ids].astype(int),
        link_body_ids=link_body_ids,
        joint_lower=model.jnt_range[joint_ids, 0].copy(),
        joint_upper=model.jnt_range[joint_ids, 1].copy(),
    )


def configure_contact_solver(model: mujoco.MjModel, max_timestep: float) -> None:
    if max_timestep <= 0.0:
        raise ValueError("--physics-timestep must be positive.")
    model.opt.timestep = min(float(model.opt.timestep), max_timestep)
    model.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_IMPLICITFAST", mujoco.mjtIntegrator.mjINT_IMPLICIT)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.iterations = max(int(model.opt.iterations), 50)
    model.opt.tolerance = 1e-10
    model.opt.impratio = max(float(model.opt.impratio), 5.0)
    model.opt.noslip_iterations = max(2, min(int(model.opt.noslip_iterations), 3))
    model.opt.noslip_tolerance = 1e-8


def configure_gripper(model: mujoco.MjModel, handles: RobotHandles, force_scale: float, max_force: float, finger_friction: float) -> None:
    if force_scale <= 0.0 or max_force <= 0.0 or finger_friction <= 0.0:
        raise ValueError("Gripper force scale, force cap and friction must be positive.")
    force_multiplier = float(handles.spec.gripper_force_multiplier)
    friction_multiplier = float(handles.spec.finger_friction_multiplier)
    scale = (
        min(force_scale, MAX_SAFE_GRIPPER_FORCE_SCALE)
        * force_multiplier
    )
    effective_max_force = float(max_force) * force_multiplier
    effective_finger_friction = (
        float(finger_friction) * friction_multiplier
    )
    model.actuator_gainprm[handles.gripper_id] *= scale
    model.actuator_biasprm[handles.gripper_id] *= scale
    model.actuator_forcelimited[handles.gripper_id] = 1
    model.actuator_forcerange[handles.gripper_id] = [
        -effective_max_force,
        effective_max_force,
    ]

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "finger" not in body_name.lower() and "pad" not in body_name.lower():
            continue
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        model.geom_friction[geom_id, 0] = max(
            float(model.geom_friction[geom_id, 0]),
            effective_finger_friction,
        )
        model.geom_friction[geom_id, 1] = max(float(model.geom_friction[geom_id, 1]), DEFAULT_FINGER_TORSIONAL_FRICTION)
        model.geom_friction[geom_id, 2] = max(float(model.geom_friction[geom_id, 2]), DEFAULT_FINGER_ROLLING_FRICTION)


def scaled_gripper_control_step(
    gripper_speed: float,
    physics_dt: float,
    speed_scale: float,
) -> float:
    """Return one physics-step gripper command change at the live speed."""
    values = np.asarray(
        (gripper_speed, physics_dt, speed_scale),
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Gripper speed inputs must be finite")
    if gripper_speed < 0.0 or physics_dt <= 0.0 or speed_scale <= 0.0:
        raise ValueError(
            "Gripper speed must be non-negative and time/scale positive",
        )
    return float(gripper_speed * physics_dt * speed_scale)


def set_gripper_fully_open(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: RobotHandles,
) -> None:
    """Place both fingers symmetrically at their maximum aperture immediately."""
    finger_joint_ids = handles.finger_joint_ids
    finger_qpos_ids = handles.finger_qpos_ids
    finger_dof_ids = model.jnt_dofadr[finger_joint_ids].astype(int)
    endpoint = 1 if handles.spec.gripper_open_at_upper else 0
    data.qpos[finger_qpos_ids] = model.jnt_range[finger_joint_ids, endpoint]
    data.qvel[finger_dof_ids] = 0.0
    data.qacc[finger_dof_ids] = 0.0
    data.qacc_warmstart[finger_dof_ids] = 0.0
    data.ctrl[handles.gripper_id] = handles.spec.gripper_open_ctrl
    mujoco.mj_forward(model, data)


def set_home(model: mujoco.MjModel, data: mujoco.MjData, handles: RobotHandles) -> None:
    data.qpos[handles.qpos_ids] = handles.spec.home_q
    data.ctrl[handles.actuator_ids] = handles.spec.home_q
    data.ctrl[handles.gripper_id] = handles.spec.gripper_open_ctrl
    mujoco.mj_forward(model, data)


def ee_pose(data: mujoco.MjData, handles: RobotHandles) -> tuple[np.ndarray, np.ndarray]:
    return (
        data.site_xpos[handles.ee_site_id].copy(),
        data.site_xmat[handles.ee_site_id].reshape(3, 3) @ handles.ee_rotation_offset,
    )
