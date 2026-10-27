"""LIBERO IPOPT motion-generator client using the native execution loop.

No physics loop lives here. IPOPT, reference integration and rotational release
are imported from the maintained implementations; robosuite owns torque and
two-finger actuation. Assets and task success predicates remain native LIBERO.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import mujoco

from ..control.ipopt import IpoptIK, IpoptSettings
from ..control.pose import TargetPose
from ..control.task_space.objective import ObjectiveSettings
from ..geometry import rotation_vector
from ..tolerance.runtime import advance_reference_pose, solve_stage_relative_target, rebase_rotation_subtask
from ..tolerance.state import RotationalToleranceState
from ..tolerance.release_state import RotationReleaseState
from .robot import RobotHandles, ee_pose, find_handles
from .robots import PANDA


def libero_handles(env) -> RobotHandles:
    robot = env.robots[0]
    model = env.sim.model._model
    joints = tuple(robot.robot_model.joints)
    joint_ids = [model.joint(name).id for name in joints]
    bodies = [model.body("robot0_link0").id] + [int(model.jnt_bodyid[j]) for j in joint_ids]
    site = int(robot.eef_site_id)
    hand = model.body(robot.robot_model.eef_name).id
    spec = replace(
        PANDA,
        joint_names=joints,
        actuator_names=tuple(robot.robot_model.actuators),
        gripper_actuator=robot.gripper.actuators[0],
        ee_site=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site),
        link_body_names=tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies),
        finger_joint_names=tuple(robot.gripper.joints),
        gripper_open_ctrl=-1.0,
        gripper_closed_ctrl=1.0,
    )
    handles = find_handles(model, spec)
    data = env.sim.data._data
    offset = data.site_xmat[site].reshape(3, 3).T @ data.xmat[hand].reshape(3, 3)
    return replace(handles, ee_rotation_offset=offset.copy())


class LiberoTaskTolSession:
    """One stateful IPOPT client session, usable by replay and policy callers.

    Actions are physical base-frame pose increments [m, rotation-vector rad]
    followed by -1=open/+1=close. They are not normalized OSC commands.
    """

    def __init__(self, env, *, kp=None, solver_settings=None, joint_control_mode="native"):
        if joint_control_mode not in ("native", "osc_inertia_matched"):
            raise ValueError("Unknown joint control mode")
        self.joint_control_mode = joint_control_mode
        self.env = env
        self.model, self.data = env.sim.model._model, env.sim.data._data
        self.handles = libero_handles(env)
        self.dt = float(env.env.control_timestep)
        self.physics_steps = int(round(self.dt / self.model.opt.timestep))
        if not np.isclose(self.physics_steps * self.model.opt.timestep, self.dt):
            raise ValueError("LIBERO control interval must contain whole physical steps")
        self.controller = env.robots[0].controller
        if self.controller.name != "JOINT_POSITION":
            raise ValueError("Native JOINT_POSITION controller is required")
        if kp is not None:
            self.controller.kp[:] = kp
            self.controller.kd[:] = 2 * np.sqrt(kp)
        if joint_control_mode == "osc_inertia_matched":
            from .libero_joint_controller import InertiaMatchedJointPositionController

            self.controller = InertiaMatchedJointPositionController.from_native(self.controller)
            env.robots[0].controller = self.controller
        # Configure only command encoding. The full joint span ensures a
        # feasible absolute IPOPT target is representable without clipping.
        self.joint_action_span = self.handles.joint_upper - self.handles.joint_lower
        self.controller.output_max = self.joint_action_span.copy()
        self.controller.output_min = -self.joint_action_span.copy()
        self.controller.action_scale = None
        self.controller.scale_action(np.zeros(7))
        self.ik = IpoptIK(
            self.model,
            self.handles,
            solver_settings or IpoptSettings(max_iterations=60, max_cpu_time_s=0.5),
            ObjectiveSettings.delta_q_squared_conditioning_safeguard(),
        )
        position, rotation = ee_pose(self.data, self.handles)
        self.target = TargetPose(position.copy(), rotation.copy())
        self.frame_index = 0
        self.tolerance = RotationalToleranceState()
        self.release = RotationReleaseState(
            rotation_for_joint=lambda q: self.ik.kinematics.evaluate(
                q, include_manipulability=False, include_link_points=False
            ).rotation
        )
        self.stage_key = None
        self.subtask_index = None

    def native_joint_action(self, joint_target, gripper_command):
        """Encode an absolute joint target for native delta-position input."""
        # Let env.step perform its own normal controller-state update;
        # encoding must not introduce extra forward/physics calls.
        self.controller.new_update = True
        delta = np.asarray(joint_target) - self.data.qpos[self.handles.qpos_ids]
        normalized = (
            delta - self.controller.action_output_transform
        ) / self.controller.action_scale + self.controller.action_input_transform
        if np.any(normalized < self.controller.input_min - 1e-10) or np.any(
            normalized > self.controller.input_max + 1e-10
        ):
            raise ValueError("Joint target cannot be encoded without clipping")
        return np.r_[normalized, np.clip(gripper_command, -1.0, 1.0)]

    def observation(self):
        self.env.sim.forward()
        self.env.env._update_observables(force=True)
        return self.env.env._get_observations()

    def step(self, action, *, phase="APP", subtask_index=0, semantic_mask="000", bounds_rad=None):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected seven finite physical pose/gripper controls")
        if phase not in ("APP", "ENG", "EXEC", "DIS", "outside_subtask"):
            raise ValueError("Unknown manipulation phase")
        if len(semantic_mask) != 3 or any(c not in "01" for c in semantic_mask):
            raise ValueError("semantic_mask must contain three binary digits")
        bounds = None if bounds_rad is None else np.asarray(bounds_rad, dtype=float)
        if bounds is not None and (bounds.shape != (6,) or not np.all(np.isfinite(bounds)) or np.any(bounds < 0)):
            raise ValueError("Directional bounds must be six finite nonnegative radians")
        before_fk = self.ik.kinematics.evaluate(
            self.data.qpos[self.handles.qpos_ids], include_manipulability=False, include_link_points=False
        )
        before_position, before_rotation = before_fk.position, before_fk.rotation
        before_q = self.data.qpos[self.handles.qpos_ids].copy()
        before_fingers = self.data.qpos[self.handles.finger_qpos_ids].copy()
        key = (subtask_index, phase)
        tolerance_enabled = bounds is not None
        rebased = False
        if tolerance_enabled and self.subtask_index is not None and subtask_index != self.subtask_index:
            rebase_rotation_subtask(
                self.target,
                self.tolerance,
                self.release,
                stage_key=key,
                accepted_rotation=self.release.accepted_rotation,
            )
            rebased = True
        if key != self.stage_key:
            self.release.begin_stage(
                key,
                optimized_rotation=(self.target.rotation if self.stage_key is None else self.release.accepted_rotation),
                nominal_rotation=self.target.rotation,
            )
        if tolerance_enabled:
            from ..tolerance.frame import box_tolerance_frame

            # No alternate heading axis is substituted at degeneracy.
            projected = self.target.rotation[:2, 1]
            guarded = bool(np.linalg.norm(projected) < 1e-6)
            mask = np.asarray([c == "1" for c in semantic_mask])
            if phase not in ("APP", "EXEC") or guarded:
                mask[:] = False
            mask &= (bounds[::2] > 0) | (bounds[1::2] > 0)
            self.tolerance.ranged = mask
            stage = {"APP": 0, "ENG": 1, "EXEC": 2, "DIS": 3, "outside_subtask": 3}[phase]
            frame = np.eye(3) if guarded else box_tolerance_frame(self.target.rotation)
            self.tolerance.configure_stage(
                stage,
                self.target.rotation,
                frame,
                np.maximum(bounds[::2], bounds[1::2]),
                negative_limits=bounds[::2],
                positive_limits=bounds[1::2],
                stage_key=key,
            )
        self.target.position, self.target.rotation = advance_reference_pose(
            self.target.position,
            self.target.rotation,
            action,
            tolerance=self.tolerance if tolerance_enabled else None,
            dt=self.dt if tolerance_enabled else None,
        )
        if tolerance_enabled:
            # Construct this cycle's frame from the advanced independent
            # nominal tool orientation. At a singular heading keep all axes
            # strict; no alternate physical axis is substituted.
            guarded = bool(np.linalg.norm(self.target.rotation[:2, 1]) < 1e-6)
            if guarded:
                self.tolerance.ranged[:] = False
            self.tolerance.frame = np.eye(3) if guarded else box_tolerance_frame(self.target.rotation)
            self.tolerance.raw_tolerance_frame = self.tolerance.frame.copy()
        active = np.ones(6, dtype=bool)
        if tolerance_enabled:
            q, diagnostics, _, release_mask, release_step = solve_stage_relative_target(
                self.ik,
                self.data,
                self.target,
                self.tolerance,
                self.release,
                active,
                tolerance_frame_mode="stage_endpoint",
            )
        else:
            for axis in range(3, 6):
                self.ik.set_axis_task(axis, None)
            q, diagnostics = self.ik.solve(self.data, self.target, active)
            release_mask = np.zeros(3, dtype=bool)
            release_step = None
            self.release.accepted_rotation = self.ik.kinematics.evaluate(q).rotation
        # Infeasible IK is a terminal recording failure; never silently fall back to OSC.
        aborted = not bool(diagnostics.feasible)
        native_action = None
        physics_steps = 0
        if not aborted:
            native_action = self.native_joint_action(q, float(action[6]))
            before_time = float(self.data.time)
            obs, _, _, _ = self.env.step(native_action.tolist())
            physics_steps = int(round((float(self.data.time) - before_time) / float(self.model.opt.timestep)))
            if not np.allclose(self.controller.goal_qpos, q, atol=1e-10, rtol=0):
                raise RuntimeError("Native joint target differs from IPOPT output")
        else:
            obs = self.env.env._get_observations()
        after_fk = self.ik.kinematics.evaluate(
            self.data.qpos[self.handles.qpos_ids], include_manipulability=False, include_link_points=False
        )
        after_position, after_rotation = after_fk.position, after_fk.rotation
        optimized = self.ik.kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
        row = dict(
            index=self.frame_index,
            joint_control_mode=self.joint_control_mode,
            time_s=self.frame_index * self.dt,
            simulation_time_s=float(self.data.time),
            phase=phase,
            subtask_index=subtask_index,
            semantic_mask=semantic_mask,
            effective_mask="".join("1" if x else "0" for x in release_mask),
            subtask_rebased=rebased,
            action=action.tolist(),
            observation_timing="pre_action",
            pre_action=dict(
                position_m=before_position.tolist(),
                rotation_vector_rad=rotation_vector(before_rotation).tolist(),
                joint_position_rad=before_q.tolist(),
                finger_qpos_m=before_fingers.tolist(),
            ),
            nominal_target=dict(
                position_m=self.target.position.tolist(),
                rotation_vector_rad=rotation_vector(self.target.rotation).tolist(),
            ),
            joint_target_rad=q.tolist(),
            post_action=dict(
                position_m=after_position.tolist(),
                rotation_vector_rad=rotation_vector(after_rotation).tolist(),
                joint_position_rad=self.data.qpos[self.handles.qpos_ids].tolist(),
                finger_qpos_m=self.data.qpos[self.handles.finger_qpos_ids].tolist(),
            ),
            optimizer=dict(
                backend="ipopt",
                solver_called=bool(diagnostics.solver_called),
                feasible=bool(diagnostics.feasible),
                status=str(diagnostics.status),
                iterations=int(diagnostics.iterations),
                elapsed_ms=float(diagnostics.elapsed_ms),
                position_residual_m=float(diagnostics.position_residual),
                strict_rotation_residual_rad=float(diagnostics.strict_rotation_residual),
            ),
            tracking_position_error_m=float(np.linalg.norm(after_position - self.target.position)),
            tracking_rotation_error_rad=float(np.linalg.norm(rotation_vector(optimized.rotation.T @ after_rotation))),
            physics_steps=physics_steps,
            native_joint_action=None if native_action is None else native_action.tolist(),
            native_joint_goal_error_rad=None if aborted else float(np.max(np.abs(self.controller.goal_qpos - q))),
            bounds_rad=None if bounds is None else bounds.tolist(),
            heading_guarded=bool(guarded) if tolerance_enabled else False,
            success=bool(self.env.check_success()),
            aborted=aborted,
        )
        if release_step is not None:
            candidate = self.release.candidate(optimized.rotation, release_step)
            row["rotation_release_rad"] = candidate.stage_relative_tolerance_release_rad.tolist()
            row["predicted_tolerance_violation_rad"] = candidate.predicted_tolerance_violation_rad.tolist()
        self.frame_index += 1
        self.stage_key = key
        self.subtask_index = subtask_index
        return obs, row
