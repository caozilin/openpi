"""Optional OSC-inertia matching for full-reference native joint control.

This changes only the inertia mapping inside native run_controller. The native
JointPositionController still owns target decoding, interpolation, state updates
and cleanup; the native robot still clips and applies torques and steps physics.
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from robosuite.controllers.joint_pos import JointPositionController
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.control_utils import opspace_matrices, orientation_error, nullspace_torques


def matched_joint_inertia(mass, jacobian):
    """Match native uncoupled OSC inertia; retain all seven joint errors.

    Uses native pseudoinverse semantics, including at rank loss. Local task-error
    equivalence assumes uniform scalar gains; this is not an exact nonlinear
    equivalence to OSC or a global stability guarantee.
    """
    full, position, rotation, _ = opspace_matrices(
        mass, jacobian, jacobian[:3], jacobian[3:]
    )
    split = np.zeros((6, 6))
    split[:3, :3] = position
    split[3:, 3:] = rotation
    effective = mass + jacobian.T @ (split - full) @ jacobian
    if not np.all(np.isfinite(effective)):
        raise FloatingPointError("Nonfinite matched joint inertia")
    return effective


class InertiaMatchedJointPositionController(JointPositionController):
    """Native joint controller with an explicit, opt-in inertia correction."""

    @classmethod
    def from_native(cls, controller):
        # Adopt initialized state without running constructor/update/reset again.
        # No dependency source, global factory or other robot is modified.
        if type(controller) is not JointPositionController:
            raise TypeError("Expected an unmodified native JointPositionController")
        if controller.impedance_mode != "fixed":
            raise ValueError("Inertia matching requires fixed impedance")
        if not (np.allclose(controller.kp, controller.kp[0], rtol=0, atol=1e-12)
                and np.allclose(controller.kd, controller.kd[0], rtol=0, atol=1e-12)):
            raise ValueError("Inertia matching requires uniform scalar kp and kd")
        result = cls.__new__(cls)
        result.__dict__.update(controller.__dict__)
        result.inertia_diagnostics = dict(
            calls=0, min_effective_eigenvalue=None, max_symmetry_error=0.0,
            correction_norm_sum_nm=0.0, max_correction_norm_nm=0.0,
            torque_clipped_calls=0, joint_error_squared_sum=0.0,
            max_joint_error_rad=0.0,
        )
        return result

    def run_controller(self):
        native_torque = super().run_controller()
        effective = matched_joint_inertia(self.mass_matrix, self.J_full)
        # Recover native PD acceleration after native interpolation. This avoids
        # maintaining a second implementation of its setpoint handling.
        acceleration = np.linalg.solve(
            self.mass_matrix, native_torque - self.torque_compensation
        )
        correction = (effective - self.mass_matrix) @ acceleration
        self.torques = native_torque + correction
        if not np.all(np.isfinite(self.torques)):
            raise FloatingPointError("Nonfinite inertia-matched joint torque")
        eigenvalue = float(np.linalg.eigvalsh((effective + effective.T) / 2).min())
        if eigenvalue <= 0:
            raise FloatingPointError("Matched joint inertia is not positive definite")
        stats = self.inertia_diagnostics
        stats["calls"] += 1
        previous = stats["min_effective_eigenvalue"]
        stats["min_effective_eigenvalue"] = eigenvalue if previous is None else min(previous, eigenvalue)
        stats["max_symmetry_error"] = max(stats["max_symmetry_error"], float(np.max(np.abs(effective - effective.T))))
        norm = float(np.linalg.norm(correction))
        stats["correction_norm_sum_nm"] += norm
        stats["max_correction_norm_nm"] = max(stats["max_correction_norm_nm"], norm)
        stats["torque_clipped_calls"] += int(np.any(
            (self.torques < self.actuator_min) | (self.torques > self.actuator_max)
        ))
        error = self.goal_qpos - self.joint_pos
        stats["joint_error_squared_sum"] += float(error @ error)
        stats["max_joint_error_rad"] = max(stats["max_joint_error_rad"], float(np.max(np.abs(error))))
        return self.torques


class FactorizedReferenceController(OperationalSpaceController):
    """Experimental task/null factors inside the native OSC execution protocol.

    The joint target is explicit seven-dimensional state, set_joint_reference().
    Native set_goal still decodes the identical physical pose/gripper command.
    This adapter is for controlled ablations, not a second execution backend.
    """

    @classmethod
    def from_native(cls, controller, *, task_feedback, null_reference, null_kp=10.0, joint_kp=150.0):
        if type(controller) is not OperationalSpaceController:
            raise TypeError("Expected original native OSC controller")
        if any(name in controller.__dict__ for name in ("run_controller", "set_goal")):
            raise ValueError("Remove instance method instrumentation before cloning controller state")
        if task_feedback not in ("joint", "osc") or null_reference not in ("joint", "initial"):
            raise ValueError("Unknown task/null feedback factor")
        if not controller.uncoupling or controller.interpolator_pos is not None or controller.interpolator_ori is not None:
            raise ValueError("Ablation requires native uncoupled OSC without interpolation")
        if null_kp <= 0 or joint_kp <= 0:
            raise ValueError("Expected positive fixed gains")
        result=cls.__new__(cls)
        result.__dict__.update({k:(v if k=="sim" else deepcopy(v)) for k,v in controller.__dict__.items()})
        result.task_feedback=task_feedback
        result.null_reference=null_reference
        result.null_kp=float(null_kp)
        result.joint_kp=float(joint_kp)
        result.joint_target=None
        result.factor_diagnostics=None
        return result

    def set_joint_reference(self, target):
        target=np.asarray(target,dtype=float)
        if target.shape!=(7,) or not np.all(np.isfinite(target)):
            raise ValueError("Expected seven finite joint-reference coordinates")
        self.joint_target=target.copy()

    def run_controller(self):
        native=super().run_controller()
        full,position,rotation,null=opspace_matrices(self.mass_matrix,self.J_full,self.J_pos,self.J_ori)
        split=np.zeros((6,6))
        split[:3,:3]=position
        split[3:,3:]=rotation
        ux=self.kp*np.r_[self.goal_pos-self.ee_pos,orientation_error(self.goal_ori,self.ee_ori_mat)]-self.kd*np.r_[self.ee_pos_vel,self.ee_ori_vel]
        if self.joint_target is None:
            if self.task_feedback=="joint" or self.null_reference=="joint":
                raise RuntimeError("Joint-reference factor requires explicit full q_d")
            uq=np.zeros(7)
        else:
            uq=self.joint_kp*(self.joint_target-self.joint_pos)-2*np.sqrt(self.joint_kp)*self.joint_vel
        task=self.J_full.T @ split @ (ux if self.task_feedback=="osc" else self.J_full @ uq)
        null_target=self.initial_joint if self.null_reference=="initial" else self.joint_target
        posture=nullspace_torques(self.mass_matrix,null,null_target,self.joint_pos,self.joint_vel,joint_kp=self.null_kp)
        self.torques=task+posture+self.torque_compensation
        if not np.all(np.isfinite(self.torques)):
            raise FloatingPointError("Nonfinite factorized-reference torque")
        native_posture=nullspace_torques(self.mass_matrix,null,self.initial_joint,self.joint_pos,self.joint_vel)
        native_reconstructed=self.J_full.T @ split @ ux+native_posture+self.torque_compensation
        self.factor_diagnostics=dict(
            task_torque=task.copy(),null_torque=posture.copy(),native_torque=native.copy(),
            native_reconstruction_residual_nm=float(np.max(np.abs(native_reconstructed-native))),
            joint_task_feedback_minus_osc=self.J_full @ uq-ux,
            joint_null_error=None if self.joint_target is None else null @ (self.joint_target-self.joint_pos),
            native_initial_joint=self.initial_joint.copy(),
            full_task_inertia=full.copy(),split_task_inertia=split.copy(),null_projector=null.copy(),
        )
        return self.torques
