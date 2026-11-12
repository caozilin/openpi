"""LIBERO actuator adapter for the canonical MuJoCo joint-reference path.

Uses the shared backend's interpolation/tracker and actual MuJoCo position
actuators. Native LIBERO still owns env.step, gripper actuation and success.
The robot API calls its ctrl output 'torques'; this adapter explicitly emits
joint angles into converted position actuators, whose force limits remain Nm.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from robosuite.controllers.base_controller import Controller
from robosuite.controllers.joint_pos import JointPositionController
from tasktol.control.joint_tracker import JointReferenceTracker
from tasktol.simulation.preset_runtime import PresetSimulationBackend
from tasktol.simulation.robots import PANDA

SHARED=Path('/media/czl/sata/franka_my_code/franka_mujoco')
SERVO_FIELDS=('actuator_dyntype','actuator_gaintype','actuator_biastype','actuator_dynprm',
    'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_ctrllimited',
    'actuator_forcerange','actuator_forcelimited')


class MuJoCoJointServoController(JointPositionController):
    @classmethod
    def from_native(cls,controller,handles,*,source_integrator=False):
        if type(controller) is not JointPositionController:
            raise TypeError('Expected native JOINT_POSITION command transport')
        if controller.impedance_mode!='fixed' or controller.interpolator is not None:
            raise ValueError('Expected fixed transport without a second interpolator')
        result=cls.__new__(cls)
        result.__dict__.update({k:v if k=='sim' else deepcopy(v) for k,v in controller.__dict__.items()})
        result.handles=handles
        model=result.sim.model._model
        source=mujoco.MjModel.from_xml_path(str(SHARED/PANDA.source_xml))
        source_ids=np.asarray([source.actuator(name).id for name in PANDA.actuator_names])
        ids=handles.actuator_ids
        np.testing.assert_array_equal(model.actuator_gear[ids],source.actuator_gear[source_ids])
        np.testing.assert_array_equal(model.actuator_trntype[ids],source.actuator_trntype[source_ids])
        assert np.all(model.actuator_actadr[ids]==-1) and np.all(source.actuator_actadr[source_ids]==-1)
        native_integrator=int(model.opt.integrator)
        original={field:getattr(model,field)[ids].copy().tolist() for field in SERVO_FIELDS}
        for field in SERVO_FIELDS:getattr(model,field)[ids]=getattr(source,field)[source_ids]
        if source_integrator:model.opt.integrator=source.opt.integrator
        result.tracker=JointReferenceTracker(handles.spec.arm_dof)
        result.reference_q=result.sim.data.qpos[handles.qpos_ids].copy()
        result.interpolation_start=result.reference_q.copy()
        result.interpolation_elapsed=1/controller.control_freq
        result.interval_s=1/controller.control_freq
        result.last_tracking=None
        result.servo_metadata=dict(source_xml=str(SHARED/PANDA.source_xml),
            output_units='radians into position-servo ctrl; robot transport still names this field torques',
            native_integrator=native_integrator,effective_integrator=int(model.opt.integrator),
            source_integrator=int(source.opt.integrator),physics_dt_s=float(model.opt.timestep),
            interpolation_duration_s=result.interval_s,tracker_settings=asdict(result.tracker.settings),
            original_actuators=original,position_servo={field:getattr(model,field)[ids].tolist() for field in SERVO_FIELDS},
            bias_compensation='qfrc_applied[arm] = qfrc_bias[arm], same as PresetSimulationBackend',
            gripper='unchanged native LIBERO',nullspace_control=False)
        result.servo_statistics=dict(calls=0,tracker_saturated_steps=0,ctrl_clipped_steps=0,
            predicted_force_limited_steps=0,max_joint_reference_error_rad=0.,max_tracker_correction_rad=0.,
            max_servo_force_nm=0.)
        result.reference_samples=[]
        return result

    def set_goal(self,action,set_qpos=None):
        super().set_goal(action,set_qpos=set_qpos)
        self.interpolation_start=self.reference_q.copy()
        self.interpolation_elapsed=0.

    def run_controller(self):
        self.update()
        data=self.sim.data._data;model=self.sim.model._model;h=self.handles
        reference,command,tracking,elapsed,alpha=PresetSimulationBackend.track_joint_reference(
            tracker=self.tracker,interpolation_start=self.interpolation_start,
            joint_target=self.goal_qpos,interpolation_elapsed=self.interpolation_elapsed,
            interpolation_duration_s=self.interval_s,physics_dt=float(model.opt.timestep),
            actual_q=data.qpos[h.qpos_ids],actual_velocity_q=data.qvel[h.dof_ids])
        self.reference_q=reference.copy();self.interpolation_elapsed=elapsed;self.last_tracking=tracking
        # Preserve the formal backend's feedforward location outside actuator
        # force saturation. MuJoCo itself evaluates all servo forces/derivatives.
        data.qfrc_applied[h.dof_ids]=data.qfrc_bias[h.dof_ids]
        ranges=model.actuator_ctrlrange[h.actuator_ids]
        clipped=np.clip(command,ranges[:,0],ranges[:,1])
        gain=model.actuator_gainprm[h.actuator_ids,0];bias=model.actuator_biasprm[h.actuator_ids,:3]
        force=gain*clipped+bias[:,0]+bias[:,1]*data.qpos[h.qpos_ids]+bias[:,2]*data.qvel[h.dof_ids]
        limits=model.actuator_forcerange[h.actuator_ids]
        stats=self.servo_statistics;stats['calls']+=1
        stats['tracker_saturated_steps']+=int(tracking.saturated)
        stats['ctrl_clipped_steps']+=int(np.any(command!=clipped))
        stats['predicted_force_limited_steps']+=int(np.any((force<limits[:,0])|(force>limits[:,1])))
        stats['max_joint_reference_error_rad']=max(stats['max_joint_reference_error_rad'],tracking.maximum_error_rad)
        stats['max_tracker_correction_rad']=max(stats['max_tracker_correction_rad'],tracking.maximum_correction_rad)
        stats['max_servo_force_nm']=max(stats['max_servo_force_nm'],float(np.max(np.abs(np.clip(force,limits[:,0],limits[:,1])))))
        self.reference_samples.append(np.r_[float(data.time),alpha,reference,command,data.qpos[h.qpos_ids],data.qvel[h.dof_ids],data.qfrc_bias[h.dof_ids]])
        self.torques=command.copy()  # Native robot clips/writes ctrl; units are rad here.
        Controller.run_controller(self)
        return self.torques
