"""Read-only replay of canonical tolerance preparation, including EMA and recovery.

No IPOPT solve, env.step, inference, or physical rollout is executed here.
"""
from copy import deepcopy
import numpy as np

class RecordedToleranceReplay:
    def __init__(self, env, initial):
        from tasktol.simulation.libero import LiberoTaskTolSession
        env.set_state(initial);env.sim.forward()
        self.session=LiberoTaskTolSession(env,joint_control_mode='native')
        s=self.session
        # The10 strict zero-action settling cycles preserve this nominal origin
        # and create the APP stage without updating the tolerance EMA.
        s.release.begin_stage((0,'APP'),optimized_rotation=s.target.rotation,nominal_rotation=s.target.rotation)
        s.stage_key=(0,'APP');s.subtask_index=0

    def verify(self, action, selection, row):
        from tasktol.tolerance.frame import box_tolerance_frame
        from tasktol.tolerance.runtime import rebase_rotation_subtask,advance_reference_pose,configure_rotation_axis_tasks
        from tasktol.geometry import rotation_matrix
        s=self.session
        names=('target','tolerance','release','stage_key','subtask_index')
        old=deepcopy({n:getattr(s,n) for n in names})
        phase=selection['phase'];subtask=selection['subtask_index'];key=(subtask,phase)
        bounds=np.asarray(selection['bounds_rad'])
        if subtask!=s.subtask_index:
            rebase_rotation_subtask(s.target,s.tolerance,s.release,stage_key=key,accepted_rotation=s.release.accepted_rotation)
        if key!=s.stage_key:
            s.release.begin_stage(key,optimized_rotation=s.target.rotation if s.stage_key is None else s.release.accepted_rotation,nominal_rotation=s.target.rotation)
        guarded=np.linalg.norm(s.target.rotation[:2,1])<1e-6
        mask=np.array([b=='1' for b in selection['semantic_mask']])
        if phase not in ('APP','EXEC') or guarded:mask[:]=False
        mask &= (bounds[::2]>0)|(bounds[1::2]>0)
        s.tolerance.ranged=mask
        stage={'APP':0,'ENG':1,'EXEC':2,'DIS':3}[phase]
        frame=np.eye(3) if guarded else box_tolerance_frame(s.target.rotation)
        s.tolerance.configure_stage(stage,s.target.rotation,frame,np.maximum(bounds[::2],bounds[1::2]),negative_limits=bounds[::2],positive_limits=bounds[1::2],stage_key=key)
        s.target.position,s.target.rotation=advance_reference_pose(s.target.position,s.target.rotation,action)
        guarded=np.linalg.norm(s.target.rotation[:2,1])<1e-6
        if guarded:s.tolerance.ranged[:]=False
        s.tolerance.frame=np.eye(3) if guarded else box_tolerance_frame(s.target.rotation)
        s.tolerance.raw_tolerance_frame=s.tolerance.frame.copy();s.tolerance.update_intent(np.asarray(action[3:6]),s.dt)
        _,mask=configure_rotation_axis_tasks(s.tolerance,())
        step=s.release.prepare_cycle(s.target.rotation,s.tolerance.frame,mask,-s.tolerance.negative_limits,s.tolerance.positive_limits,allow_boundary_governor=phase not in ('ENG','DIS'))
        if row['optimizer']['feasible']:
            rotation=s.ik.kinematics.evaluate(np.asarray(row['joint_target_rad']),include_manipulability=False,include_link_points=False).rotation
            candidate=s.release.candidate(rotation,step)
            np.testing.assert_allclose(candidate.stage_relative_tolerance_release_rad,row['rotation_release_rad'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(candidate.predicted_tolerance_violation_rad,row['predicted_tolerance_violation_rad'],atol=1e-10,rtol=0)
            assert row['effective_mask']==''.join('1' if b else '0' for b in step.mask)
            coordinates=candidate.stage_relative_error_fixed_xyz_rad
            violation=np.where(step.mask,np.maximum.reduce([step.effective_lower_limits-coordinates,coordinates-step.effective_upper_limits,np.zeros(3)]),0)
            assert np.max(violation)<=s.ik.settings.inequality_tolerance+1e-9,(row['index'],violation)
            assert np.linalg.norm(coordinates[~step.mask])<=s.ik.settings.rotation_tolerance+1e-9
            np.testing.assert_allclose(s.target.position,row['nominal_target']['position_m'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(s.target.rotation,rotation_matrix(np.asarray(row['nominal_target']['rotation_vector_rad'])),atol=1e-10,rtol=0)
            s.release.commit(rotation,step);s.stage_key=key;s.subtask_index=subtask
        else:
            for name,value in old.items():setattr(s,name,value)
