"""Two zero-action offline checks; no policy inference or benchmark rollout."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime'),str(ROOT),str(ROOT/'third_party/libero')]
import numpy as np
from libero.libero.envs.env_wrapper import ControlEnv
from tasktol.simulation.libero import LiberoTaskTolSession
from examples.libero.confirmed_mask_policy import ConfirmedMaskPolicy,step_with_selection
from experiments.libero_tasktol_compat.mujoco_joint_servo import MuJoCoJointServoController
from examples.libero.solver_failure_hold import SolverFailureHoldSession
from tasktol.control.ipopt import IpoptSettings
from tasktol.control.task_space.objective import ObjectiveSettings


class Classifier:
    phase = 'pregrasp'
    def update(self,fingers,closed):
        return NS(phase=NS(key=self.phase),subtask_index=0,aperture_m=float(sum(fingers)),motion_m=0.)


records = []
tasks = json.loads((ROOT/'data/libero_evaluation/pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104/official_tasks.json').read_text())
for suite,tid,expected in [('libero_spatial',4,[0,5,0]),('libero_10',6,[5,5,15])]:
    task = next(t for t in tasks if t['suite']==suite and t['task_id']==tid)
    env = ControlEnv(bddl_file_name=str(ROOT/'third_party/libero/libero/libero/bddl_files'/task['problem_folder']/task['bddl_file']),controller='JOINT_POSITION',use_camera_obs=False,has_offscreen_renderer=False,control_freq=20)
    try:
        env.seed(0);obs=env.reset()
        session = LiberoTaskTolSession(env,joint_control_mode='native',kp=None,solver_settings=IpoptSettings())
        assert session.ik.objective_settings == ObjectiveSettings.delta_q_squared_conditioning_safeguard()
        control = MuJoCoJointServoController.from_native(session.controller,session.handles)
        session.controller=control;env.robots[0].controller=control
        session = SolverFailureHoldSession(session)
        c = Classifier();policy = ConfirmedMaskPolicy(suite,tid,classifier=c)
        if suite == 'libero_10':
            position = env.sim.data.body_xpos[env.env.obj_body_id['porcelain_mug_1']].copy()
            policy.resolve(session.data.qpos[session.handles.finger_qpos_ids],np.zeros(7),env.env,position)
            c.phase = 'postgrasp'
        selection = policy.resolve(session.data.qpos[session.handles.finger_qpos_ids],np.zeros(7),env.env,np.asarray(obs['robot0_eef_pos']))
        assert selection['semantic_tolerance_deg'] == expected,selection
        _,row = step_with_selection(session,np.r_[np.zeros(6),-1.],selection)
        assert row['physics_steps'] == 25
        assert row['optimizer']['feasible'],row['optimizer']
        np.testing.assert_allclose(row['bounds_rad'],np.deg2rad(np.repeat(expected,2)),atol=1e-15,rtol=0)
        np.testing.assert_allclose(session.tolerance.raw_negative_limits,np.deg2rad(expected),atol=1e-15,rtol=0)
        np.testing.assert_allclose(session.tolerance.raw_positive_limits,np.deg2rad(expected),atol=1e-15,rtol=0)
        np.testing.assert_allclose(session.release._accepted_step.lower_limits,-np.deg2rad(expected),atol=1e-15,rtol=0)
        np.testing.assert_allclose(session.release._accepted_step.upper_limits,np.deg2rad(expected),atol=1e-15,rtol=0)
        records.append(dict(suite=suite,task_id=tid,phase=row['phase'],degrees=expected,bounds_rad=row['bounds_rad'],semantic_mask=row['semantic_mask'],effective_mask=row['effective_mask'],optimizer=row['optimizer'],physics_steps=row['physics_steps']))
    finally:
        env.close()
output = Path(__file__).with_name('native_control_integration.json')
output.write_text(json.dumps(dict(complete=True,scope=__doc__,records=records),indent=2)+'\n')
print(json.dumps(dict(complete=True,checks=len(records),report=str(output))))
