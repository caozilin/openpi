"""Read native controller calls without changing targets, torques or physics."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL=ROOT.parent/'franka_mujoco'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['ipopt','osc'],required=True)
    parser.add_argument('--cases',type=int,nargs='+',default=[0,25,30])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--joint-control-mode',choices=['native','osc_inertia_matched','osc_task_joint_null'],default='native')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(args.output/'numba_cache')
    os.environ['MPLCONFIGDIR']=str(args.output/'matplotlib')
    sys.path[:0]=[str(ROOT),str(TASKTOL),str(ROOT/'third_party/libero')]
    import numpy as np
    import mujoco
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.benchmark.libero import run_replay
    from experiments.libero_tasktol_compat.native_alignment import replay
    selected=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    original_step=OffScreenRenderEnv.step
    samples=[];frame=-1

    def observed_step(env,action):
        nonlocal frame
        frame+=1
        ctrl=env.robots[0].controller
        if not getattr(ctrl,'_tasktol_tracking_observed',False):
            native_run=ctrl.run_controller
            robot=env.robots[0]
            def observed_run():
                tau=native_run()
                model,data=env.sim.model._model,env.sim.data._data
                contacts=0
                for contact in data.contact[:data.ncon]:
                    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
                           for g in (contact.geom1,contact.geom2)]
                    is_robot=[name.startswith(('robot0','gripper0')) for name in names]
                    contacts+=int(is_robot[0]!=is_robot[1])
                samples.append(dict(
                    frame=frame,time_s=float(data.time),q=ctrl.joint_pos.copy(),v=ctrl.joint_vel.copy(),
                    q_goal=ctrl.joint_target.copy() if getattr(ctrl,'joint_target',None) is not None else ctrl.goal_qpos.copy() if ctrl.name=='JOINT_POSITION' else np.full(7,np.nan),
                    initial_joint=ctrl.initial_joint.copy(),
                    task_torque=getattr(ctrl,'factor_diagnostics',None)['task_torque'].copy() if getattr(ctrl,'factor_diagnostics',None) is not None else np.full(7,np.nan),
                    null_torque=getattr(ctrl,'factor_diagnostics',None)['null_torque'].copy() if getattr(ctrl,'factor_diagnostics',None) is not None else np.full(7,np.nan),
                    goal_p=np.asarray(ctrl.goal_pos).copy() if ctrl.name=='OSC_POSE' else np.full(3,np.nan),
                    goal_r=ctrl.goal_ori.copy() if ctrl.name=='OSC_POSE' else np.full((3,3),np.nan),
                    ee_p=ctrl.ee_pos.copy(),ee_r=ctrl.ee_ori_mat.copy(),
                    ee_v=np.r_[ctrl.ee_pos_vel,ctrl.ee_ori_vel],
                    tau=np.asarray(tau).copy(),tau_clipped=np.clip(tau,*robot.torque_limits),
                    mass=ctrl.mass_matrix.copy(),jac=ctrl.J_full.copy(),bias=ctrl.torque_compensation.copy(),
                    kp=ctrl.kp.copy(),kd=ctrl.kd.copy(),robot_external_contacts=contacts,
                ))
                return tau
            ctrl.run_controller=observed_run
            ctrl._tasktol_tracking_observed=True
        return original_step(env,action)

    OffScreenRenderEnv.step=observed_step
    summaries=[]
    try:
        for case in args.cases:
            samples.clear();frame=-1
            task=selected[case];dest=args.output/task['suite']/f"{task['task_id']:02d}"
            if args.mode=='ipopt':
                report=run_replay(openpi_root=ROOT,output=dest,suite=task['suite'],task_id=task['task_id'],kp=150,
                                  joint_control_mode=args.joint_control_mode,
                                  reference_mode='relabelled_actions',reference_actions_path=ROOT/'data/libero_reference_actions/20261020_osc_reference_v1'/task['suite']/f"{task['task_id']:02d}"/'reference_actions.hdf5')
            else:
                report=replay(case,dest,refresh_controller=True,source_reset_state=True,source_reset_history=True,
                              osc_reference_mode='accumulated_reference',reference_actions_path=ROOT/'data/libero_reference_actions/20261020_osc_reference_v1'/task['suite']/f"{task['task_id']:02d}"/'reference_actions.hdf5')
            arrays={key:np.asarray([s[key] for s in samples]) for key in samples[0]}
            np.savez_compressed(dest/'control_trace.npz',**arrays)
            clipped=np.any(np.abs(arrays['tau']-arrays['tau_clipped'])>1e-8,axis=1)
            summary=dict(case=case,task=f"{task['suite']}/{task['task_id']:02d}",success=report['task_success'],
                         controller_calls=len(samples),physics_period_s=float(np.median(np.diff(arrays['time_s']))),
                         torque_clipped_samples=int(clipped.sum()),robot_contact_samples=int(np.sum(arrays['robot_external_contacts']>0)),
                         instrumentation='read-only wrapping native run_controller and env.step; no extra controller or physics calls')
            summaries.append(summary)
            (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
            print(json.dumps(summary),flush=True)
    finally:
        OffScreenRenderEnv.step=original_step


if __name__=='__main__':
    main()
