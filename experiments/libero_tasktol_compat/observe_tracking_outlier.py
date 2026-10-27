"""Read-only native controller/contact observations during maintained cold replay."""
import json
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/libero_tasktol_compat'),str(ROOT.parent/'franka_mujoco'),str(ROOT/'third_party/libero')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-frames',action='store_true');args=parser.parse_args()
    import mujoco
    import numpy as np
    from replay_state_dataset import replay
    from tasktol.simulation.libero import LiberoTaskTolSession
    from tasktol.simulation.libero_joint_controller import FactorizedReferenceController
    out=ROOT/('experiments/libero_tasktol_compat/results/tracking_outlier32_native_diagnostic_all_v2'if args.all_frames else'experiments/libero_tasktol_compat/results/tracking_outlier32_native_diagnostic_v1');out.mkdir(exist_ok=False)
    records=[];active=[None]
    step=LiberoTaskTolSession.step;control=FactorizedReferenceController.run_controller
    def observe_step(session,*args,**kwargs):
        active[0]=session.frame_index
        try:return step(session,*args,**kwargs)
        finally:active[0]=None
    def observe_control(controller):
        torque=control(controller)
        if active[0] is not None and (args.all_frames or 106<=active[0]<=116):
            model=controller.sim.model._model;data=controller.sim.data._data
            contacts=[]
            for i,contact in enumerate(data.contact):
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))for g in [contact.geom1,contact.geom2]]
                if not any(n and n.startswith(('robot0_','gripper0_'))for n in names):continue
                force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                contacts.append(dict(bodies=names,distance_m=float(contact.dist),force_norm_n=float(np.linalg.norm(force[:3]))))
            contacts.sort(key=lambda c:c['force_norm_n'],reverse=True)
            records.append(dict(frame=active[0],physics_time_s=float(data.time),raw_torque_nm=torque.tolist(),
                task_torque_nm=controller.factor_diagnostics['task_torque'].tolist(),
                null_torque_nm=controller.factor_diagnostics['null_torque'].tolist(),
                sigma_min=float(np.linalg.svd(controller.J_full,compute_uv=False)[-1]),
                clipped=bool(np.any((torque<controller.actuator_min)|(torque>controller.actuator_max))),
                goal_minus_actual_position_m=(controller.goal_pos-controller.ee_pos).tolist(),
                actual_q_rad=controller.joint_pos.tolist(),reference_q_rad=controller.joint_target.tolist(),
                max_abs_joint_error_rad=float(np.max(np.abs(controller.joint_target-controller.joint_pos))),
                torque_limits_low=controller.actuator_min.tolist(),torque_limits_high=controller.actuator_max.tolist(),
                robot_contacts=contacts[:12]))
        return torque
    LiberoTaskTolSession.step=observe_step;FactorizedReferenceController.run_controller=observe_control
    try:
        replay(ROOT/'data/libero_tasktol_state/20261024_tracking_outlier32_pilot',32,'demo_21',out/'replay',include_failure=True)
    finally:
        LiberoTaskTolSession.step=step;FactorizedReferenceController.run_controller=control
    summary=[]
    for frame in sorted({r['frame']for r in records}):
        group=[r for r in records if r['frame']==frame]
        contacts=[c for r in group for c in r['robot_contacts']]
        strongest=max(contacts,key=lambda c:c['force_norm_n'])if contacts else None
        summary.append(dict(frame=frame,physics_steps=len(group),torque_clipped_steps=sum(r['clipped']for r in group),
            max_goal_position_error_mm=max(np.linalg.norm(r['goal_minus_actual_position_m'])*1000 for r in group),
            max_joint_reference_error_rad=max(r['max_abs_joint_error_rad']for r in group),strongest_robot_contact=strongest))
    (out/'native_diagnostic.json').write_text(json.dumps(dict(controller_modified=False,settings_modified=False,records=records,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
