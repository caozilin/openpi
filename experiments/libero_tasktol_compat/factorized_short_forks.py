"""Same-state task/null ablations, using only native LIBERO env.step.

Capture the existing canonical OSC replay; then execute ONE native policy step
per copied state and factor combination. No alternate physics/rollout loop.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL=ROOT.parent/'franka_mujoco'
sys.path[:0]=[str(ROOT),str(TASKTOL),str(ROOT/'third_party/libero')]
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/tasktol_factor_numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/tasktol_factor_matplotlib')

import h5py
import mujoco
import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
from tasktol.control.ipopt import IpoptIK,IpoptSettings
from tasktol.control.pose import TargetPose
from tasktol.control.task_space.objective import ObjectiveSettings
from tasktol.geometry import rotation_vector
from tasktol.simulation.libero import libero_handles
from tasktol.simulation.libero_joint_controller import FactorizedReferenceController
from experiments.libero_tasktol_compat.native_alignment import replay

MODES=dict(
    native=None,
    A10=dict(task_feedback='joint',null_reference='joint',null_kp=10),
    B10=dict(task_feedback='osc',null_reference='joint',null_kp=10),
    C10=dict(task_feedback='joint',null_reference='initial',null_kp=10),
    D10=dict(task_feedback='osc',null_reference='initial',null_kp=10),
    old_A150=dict(task_feedback='joint',null_reference='joint',null_kp=150),
)
MODEL_FIELDS=('body_pos','body_quat','body_mass','body_inertia','jnt_pos','jnt_axis',
              'dof_damping','dof_armature','geom_pos','geom_quat','geom_size','geom_friction',
              'geom_contype','geom_conaffinity','geom_solref','geom_solimp',
              'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_ctrlrange','eq_data')


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def contacts(env):
    model,data=env.sim.model._model,env.sim.data._data
    values=[]
    for contact_index,contact in enumerate(data.contact[:data.ncon]):
        bodies=[int(model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or '' for b in bodies]
        robot=[n.startswith(('robot0','gripper0')) for n in names]
        if robot[0]==robot[1]:continue
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,contact_index,wrench)
        moving=[]
        for body in bodies:
            ancestor=body
            while ancestor and not model.body_jntnum[ancestor]:ancestor=int(model.body_parentid[ancestor])
            moving.append(bool(ancestor and model.body_jntnum[ancestor]))
        values.append(dict(bodies=names,position=contact.pos.copy(),frame=contact.frame.copy(),
                           wrench_contact=wrench,gripper_movable=any(n.startswith('gripper0') for n in names) and any(m and not r for m,r in zip(moving,robot))))
    return values


def integration(env):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    result=np.empty(mujoco.mj_stateSize(env.sim.model._model,spec))
    mujoco.mj_getState(env.sim.model._model,env.sim.data._data,result,spec)
    return result


def capture(case,folder,reference):
    snapshots=[];frame=-1;source_model={};first_touch=None
    original_step=OffScreenRenderEnv.step

    def observed(env,action):
        nonlocal frame,first_touch
        frame+=1
        ctrl=env.robots[0].controller
        if frame==0:
            source_model.update({k:np.array(getattr(env.sim.model._model,k)).copy() for k in MODEL_FIELDS})
        snapshot=dict(index=frame,state=integration(env),action=np.asarray(action).copy(),
                      ctrl_state=deepcopy({k:v for k,v in ctrl.__dict__.items() if k!='sim'}),
                      gripper=env.robots[0].gripper.current_action.copy(),
                      timestep=env.env.timestep,cur_time=env.env.cur_time,done=env.env.done,
                      contacts_before=contacts(env))
        original_sim_step=env.sim.step
        touching=[]

        def observe_physics():
            result=original_sim_step()
            touching.extend(c for c in contacts(env) if c['gripper_movable'])
            return result

        env.sim.step=observe_physics
        try:result=original_step(env,action)
        finally:env.sim.step=original_sim_step
        snapshot['gripper_object_contacts_in_cycle']=bool(touching)
        snapshots.append(snapshot)
        if touching and first_touch is None:first_touch=frame
        return result

    OffScreenRenderEnv.step=observed
    try:
        report=replay(case,folder,refresh_controller=True,source_reset_state=True,source_reset_history=True,
                      osc_reference_mode='accumulated_reference',reference_actions_path=reference)
    finally:OffScreenRenderEnv.step=original_step
    np.savez_compressed(folder/'integration_snapshots.npz',states=np.asarray([s['state'] for s in snapshots]),
                        native_actions=np.asarray([s['action'] for s in snapshots]),state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
                        **{'model_'+k:v for k,v in source_model.items()})
    return snapshots,source_model,first_touch,report


def post_physics(env,scratch,handles):
    model,data=env.sim.model._model,env.sim.data._data
    scratch.qpos[:]=data.qpos;scratch.qvel[:]=data.qvel
    scratch.mocap_pos[:]=data.mocap_pos;scratch.mocap_quat[:]=data.mocap_quat
    mujoco.mj_kinematics(model,scratch);mujoco.mj_comPos(model,scratch)
    jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,scratch,jp,jr,handles.ee_site_id)
    p=scratch.site_xpos[handles.ee_site_id].copy()
    r=scratch.site_xmat[handles.ee_site_id].reshape(3,3) @ handles.ee_rotation_offset
    object_poses={name:dict(position=scratch.xpos[model.body(obj.root_body).id].copy(),rotation=scratch.xquat[model.body(obj.root_body).id].copy()) for name,obj in env.env.objects_dict.items()}
    return dict(position=p,rotation=r,velocity=np.r_[jp@data.qvel,jr@data.qvel],object_poses=object_poses)


def run_case(case,output,joint_reference_source):
    selected=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks'][case]
    rel=Path(selected['suite'])/f"{selected['task_id']:02d}";dest=output/rel;dest.mkdir(parents=True,exist_ok=False)
    reference=ROOT/'data/libero_reference_actions/20261020_osc_reference_v1'/rel/'reference_actions.hdf5'
    snapshots,model_values,touch,source_report=capture(case,dest/'source_osc',reference)
    with h5py.File(reference,'r') as file:
        labels=file['data'][selected['demo_id']];actions=labels['actions'][()];goals_p=labels['goal_positions_m'][()];goals_r=labels['goal_tool_rotations'][()]
    recorded_path=ROOT/'experiments/libero_tasktol_compat/results/ipopt_relabelled_kp150_all40_v1/kp_150'/rel/'trajectory.json'
    recorded=json.loads(recorded_path.read_text())['frames']
    recorded_report=json.loads(recorded_path.with_name('report.json').read_text())
    assert recorded_report['optimizer_backend']=='ipopt' and recorded_report['reference_actions']['sha256']==hashlib.sha256(reference.read_bytes()).hexdigest()
    close=next((i for i,a in enumerate(actions) if a[6]>0),None)
    requested=dict(start=0,early_motion=min(10,len(actions)-1))
    if close is not None:requested.update(pre_close=max(0,close-1),close=close)
    if touch is not None:requested.update(pre_first_gripper_contact=max(0,touch-1),first_gripper_contact=touch,post_first_gripper_contact=min(touch+1,len(actions)-1))
    if case==30:
        requested.update({f'long00_critical_{i}':i for i in [123,124,127,128,195,196]})
    events={}
    for label,index in requested.items():events.setdefault(index,[]).append(label)
    env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')),
                   controller='OSC_POSE',use_camera_obs=False,has_offscreen_renderer=False,control_freq=20)
    records=[]
    try:
        env.reset()
        for key,value in model_values.items():getattr(env.sim.model._model,key)[:]=value
        assert all(np.array_equal(getattr(env.sim.model._model,k),v) for k,v in model_values.items())
        template=env.robots[0].controller
        handles=libero_handles(env)
        ik=IpoptIK(env.sim.model._model,handles,IpoptSettings(max_iterations=60,max_cpu_time_s=0.5),ObjectiveSettings.delta_q_squared_conditioning_safeguard())
        scratch=mujoco.MjData(env.sim.model._model)

        def restore(snapshot):
            mujoco.mj_setState(env.sim.model._model,env.sim.data._data,snapshot['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
            env.env.timestep=snapshot['timestep'];env.env.cur_time=snapshot['cur_time'];env.env.done=snapshot['done']
            env.robots[0].gripper.current_action=snapshot['gripper'].copy()
            template.__dict__.clear()
            template.__dict__.update(deepcopy(snapshot['ctrl_state']))
            template.sim=env.sim
            env.robots[0].controller=template
            # Observations do not drive any controller; clear previous cache.
            env.env._obs_cache={}
            assert np.array_equal(integration(env),snapshot['state'])

        for index,labels_event in sorted(events.items()):
            snapshot=snapshots[index];spot=dest/f'frame_{index:04d}';spot.mkdir()
            (spot/'source_program_state.json').write_text(json.dumps(plain({k:v for k,v in snapshot.items() if k!='state'}),indent=2)+'\n')
            restore(snapshot)
            if joint_reference_source=='fresh_snapshot_ik':
                qd,solve=ik.solve(env.sim.data._data,TargetPose(goals_p[index],goals_r[index]),np.ones(6,dtype=bool))
            else:
                source_row=recorded[index]
                assert np.array_equal(source_row['action'],actions[index])
                qd=np.asarray(source_row['joint_target_rad']).copy()
                solve=SimpleNamespace(feasible=source_row['optimizer']['feasible'],status=source_row['optimizer']['status'],elapsed_ms=source_row['optimizer']['elapsed_ms'])
            fk=ik.kinematics.evaluate(qd,include_manipulability=False,include_link_points=False)
            qd_info=dict(feasible=bool(solve.feasible),status=str(solve.status),elapsed_ms=float(solve.elapsed_ms),joint_reference=qd.copy(),
                         source=joint_reference_source,recorded_ipopt_path=str(recorded_path),recorded_ipopt_sha256=hashlib.sha256(recorded_path.read_bytes()).hexdigest(),
                         fk_position_difference_m=float(np.linalg.norm(fk.position-goals_p[index])),fk_rotation_difference_rad=float(np.linalg.norm(rotation_vector(fk.rotation @ goals_r[index].T))))
            for mode,settings in MODES.items():
                if settings is not None and (settings['task_feedback']=='joint' or settings['null_reference']=='joint') and not solve.feasible:
                    records.append(dict(task=str(rel),frame=index,events=labels_event,mode=mode,skipped='ipopt_rejected_shared_target',solver=plain(qd_info)))
                    continue
                restore(snapshot)
                ctrl=template if settings is None else FactorizedReferenceController.from_native(template,**settings)
                if settings is not None and solve.feasible:ctrl.set_joint_reference(qd)
                env.robots[0].controller=ctrl
                trace=[];contact_trace=[];native_run=ctrl.run_controller;native_sim_step=env.sim.step
                initial_post=post_physics(env,scratch,handles)
                current={}

                def observed_control():
                    tau=native_run()
                    diag=getattr(ctrl,'factor_diagnostics',None)
                    current.update(tau=np.asarray(tau).copy(),tau_clipped=np.clip(tau,*env.robots[0].torque_limits),
                                   q_pre=ctrl.joint_pos.copy(),v_pre=ctrl.joint_vel.copy(),mass=ctrl.mass_matrix.copy(),jac=ctrl.J_full.copy(),bias=ctrl.torque_compensation.copy(),
                                   goal_position=ctrl.goal_pos.copy(),goal_rotation=ctrl.goal_ori.copy(),
                                   native_reconstruction_residual_nm=0.0 if diag is None else diag['native_reconstruction_residual_nm'],
                                   task_torque=np.full(7,np.nan) if diag is None else diag['task_torque'].copy(),
                                   null_torque=np.full(7,np.nan) if diag is None else diag['null_torque'].copy(),
                                   split_inertia=np.full((6,6),np.nan) if diag is None else diag['split_task_inertia'].copy(),
                                   null_projector=np.full((7,7),np.nan) if diag is None else diag['null_projector'].copy(),
                                   feedback_mismatch=np.full(6,np.nan) if diag is None else diag['joint_task_feedback_minus_osc'].copy(),
                                   native_torque=np.asarray(tau).copy() if diag is None else diag['native_torque'].copy())
                    return tau

                def observed_physics():
                    result=native_sim_step()
                    actual=post_physics(env,scratch,handles);data=env.sim.data._data
                    contact_trace.append(contacts(env))
                    previous_velocity=initial_post['velocity'] if not trace else trace[-1]['ee_velocity']
                    trace.append(dict(**current,time_s=float(data.time),q_post=data.qpos[handles.qpos_ids].copy(),v_post=data.qvel[handles.dof_ids].copy(),
                                      qacc=data.qacc[handles.dof_ids].copy(),ee_position=actual['position'],ee_rotation=actual['rotation'],ee_velocity=actual['velocity'],
                                      ee_acceleration_fd=(actual['velocity']-previous_velocity)/env.sim.model.opt.timestep,
                                      finger_qpos=data.qpos[handles.finger_qpos_ids].copy(),object_poses=actual['object_poses']))
                    return result

                ctrl.run_controller=observed_control;env.sim.step=observed_physics
                try:env.step(snapshot['action'].tolist())
                finally:
                    env.sim.step=native_sim_step
                    del ctrl.__dict__['run_controller']
                folder=spot/mode;folder.mkdir()
                arrays={k:np.asarray([r[k] for r in trace]) for k in trace[0] if k!='object_poses'}
                assert len(trace)==25
                if settings is not None:
                    assert ctrl.factor_diagnostics is not None
                    assert np.all(np.isfinite(arrays['task_torque'])) and np.all(np.isfinite(arrays['null_torque']))
                    assert arrays['native_reconstruction_residual_nm'].max()<1e-9
                    if mode=='D10':
                        assert np.max(np.abs(arrays['tau']-arrays['native_torque']))<1e-9
                np.savez_compressed(folder/'physics_trace.npz',**arrays,joint_reference=qd)
                (folder/'contact_and_object_trace.json').write_text(json.dumps(plain(dict(contacts=contact_trace,object_poses=[r['object_poses'] for r in trace])),indent=2)+'\n')
                impulse=sum(abs(c['wrench_contact'][0])*env.sim.model.opt.timestep for items in contact_trace for c in items)
                clipping=int(np.sum(np.any(np.abs(arrays['tau']-arrays['tau_clipped'])>1e-10,axis=1)))
                record=dict(task=str(rel),frame=index,events=labels_event,mode=mode,settings=settings,solver=plain(qd_info),
                            control_calls=len(trace),duration_s=float(env.sim.data._data.time-snapshot['state'][0]),
                            final_position=arrays['ee_position'][-1].tolist(),final_rotation=arrays['ee_rotation'][-1].tolist(),final_velocity=arrays['ee_velocity'][-1].tolist(),
                            final_q=arrays['q_post'][-1].tolist(),first_tau=arrays['tau'][0].tolist(),first_task_torque=arrays['task_torque'][0].tolist(),first_null_torque=arrays['null_torque'][0].tolist(),
                            native_reconstruction_max_residual_nm=float(arrays['native_reconstruction_residual_nm'].max()),
                            contact_normal_impulse_abs_ns=float(impulse),contact_solve_samples=sum(bool(c) for c in contact_trace),torque_clipped_calls=clipping,
                            task_success=bool(env.check_success()),data_directory=str(folder),source_state_sha256=hashlib.sha256(snapshot['state'].tobytes()).hexdigest())
                records.append(record);(folder/'report.json').write_text(json.dumps(plain(record),indent=2)+'\n')
            print(f'{rel} frame {index} {labels_event}: branches complete',flush=True)
    finally:env.close()
    result=dict(task=str(rel),case=case,events=requested,source_task_success=source_report['task_success'],source_reference=source_report['reference_actions'],records=records)
    (dest/'short_forks.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cases',type=int,nargs='+',default=[0,25,30,36])
    parser.add_argument('--joint-reference-source',choices=['recorded_full_replay','fresh_snapshot_ik'],default='recorded_full_replay')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(args.output/'numba');os.environ['MPLCONFIGDIR']=str(args.output/'matplotlib')
    results=[]
    for case in args.cases:
        results.append(run_case(case,args.output,args.joint_reference_source))
        (args.output/'summary.json').write_text(json.dumps(plain(dict(complete=len(results)==len(args.cases),joint_reference_source=args.joint_reference_source,scope='One native 50ms step per same-state branch; balanced null kp10, old kp150 retained separately; shared accepted IPOPT q_d',results=results)),indent=2)+'\n')


if __name__=='__main__':main()
