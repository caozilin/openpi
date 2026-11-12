"""Instrument the existing offline backend; never modify the frozen batch.

All execution uses run_counterfactual_trial. Patches below are temporary,
process-local controlled interventions at its shared IK interface.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
PROJECT = Path('/media/czl/sata/franka_my_code/franka_mujoco')
sys.path.insert(0, str(PROJECT))
import argparse
import hashlib
import json
from dataclasses import asdict, replace
from unittest.mock import patch
import mujoco
import numpy as np
from tasktol.counterfactual.runner import CounterfactualTrialSpec, PreparedNominalTrial, run_counterfactual_trial
from tasktol.counterfactual.stored import StoredKinematicContext, StoredStageReference, StoredSubtaskReference
from tasktol.data.libero_annotation.scene import native_handles
from tasktol.geometry import rotation_matrix, rotation_vector
from tasktol.tasks.planning.types import TaskFrame
from tasktol.tolerance.annotations import box_tolerance_frame
from tasktol.control.ipopt.controller import IpoptIK
from tasktol.control.ipopt.adapter import IpoptAdapter
from tasktol.control.ipopt import controller as controller_module

RUN = PROJECT / 'analysis_results/libero_relative_clearance_eps01_dsp16_first5pertask_20261106'
CASE = RUN / 'libero_spatial/task_04/demo_1'
OUT = Path(__file__).resolve().parent / 'replay_diagnosis'

def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def prepared(document):
    frames = tuple(document['frames'])
    subtask = document['subtasks'][0]
    model = mujoco.MjModel.from_xml_string(document['metadata']['robot_xml'])
    data = mujoco.MjData(model)
    first = frames[0]
    for i,v in enumerate(first['joint_state']['position_rad'],1):
        data.qpos[model.joint(f'robot0_joint{i}').qposadr[0]] = v
    for i,v in enumerate(first['gripper_state']['raw_joint_positions_m'],1):
        data.qpos[model.joint(f'gripper0_finger_joint{i}').qposadr[0]] = v
    mujoco.mj_forward(model,data)
    handles = native_handles(model,data)
    masks = (tuple(subtask['APP_mask']), tuple(subtask['EXEC_mask']))
    refs=[]
    for phase, mask in zip(('pregrasp','postgrasp'),masks):
        end=next((f for f in reversed(frames) if f['phase']==phase),frames[-1])
        rot=rotation_matrix(end['nominal_target_pose']['rotation_vector_base_rad'])
        refs.append(StoredStageReference(rot,box_tolerance_frame(rot),mask))
    goal=first['nominal_target_pose']; action=np.asarray(first['action'])
    context=StoredKinematicContext(handles,.05,('work_surface',),
        np.asarray(goal['position_m'])-action[:3],
        rotation_matrix(-action[3:6])@rotation_matrix(goal['rotation_vector_base_rad']),
        (StoredSubtaskReference(subtask['label'],*refs),))
    plan=tuple(TaskFrame(np.asarray(f['nominal_target_pose']['position_m']),
        rotation_matrix(f['nominal_target_pose']['rotation_vector_base_rad']),
        bool(f['action'][6]>0),f['phase']) for f in frames)
    return PreparedNominalTrial(model,data,'panda',plan,document['metadata'],context), frames

def run(mode):
    doc=json.loads((CASE/'prepared_input.json').read_text())
    rows=json.loads((Path(doc['source']['folder'])/'trajectory.json').read_text())
    frozen=json.loads(next((CASE/'zero_baselines').rglob('trial.json')).read_text())
    spec=CounterfactualTrialSpec(**frozen['result']['spec'])
    prep, frames=prepared(doc)
    trace=[]; raw=[]; at={'frame':-1}
    solve=IpoptIK.solve; adapter=IpoptAdapter.solve
    chart=controller_module.rotation_release_error_and_jacobian
    def chart_override(kinematics,state,step):
        if 'so3_chart' not in mode: return chart(kinematics,state,step)
        if 'at135' in mode and at['frame']!=135: return chart(kinematics,state,step)
        reference=step.stage_reference_rotation()
        return (kinematics.pose_error(state,np.zeros(3),reference)[3:],
                kinematics.pose_error_jacobian(state,reference)[3:])
    def observe(ctx): at['frame']=ctx.frame_index
    def capture_adapter(self,*args,**kwargs):
        value=adapter(self,*args,**kwargs)
        item=dict(frame=at['frame'],seed=np.asarray(args[4]).copy(),
            q=value.q.copy(),iterations=value.iterations,status=value.status,
            status_code=value.status_code,validation=asdict(value.validation),
            constraints=asdict(value.constraints),cost=value.cost)
        if 'derivatives' in mode and at['frame']==135:
            checks=[]
            for label,point in [('seed',np.asarray(args[4])),('last_iterate',value.q)]:
                c=args[2](point); analytic=np.vstack(args[3](point,c))
                for eps in (1e-5,1e-6,1e-7):
                    numeric=np.empty_like(analytic)
                    for j in range(point.size):
                        delta=np.eye(point.size)[j]*eps
                        plus=args[2](point+delta); minus=args[2](point-delta)
                        numeric[:,j]=(np.r_[plus.equality,plus.inequality]-np.r_[minus.equality,minus.inequality])/(2*eps)
                    checks.append(dict(point=label,epsilon_rad=eps,
                        max_absolute_jacobian_error=float(np.max(np.abs(analytic-numeric))),
                        relative_frobenius_error=float(np.linalg.norm(analytic-numeric)/np.linalg.norm(numeric))))
            item['finite_difference_checks']=checks
        raw.append(item)
        return value
    def capture_solve(self,data,target,active_dofs,tolerance_rotation=None,
                      rotation_release_step=None,**kwargs):
        i=at['frame']; source=rows[i]; seed_before=None if self._xopt is None else self._xopt.copy()
        if 'source_history' in mode:
            initial=np.asarray(rows[0]['pre_action']['joint_position_rad'])
            self._xopt=None if i==0 else np.asarray(rows[i-1]['joint_target_rad'])
            self._previous_state=None if i==0 else (np.asarray(rows[i-2]['joint_target_rad']) if i>=2 else initial.copy())
            self._previous_state2=None if i==0 else (np.asarray(rows[i-3]['joint_target_rad']) if i>=3 else initial.copy())
            self._warm_start=None
        if 'source_state' in mode:
            data.qpos[self.handles.qpos_ids]=source['pre_action']['joint_position_rad']
        if 'source_target' in mode:
            target.position=np.asarray(source['nominal_target']['position_m'])
            target.rotation=rotation_matrix(source['nominal_target']['rotation_vector_rad'])
        if 'align_handoff' in mode and rotation_release_step is not None:
            rotation_release_step=replace(rotation_release_step,
                stage_handoff_rotation=rotation_release_step.stage_handoff_nominal_rotation.copy())
        if 'absolute' in mode:
            rotation_release_step=None; tolerance_rotation=None
            for axis in range(3,6): self.set_axis_task(axis,None)
        problem=self._prepare_single_step_nlp(data,target,active_dofs,tolerance_rotation,rotation_release_step)
        witness=problem.constraint_evaluation(np.asarray(source['joint_target_rad']))[0]
        entry=dict(frame=i,source_phase=source['phase'],seed=problem.seed.copy(),
            prior_xopt=seed_before,target_position=target.position.copy(),target_rotation=target.rotation.copy(),
            target_position_error_vs_source=float(np.max(np.abs(target.position-source['nominal_target']['position_m']))),
            target_rotation_error_vs_source=float(np.linalg.norm(rotation_vector(rotation_matrix(source['nominal_target']['rotation_vector_rad']).T@target.rotation))),
            has_release_step=rotation_release_step is not None,
            source_witness_constraints=asdict(witness))
        if rotation_release_step is not None:
            entry['stage_reference_error_vs_nominal_rad']=float(np.linalg.norm(rotation_vector(target.rotation.T@rotation_release_step.stage_reference_rotation())))
            entry['stage_handoff_rotation']=rotation_release_step.stage_handoff_rotation.copy()
            entry['stage_handoff_nominal_rotation']=rotation_release_step.stage_handoff_nominal_rotation.copy()
        command,diag=solve(self,data,target,active_dofs,tolerance_rotation,rotation_release_step,**kwargs)
        entry.update(command=command.copy(),max_command_difference_vs_source=float(np.max(np.abs(command-source['joint_target_rad']))),
            diag=dict(feasible=diag.feasible,status=diag.status,iterations=diag.iterations,
                position_residual=diag.position_residual,rotation_residual=diag.strict_rotation_residual))
        trace.append(entry)
        return command,diag
    if mode.endswith('_iter100'): spec=replace(spec,ipopt_max_iterations=100)
    if mode.endswith('_iter21'): spec=replace(spec,ipopt_max_iterations=21)
    with (patch('mujoco.mj_step',side_effect=AssertionError('No physics allowed')),
          patch.object(IpoptIK,'solve',capture_solve),patch.object(IpoptAdapter,'solve',capture_adapter),
          patch.object(controller_module,'rotation_release_error_and_jacobian',chart_override)):
        result=run_counterfactual_trial(spec,prepared_nominal=prep,dataset_nominal_frames=frames,
            kinematic_step_observer=observe).as_dict()
    report=dict(mode=mode,physics_steps=0,trace=trace,raw_solver_results=raw,
        evaluation={k:v for k,v in result['evaluation'].items() if k not in ('environment',)},
        mujoco_version=mujoco.__version__,source_prepared_sha256=hashlib.sha256((CASE/'prepared_input.json').read_bytes()).hexdigest())
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{mode}.json').write_text(json.dumps(report,default=serial,indent=2)+'\n')
    divergences={str(tol):next((t['frame'] for t in trace if t['max_command_difference_vs_source']>tol),None) for tol in (1e-9,1e-6,1e-4,.01)}
    print(json.dumps(dict(mode=mode,frames=len(trace),first_infeasible=next((t['frame'] for t in trace if not t['diag']['feasible']),None),first_q_divergence=divergences,
        max_target_position_error=max(t['target_position_error_vs_source'] for t in trace),
        max_target_rotation_error=max(t['target_rotation_error_vs_source'] for t in trace),
        last_raw_validation=raw[-1]['validation']),default=serial),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('modes',nargs='+');args=p.parse_args()
    for mode in args.modes: run(mode)
