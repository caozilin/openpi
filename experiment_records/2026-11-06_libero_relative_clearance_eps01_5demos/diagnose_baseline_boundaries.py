"""Read-only two-case IK boundary diagnosis using the shared replay backend.

No physics loop or replacement optimizer. Interventions are process-local
patches of the existing shared controller, stored separately from batch data.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
import importlib.util
import json
import hashlib
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_diagnosis',HERE/'diagnose_zero_replay.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
import mujoco
import numpy as np
from tasktol.control.ipopt.controller import IpoptIK
from tasktol.control.ipopt.adapter import IpoptAdapter
from tasktol.control.ipopt import controller as module
from tasktol.counterfactual.runner import CounterfactualTrialSpec,run_counterfactual_trial
from tasktol.geometry import rotation_matrix,rotation_vector
from tasktol.control.task_space.conditioning import velocity_normalized_sigma_min

CASES=['libero_spatial/task_04/demo_1','libero_10/task_08/demo_4']
ROOT=previous.RUN
OUTPUT=HERE/'two_case_boundary_diagnosis'

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,default=previous.serial,ensure_ascii=False,indent=2)+'\n')

def run(case,mode):
    folder=ROOT/case
    document=json.loads((folder/'prepared_input.json').read_text())
    status=json.loads((folder/'annotation_status.json').read_text())
    source_rows=json.loads((Path(document['source']['folder'])/'trajectory.json').read_text())
    original=json.loads(Path(status['zero_trial']).read_text())
    block=status['failed_subtask']
    subset=dict(document,frames=[f for f in document['frames'] if f['subtask_index']==block],
                subtasks=[document['subtasks'][block]])
    prepared,frames=previous.prepared(subset)
    trial_spec=CounterfactualTrialSpec(**original['result']['spec'])
    if mode=='iter21':trial_spec=replace(trial_spec,ipopt_max_iterations=21)
    if mode=='iter100':trial_spec=replace(trial_spec,ipopt_max_iterations=100)
    at={'local':-1}; trace=[];solver=[]
    actual_solve=IpoptIK.solve;actual_adapter=IpoptAdapter.solve
    actual_chart=module.rotation_release_error_and_jacobian
    def observe(ctx):at['local']=ctx.frame_index
    def geometry(controller,q):
        q=np.asarray(q);state=controller.kinematics.evaluate(q,include_manipulability=True,include_link_points=False)
        singular=np.linalg.svd(state.jacobian,compute_uv=False)
        lower=q-controller.handles.joint_lower;upper=controller.handles.joint_upper-q
        margins=np.minimum(lower,upper)
        return dict(q_rad=q,nearest_joint_limit_rad=float(min(margins)),
            nearest_joint_limit_deg=float(np.rad2deg(min(margins))),nearest_joint=int(np.argmin(margins))+1,
            limit_margins_deg=np.rad2deg(margins),singular_values=singular,
            sigma_min_full=float(singular[-1]),jacobian_condition=float(singular[0]/singular[-1]),
            sigma_min_velocity_normalized=float(velocity_normalized_sigma_min(state.jacobian,controller.handles.spec.joint_velocity_limits)),
            position_m=state.position,rotation=state.rotation)
    def chart(kin,state,step):
        if mode!='same_frame_log': return actual_chart(kin,state,step)
        ref=step.stage_reference_rotation()
        transform=step.tolerance_frame.T@ref
        return (transform@kin.pose_error(state,np.zeros(3),ref)[3:],
                transform@kin.pose_error_jacobian(state,ref)[3:])
    def adapter(self,*args,**kwargs):
        result=actual_adapter(self,*args,**kwargs)
        solver.append(dict(local_frame=at['local'],source_frame=frames[at['local']]['source_frame_index'],
            raw_q=result.q.copy(),iterations=result.iterations,status=result.status,status_code=result.status_code,
            validation=asdict(result.validation),optimality=result.optimality,cost=result.cost))
        return result
    def solve(self,data,target,active_dofs,tolerance_rotation=None,rotation_release_step=None,**kwargs):
        i=at['local'];source_i=frames[i]['source_frame_index'];row=source_rows[source_i]
        if mode=='align_handoff':
            rotation_release_step=replace(rotation_release_step,
                stage_handoff_rotation=rotation_release_step.stage_handoff_nominal_rotation.copy())
        if mode=='absolute':
            rotation_release_step=None;tolerance_rotation=None
            for axis in range(3,6):self.set_axis_task(axis,None)
        if mode=='source_history':
            first=frames[0]['source_frame_index'];initial=np.asarray(source_rows[first]['pre_action']['joint_position_rad'])
            self._xopt=None if i==0 else np.asarray(source_rows[source_i-1]['joint_target_rad'])
            self._previous_state=None if i==0 else (np.asarray(source_rows[source_i-2]['joint_target_rad']) if i>=2 else initial)
            self._previous_state2=None if i==0 else (np.asarray(source_rows[source_i-3]['joint_target_rad']) if i>=3 else initial)
            self._warm_start=None
        problem=self._prepare_single_step_nlp(data,target,active_dofs,tolerance_rotation,rotation_release_step)
        source_witness=np.asarray(row['joint_target_rad'])
        constraints=problem.constraint_evaluation(source_witness)[0]
        entry=dict(local_frame=i,source_frame=source_i,time_s=source_i*.05,
            source_phase=row['phase'],annotation_phase=frames[i]['phase'],original_success=bool(row['success']),
            original_optimizer=row['optimizer'],source_witness_constraints=asdict(constraints),
            seed=problem.seed.copy(),target_position=target.position.copy(),target_rotation=target.rotation.copy(),
            original_joint_witness=geometry(self,source_witness),seed_geometry=geometry(self,problem.seed),
            source_pre_actual_geometry=geometry(self,row['pre_action']['joint_position_rad']),
            position_target_difference_vs_source_m=float(np.max(np.abs(target.position-row['nominal_target']['position_m']))))
        if rotation_release_step is not None:
            entry.update(stage_reference_error_vs_nominal_rad=float(np.linalg.norm(rotation_vector(target.rotation.T@rotation_release_step.stage_reference_rotation()))),
                stage_reference_rotation=rotation_release_step.stage_reference_rotation(),tolerance_frame=rotation_release_step.tolerance_frame.copy())
        before=self.fork() if mode.startswith('multistart') else None
        command,diagnostics=actual_solve(self,data,target,active_dofs,tolerance_rotation,rotation_release_step,**kwargs)
        entry.update(feasible=diagnostics.feasible,command=command.copy(),raw_candidate_geometry=geometry(self,solver[-1]['raw_q']))
        if before is not None and not diagnostics.feasible:
            seeds=[('original_previous_command',problem.seed.copy()),('original_saved_witness',source_witness.copy()),
                   ('last_raw_candidate',solver[-1]['raw_q'].copy()),('recorded_actual_state',np.asarray(row['pre_action']['joint_position_rad']))]
            rng=np.random.default_rng(20261106)
            for k in range(8):
                seeds.append((f'witness_perturb_{k}',np.clip(source_witness+rng.normal(0,.35,source_witness.size),self.handles.joint_lower+1e-6,self.handles.joint_upper-1e-6)))
            probes=[];probe_raw=[]
            def capture_probe(adapter_self,*adapter_args,**adapter_kwargs):
                result=actual_adapter(adapter_self,*adapter_args,**adapter_kwargs)
                probe_raw.append(dict(q=result.q.copy(),validation=asdict(result.validation),status_code=result.status_code))
                return result
            with patch.object(IpoptAdapter,'solve',capture_probe):
                for label,seed in seeds:
                    clone=before.fork();clone.settings=replace(clone.settings,max_iterations=300,max_cpu_time_s=1e6)
                    clone.set_warm_start(seed)
                    q,diag=actual_solve(clone,data,target,active_dofs,tolerance_rotation,rotation_release_step,**kwargs)
                    probes.append(dict(seed_label=label,seed=seed,iterations=diag.iterations,feasible=diag.feasible,
                        position_residual_m=diag.position_residual,rotation_residual_rad=diag.strict_rotation_residual,
                        status=diag.status,published_q=q.copy(),raw_candidate=probe_raw[-1],raw_geometry=geometry(clone,probe_raw[-1]['q'])))
            entry['same_problem_multistart']=probes
        trace.append(entry)
        return command,diagnostics
    with (patch('mujoco.mj_step',side_effect=AssertionError('No physics allowed')),
          patch.object(IpoptIK,'solve',solve),patch.object(IpoptAdapter,'solve',adapter),
          patch.object(module,'rotation_release_error_and_jacobian',chart)):
        result=run_counterfactual_trial(trial_spec,prepared_nominal=prepared,dataset_nominal_frames=frames,
                                        kinematic_step_observer=observe).as_dict()
    value=dict(case=case,mode=mode,failed_subtask=block,frame_count=len(frames),physics_steps=0,
        source_frame_start=frames[0]['source_frame_index'],source_folder=document['source']['folder'],
        original_trial=status['zero_trial'],trace=trace,solver_results=solver,
        hard_failures=result['evaluation']['hard_failure_details'],
        complete=result['evaluation']['trajectory_complete'],
        kinematic_feasible_without_table=result['evaluation']['kinematic_feasible_without_table'],
        mujoco_version=mujoco.__version__)
    dest=OUTPUT/case/f'{mode}.json';dump(dest,value)
    print(json.dumps(dict(case=case,mode=mode,frames=len(trace),complete=value['complete'],
        last_source_frame=trace[-1]['source_frame'],last_candidate=solver[-1]['validation'],
        sigma_min=trace[-1]['raw_candidate_geometry']['sigma_min_full'],
        nearest_limit_deg=trace[-1]['raw_candidate_geometry']['nearest_joint_limit_deg'])),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES,action='append');parser.add_argument('--modes',nargs='+',default=['baseline','iter21','iter100','same_frame_log','align_handoff','absolute','source_history'])
    args=parser.parse_args()
    for case in args.case or CASES:
        for mode in args.modes:run(case,mode)
