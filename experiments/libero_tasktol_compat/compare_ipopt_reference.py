"""Pair direct relabelled-action IPOPT execution with native accumulated OSC."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tasktol.geometry import rotation_matrix, rotation_vector


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values,95)), max=float(values.max()))


def compare(osc, ipopt, dataset, joint_control_mode='native'):
    tasks=[];paired_position=[];paired_rotation=[];source_position=[];tracking_position=[]
    accepted_position_residual=[];accepted_rotation_residual=[];solver_times=[]
    for osc_path in sorted(osc.glob('*/*/report.json')):
        rel=osc_path.parent.relative_to(osc)
        osc_report=json.loads(osc_path.read_text())
        osc_frames=json.loads((osc_path.parent/'trace.json').read_text())
        report=json.loads((ipopt/rel/'report.json').read_text())
        frames=json.loads((ipopt/rel/'trajectory.json').read_text())['frames']
        with h5py.File(dataset/rel/'reference_actions.hdf5','r') as file:
            labels=file['data'][osc_report['demo']]
            expected_actions=labels['actions'][()]
            expected_p=labels['goal_positions_m'][()]
            expected_r=labels['goal_tool_rotations'][()]
        action_error=float(np.max(np.abs(np.array([r['action'] for r in frames])-expected_actions[:len(frames)])))
        nominal_position_error=float(np.max(np.linalg.norm(
            np.array([r['nominal_target']['position_m'] for r in frames])-expected_p[:len(frames)],axis=1)))
        nominal_rotation_error=float(max(np.linalg.norm(rotation_vector(
            rotation_matrix(np.array(r['nominal_target']['rotation_vector_rad'])) @ expected_r[i].T))
            for i,r in enumerate(frames)))
        valid=[(i,r) for i,r in enumerate(frames) if not r['aborted']]
        position=[np.linalg.norm(np.array(r['post_action']['position_m'])-osc_frames[i]['actual_position_m'])*1000 for i,r in valid]
        rotation=[np.linalg.norm(rotation_vector(rotation_matrix(np.array(r['post_action']['rotation_vector_rad'])) @
                   np.array(osc_frames[i]['actual_tool_rotation']).T)) for i,r in valid]
        source=[r['source_replay_position_error_m']*1000 for _,r in valid]
        target_tracking=[r['tracking_position_error_m']*1000 for _,r in valid]
        residual=[r['optimizer']['position_residual_m']*1000 for _,r in valid]
        r_residual=[r['optimizer']['strict_rotation_residual_rad'] for _,r in valid]
        checks=dict(
            source_initial_state_identical=report['initial_state_sha256']==osc_report['initial_state_sha256'],
            fixtures_identical=report['initialization']['fixtures']==osc_report['fixtures'],
            source_reset_rng_verified=report['initialization']['reset_rng_matches_source'] and osc_report['reset_rng_matches_source'],
            reference_mode=report['reference_mode']=='relabelled_actions',
            hdf5_actions_used_directly=action_error==0.0,
            reference_goals_match=nominal_position_error<1e-10 and nominal_rotation_error<1e-9,
            ipopt_called_every_attempt=all(r['optimizer']['solver_called'] for r in frames),
            native_physics_steps=all(r['physics_steps']==(0 if r['aborted'] else 25) for r in frames),
            native_joint_goals_match=all(r['native_joint_goal_error_rad'] is None or r['native_joint_goal_error_rad']<1e-10 for r in frames),
            tolerance_disabled=report['tolerance']['mode']=='off' and all(r['effective_mask']=='000' for r in frames),
            lengths_consistent=len(osc_frames)==len(expected_actions)==report['expected_frames'],
            no_osc_fallback=report['optimizer_backend']=='ipopt' and report['controller']=={
                'native':'JointPositionController','osc_inertia_matched':'InertiaMatchedJointPositionController',
                'osc_task_joint_null':'FactorizedReferenceController'}[joint_control_mode],
        )
        if joint_control_mode == 'osc_inertia_matched':
            diagnostics=report['inertia_diagnostics']
            checks.update(
                requested_joint_control_mode=report['joint_control_mode']==joint_control_mode,
                matched_every_physics_step=diagnostics['calls']==sum(r['physics_steps'] for r in frames),
                positive_effective_inertia=diagnostics['min_effective_eigenvalue']>0,
                uniform_fixed_gain=report['native_controller_kp']==[150.0]*7,
            )
        elif joint_control_mode == 'osc_task_joint_null':
            diagnostics=report['factor_statistics']
            checks.update(
                requested_joint_control_mode=report['joint_control_mode']==joint_control_mode,
                factorized_every_physics_step=diagnostics['calls']==sum(r['physics_steps'] for r in frames),
                native_task_formula_verified=diagnostics['max_native_reconstruction_residual_nm']<1e-9,
                uniform_fixed_task_gain=report['native_controller_kp']==[150.0]*6,
                fixed_null_gain=report['native_null_kp']==diagnostics['null_kp']==10,
                complete_joint_reference_feedback=diagnostics['null_reference']=='joint' and diagnostics['task_feedback']=='osc' and all(r['joint_reference_transport']=='explicit full q_d plus native FK pose command' for r in frames),
            )
        rejected=[dict(index=i,source_index=r['source_index'],status=r['optimizer']['status'],
                       position_residual_mm=r['optimizer']['position_residual_m']*1000,
                       rotation_residual_rad=r['optimizer']['strict_rotation_residual_rad'])
                  for i,r in enumerate(frames) if r['aborted']]
        osc_success_frames=[i for i,r in enumerate(osc_frames) if r['success']]
        ipopt_success_frames=[i for i,r in enumerate(frames) if r['success'] and not r['aborted']]
        tasks.append(dict(task=str(rel),demo=osc_report['demo'],osc_success=osc_report['task_success'],
                          ipopt_success=report['task_success'],status=report['status'],
                          osc_ever_success=bool(osc_success_frames),ipopt_ever_success=bool(ipopt_success_frames),
                          ipopt_first_success_frame=ipopt_success_frames[0] if ipopt_success_frames else None,
                          ipopt_last_success_frame=ipopt_success_frames[-1] if ipopt_success_frames else None,
                          expected_frames=len(osc_frames),attempted_frames=len(frames),executed_frames=len(valid),
                          failure_type=None if report['task_success'] else ('ik_rejected_target' if rejected else 'native_execution_task_failure'),
                          rejected_targets=rejected,paired_eef_position_difference_mm=stats(position),
                          paired_eef_rotation_difference_rad=stats(rotation),source_position_error_mm=stats(source),
                          nominal_target_tracking_error_mm=stats(target_tracking),
                          accepted_ik_position_residual_mm=stats(residual),
                          max_reference_action_difference=action_error,max_nominal_position_difference_m=nominal_position_error,
                          max_nominal_rotation_difference_rad=nominal_rotation_error,checks=checks))
        paired_position.extend(position);paired_rotation.extend(rotation);source_position.extend(source)
        tracking_position.extend(target_tracking);accepted_position_residual.extend(residual);accepted_rotation_residual.extend(r_residual)
        solver_times.extend(r['optimizer']['elapsed_ms'] for r in frames)
    return dict(
        scope=f'40 fixed selected demonstrations; direct physical relabelled HDF5 increments; tolerance off; full joint reference kp150; joint_control_mode={joint_control_mode}; no VLA checkpoint',
        joint_control_mode=joint_control_mode,
        tasks=len(tasks),osc_success=sum(r['osc_success'] for r in tasks),ipopt_success=sum(r['ipopt_success'] for r in tasks),
        success_evaluation=dict(primary='Native success predicate at the final recorded frame, matching preceding full-trajectory comparisons',
                                secondary='Native success predicate reached at an executed frame; corresponds to stopping at first success; this rollout itself retained the full reference sequence'),
        osc_ever_success=sum(r['osc_ever_success'] for r in tasks),ipopt_ever_success=sum(r['ipopt_ever_success'] for r in tasks),
        transient_success_tasks=[r['task'] for r in tasks if r['ipopt_ever_success'] and not r['ipopt_success']],
        never_successful_tasks=[r['task'] for r in tasks if not r['ipopt_ever_success']],
        lost_tasks=[r['task'] for r in tasks if r['osc_success'] and not r['ipopt_success']],
        gained_tasks=[r['task'] for r in tasks if r['ipopt_success'] and not r['osc_success']],
        expected_frames=sum(r['expected_frames'] for r in tasks),attempted_frames=sum(r['attempted_frames'] for r in tasks),
        executed_frames=sum(r['executed_frames'] for r in tasks),
        ik_rejected_tasks=sum(r['failure_type']=='ik_rejected_target' for r in tasks),
        execution_failed_tasks=sum(r['failure_type']=='native_execution_task_failure' for r in tasks),
        metric_scope='Only frames actually executed by IPOPT are paired with OSC; rejected frames and unexecuted suffixes are excluded from trajectory errors',
        paired_eef_position_difference_mm=stats(paired_position),paired_eef_rotation_difference_rad=stats(paired_rotation),
        source_position_error_mm=stats(source_position),nominal_target_tracking_error_mm=stats(tracking_position),
        accepted_ik_position_residual_mm=stats(accepted_position_residual),accepted_ik_rotation_residual_rad=stats(accepted_rotation_residual),
        solver_elapsed_ms=stats(solver_times),
        checks_passed=sum(sum(t['checks'].values()) for t in tasks),checks_total=sum(len(t['checks']) for t in tasks),
        results=tasks,
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--osc',type=Path,required=True)
    parser.add_argument('--ipopt',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--joint-control-mode',choices=['native','osc_inertia_matched','osc_task_joint_null'],default='native')
    args=parser.parse_args();result=compare(args.osc,args.ipopt,args.dataset,args.joint_control_mode)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    if result['tasks']!=40 or result['checks_passed']!=result['checks_total']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
