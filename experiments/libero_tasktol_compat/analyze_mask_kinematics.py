"""Read saved paired trajectories; verify native FK without physics stepping.

Separate source-to-reference offsets, IK residuals, and execution tracking.
Stage release is the existing fixed-XYZ stage-relative coordinate, not the sum
of its successive values and not a world-axis rotation vector.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT.parent/'franka_mujoco'
sys.path[:0]=[str(ROOT),str(SHARED),str(ROOT/'third_party/libero')]


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def analyze(run):
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    from examples.libero import regenerate_libero_dataset as recorder
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from tasktol.control.task_space.kinematics import RobotKinematics
    from tasktol.simulation.libero import libero_handles
    from tasktol.geometry import rotation_tolerance_coordinates,stage_reference_rotation
    from tasktol.tolerance.frame import box_tolerance_frame

    plan=json.loads((run/'plan.json').read_text());summary=json.loads((run/'summary.json').read_text())
    assert summary['complete']
    # Relevant FK/geometry code must match the recording. Optional SO3-box
    # preprocessing is not executed by this read-only kinematics analysis.
    for rel in ['tasktol/geometry.py','tasktol/control/task_space/kinematics.py',
                'tasktol/simulation/libero.py','tasktol/simulation/robot.py',
                'tasktol/tolerance/frame.py','tasktol/tolerance/release_state.py']:
        path=SHARED/rel
        assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['source_sha256'][str(path)],rel
    out=run/'kinematic_analysis';out.mkdir(exist_ok=False)
    task=plan['task'];source_task=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks'][plan['case']]
    all_frames=[];stages=[];episodes=[];verifications=[];pair_records=[]

    def stats(values):
        a=np.asarray(values,dtype=float)
        return dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()),min=float(a.min()),count=len(a)) if a.size else None
    def matrix(rv):return Rotation.from_rotvec(rv).as_matrix()
    def angle(r):return float(np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec())))
    def norm(v):return float(np.linalg.norm(v))

    for seed,job in enumerate(plan['jobs']):
        demo=job['demo'];trial=Path(job['selected_root'])/task['suite']/f"{task['task_id']:02d}"/demo
        with h5py.File(trial/'state_reference.hdf5') as f:
            source_p=f['source_positions_m'][()];anchor_r=f['rotations'][0]
            reference_p=f['positions'][()]
        with h5py.File(trial/'ipopt_state_B10/recording.hdf5') as f:initial=f['attempts'][demo]['tasktol/initial_state'][()]
        with h5py.File(source_task['processed_hdf5']) as f:
            config=json.loads(f.attrs['configuration']);order=config['demo_ids'];number=order.index(demo)
            before=json.loads(f['regeneration/episodes'][order[number-1]].attrs['rng_state']) if number else None
            after=json.loads(f['regeneration/episodes'][demo].attrs['rng_state'])
        env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task['suite']/(task['task_name']+'.bddl')),
            controller='OSC_POSE',use_camera_obs=False,has_offscreen_renderer=False,control_freq=20)
        paired={};seed_frames={}
        try:
            env.seed(config['seed']);env.env.object_property_initializers*=number+1
            recorder._restore_rng(before if before is not None else dict(numpy=recorder._rng_state()['numpy'],python=after['python']))
            env.reset();env.set_init_state(initial)
            model=env.sim.model._model
            model_hash=hashlib.sha256(b''.join(np.asarray(getattr(model,k)).tobytes() for k in ['body_pos','body_quat','body_mass','body_inertia','geom_friction','jnt_range'])).hexdigest()
            handles=libero_handles(env);kin=RobotKinematics(model,handles)
            for mode in ['off','mask30']:
                folder=run/f'seed_{seed:02d}'/mode
                rows=json.loads((folder/'trajectory.json').read_text());paired[mode]=rows
                report=json.loads((folder/'report.json').read_text());assert model_hash==report['model_sha256']
                records=[];previous_q=np.asarray(rows[0]['pre_action']['joint_position_rad']);previous_step=np.zeros(7)
                previous_nominal=anchor_r.copy();previous_optimized=anchor_r.copy();previous_key=None;group=[]
                def finish_stage(group):
                    if not group:return
                    vectors=np.asarray([r['release_deg'] for r in group])
                    actual=np.asarray([r['actual_release_deg'] for r in group])
                    last=group[-1]
                    stages.append(dict(seed=seed,demo=demo,mode=mode,phase=last['phase'],subtask_index=last['subtask_index'],
                        start_frame=group[0]['frame'],end_frame_exclusive=last['frame']+1,duration_s=len(group)/20,
                        release_end_deg=vectors[-1].tolist(),release_peak_abs_deg=np.max(np.abs(vectors),axis=0).tolist(),
                        release_coordinate_total_variation_deg=np.abs(np.diff(np.vstack([np.zeros(3),vectors]),axis=0)).sum(axis=0).tolist(),
                        actual_stage_relative_end_deg=actual[-1].tolist(),actual_stage_relative_peak_abs_deg=np.max(np.abs(actual),axis=0).tolist(),
                        command_global_nominal_deviation_end_deg=last['command_nominal_rotation_error_deg'],
                        actual_global_nominal_deviation_end_deg=last['actual_nominal_rotation_error_deg'],
                        ik_position_residual_mm=stats([r['ik_residual_mm'] for r in group]),
                        execution_tracking_mm=stats([r['execution_tracking_mm'] for r in group]),
                        source_tracking_mm=stats([r['source_error_mm'] for r in group])))
                for row in rows:
                    i=row['index']
                    if row['aborted']:
                        finish_stage(group);group=[];break
                    q=np.asarray(row['joint_target_rad']);actual_q=np.asarray(row['post_action']['joint_position_rad'])
                    command=kin.evaluate(q,include_manipulability=False,include_link_points=False)
                    actual=kin.evaluate(actual_q,include_manipulability=False,include_link_points=False)
                    p_ref=np.asarray(row['nominal_target']['position_m']);p_src=source_p[i+1]
                    p_actual=np.asarray(row['post_action']['position_m'])
                    np.testing.assert_allclose(command.position,row['native_joint_action'][:3],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(actual.position,p_actual,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(p_ref,reference_p[i+1],rtol=0,atol=1e-10)
                    ik=command.position-p_ref;execution=p_actual-command.position;ref=p_ref-p_src;total=p_actual-p_src
                    np.testing.assert_allclose(total,ref+ik+execution,rtol=0,atol=1e-12)
                    scaled=command.jacobian.copy();scaled[:3]/=.3
                    singular=float(np.linalg.svd(scaled,compute_uv=False)[-1])
                    actual_scaled=actual.jacobian.copy();actual_scaled[:3]/=.3
                    step=q-previous_q
                    nominal=matrix(row['nominal_target']['rotation_vector_rad'])
                    key=(row['subtask_index'],row['phase'])
                    if row['subtask_rebased']:previous_nominal=previous_optimized.copy()
                    if key!=previous_key:
                        finish_stage(group);group=[]
                        handoff=previous_optimized.copy();handoff_nominal=previous_nominal.copy()
                    release=np.asarray(row.get('rotation_release_rad',[0,0,0]))
                    actual_release=np.asarray(row.get('actual_rotation_release_rad',[0,0,0]))
                    if mode=='mask30':
                        frame=box_tolerance_frame(nominal)
                        stage_ref=stage_reference_rotation(handoff,handoff_nominal,nominal)
                        coords=rotation_tolerance_coordinates(command.rotation,stage_ref,frame)
                        effective=np.asarray([c=='1' for c in row['effective_mask']])
                        np.testing.assert_allclose(np.where(effective,coords,0),release,rtol=0,atol=1e-9)
                        np.testing.assert_allclose(rotation_tolerance_coordinates(actual.rotation,stage_ref,frame),actual_release,rtol=0,atol=1e-9)
                    record=dict(seed=seed,demo=demo,mode=mode,frame=i,phase=row['phase'],subtask_index=row['subtask_index'],
                        source_error_mm=norm(total)*1000,reference_offset_from_source_mm=norm(ref)*1000,
                        ik_residual_mm=norm(ik)*1000,execution_tracking_mm=norm(execution)*1000,
                        reference_position_m=p_ref.tolist(),command_position_m=command.position.tolist(),actual_position_m=p_actual.tolist(),
                        ik_residual_vector_mm=(ik*1000).tolist(),execution_tracking_vector_mm=(execution*1000).tolist(),source_error_vector_mm=(total*1000).tolist(),
                        sigma_min_scaled=singular,actual_sigma_min_scaled=float(np.linalg.svd(actual_scaled,compute_uv=False)[-1]),
                        joint_limit_margin_deg=float(np.degrees(np.min(np.minimum(q-handles.joint_lower,handles.joint_upper-q)))),
                        joint_reference_step_norm_rad=norm(step),joint_reference_step_max_deg=float(np.degrees(np.max(np.abs(step)))),
                        joint_reference_velocity_max_rad_s=float(np.max(np.abs(step))/.05),
                        joint_reference_acceleration_max_rad_s2=float(np.max(np.abs(step-previous_step))/.05**2),
                        joint_tracking_error_norm_rad=norm(actual_q-q),
                        release_deg=np.degrees(release).tolist(),actual_release_deg=np.degrees(actual_release).tolist(),
                        command_nominal_rotation_error_deg=angle(command.rotation@nominal.T),
                        actual_nominal_rotation_error_deg=angle(actual.rotation@nominal.T))
                    records.append(record);group.append(record)
                    previous_q=q.copy();previous_step=step.copy();previous_nominal=nominal.copy();previous_optimized=command.rotation.copy();previous_key=key
                finish_stage(group)
                seed_frames[mode]=records;all_frames.extend(records)
                episodes.append(dict(seed=seed,demo=demo,mode=mode,success=report['success'],executed_frames=len(records),
                    metrics={name:stats([r[name] for r in records]) for name in ['source_error_mm','reference_offset_from_source_mm','ik_residual_mm','execution_tracking_mm',
                        'sigma_min_scaled','actual_sigma_min_scaled','joint_limit_margin_deg','joint_reference_step_max_deg','joint_reference_velocity_max_rad_s','joint_reference_acceleration_max_rad_s2','joint_tracking_error_norm_rad']},
                    joint_command_path_length_rad=sum(r['joint_reference_step_norm_rad'] for r in records)))
            common=min(len(seed_frames['off']),len(seed_frames['mask30']))
            reference_difference=[];command_difference=[];actual_difference=[];identity=[]
            for i in range(common):
                a,b=(seed_frames[m][i] for m in ['off','mask30'])
                dr=np.asarray(b['reference_position_m'])-a['reference_position_m'];dc=np.asarray(b['command_position_m'])-a['command_position_m'];da=np.asarray(b['actual_position_m'])-a['actual_position_m']
                di=np.asarray(b['ik_residual_vector_mm'])-a['ik_residual_vector_mm'];de=np.asarray(b['execution_tracking_vector_mm'])-a['execution_tracking_vector_mm']
                reference_difference.append(norm(dr)*1000);command_difference.append(norm(dc)*1000);actual_difference.append(norm(da)*1000)
                identity.append(norm(da*1000-dr*1000-di-de))
            pair_records.append(dict(seed=seed,demo=demo,common_frames=common,reference_on_minus_off_mm=stats(reference_difference),
                command_on_minus_off_mm=stats(command_difference),actual_on_minus_off_mm=stats(actual_difference),decomposition_closure_mm=stats(identity)))
            verifications.append(dict(seed=seed,model_hash_matches_recording=True,native_command_equals_fk=True,post_state_equals_fk=True,nominal_matches_reconstruction=True,stage_release_recomputed=True))
        finally:env.close()
        print(f'{seed}: {demo} FK and position decomposition verified',flush=True)
    complete_seeds=[p['seed'] for p in summary['pairs'] if p['off_success'] and p['mask30_success']]
    aggregate={}
    for mode in ['off','mask30']:
        cohort=[r for r in all_frames if r['mode']==mode and r['seed'] in complete_seeds]
        aggregate[mode]={name:stats([r[name] for r in cohort]) for name in ['source_error_mm','reference_offset_from_source_mm','ik_residual_mm','execution_tracking_mm',
            'sigma_min_scaled','actual_sigma_min_scaled','joint_limit_margin_deg','joint_reference_step_max_deg','joint_tracking_error_norm_rad']}
    def csv_write(path,rows):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for r in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    csv_write(out/'frame_metrics.csv',all_frames);csv_write(out/'stage_release.csv',[s for s in stages if s['mode']=='mask30'])
    result=dict(complete=True,physics_steps=0,solver_calls=0,comparable_seeds=complete_seeds,aggregate=aggregate,episodes=episodes,
        pairs=pair_records,stages=stages,verifications=verifications,
        definitions=dict(position_identity='p_actual-p_source = (p_ref-p_source)+(FK(q_cmd)-p_ref)+(p_actual-FK(q_cmd))',
            release='Signed stage-relative fixed-XYZ coordinates in the shared gravity-aligned F frame, relative to transported physical stage handoff; terminal value is net stage release, not sum of per-frame values.',
            variation='Sum of abs coordinate differences from stage zero; a path diagnostic in changing F charts, not net SO3 release or an additional admissible tolerance budget.',
            strict='ENG/DIS have zero additional optimized stage release; inherited APP/EXEC orientation is transported, not reset to original nominal.',
            sigma='6x7 geometric Jacobian with translational rows divided by 0.3m. Small values diagnose conditioning, not global unreachability.'),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'analysis.json',result)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run.resolve())
