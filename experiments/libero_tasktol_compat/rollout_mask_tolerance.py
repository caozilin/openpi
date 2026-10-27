"""Paired current-client LIBERO replay with task masks and +/-30deg boxes.

Only orchestration and observation live here. LiberoTaskTolSession owns solving,
reference integration, and native env.step; no custom physics/controller loop.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT.parent/'franka_mujoco'
RUN=ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_fixed_v2'
SELECTION=ROOT/'data/libero40_mask_annotation/20261019_mainview_v1'
sys.path[:0]=[str(ROOT),str(SHARED),str(ROOT/'third_party/libero'),str(ROOT/'examples/libero')]


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('w') as file:
        file.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        file.flush();os.fsync(file.fileno())
    tmp.replace(path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources():
    # All shared source files are hashed; only this experiment writes outputs.
    paths=[*sorted((SHARED/'tasktol').rglob('*.py')),Path(__file__),
        ROOT/'examples/libero/mask_viewer/phase_adapter.py',
        ROOT/'examples/libero/regenerate_libero_dataset.py',
        Path(__file__).with_name('observe_online_ipopt.py')]
    return {str(p):sha(p) for p in paths}


def worker(output,seed,mode,reference_transform=None,observation_callback=None):
    import h5py
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.simulation.libero import LiberoTaskTolSession
    from tasktol.control.ipopt import IpoptSettings
    from tasktol.control.task_space.objective import ObjectiveSettings
    from mask_viewer.phase_adapter import shared_classifier
    from examples.libero import regenerate_libero_dataset as recorder
    from experiments.libero_tasktol_compat.observe_online_ipopt import OnlineIpoptObserver
    assert mujoco.__version__=='3.2.3'
    plan=json.loads((output/'plan.json').read_text())
    assert freeze_sources()==plan['source_sha256'],'Sources changed since plan was frozen'
    job=plan['jobs'][seed]
    task=plan['task'];demo=job['demo'];root=Path(job['selected_root']) if job.get('selected_root') else None
    trial=root/task['suite']/f"{task['task_id']:02d}"/demo if root else None
    reference_path=Path(job['reference_path']) if job.get('reference_path') else trial/'state_reference.hdf5' if trial else None
    reference=None
    if reference_path:
        with h5py.File(reference_path) as f:reference={k:v[()] for k,v in f.items()}
    with h5py.File(job['source_hdf5']) as f:initial=f['data'][demo]['states'][0]
    task_source=json.loads((SELECTION/'selection_manifest.json').read_text())['tasks'][plan['case']]
    with h5py.File(task_source['processed_hdf5']) as f:
        config=json.loads(f.attrs['configuration']);order=config['demo_ids'];number=order.index(demo)
        before_rng=json.loads(f['regeneration/episodes'][order[number-1]].attrs['rng_state']) if number else None
        after_rng=json.loads(f['regeneration/episodes'][demo].attrs['rng_state'])
    dest=output/f'seed_{seed:02d}'/mode
    dest.mkdir(parents=True,exist_ok=False)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task['suite']/(task['task_name']+'.bddl')),
        controller='OSC_POSE',camera_names=['agentview','robot0_eye_in_hand'] if observation_callback else ['agentview'],
        camera_heights=256,camera_widths=256,control_freq=20)
    rows=[];states=[];observer=OnlineIpoptObserver();ever=False
    try:
        # Restore the validated recorded-model initialization exactly; the
        # native XML string's float serialization is not used as a state reset.
        env.seed(config['seed'])
        env.env.object_property_initializers*=number+1
        recorder._restore_rng(before_rng if before_rng is not None else dict(numpy=recorder._rng_state()['numpy'],python=after_rng['python']))
        env.reset()
        obs=env.set_init_state(initial)
        np.testing.assert_array_equal(env.sim.get_state().flatten(),initial)
        model=env.sim.model._model
        physical_hash=hashlib.sha256(b''.join(np.asarray(getattr(model,k)).tobytes()
            for k in ['body_pos','body_quat','body_mass','body_inertia','geom_friction','jnt_range'])).hexdigest()
        if trial:
            prior=json.loads((trial/'ipopt_state_B10/report.json').read_text())
            assert physical_hash==prior['model_sha256'],'Saved scene differs from selected recording'
        # Rollout RNG seed is distinct from the initialization replay RNG.
        # Each seed uses a different recorded demo initial state and action path.
        env.seed(seed)
        session=LiberoTaskTolSession(env,kp=150,joint_control_mode='osc_task_joint_null',solver_settings=IpoptSettings())
        assert session.ik.settings==IpoptSettings()
        assert session.ik.objective_settings==ObjectiveSettings.delta_q_squared_conditioning_safeguard()
        if reference_transform is not None:
            reference=reference_transform(reference,session)
            with h5py.File(dest/'state_reference.hdf5','w') as f:
                for k,v in reference.items():f.create_dataset(k,data=v)
                f.attrs.update(source=job['source_hdf5'],demo=demo,dt_s=.05,time_scale=1,
                    gripper_policy='preserve',projection_posture_weight=0.)
        assert reference is not None,'An offline reference or reference factory is required'
        actions=reference['actions'];anchor_p=reference['positions'][0];anchor_r=reference['rotations'][0]
        source_p=reference['source_positions_m'];source_r=reference['source_rotations']
        session.target.position=anchor_p.copy();session.target.rotation=anchor_r.copy()
        session.release.accepted_rotation=anchor_r.copy()
        classifier_type,phase_model,_=shared_classifier();classifier=classifier_type(phase_model)
        observer.ik=session.ik
        Image.fromarray(obs['agentview_image'][::-1,::-1].copy()).save(dest/'initial.png')
        with imageio.get_writer(dest/'main_rgb.mp4',fps=20,codec='libx264',quality=7) as video,observer:
            for i,action in enumerate(actions):
                if observation_callback is not None:observation_callback(i,action,obs,env)
                states.append(env.sim.get_state().flatten().copy())
                video.append_data(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
                fingers=env.sim.data.qpos[session.handles.finger_qpos_ids].copy()
                phase=classifier.update(np.clip(fingers*[1.,-1.],0,.04),bool(action[6]>0))
                protocol_phase={'pregrasp':'APP','grasp':'ENG','postgrasp':'EXEC','release':'DIS'}[phase.phase.key]
                semantic=task['masks'].get(protocol_phase,'000') if phase.subtask_index==0 else '000'
                if phase.subtask_index>0:protocol_phase='outside_subtask'
                obs,row=session.step(action,phase=protocol_phase,subtask_index=phase.subtask_index,
                    semantic_mask=semantic,bounds_rad=np.full(6,np.deg2rad(30)) if mode=='mask30' else None)
                row.update(observed_phase=phase.phase.key,phase_aperture_m=phase.aperture_m,phase_motion_m=phase.motion_m,
                    observed_subtask_index=phase.subtask_index,mask_prediction=semantic)
                if not row['aborted']:
                    assert row['physics_steps']==25
                    after_r=Rotation.from_rotvec(row['post_action']['rotation_vector_rad']).as_matrix()
                    target_r=Rotation.from_rotvec(row['nominal_target']['rotation_vector_rad']).as_matrix()
                    row.update(source_next_position_error_mm=float(np.linalg.norm(np.asarray(row['post_action']['position_m'])-source_p[i+1])*1000),
                        source_next_rotation_error_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(after_r@source_r[i+1].T).as_rotvec()))),
                        nominal_rotation_error_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(after_r@target_r.T).as_rotvec()))))
                    if mode=='mask30':
                        step=session.release._accepted_step
                        assert step is not None
                        coords=step.stage_relative_coordinates(after_r)
                        row['actual_rotation_release_rad']=coords.tolist()
                        row['actual_tolerance_violation_rad']=step.predicted_violation(coords).tolist()
                else:assert row['physics_steps']==0
                if protocol_phase not in ('APP','EXEC') or mode=='off':assert row['effective_mask']=='000'
                assert all(a=='0' or b=='1' for a,b in zip(row['effective_mask'],semantic,strict=True))
                rows.append(row);ever|=row['success']
                if row['aborted']:break
        final=env.sim.get_state().flatten().copy()
        Image.fromarray(obs['agentview_image'][::-1,::-1].copy()).save(dest/'final.png')
        final_goal=bool(env.check_success())
        predicates=[dict(predicate=p,satisfied=bool(env.env._eval_predicate(p))) for p in env.env.parsed_problem['goal_state']]
        report=dict(seed=seed,demo=demo,mode=mode,success=bool(final_goal and len(rows)==len(actions) and not rows[-1]['aborted']),
            native_goal_at_stop=final_goal,ever_success=bool(ever),planned_frames=len(actions),executed_frames=sum(not r['aborted'] for r in rows),
            execution_rejected=rows[-1]['aborted'],rejected_frame=rows[-1]['index'] if rows[-1]['aborted'] else None,
            solver_settings=asdict(session.ik.settings),objective_settings=asdict(session.ik.objective_settings),
            initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),model_sha256=physical_hash,
            initialization_protocol='validated original reset sampler/RNG + set_init_state; exact physical model hash matched; rollout seed applied after initialization',
            actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),selected_root=str(root) if root else None,
            native_predicates=predicates,phase_rule=dict(frequency_hz=20,stable_frames=3,motion_threshold_m=.0002),
            tolerance_protocol=dict(masks=task['masks'],bounds_deg=[30]*6 if mode=='mask30' else None,
                strict_phases=['ENG','DIS','outside_subtask'],frame='shared client stage_endpoint',subtask_rebase='shared client, on observed cycle change'),
            reference_scope='Saved selected independent physical increments; current-state feedback for online phase and joint execution; no VLA inference; no future states supplied to controller')
    finally:env.close()
    executed=[r for r in rows if not r['aborted']]
    def stats(values):
        if not values:return None
        a=np.asarray(values,dtype=float)
        return dict(mean=float(np.mean(a)),p95=float(np.percentile(a,95)),max=float(np.max(a)))
    report['tracking']=dict(source_position_mm=stats([r['source_next_position_error_mm'] for r in executed]),
        source_rotation_deg=stats([r['source_next_rotation_error_deg'] for r in executed]),
        nominal_position_mm=stats([r['tracking_position_error_m']*1000 for r in executed]),
        nominal_rotation_deg=stats([r['nominal_rotation_error_deg'] for r in executed]),
        solver_wall_ms=stats([r['optimizer']['elapsed_ms'] for r in rows]),
        optimized_tolerance_violation_deg=stats([np.degrees(max(r.get('predicted_tolerance_violation_rad',[0]))) for r in executed]),
        actual_tolerance_violation_deg=stats([np.degrees(max(r.get('actual_tolerance_violation_rad',[0]))) for r in executed]))
    report['effective_mask_counts']=dict(__import__('collections').Counter(r['effective_mask'] for r in rows))
    report['phase_counts']=dict(__import__('collections').Counter(r['phase'] for r in rows))
    diagnostics=observer.records(rows)
    report['solver_status_counts']=dict(__import__('collections').Counter(str(r.get('status_code','not_called')) for r in diagnostics))
    report['cpu_timeout_frames']=[r['index'] for r in diagnostics if r.get('status_code')==-4]
    report['source_unchanged']=freeze_sources()==plan['source_sha256']
    assert report['source_unchanged']
    np.savez_compressed(dest/'states.npz',pre_action_states=np.asarray(states),final_state=final)
    dump(dest/'trajectory.json',rows);dump(dest/'solver_observations.json',diagnostics);dump(dest/'report.json',report)
    print(json.dumps(dict(seed=seed,demo=demo,mode=mode,success=report['success'],executed=report['executed_frames'],rejected=report['rejected_frame'])),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--mode',choices=['off','mask30']);p.add_argument('--case',type=int,default=0)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    args=p.parse_args();out=args.output.resolve()
    if args.worker:worker(out,args.seed,args.mode);return
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False)
        task=json.loads((SELECTION/'selection_manifest.json').read_text())['tasks'][args.case]
        annotation=next(t for t in json.loads((SELECTION/'mask_annotations.json').read_text())['tasks']
            if (t['suite'],t['task_id'])==(task['suite'],task['task_id']))
        assert len(annotation['subtasks'])==1,'This probe deliberately uses one semantic subtask'
        task={k:task[k] for k in ['suite','task_id','task_name','language']}
        task['masks']={phase:annotation['subtasks'][0][phase]['effective_mask'] for phase in ['APP','EXEC']}
        jobs=sorted([j for j in json.loads((RUN/'summary.json').read_text())['results'] if j['case']==args.case],key=lambda j:int(j['demo'][5:]))
        assert len(jobs)==10
        sources=freeze_sources();snap=out/'source_snapshot';snap.mkdir()
        for i,path in enumerate(sources):shutil.copy2(path,snap/f'{i:03d}_{Path(path).name}')
        dump(out/'plan.json',dict(case=args.case,task=task,jobs=jobs,source_sha256=sources,seeds=list(range(10)),
            design='10 distinct recorded source initial states and selected action sequences, seed 0..9 paired by sorted demo. Validated reset RNG protocol restores the exact selected physical model; rollout seed is applied after initialization. Not a new VLA policy evaluation.',
            source_summary_sha256=sha(RUN/'summary.json'),annotation_sha256=sha(SELECTION/'mask_annotations.json')))
    plan=json.loads((out/'plan.json').read_text());assert freeze_sources()==plan['source_sha256']
    assert all(0<=s<10 for s in args.seeds)
    entries=[(seed,mode) for seed in args.seeds for mode in ['off','mask30']]
    def run(entry):
        seed,mode=entry;folder=out/f'seed_{seed:02d}'/mode;report=folder/'report.json'
        if args.resume and report.is_file():return json.loads(report.read_text())
        log=out/f'seed_{seed:02d}_{mode}.log'
        with log.open('w') as stream:
            done=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--output',str(out),'--seed',str(seed),'--mode',mode],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:return dict(seed=seed,mode=mode,infrastructure_error=True,returncode=done.returncode,log=str(log))
        return json.loads(report.read_text())
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run,entries):
            records.append(record)
            dump(out/'summary.json',dict(complete=False,completed=len(records),results=records))
            print(json.dumps({k:record[k] for k in ['seed','mode','success','execution_rejected','infrastructure_error'] if k in record}),flush=True)
    valid=[r for r in records if not r.get('infrastructure_error')]
    pairs=[]
    for seed in range(10):
        values={r['mode']:r for r in valid if r['seed']==seed}
        if len(values)==2:
            for key in ['initial_state_sha256','model_sha256','actions_sha256']:assert values['off'][key]==values['mask30'][key]
            pairs.append(dict(seed=seed,demo=values['off']['demo'],off_success=values['off']['success'],mask30_success=values['mask30']['success']))
    summary=dict(complete=len(valid)==20,completed=len(records),task=plan['task'],pairs=pairs,
        counts={mode:dict(success=sum(r['success'] for r in valid if r['mode']==mode),
            rejected=sum(r['execution_rejected'] for r in valid if r['mode']==mode),total=sum(r['mode']==mode for r in valid)) for mode in ['off','mask30']},
        source_unchanged=freeze_sources()==plan['source_sha256'],results=records)
    dump(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['complete','counts','source_unchanged']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
