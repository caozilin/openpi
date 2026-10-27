"""Collect 10 distinct successful reconstructed demos for every LIBERO task.

All execution is delegated to the maintained shared-servo rollout. Trial files
are temporary; only selected complete successes survive. Failure ledgers never
contain state/action/image arrays. Default IPOPT and native timings are frozen.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'franka_mujoco')]
from experiments.libero_tasktol_compat import reconstruct_joint_servo as reconstruction
from experiments.libero_tasktol_compat import rollout_mujoco_joint_servo as servo
from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner

SELECTION=ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'
OLD=HERE/'results/task_pose_default_ipopt400_fixed_v2/summary.json'
GAINS=[1.,.75,.5,.5]
base_freeze=reconstruction.freeze


def freeze():
    return dict(base_freeze(),**{str(Path(__file__)):runner.sha(Path(__file__))})


runner.freeze_sources=freeze


def select_candidate(previous,trial):
    if not trial['full_length'] or trial['rejected']:return False
    if previous is None or not previous['full_length'] or previous['rejected']:return True
    if trial['success'] and not previous['success']:return True
    if previous['success'] and not trial['success']:return False
    return trial['objective']<previous['objective']


class ObservationWriter:
    """Native pre-action camera/state observer, without execution logic."""
    def __init__(self,path,demo):
        import h5py
        self.file=h5py.File(path,'w');self.group=self.file.create_group('data').create_group(demo)
        self.obs=self.group.create_group('obs');self.count=0

    def append(self,group,name,value):
        import numpy as np
        a=np.asarray(value)
        if name not in group:
            group.create_dataset(name,shape=(0,)+a.shape,maxshape=(None,)+a.shape,
                chunks=(1,)+a.shape,dtype=a.dtype,compression='lzf')
        d=group[name];d.resize(self.count+1,axis=0);d[self.count]=a

    def __call__(self,index,action,obs,env):
        import numpy as np
        from scipy.spatial.transform import Rotation
        assert index==self.count
        ee=np.r_[obs['robot0_eef_pos'],Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec()]
        for name,value in dict(agentview_rgb=obs['agentview_image'],eye_in_hand_rgb=obs['robot0_eye_in_hand_image'],
            gripper_states=obs['robot0_gripper_qpos'],joint_states=obs['robot0_joint_pos'],
            ee_states=ee,ee_pos=ee[:3],ee_ori=ee[3:]).items():self.append(self.obs,name,value)
        self.append(self.group,'actions',action)
        self.append(self.group,'states',env.sim.get_state().flatten())
        self.append(self.group,'robot_states',np.r_[obs['robot0_gripper_qpos'],obs['robot0_eef_pos'],obs['robot0_eef_quat']])
        if index==0:self.group.attrs['source_native_model_file']=env.sim.model.get_xml()
        self.count+=1

    def finish(self,report,folder,job,task):
        import numpy as np
        assert report['success'] and self.count==report['planned_frames']==report['executed_frames']
        with np.load(folder/'states.npz')as f:
            np.testing.assert_array_equal(self.group['states'][()],f['pre_action_states'])
            states=np.concatenate([f['pre_action_states'],f['final_state'][None]])
            self.group.create_dataset('next_states',data=states[1:],compression='lzf')
            trace=self.group.create_group('tasktol');trace.create_dataset('initial_state',data=states[0])
            trace.create_dataset('final_state',data=states[-1]);trace.create_dataset('source_step_indices',data=np.arange(self.count))
        np.testing.assert_allclose(np.diff(states[:,0]),.05,rtol=0,atol=1e-10)
        terminal=np.zeros(self.count,dtype=np.uint8);terminal[-1]=1
        for name in ['rewards','dones']:self.group.create_dataset(name,data=terminal)
        self.group.attrs.update(num_samples=self.count,success=True,source_demo=job['demo'],
            source_hdf5=job['source_hdf5'],language=task['language'],report_json=json.dumps(report))
        self.file.attrs.update(control_hz=20,physics_hz=500,observation_timing='pre_action',
            image_orientation='native LIBERO; rotate once at training export',
            action_units='xyz metres, world rotvec radians, original gripper',
            controller='shared MuJoCo joint servo; requires saved actuator overrides, not stock OSC',
            source_native_xml_requires_servo_override=True)
        self.file.flush()

    def close(self):self.file.close()


def pose_metrics(model,folder,reference,source_states,report):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from experiments.libero_tasktol_compat.task_pose_object_metrics import task_cost
    rows=[r for r in json.loads((folder/'trajectory.json').read_text())if not r['aborted']]
    with np.load(folder/'states.npz')as f:actual=np.concatenate([f['pre_action_states'][:len(rows)],f['final_state'][None]])
    p=np.asarray([reference['source_positions_m'][0]]+[r['post_action']['position_m']for r in rows])
    r=np.concatenate([reference['source_rotations'][:1],Rotation.from_rotvec([row['post_action']['rotation_vector_rad']for row in rows]).as_matrix()]) if rows else reference['source_rotations'][:1]
    n=len(actual)
    result=task_cost(model,actual,source_states[:n],p,r,reference['source_positions_m'][:n],reference['source_rotations'][:n])
    result.update(success=report['success'],rejected=report['execution_rejected'],
        full_length=report['executed_frames']==report['planned_frames'] and not report['execution_rejected'])
    return result


def attempt(out,case,index):
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation
    from probe_original_speed_reference import make_reference
    from rerecord_state_reference import state_reference
    from experiments.libero_tasktol_compat.osc_reference import decode_source_goals
    from tasktol.control.ipopt import IpoptSettings
    from tasktol.control.task_space.objective import ObjectiveSettings
    master=json.loads((out/'plan.json').read_text());entry=master['tasks'][case];job=entry['candidates'][index]
    demo=job['demo'];task=entry['task'];key=f'case_{case:02d}_{demo}'
    scratch=out/'.scratch'/key
    if scratch.exists():shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    started=time.monotonic();ledger=[];parent=None;best=None;seen=set()
    with h5py.File(job['source_hdf5'])as f:
        g=f['data'][demo];source_states=g['states'][()];source_actions=g['actions'][()]
        source_hash=hashlib.sha256(source_states.tobytes()+source_actions.tobytes()+str(g.attrs['model_file']).encode()).hexdigest()
    assert source_hash==job['source_input_sha256']
    if job.get('selected_root'):
        initial_reference=reconstruction.load_reference(Path(job['selected_root'])/task['suite']/f"{task['task_id']:02d}"/demo/'state_reference.hdf5')
    else:initial_reference=None
    try:
        for iteration,gain in enumerate([None]+GAINS):
            if parent is not None:
                if not any(not r['aborted'] for r in parent['rows']):
                    ledger.append(dict(iteration=iteration,skipped=True,reason='no executed prefix for correction'));break
                signature=hashlib.sha256(parent['reference']['actions'].tobytes()+str(gain).encode()).hexdigest()
                if signature in seen:
                    ledger.append(dict(iteration=iteration,skipped=True,reason='identical parent and gain already evaluated'));continue
                seen.add(signature)
                cp,cr,measured=reconstruction.corrected(parent['reference'],parent['rows'],gain)
            trial=scratch/f'iteration_{iteration}';trial.mkdir()
            local_plan=dict(case=case,task=task,jobs=[job],source_sha256=master['source_sha256'])
            runner.dump(trial/'joint_native/plan.json',local_plan)
            capture={}
            def reference_factory(original,session):
                capture['model']=session.model
                if parent is None:
                    if initial_reference is not None:result={k:v.copy()for k,v in initial_reference.items()}
                    else:
                        base=state_reference(source_states,source_actions,session.handles,session.ik.kinematics,1,1)
                        sp,sr,sq=[base[k].copy()for k in ['positions','rotations','q']]
                        gp,gr=decode_source_goals(sp[:-1],sr[:-1],np.clip(source_actions[:-1,:6],-1,1)*[.05,.05,.05,.5,.5,.5],sr[0])
                        base.update(positions=np.concatenate([sp[:1],gp]),rotations=np.concatenate([sr[:1],gr]))
                        result=make_reference(base,source_actions,session.handles,session.ik.kinematics,'preview',1,0,posture_weight=0.)
                        result.update(source_positions_m=sp,source_rotations=sr,q=sq)
                else:
                    base={k:v.copy()for k,v in parent['reference'].items()}
                    sp,sr,sq=[base[k].copy()for k in ['source_positions_m','source_rotations','q']]
                    base['positions']=np.concatenate([base['positions'][:1],base['positions'][1:]+cp])
                    base['rotations']=np.concatenate([base['rotations'][:1],Rotation.from_rotvec(cr).as_matrix()@base['rotations'][1:]])
                    base['q']=np.concatenate([sq[:1],base['goal_joint_witness_rad']])
                    result=make_reference(base,source_actions,session.handles,session.ik.kinematics,'preview',1,0,posture_weight=0.)
                    result.update(source_positions_m=sp,source_rotations=sr,q=sq,
                        iteration_correction_position_m=cp,iteration_correction_rotation_rad=cr)
                np.testing.assert_array_equal(result['actions'][:,6],source_actions[:-1,6])
                capture['reference']=result
                return result
            try:
                report=servo.worker(trial,0,'joint_native',reference_transform=reference_factory)
                folder=trial/'joint_native/seed_00/off'
                assert report['solver_settings']==asdict(IpoptSettings())
                assert report['objective_settings']==asdict(ObjectiveSettings.delta_q_squared_conditioning_safeguard())
                rows=json.loads((folder/'trajectory.json').read_text())
                metrics=pose_metrics(capture['model'],folder,capture['reference'],source_states,report)
                diagnostics=json.loads((folder/'solver_observations.json').read_text())
                failures=[dict(index=d['index'],status_code=d.get('status_code'),status=d.get('status'),
                    iterations=d.get('iterations'),wall_ms=d.get('problem_solve_wall_ms'),cpu_ms=d.get('adapter_wrapper_cpu_ms'),
                    validation=d.get('raw_candidate_validation'))for d in diagnostics if not d['published'] or d.get('status_code')==-4]
                accepted=select_candidate(parent['metrics'] if parent else None,metrics)
                record=dict(iteration=iteration,gain=gain,success=report['success'],accepted=accepted,
                    execution_rejected=report['execution_rejected'],rejected_frame=report['rejected_frame'],
                    executed_frames=report['executed_frames'],planned_frames=report['planned_frames'],
                    native_predicates=report['native_predicates'],tracking=report['tracking'],metrics=metrics,
                    cpu_timeout_frames=report['cpu_timeout_frames'],solver_status_counts=report['solver_status_counts'],
                    failure_solver_records=failures,
                    max_ik_wall_ms=max(d['ik_wall_ms']for d in diagnostics),
                    max_ik_cpu_ms=max(d['ik_cpu_ms']for d in diagnostics),
                    source_initial_state_sha256=report['initial_state_sha256'],model_sha256=report['model_sha256'])
                ledger.append(record)
            finally:pass
            if accepted:
                parent=dict(reference=capture['reference'],rows=rows,metrics=metrics)
                if report['success']:
                    if best:shutil.rmtree(best['trial'])
                    best=dict(trial=trial,folder=folder,report=report,metrics=metrics,iteration=iteration,reference=capture['reference'])
                    runner.dump(folder/'task_pose_object_metrics.json',metrics)
            elif parent is None:
                # A rejected baseline is still a measured rollout prefix.
                # Keep it in RAM as the correction source; never publish it.
                parent=dict(reference=capture['reference'],rows=rows,metrics=metrics)
            # Failed/unselected arrays exist only during this attempt; measured
            # parent arrays needed for the next update are retained in RAM.
            if not best or trial!=best['trial']:shutil.rmtree(trial)
        result=dict(case=case,suite=task['suite'],task_id=task['task_id'],demo=demo,
            source_input_sha256=source_hash,success=best is not None,trials=ledger,wall_s=time.monotonic()-started)
        if best:
            # Record full-quality training observations only for the selected
            # success, using a fresh execution of its frozen final actions.
            replay=scratch/'selected_replay';replay.mkdir()
            runner.dump(replay/'joint_native/plan.json',local_plan)
            writer=ObservationWriter(replay/'recording.hdf5',demo)
            try:
                replay_report=servo.worker(replay,0,'joint_native',
                    reference_transform=lambda original,session:best['reference'],observation_callback=writer)
                replay_folder=replay/'joint_native/seed_00/off'
                assert replay_report['success']
                for name in ['states.npz','servo_physics.npz']:
                    with np.load(replay_folder/name)as a,np.load(best['folder']/name)as b:
                        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
                before=json.loads((best['folder']/'trajectory.json').read_text())
                after=json.loads((replay_folder/'trajectory.json').read_text())
                for a,b in zip(before,after,strict=True):
                    for key in ['action','pre_action','post_action','joint_target_rad','nominal_target','phase','success']:
                        assert a[key]==b[key]
                writer.finish(replay_report,replay_folder,job,task)
            finally:writer.close()
            shutil.move(str(replay/'recording.hdf5'),replay_folder/'recording.hdf5')
            runner.dump(replay_folder/'task_pose_object_metrics.json',best['metrics'])
            result['independent_replay_bitwise_identical']=True
            destination=out/'episodes'/task['suite']/f"{task['task_id']:02d}"/demo
            destination.parent.mkdir(parents=True,exist_ok=True)
            assert not destination.exists()
            # Commit successful files before their durable ledger entry. This
            # keeps a host restart from turning a reported success into holes.
            for artifact in replay_folder.iterdir():
                if artifact.is_file():
                    with artifact.open('rb')as file:os.fsync(file.fileno())
            shutil.move(str(replay_folder),destination)
            directory=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
            result.update(folder=str(destination),selected_iteration=best['iteration'],metrics=best['metrics'],report=replay_report)
        runner.dump(out/'records'/f'case_{case:02d}'/(demo+'.json'),result)
        return result
    finally:
        if scratch.exists():shutil.rmtree(scratch)


def make_plan(out):
    import h5py
    import numpy as np
    tasks=json.loads(SELECTION.read_text())['tasks'];old=json.loads(OLD.read_text())['results']
    entries=[]
    for case,t in enumerate(tasks):
        previous=sorted([j for j in old if j['case']==case],key=lambda j:int(j['demo'][5:]))
        by_demo={j['demo']:j for j in previous}
        with h5py.File(t['source_hdf5'])as f,h5py.File(t['processed_hdf5'])as processed:
            available=set(processed['regeneration/episodes'])
            remaining=sorted(set(f['data'])-set(by_demo),key=lambda d:int(d[5:]))
            np.random.default_rng(20261025+case).shuffle(remaining)
            order=[j['demo']for j in previous]+remaining;candidates=[]
            for demo in order:
                assert demo in available
                g=f['data'][demo];states=g['states'][()];actions=g['actions'][()]
                np.testing.assert_allclose(np.diff(states[:,0]),.05,rtol=0,atol=1e-10)
                job=dict(case=case,suite=t['suite'],task_id=t['task_id'],demo=demo,source_hdf5=t['source_hdf5'],
                    planned_frames=len(states)-1,source_input_sha256=hashlib.sha256(states.tobytes()+actions.tobytes()+str(g.attrs['model_file']).encode()).hexdigest())
                if demo in by_demo:job['selected_root']=by_demo[demo]['selected_root']
                candidates.append(job)
        task={k:t[k]for k in ['suite','task_id','task_name','language']};task['masks']={p:'000'for p in ['APP','ENG','EXEC','DIS']}
        entries.append(dict(case=case,task=task,candidates=candidates))
    sources=freeze()
    policy=dict(target_successes_per_task=10,unique_source_demos=True,gains=GAINS,control_hz=20,physics_hz=500,
        gripper='original unchanged',tolerance='off',online_optimizer='unchanged defaults and objective',
        robot_joint_tracking_penalty=False,finger_joint_tracking_penalty=False,
        candidate_order='previous fixed ten first, then remaining 40 in deterministic RNG order seed20261025+case',
        acceptance='complete feasible; prioritize recovering task success; within same success class reduce EEF+object cost',
        failure_storage='scalar diagnostics, failure frames and reason only; delete transient trajectory/video/state/image files',
        success_storage='one selected complete success per distinct source demo; original pre-action dual-camera images',
        iterations='baseline + up to four uniform corrections; identical parent/gain trials skipped')
    plan=dict(tasks=entries,source_sha256=sources,policy=policy,selection_sha256=runner.sha(SELECTION),old_summary_sha256=runner.sha(OLD))
    runner.dump(out/'plan.json',plan)
    for i,path in enumerate(sources):
        (out/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(path,out/'source_snapshot'/f'{i:03d}_{Path(path).name}')
    return plan


def summarize(out,plan):
    tasks=[]
    for entry in plan['tasks']:
        records=[json.loads(p.read_text())for p in sorted((out/'records'/f"case_{entry['case']:02d}").glob('*.json'))]
        successful=[r for r in records if r.get('success')]
        tasks.append(dict(case=entry['case'],suite=entry['task']['suite'],task_id=entry['task']['task_id'],
            success=len(successful),target=10,candidates_attempted=len(records),
            process_errors=sum('error'in r for r in records),
            failed_trials=sum(not t.get('success',True)for r in records for t in r.get('trials',[])if not t.get('skipped')),
            cpu_timeouts=sum(len(t.get('cpu_timeout_frames',[]))for r in records for t in r.get('trials',[])),
            successful_demos=[r['demo']for r in successful]))
    value=dict(complete=all(t['success']==10 for t in tasks),total_success=sum(t['success']for t in tasks),
        tasks_complete=sum(t['success']==10 for t in tasks),tasks=tasks,source_unchanged=freeze()==plan['source_sha256'])
    runner.dump(out/'summary.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--case',type=int);p.add_argument('--index',type=int)
    p.add_argument('--cases',nargs='+',type=int,default=list(range(40)));p.add_argument('--workers',type=int,default=6)
    p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if a.worker:
        try:attempt(out,a.case,a.index)
        except Exception as e:
            traceback.print_exc()
            plan=json.loads((out/'plan.json').read_text());job=plan['tasks'][a.case]['candidates'][a.index]
            runner.dump(out/'records'/f'case_{a.case:02d}'/(job['demo']+'.json'),dict(case=a.case,demo=job['demo'],success=False,error=str(e)))
            raise
        return
    if not a.resume:out.mkdir(parents=True,exist_ok=False);plan=make_plan(out)
    else:plan=json.loads((out/'plan.json').read_text())
    assert freeze()==plan['source_sha256'],'Sources changed since plan was frozen'
    if a.prepare_only:return
    lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
    def task(case):
        entry=plan['tasks'][case]
        success={json.loads(p.read_text())['demo']for p in (out/'records'/f'case_{case:02d}').glob('*.json') if json.loads(p.read_text()).get('success')}
        if len(success)==10:return case
        for index,job in enumerate(entry['candidates']):
            record=out/'records'/f'case_{case:02d}'/(job['demo']+'.json')
            if record.exists():row=json.loads(record.read_text())
            else:
                logfile=out/'logs'/f"case_{case:02d}_{job['demo']}.log";logfile.parent.mkdir(exist_ok=True)
                with logfile.open('w')as log:
                    done=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--output',str(out),
                        '--case',str(case),'--index',str(index)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if not record.exists():raise RuntimeError(f'Missing worker record: {logfile}, return {done.returncode}')
                row=json.loads(record.read_text())
                if 'error'in row:raise RuntimeError(f'Worker error: {logfile}: {row["error"]}')
            if row['success']:success.add(job['demo'])
            print(json.dumps(dict(case=case,demo=job['demo'],success=len(success),attempted=index+1)),flush=True)
            if len(success)==10:break
        runner.dump(out/'task_status'/f'case_{case:02d}.json',dict(case=case,success=len(success),exhausted=len(success)<10))
        return case
    with ThreadPoolExecutor(max_workers=a.workers)as pool:
        pending={pool.submit(task,case):case for case in a.cases}
        while pending:
            from concurrent.futures import wait,FIRST_COMPLETED
            completed,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for f in completed:f.result();pending.pop(f)
            s=summarize(out,plan)
            print(json.dumps(dict(total_success=s['total_success'],tasks_complete=s['tasks_complete'])),flush=True)
    summarize(out,plan)


if __name__=='__main__':main()
