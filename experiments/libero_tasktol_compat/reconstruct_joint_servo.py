"""Offline pose-only reference reconstruction for the unchanged shared servo.

One fixed rule applies to every demo. Future source poses are label-generation
inputs only. Online execution remains the maintained LIBERO session/worker.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT.parent/'franka_mujoco')]
from experiments.libero_tasktol_compat import rollout_mujoco_joint_servo as servo
from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner

BASE=HERE/'results/spatial00_direct_mujoco_no_tol_20261025_v1'
GAINS=[1.,.75,.5,.5]
original_freeze=servo.freeze


def freeze():
    paths=[Path(__file__),HERE/'probe_original_speed_reference.py',HERE/'osc_reference.py',
           HERE/'task_pose_object_metrics.py',HERE/'closed_loop_state_metrics.py']
    return dict(original_freeze(),**{str(p):runner.sha(p) for p in paths})


runner.freeze_sources=freeze


def load_reference(path):
    import h5py
    with h5py.File(path) as f:return {k:v[()] for k,v in f.items()}


def corrected(reference,rows,gain):
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation
    valid=[r for r in rows if not r['aborted']]
    assert valid
    n=len(valid);count=len(reference['actions'])
    ep=reference['source_positions_m'][1:n+1]-np.asarray([r['post_action']['position_m'] for r in valid])
    actual_r=Rotation.from_rotvec([r['post_action']['rotation_vector_rad'] for r in valid]).as_matrix()
    er=Rotation.from_matrix(reference['source_rotations'][1:n+1]@actual_r.transpose(0,2,1)).as_rotvec()
    window=min(5,n if n%2 else n-1)
    if window>=3:
        ep=savgol_filter(ep,window,2,axis=0);er=savgol_filter(er,window,2,axis=0)
    # Same one-frame anticipation for all demos. Missing measured suffixes use
    # the last measured correction, explicitly recorded, never invented states.
    indices=np.minimum(np.arange(count)+1,n-1)
    def bound(a,limit):return a*np.minimum(1.,limit/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-15))
    cp=bound(gain*ep[indices],.020 if gain==1. else .010)
    cr=bound(gain*er[indices],.10 if gain==1. else .05)
    return cp,cr,n


def accepts(previous,trial):
    return (trial['full_length'] and not trial['rejected']
            and (not previous['success'] or trial['success'])
            and trial['objective']<previous['objective'])


def score(folder,job,task,reference_path):
    import h5py
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from experiments.libero_tasktol_compat.task_pose_object_metrics import task_cost
    ref=load_reference(reference_path)
    report=json.loads((folder/'report.json').read_text())
    rows=[r for r in json.loads((folder/'trajectory.json').read_text()) if not r['aborted']]
    with np.load(folder/'states.npz') as f:
        actual=np.concatenate([f['pre_action_states'][:len(rows)],f['final_state'][None]])
    with h5py.File(job['source_hdf5']) as f:source=f['data'][job['demo']]['states'][()]
    old=Path(job['selected_root'])/task['suite']/f"{task['task_id']:02d}"/job['demo']/'ipopt_state_B10/recording.hdf5'
    with h5py.File(old) as f:model=mujoco.MjModel.from_xml_string(f['attempts'][job['demo']].attrs['model_file'])
    p=np.asarray([ref['source_positions_m'][0]]+[r['post_action']['position_m'] for r in rows])
    rotations=np.concatenate([ref['source_rotations'][:1],Rotation.from_rotvec([r['post_action']['rotation_vector_rad'] for r in rows]).as_matrix()])
    count=len(actual)
    result=task_cost(model,actual,source[:count],p,rotations,ref['source_positions_m'][:count],ref['source_rotations'][:count])
    result.update(success=report['success'],rejected=report['execution_rejected'],full_length=len(rows)==len(ref['actions']) and not report['execution_rejected'])
    if not result['full_length']:result['objective']+=1e12
    runner.dump(folder/'task_pose_object_metrics.json',result)
    return result


def worker(out,seed):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from probe_original_speed_reference import make_reference
    config=json.loads((out/'reconstruction.json').read_text())
    previous=load_reference(Path(config['reference_path']))
    rows=json.loads((Path(config['parent_folder'])/'trajectory.json').read_text())
    cp,cr,measured=corrected(previous,rows,config['gain'])
    def transform(original,session):
        base={k:v.copy() for k,v in previous.items()}
        source_p=base['source_positions_m'].copy();source_r=base['source_rotations'].copy();source_q=base['q'].copy()
        base['positions']=np.concatenate([base['positions'][:1],base['positions'][1:]+cp])
        base['rotations']=np.concatenate([base['rotations'][:1],Rotation.from_rotvec(cr).as_matrix()@base['rotations'][1:]])
        base['q']=np.concatenate([source_q[:1],previous['goal_joint_witness_rad']])
        result=make_reference(base,previous['actions'],session.handles,session.ik.kinematics,'preview',1,0,posture_weight=0.)
        result.update(source_positions_m=source_p,source_rotations=source_r,q=source_q,
                      iteration_correction_position_m=cp,iteration_correction_rotation_rad=cr)
        np.testing.assert_array_equal(result['actions'][:,6],original['actions'][:,6])
        return result
    report=servo.worker(out,seed,'joint_native',reference_transform=transform)
    folder=out/'joint_native'/f'seed_{seed:02d}/off'
    plan=json.loads((out/'plan.json').read_text());job=plan['jobs'][seed]
    metrics=score(folder,job,plan['task'],folder/'state_reference.hdf5')
    report.update(reference_reconstruction=dict(gain=config['gain'],parent_folder=config['parent_folder'],
        measured_prefix_frames=measured,unmeasured_suffix_frames=report['planned_frames']-measured,
        offline_future_only=True,projection_posture_weight=0.,time_scale=1,gripper='unchanged'))
    runner.dump(folder/'report.json',report)
    runner.dump(out/'result.json',dict(folder=str(folder),reference_path=str(folder/'state_reference.hdf5'),metrics=metrics,report=report))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if a.worker:worker(out,a.seed);return
    if not a.resume:
        out.mkdir(parents=True,exist_ok=False)
        plan=json.loads((BASE/'plan.json').read_text());oldhash=plan['source_sha256']
        for path,digest in oldhash.items():
            if '/franka_mujoco/' in path:assert runner.sha(path)==digest,f'Shared runtime changed: {path}'
        plan.update(source_sha256=freeze(),reconstruction_policy=dict(gains=GAINS,anticipation_frames=1,
            position_update_caps_m=[.020,.010,.010,.010],rotation_update_caps_rad=[.10,.05,.05,.05],
            acceptance='complete feasible; retain task success; decrease fixed EEF+object cost',
            robot_joint_tracking_penalty=False,finger_joint_tracking_penalty=False,optimizer='unchanged default',
            control_hz=20,physics_hz=500,time_scale=1,gripper='unchanged',apply_to='all selected demos at every iteration',
            missing_suffix_correction='hold last measured correction; no fabricated source/actual samples'),
            baseline=str(BASE))
        runner.dump(out/'plan.json',plan)
        for i,path in enumerate(plan['source_sha256']):
            (out/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(path,out/'source_snapshot'/f'{i:03d}_{Path(path).name}')
    plan=json.loads((out/'plan.json').read_text());assert freeze()==plan['source_sha256']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
    def run(seed):
        job=plan['jobs'][seed];task=plan['task']
        reference_path=Path(job['selected_root'])/task['suite']/f"{task['task_id']:02d}"/job['demo']/'state_reference.hdf5'
        folder=BASE/'joint_native'/f'seed_{seed:02d}/off'
        # Baseline diagnostics are written into this new experiment, preserving
        # all previously published recordings byte-for-byte.
        baseline_score_path=out/f'seed_{seed:02d}_baseline_metrics.json'
        if baseline_score_path.exists():metrics=json.loads(baseline_score_path.read_text())
        else:
            # score() writes beside folder; suppress that write for cached runs.
            metrics=score_cached(folder,job,task,reference_path,out/f'baseline_seed_{seed:02d}')
            runner.dump(baseline_score_path,metrics)
        baseline=metrics;selected=dict(folder=str(folder),reference_path=str(reference_path),metrics=metrics)
        trials=[]
        for index,gain in enumerate(GAINS):
            trial_out=out/f'seed_{seed:02d}'/f'iteration_{index+1}'
            if not trial_out.exists():
                trial_out.mkdir(parents=True)
                runner.dump(trial_out/'plan.json',plan);runner.dump(trial_out/'joint_native/plan.json',plan)
                runner.dump(trial_out/'reconstruction.json',dict(parent_folder=selected['folder'],reference_path=selected['reference_path'],gain=gain))
            if not (trial_out/'result.json').exists():
                with (trial_out/'worker.log').open('w') as log:
                    done=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--output',str(trial_out),'--seed',str(seed)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                assert done.returncode==0,f'Trial failed: {trial_out}/worker.log'
            trial=json.loads((trial_out/'result.json').read_text())
            accepted=accepts(selected['metrics'],trial['metrics'])
            decision=dict(iteration=index+1,gain=gain,accepted=accepted,parent_folder=selected['folder'],
                previous_objective=selected['metrics']['objective'],**trial)
            runner.dump(trial_out/'acceptance.json',decision);trials.append(decision)
            if accepted:selected=trial
            print(json.dumps(dict(seed=seed,iteration=index+1,accepted=accepted,success=trial['metrics']['success'],
                objective=trial['metrics']['objective'],source_position_mm=trial['report']['tracking']['source_position_mm'])),flush=True)
        result=dict(seed=seed,demo=job['demo'],baseline=baseline,selected=selected,trials=trials)
        runner.dump(out/f'seed_{seed:02d}/selection.json',result)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,s) for s in a.seeds]):
            results.append(f.result());runner.dump(out/'summary.json',dict(complete=len(results)==10,results=results,
                baseline_success=sum(r['baseline']['success']for r in results),success=sum(r['selected']['metrics']['success']for r in results)))
    assert freeze()==plan['source_sha256']


def score_cached(folder,job,task,reference_path,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['report.json','trajectory.json','states.npz']:
        if not (destination/name).exists():os.link(folder/name,destination/name)
    return score(destination,job,task,reference_path)


if __name__=='__main__':main()
