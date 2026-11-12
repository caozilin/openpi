"""Evaluate the audited joint13 model using learned tolerance bounds, six local clients."""
from pathlib import Path
import argparse,collections,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110'
PYTHON=ROOT.parent/'franka_mujoco/experiments/libero_closed_loop/.venv/bin/python'
FROZEN=DEFAULT/'frozen_shared_runtime'
IDENTITY='libero_joint13_successmix20_step10000_predtol_seed0to9_20261110'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10'];LIMITS=[220,280,300,520]
PORTS=list(range(18760,18766))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);q=path.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(path)
def assignments():
 jobs=[dict(suite=s,task_id=t,seed=z) for s in SUITES for t in range(10) for z in range(10)]
 return {str(i):jobs[i::6] for i in range(6)}
def validate(request):
 expected={(s,t,z) for s in SUITES for t in range(10) for z in range(10)}
 selected={(r['suite'],r['task_id'],r['seed']) for r in request['selected_episodes']}
 assert selected==expected and len(request['selected_episodes'])==400
 assert request['worker_assignments']==assignments() and request['expected_episodes']==400
 assert request['checkpoint_step']==10000 and request['seeds']==list(range(10))
 assert request['client_location']=='local' and request['tolerance_enabled']
 assert request['max_consecutive_solver_failures']==20
 assert request['tolerance_bounds_policy']=='model_joint13_directional_bounds_with_shared_phase_guard'
 assert request['success_criterion']=='first_native_success' and request['post_success_steps']==20
 for path,expected_sha in request['source_sha256'].items():assert sha(Path(path))==expected_sha,path
 for path,expected_sha in request['preserved_artifact_sha256'].items():assert sha(Path(path))==expected_sha,path

def worker(run,seed,max_consecutive_solver_failures=20,*,replica_index=None,episode_keys=None):
 import random
 import numpy as np
 import imageio.v2 as imageio
 from scipy.spatial.transform import Rotation
 from libero.libero import get_libero_path
 from types import SimpleNamespace
 from libero.libero.envs import OffScreenRenderEnv
 from tasktol.simulation.libero import LiberoTaskTolSession
 from tasktol.control.ipopt import IpoptSettings
 from tasktol.control.task_space.objective import ObjectiveSettings
 from experiments.libero_tasktol_compat.mujoco_joint_servo import MuJoCoJointServoController
 from examples.libero.tasktol_actions import decode_tasktol_action
 from examples.libero.solver_failure_hold import SolverFailureHoldSession
 from examples.libero.success_continuation import SuccessContinuation,validate_request as validate_success_request
 from examples.libero.predicted_tolerance_policy import PredictedTolerancePolicy,decode_prediction,step_with_prediction,POLICY
 from openpi_client import msgpack_numpy
 import websockets.sync.client
 class EvaluationClient:
  def __init__(self,host,port):
   self._connection=websockets.sync.client.connect(f'ws://{host}:{port}',compression=None,max_size=None,ping_interval=None,ping_timeout=None,open_timeout=300)
   self._ws=self._connection.__enter__()
   self._packer=msgpack_numpy.Packer()
   self._metadata=msgpack_numpy.unpackb(self._ws.recv(timeout=600))
  def get_server_metadata(self):return self._metadata
  def infer(self,packet):
   self._ws.send(self._packer.pack(packet));response=self._ws.recv(timeout=600)
   if isinstance(response,str):raise RuntimeError(response)
   return msgpack_numpy.unpackb(response)
 import mujoco
 assert mujoco.__version__=='3.2.3'
 run_request=json.loads((run/'evaluation_request.json').read_text())
 success_settings=validate_success_request(run_request)
 tolerance_settings={'tolerance_bounds_policy':POLICY}
 selected={tuple((e['suite'],e['task_id'],e['seed'])) for e in run_request['selected_episodes']} if 'selected_episodes' in run_request else None
 result_identity=run_request.get('eval_identity',IDENTITY)
 replica=seed if replica_index is None else replica_index
 if episode_keys is not None:selected=set(episode_keys) if selected is None else selected & set(episode_keys)
 client=EvaluationClient('127.0.0.1',PORTS[replica])
 metadata=client.get_server_metadata()
 assert metadata['eval_identity']==run_request.get('server_eval_identity',IDENTITY) and metadata['replica']==replica
 connection_name=f'client_{seed}_connection.json' if replica_index is None else f'client_{replica}_seed_{seed}_connection.json'
 dump(run/connection_name,{'seed':seed,'replica':replica,'server_metadata':metadata,'pid':os.getpid()})
 task_rows=json.loads((run/'official_tasks.json').read_text())
 initial_states=np.load(run/'official_initial_states.npz')
 for si,suite in enumerate(SUITES):
  for task_id in range(10):
   if selected is not None and (suite,task_id,seed) not in selected:continue
   case=si*10+task_id;out=run/'episodes'/suite/f'task_{task_id:02d}'/f'seed_{seed}'
   result_path=out/'result.json'
   if result_path.exists():
    old=json.loads(result_path.read_text());assert old['complete'] and old['eval_identity']==result_identity
    if any(old.get(key)!=value for key,value in success_settings.items()):raise ValueError('Existing result has a different success continuation policy; use a new run directory')
    assert all(old.get(key)==value for key,value in tolerance_settings.items())
    continue
   out.mkdir(parents=True,exist_ok=True)
   task=SimpleNamespace(**next(t for t in task_rows if t['suite']==suite and t['task_id']==task_id));initial=initial_states[f'{suite}_{task_id}_{seed}']
   np.random.seed(seed);random.seed(seed)
   env=None;video=None;rows=[];actions=[];states=[];started=time.monotonic();success=False;reason='step_limit';infer_times=[]
   try:
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),controller='JOINT_POSITION',camera_names=['agentview','robot0_eye_in_hand'],camera_heights=256,camera_widths=256,control_freq=20)
    env.seed(seed);env.reset();obs=env.set_init_state(initial)
    session=LiberoTaskTolSession(env,joint_control_mode='native',kp=None,solver_settings=IpoptSettings())
    assert session.ik.settings==IpoptSettings() and session.ik.objective_settings==ObjectiveSettings.delta_q_squared_conditioning_safeguard()
    control=MuJoCoJointServoController.from_native(session.controller,session.handles)
    session.controller=control;env.robots[0].controller=control
    assert session.physics_steps==25 and session.dt==.05
    session=SolverFailureHoldSession(session,max_consecutive_solver_failures=max_consecutive_solver_failures)
    for _ in range(10):
     obs,row=session.step(np.r_[np.zeros(6),-1.])
     if row['aborted']:raise RuntimeError('initial settling reached consecutive solver failure threshold')
    mask_policy=PredictedTolerancePolicy()
    video=imageio.get_writer(out/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16)
    plan=collections.deque();chunk=0;continuation=SuccessContinuation(success_settings['post_success_seconds']);step=0
    while continuation.should_run(step,LIMITS[si]):
     img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
     video.append_data(np.concatenate([img,wrist],axis=1))
     if not plan:
      state=np.r_[obs['robot0_eef_pos'],Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec(),obs['robot0_gripper_qpos']].astype(np.float32)
      packet={'observation/image':img,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language,'_libero_eval_noise_seed':int(seed*1000000+case*1000+chunk)}
      tic=time.monotonic();pred=client.infer(packet);infer_times.append(time.monotonic()-tic)
      plan.extend(decode_prediction(pred));chunk+=1
     action,raw_bounds,raw_action=plan.popleft();states.append(env.sim.get_state().flatten().copy());actions.append(action.copy())
     selection=mask_policy.resolve(session.data.qpos[session.handles.finger_qpos_ids].copy(),action,raw_bounds)
     obs,row=step_with_prediction(session,action,selection)
     row.update(model_action_raw=raw_action.tolist(),inference_chunk=chunk-1,inference_chunk_offset=step%5)
     rows.append(row)
     stop=continuation.observe(step,row);step+=1
     if stop:break
    success=continuation.success;reason=continuation.termination
    video.close();video=None
    np.savez_compressed(out/'trajectory.npz',actions=np.asarray(actions),pre_action_states=np.asarray(states),initial_state=np.asarray(initial),final_state=env.sim.get_state().flatten())
    dump(out/'trajectory.json',rows)
    result={'complete':True,'eval_identity':IDENTITY,'suite':suite,'task_id':task_id,'task_name':task.name,'language':task.language,'seed':seed,'initial_state_index':seed,'initial_state_sha256':hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),'success':success,'termination':reason,'steps':len(rows),'max_steps':LIMITS[si],'settle_steps':10,'replan_steps':5,'inference_calls':chunk,'inference_wall_seconds':sum(infer_times),'elapsed_seconds':time.monotonic()-started,'controller':'frozen shared MuJoCo joint servo + default IPOPT, confirmed task masks +/-30deg','control_hz':20,'physics_hz':500,'gripper_deadband':.5,'state_input_enabled':False,'server_metadata':metadata,'servo':control.servo_metadata,'servo_statistics':control.servo_statistics,'finished_at':datetime.datetime.now().astimezone().isoformat()}
    result.update(session.policy_metadata())
    result.update(continuation.metadata())
    result['controller']='shared MuJoCo joint servo + default IPOPT, model predicted directional bounds and causal phase guard'
    result.update(tolerance_settings)
    result.update(eval_identity=result_identity,tolerance_enabled=True,training_request_sha256=run_request['training_request_sha256'])
    result['tolerance_order']=['rx-','rx+','ry-','ry+','rz-','rz+']
    dump(result_path,result);print(json.dumps({k:result[k] for k in ['suite','task_id','seed','success','termination','steps']}),flush=True)
   except Exception:
    dump(out/'infrastructure_error.json',{'suite':suite,'task_id':task_id,'seed':seed,'error':traceback.format_exc()});raise
   finally:
    if video is not None:video.close()
    if env is not None:env.close()

def summarize(run, request):
    rows = [read(p) for p in sorted(run.glob('episodes/*/task_*/seed_*/result.json'))]
    keys = [(r['suite'], r['task_id'], r['seed']) for r in rows]
    expected = {(r['suite'], r['task_id'], r['seed']) for r in request['selected_episodes']}
    assert len(keys) == len(set(keys)) and set(keys) <= expected
    assert all(r['complete'] and r['eval_identity'] == request['eval_identity'] for r in rows)
    errors = [str(p) for p in run.glob('episodes/*/task_*/seed_*/infrastructure_error.json')]
    summary = dict(complete=set(keys) == expected and not errors, expected_episodes=400,
                   completed_episodes=len(rows), successes=sum(r['success'] for r in rows),
                   infrastructure_errors=len(errors), infrastructure_error_files=errors,
                   per_suite={s: dict(episodes=sum(r['suite'] == s for r in rows),
                                      successes=sum(r['suite'] == s and r['success'] for r in rows)) for s in SUITES},
                   updated_at=datetime.datetime.now().astimezone().isoformat())
    dump(run / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=DEFAULT)
    parser.add_argument('--worker', type=int, choices=range(6))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    request = read(run / 'evaluation_request.json')
    sys.path.insert(0, str(ROOT))
    from examples.libero.success_continuation import validate_request as validate_success_request
    validate_success_request(request)
    validate(request)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(run/'frozen_shared_runtime'), str(ROOT), str(ROOT / 'third_party/libero'), str(ROOT / 'packages/openpi-client/src')]
    from examples.libero.solver_failure_hold import validate_failure_policy_request
    validate_failure_policy_request(request, 20)
    if args.worker is not None:
        jobs = request['worker_assignments'][str(args.worker)]
        episode_keys = {(x['suite'], x['task_id'], x['seed']) for x in jobs}
        for seed in sorted({job['seed'] for job in jobs}):
            worker(run, seed, 20, replica_index=args.worker, episode_keys=episode_keys)
        return
    lock = (run / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump(run / 'evaluation_process.json', dict(pid=os.getpid(), command=sys.argv,
         started_at=datetime.datetime.now().astimezone().isoformat()))
    import urllib.request
    deadline = time.monotonic() + 1800
    for port in request['ports']:
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=5) as response:
                    assert response.status == 200
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError('Inference readiness timed out') from None
                time.sleep(5)
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):signal.signal(sig,stop)
    try:
        for replica in range(6):
            with (run / f'client_{replica}.log').open('a') as log:
                p = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()), '--run-dir', str(run),
                                      '--worker', str(replica)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(p)
            dump(run / f'client_{replica}_process.json', dict(pid=p.pid, replica=replica,
                 command=str(Path(__file__).resolve()), assigned_episodes=len(request['worker_assignments'][str(replica)])))
        dump(run / 'evaluation_state.json', dict(stage='evaluating', pids=[p.pid for p in children], expected_episodes=400))
        while any(p.poll() is None for p in children):
            summarize(run, request)
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Client infrastructure error; inspect per-client logs')
            time.sleep(5)
        summary = summarize(run, request)
        assert summary['complete'] and all(p.returncode == 0 for p in children)
        dump(run / 'evaluation_completion.json', dict(summary, exit_codes=[p.returncode for p in children], eval_identity=request['eval_identity']))
        dump(run / 'evaluation_state.json', dict(stage='complete', episodes=400, successes=summary['successes']))
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == '__main__':
    main()
