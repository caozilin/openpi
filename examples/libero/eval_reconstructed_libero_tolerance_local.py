"""Matched-seed LIBERO comparison: local shared servo, all-axis +/-30 degree tolerance."""
from pathlib import Path
import argparse,collections,datetime,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
ROOT=Path('/media/czl/sata/franka_my_code/openpi')
DEFAULT=ROOT/'data/libero_evaluation/pi05_reconstructed_10000_tol30_mask111_40tasks_6seeds_20261102'
PYTHON=ROOT.parent/'franka_mujoco/experiments/libero_closed_loop/.venv/bin/python'
FROZEN=ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime'
SERVER_IDENTITY='libero_reconstructed_step10000_40tasks_seed0to5_local_client_20261102'
IDENTITY='libero_reconstructed_step10000_tol30_mask111_40tasks_seed0to5_local_client_20261102'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10'];LIMITS=[220,280,300,520]
PORTS=list(range(18760,18766))
CHILDREN=[]
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);q=path.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n');q.replace(path)
def worker(run,seed,max_consecutive_solver_failures=20):
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
 client=EvaluationClient('127.0.0.1',PORTS[seed])
 metadata=client.get_server_metadata()
 assert metadata['eval_identity']==SERVER_IDENTITY and metadata['replica']==seed
 dump(run/f'client_{seed}_connection.json',{'seed':seed,'server_metadata':metadata,'pid':os.getpid()})
 task_rows=json.loads((run/'official_tasks.json').read_text())
 initial_states=np.load(run/'official_initial_states.npz')
 for si,suite in enumerate(SUITES):
  for task_id in range(10):
   case=si*10+task_id;out=run/'episodes'/suite/f'task_{task_id:02d}'/f'seed_{seed}'
   result_path=out/'result.json'
   if result_path.exists():
    old=json.loads(result_path.read_text());assert old['complete'] and old['eval_identity']==IDENTITY;continue
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
    video=imageio.get_writer(out/'rollout.mp4',fps=20,codec='libx264',quality=7,macro_block_size=16)
    plan=collections.deque();chunk=0
    for step in range(LIMITS[si]):
     img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
     video.append_data(np.concatenate([img,wrist],axis=1))
     if not plan:
      state=np.r_[obs['robot0_eef_pos'],Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec(),obs['robot0_gripper_qpos']].astype(np.float32)
      packet={'observation/image':img,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language,'_libero_eval_noise_seed':int(seed*1000000+case*1000+chunk)}
      tic=time.monotonic();pred=client.infer(packet);infer_times.append(time.monotonic()-tic)
      a=np.asarray(pred['actions']);assert a.shape==(10,7) and np.isfinite(a).all()
      plan.extend(decode_tasktol_action(a[:5]));chunk+=1
     action=plan.popleft();states.append(env.sim.get_state().flatten().copy());actions.append(action.copy())
     obs,row=session.step(action,phase='APP',semantic_mask='111',bounds_rad=np.full(6,np.deg2rad(30.)))
     assert row['semantic_mask']=='111' and np.allclose(row['bounds_rad'],np.deg2rad(30.),rtol=0,atol=1e-15)
     rows.append(row)
     if row['aborted']:reason='ik_failure_streak';break
     if row['success']:success=True;reason='task_success';break
    video.close();video=None
    np.savez_compressed(out/'trajectory.npz',actions=np.asarray(actions),pre_action_states=np.asarray(states),initial_state=np.asarray(initial),final_state=env.sim.get_state().flatten())
    dump(out/'trajectory.json',rows)
    result={'complete':True,'eval_identity':IDENTITY,'suite':suite,'task_id':task_id,'task_name':task.name,'language':task.language,'seed':seed,'initial_state_index':seed,'initial_state_sha256':hashlib.sha256(np.asarray(initial).tobytes()).hexdigest(),'success':success,'termination':reason,'steps':len(rows),'max_steps':LIMITS[si],'settle_steps':10,'replan_steps':5,'inference_calls':chunk,'inference_wall_seconds':sum(infer_times),'elapsed_seconds':time.monotonic()-started,'controller':'frozen shared MuJoCo joint servo + default IPOPT, tolerance +/-30deg mask111','tolerance_enabled':True,'semantic_mask':'111','tolerance_bounds_deg':[30.]*6,'tolerance_frame_mode':'stage_endpoint','phase':'APP','baseline_run':str(DEFAULT.parent/'pi05_reconstructed_10000_40tasks_6seeds_20261102'),'control_hz':20,'physics_hz':500,'gripper_deadband':.5,'state_input_enabled':False,'server_metadata':metadata,'servo':control.servo_metadata,'servo_statistics':control.servo_statistics,'finished_at':datetime.datetime.now().astimezone().isoformat()}
    result.update(session.policy_metadata())
    dump(result_path,result);print(json.dumps({k:result[k] for k in ['suite','task_id','seed','success','termination','steps']}),flush=True)
   except Exception:
    dump(out/'infrastructure_error.json',{'suite':suite,'task_id':task_id,'seed':seed,'error':traceback.format_exc()});raise
   finally:
    if video is not None:video.close()
    if env is not None:env.close()
def summarize(run):
 rows=[json.loads(p.read_text()) for p in sorted((run/'episodes').glob('*/task_*/seed_*/result.json'))]
 keys={(r['suite'],r['task_id'],r['seed']) for r in rows};expected={(s,t,z) for s in SUITES for t in range(10) for z in range(6)}
 assert len(keys)==len(rows) and keys<=expected
 errors=list((run/'episodes').glob('*/task_*/seed_*/infrastructure_error.json'))
 result={'complete':keys==expected and not errors,'expected_episodes':240,'completed_episodes':len(rows),'successes':sum(r['success'] for r in rows),'infrastructure_errors':len(errors),'per_suite':{s:{'episodes':sum(r['suite']==s for r in rows),'successes':sum(r['suite']==s and r['success'] for r in rows)} for s in SUITES},'per_task':[{'suite':s,'task_id':t,'episodes':sum(r['suite']==s and r['task_id']==t for r in rows),'successes':sum(r['suite']==s and r['task_id']==t and r['success'] for r in rows)} for s in SUITES for t in range(10)],'updated_at':datetime.datetime.now().astimezone().isoformat()}
 dump(run/'summary.json',result);return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,default=DEFAULT);ap.add_argument('--worker',type=int);ap.add_argument('--max-consecutive-solver-failures',type=int,default=20);a=ap.parse_args();run=a.run_dir.resolve()
 if a.max_consecutive_solver_failures<=0:ap.error('--max-consecutive-solver-failures must be positive')
 if (run/'configuration_error_stop.json').exists():raise RuntimeError('This run was stopped for invalid configuration; use a new reviewed request and run directory')
 os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 sys.path[:0]=[str(FROZEN),str(ROOT),str(ROOT/'third_party/libero'),str(ROOT/'packages/openpi-client/src')]
 request=json.loads((run/'evaluation_request.json').read_text())
 assert request['client_location']=='local' and request['expected_episodes']==240
 assert request['tolerance_enabled'] and request['semantic_mask']=='111' and request['tolerance_bounds_deg']==[30.]*6
 from examples.libero.solver_failure_hold import validate_failure_policy_request
 validate_failure_policy_request(request,a.max_consecutive_solver_failures)
 for path,sha in request['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
 if a.worker is not None:worker(run,a.worker,a.max_consecutive_solver_failures);return
 run.mkdir(parents=True,exist_ok=True);lock=open(run/'evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dump(run/'evaluation_process.json',{'pid':os.getpid(),'command':sys.argv})
 dump(run/'evaluation_state.json',{'stage':'waiting_for_inference','client_location':'local','expected_episodes':240})
 import urllib.request
 deadline=time.monotonic()+1800
 for port in PORTS:
  while True:
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=5) as f:assert f.status==200
    break
   except Exception:
    if time.monotonic()>deadline:raise TimeoutError('inference replicas not ready')
    time.sleep(10)
 children=CHILDREN
 for seed in range(6):
  log=open(run/f'client_{seed}.log','a');p=subprocess.Popen([str(PYTHON),'-u',str(Path(__file__).resolve()),'--run-dir',str(run),'--worker',str(seed),'--max-consecutive-solver-failures',str(a.max_consecutive_solver_failures)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  children.append(p);dump(run/f'client_{seed}_process.json',{'pid':p.pid,'seed':seed,'port':PORTS[seed]})
 dump(run/'evaluation_state.json',{'stage':'evaluating','client_location':'local','pids':[p.pid for p in children],'expected_episodes':240})
 while any(p.poll() is None for p in children):
  summary=summarize(run)
  if any(p.poll() not in (None,0) for p in children):raise RuntimeError('evaluation client failed; inspect per-client log')
  time.sleep(15)
 summary=summarize(run)
 assert all(p.returncode==0 for p in children) and summary['complete']
 dump(run/'evaluation_completion.json',dict(summary,exit_codes=[p.returncode for p in children],eval_identity=IDENTITY))
 dump(run/'evaluation_state.json',{'stage':'complete','episodes':240,'successes':summary['successes']})
if __name__=='__main__':
 try:main()
 except BaseException as error:
  for child in CHILDREN:
   if child.poll() is None:child.terminate()
  for child in CHILDREN:
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  if isinstance(error,SystemExit):raise
  traceback.print_exc();sys.exit(1)
