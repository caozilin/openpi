"""Read-only audit: labels, actions, native final goal, states and both MP4s."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,datetime,hashlib,json,os,subprocess,sys,multiprocessing,fcntl
ROOT=Path(__file__).resolve().parents[2]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')
sys.path[:0]=[str(ROOT),str(ROOT.parent/'franka_mujoco'),str(ROOT/'third_party/libero')]
from experiments.libero_tasktol_compat.annotated_tolerance_policy import label_policy

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def check_one(arg):
 import numpy as np,h5py,mujoco
 from libero.libero.envs.env_wrapper import ControlEnv
 from libero.libero import get_libero_path
 from examples.libero import regenerate_libero_dataset as recorder
 from tasktol.simulation.libero import libero_handles
 out,key=arg;out=Path(out);plan=read(out/'plan.json');e=next(x for x in plan['entries'] if x['key']==key)
 record=read(out/'records'/(key+'.json'));assert not record.get('infrastructure_error')
 folder=out/'episodes'/key;report=read(folder/'report.json');rows=read(folder/'trajectory.json');labels=read(e['labels']);policy=label_policy(labels)
 assert sha(e['labels'])==e['labels_sha256']==record['labels_sha256']==report['label_source_sha256']
 assert len(rows)==record['video_frames']>0 and len(rows)<=e['source_frames']
 assert not list(folder.glob('*.hdf5')) and not (folder/'servo_physics.npz').exists()
 for name,digest in record['artifact_sha256'].items():assert sha(folder/name)==digest,(key,name)
 with h5py.File(e['job']['reference_path']) as f:actions=f['actions'][()]
 assert sha(e['job']['reference_path'])==e['reference_sha256']
 assert hashlib.sha256(actions.tobytes()).hexdigest()==e['baseline_actions_sha256']==report['actions_sha256']
 with np.load(folder/'states.npz',allow_pickle=False) as z:
  states=z['pre_action_states'];final=z['final_state']
 assert len(states)==len(rows) and np.isfinite(states).all() and np.isfinite(final).all()
 assert hashlib.sha256(states[0].tobytes()).hexdigest()==e['initial_state_sha256']==report['initial_state_sha256']
 for i,row in enumerate(rows):
  want=policy(i,None,None);np.testing.assert_array_equal(row['action'],actions[i])
  assert row['phase']==want['phase'] and row['bounds_rad']==want['bounds_rad']
  assert row['subtask_index']==want['label_subtask_index'] and row['label_source_frame_index']==i
  assert row['semantic_mask']==want['semantic_mask']
  assert all(a=='0' or b=='1' for a,b in zip(row['effective_mask'],row['semantic_mask'],strict=True))
  after=states[i+1] if i+1<len(states) else final
  if row['aborted']:
   assert i==len(rows)-1 and row['physics_steps']==0 and not row['optimizer']['feasible']
   np.testing.assert_array_equal(states[i],after)
  else:
   assert row['physics_steps']==25 and row['optimizer']['feasible']
   assert abs(after[0]-states[i][0]-.05)<1e-9
 assert report['executed_frames']==sum(not r['aborted'] for r in rows)
 assert report['shared_joint_servo_statistics']['calls']==25*report['executed_frames']
 assert report['execution_rejected']==rows[-1]['aborted']
 # Restore the original fixture model from the same recorded reset/RNG chain.
 task=e['task'];selection=read(ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json')['tasks'][e['case']]
 with h5py.File(selection['processed_hdf5']) as f:
  config=json.loads(f.attrs['configuration']);order=config['demo_ids'];number=order.index(e['job']['demo'])
  before=json.loads(f['regeneration/episodes'][order[number-1]].attrs['rng_state']) if number else None
  after_rng=json.loads(f['regeneration/episodes'][e['job']['demo']].attrs['rng_state'])
 env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task['suite']/(task['task_name']+'.bddl')),
  controller='JOINT_POSITION',use_camera_obs=False,has_offscreen_renderer=False,control_freq=20)
 try:
  env.seed(config['seed']);env.env.object_property_initializers*=number+1
  recorder._restore_rng(before if before is not None else dict(numpy=recorder._rng_state()['numpy'],python=after_rng['python']))
  env.reset();env.set_state(states[0]);env.sim.forward()
  m=env.sim.model._model;d=env.sim.data._data
  model_digest=hashlib.sha256(b''.join(np.asarray(getattr(m,k)).tobytes() for k in ['body_pos','body_quat','body_mass','body_inertia','geom_friction','jnt_range'])).hexdigest()
  assert model_digest==e['job']['expected_model_sha256']==report['model_sha256'],key
  handles=libero_handles(env)
  for i,row in enumerate(rows):
   np.testing.assert_allclose(states[i,1+handles.qpos_ids],row['pre_action']['joint_position_rad'],atol=1e-12,rtol=0)
   after=states[i+1] if i+1<len(states) else final
   np.testing.assert_allclose(after[1+handles.qpos_ids],row['post_action']['joint_position_rad'],atol=1e-12,rtol=0)
  env.set_state(final);env.sim.forward();fresh=bool(env.check_success())
  if report['executed_frames']:
   assert int(m.opt.integrator)==report['shared_joint_servo']['effective_integrator']==0
   qpos=d.qpos.copy();qvel=d.qvel.copy();mujoco.mj_integratePos(m,d.qpos,qvel,-m.opt.timestep)
   roundtrip=d.qpos.copy();mujoco.mj_integratePos(m,roundtrip,qvel,m.opt.timestep)
   np.testing.assert_allclose(roundtrip,qpos,atol=1e-12,rtol=0)
   env.sim.forward();d.qpos[:]=qpos
  goal=bool(env.check_success());predicates=[dict(predicate=p,satisfied=bool(env.env._eval_predicate(p))) for p in env.env.parsed_problem['goal_state']]
  assert goal==report['native_goal_at_stop'] and predicates==report['native_predicates'],(key,goal,report['native_goal_at_stop'])
  success=bool(goal and len(rows)==len(actions) and not rows[-1]['aborted'])
  assert success==report['success']==record['success']
 finally:env.close()
 videos=[]
 for name in ('main_rgb.mp4','wrist_rgb.mp4'):
  video=folder/name
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames','-of','json',str(video)]))['streams'][0]
  assert info['codec_name']=='h264' and (info['width'],info['height'])==(256,256) and info['r_frame_rate']=='20/1'
  assert int(info['nb_read_frames'])==len(rows)
  subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  videos.append(dict(name=name,frames=len(rows),sha256=sha(video)))
 return dict(key=key,success=success,execution_rejected=report['execution_rejected'],frames=len(rows),executed_frames=report['executed_frames'],
   labels_sha256=e['labels_sha256'],native_success=goal,fresh_forward_success=fresh,fixture_model_verified=True,per_frame_labels_verified=True,
   original_actions_initial_state_verified=True,videos=videos)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--keys',nargs='+');p.add_argument('--workers',type=int,default=6)
 a=p.parse_args();out=a.output.resolve();plan=read(out/'plan.json');keys=a.keys or [e['key'] for e in plan['entries']]
 full=a.keys is None;name='independent_final_audit' if full else 'pilot_integration_audit'
 lock=(out/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(out/(name+'_process.json'),dict(pid=os.getpid(),command=sys.argv))
 results=[]
 try:
  with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
   futures=[pool.submit(check_one,(str(out),key)) for key in keys]
   for f in as_completed(futures):
    results.append(f.result());save(out/(name+'_state.json'),dict(stage='auditing',checked=len(results),expected=len(keys)))
  if full:
   assert read(out/'generation_completion.json')['complete']
   assert len(results)==1851 and len({r['key'] for r in results})==1851
   assert read(out/'summary.json')['infrastructure_errors']==0
  result=dict(complete=True,episodes=len(results),success=sum(r['success'] for r in results),rejected=sum(r['execution_rejected'] for r in results),
   checked_at=datetime.datetime.now().astimezone().isoformat(),plan_sha256=sha(out/'plan.json'),scope='full1851' if full else 'pilot only',results=sorted(results,key=lambda r:r['key']))
  save(out/(name+'.json'),result);save(out/(name+'_state.json'),dict(stage='complete',episodes=len(results)));print(json.dumps({k:v for k,v in result.items() if k!='results'}))
 except BaseException:
  import traceback;save(out/(name+'_state.json'),dict(stage='failed',checked=len(results),error=traceback.format_exc()));raise

if __name__=='__main__':main()
