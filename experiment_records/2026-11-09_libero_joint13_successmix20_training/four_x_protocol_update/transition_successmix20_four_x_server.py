"""Versioned user-authorized4x protocol update after the original normalization actually exits."""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,signal,shutil,subprocess,sys,time,traceback
from pathlib import Path

def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def proc(pid):
 d=Path('/proc')/str(pid)
 try:
  rest=(d/'stat').read_text().rsplit(')',1)[1].split();cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  return dict(pid=pid,state=rest[0],ppid=int(rest[1]),start_ticks=int(rest[19]),exit_code=int(rest[49]) if len(rest)>49 else None,command=cmd)
 except FileNotFoundError:return None
def stop_verified(pid,run,script,stopped=False):
 p=proc(pid);assert p and str(run) in p['command'] and script in p['command'],p
 os.kill(pid,signal.SIGTERM)
 if stopped:os.kill(pid,signal.SIGCONT)
 for _ in range(100):
  p=proc(pid)
  if not p or p['state']=='Z':return p
  time.sleep(.1)
 raise RuntimeError(('Verified process did not exit',p))

def main(run):
 update=run/'four_x_protocol_update';update.mkdir(exist_ok=True)
 with (update/'transition.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  write(update/'transition_process.json',dict(pid=os.getpid(),command=sys.argv,started_at=now()))
  assert not (update/'transition_completion.json').exists(),'Already transitioned; inspect current training rather than repeat'
  old_request=read(run/'training_request.json');old_request_sha=sha(run/'training_request.json')
  assert old_request_sha=='9077cac73ceae17e0e9226af65d5221cd1b0089b6a681ba978bee22a6bcb9917'
  assert sha(run/'train_successmix20_joint13_pi05_server.py')==old_request['server_script_sha256']
  assert sha(run/'overlay_manifest.json')==old_request['overlay_manifest_sha256']
  for name,h in read(run/'overlay_manifest.json')['sha256'].items():assert sha(run/'source_overlay'/name)==h,name
  payload=read(update/'payload_manifest.json')
  for name,h in payload['sha256'].items():assert sha(update/'payload'/name)==h,name
  parent=proc(3312670);norm=proc(3312672)
  assert parent and parent['state']=='T' and str(run) in parent['command'] and '--stage' not in parent['command'],parent
  assert norm and norm['ppid']==3312670 and (norm['state']=='Z' or '--stage norm' in norm['command']),norm
  write(update/'hold_verification.json',dict(at=now(),parent=parent,normalization=norm,reason='User requested4x before any GPU training',original_request_sha256=old_request_sha,normalization_continues=True))
  while True:
   norm=proc(3312672)
   assert norm and norm['ppid']==3312670,norm
   if norm['state']=='Z':break
   write(update/'transition_state.json',dict(stage='waiting_for_original_normalization',at=now(),normalization=norm,parent_stopped_intentionally=True));time.sleep(15)
  assert norm['exit_code']==0,norm
  for p in Path('/proc').iterdir():
   if p.name.isdigit():
    detail=proc(int(p.name))
    assert not (detail and detail['ppid']==3312672 and detail['state']!='Z'),detail
  assert '4338/4338' in (run/'normalization.log').read_text(errors='replace')[-20000:]
  assert not (run/'train.log').exists() or 'Step ' not in (run/'train.log').read_text(errors='replace')
  module_spec=importlib.util.spec_from_file_location('original_normalization_receipt',run/'train_successmix20_joint13_pi05_server.py');entry=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(entry)
  verify=read(run/'lerobot_upload_verification.json');receipt=entry.normalization_receipt(run,old_request,verify)
  write(run/'normalization_completion.json',receipt)
  write(update/'original_normalization_exit.json',dict(complete=True,at=now(),process=norm,planned_batches=4338,log_sha256=sha(run/'normalization.log'),receipt=receipt,request_sha256=old_request_sha))
  write(update/'transition_state.json',dict(stage='archiving_pretraining_protocol',at=now()))
  stopped_parent=stop_verified(3312670,run,'train_successmix20_joint13_pi05_server.py',stopped=True)
  stopped_finisher=stop_verified(3210109,run,'finish_successmix20_joint13_training_server.py')
  with (run/'training.lock').open('a') as training_lock:fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with (run/'finish_training.lock').open('a') as finish_lock:fcntl.flock(finish_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  archive=update/'before_4x';archive.mkdir()
  names=['training_request.json','actual_config.json','training_launch.json','training_process.json','training_state.json','overlay_manifest.json','overlay_preparation.json','train_successmix20_joint13_pi05_server.py','audit_successmix20_joint13_training_server.py','finish_successmix20_joint13_training_server.py','finish_training_process.json','finish_training_state.json','training_live_process_snapshot.json','normalization_completion.json']
  archived={}
  for name in names:
   p=run/name
   if p.exists():shutil.copy2(p,archive/name);archived[name]=sha(p)
  (run/'source_overlay').rename(archive/'source_overlay')
  shutil.copytree(archive/'source_overlay',run/'source_overlay',ignore=shutil.ignore_patterns('__pycache__'))
  write(update/'prior_protocol_retirement.json',dict(complete=True,at=now(),reason='User4x configuration update before first training step; original normalization completed normally and preserved',stopped_parent=stopped_parent,stopped_finisher=stopped_finisher,original_parent_pid=3312670,original_finisher_pid=3210109,archive_sha256=archived,normalization_receipt_sha256=sha(run/'normalization_completion.json'),gpu_training_steps=0))
  for name in payload['sha256']:
   destination=run/'source_overlay'/name.removeprefix('src/') if name.startswith('src/') else run/Path(name).name
   # src/openpi maps to overlay/openpi.
   destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(update/'payload'/name,destination)
  overlay=read(archive/'overlay_manifest.json');overlay['sha256']={str(p.relative_to(run/'source_overlay')):sha(p) for p in (run/'source_overlay').rglob('*.py')};overlay['four_x_payload_manifest_sha256']=sha(update/'payload_manifest.json');write(run/'overlay_manifest.json',overlay)
  request=json.loads(json.dumps(old_request));request['config']['data']['tolerance_trajectory_action_loss_weight']=4.0
  request.update(status='frozen_four_x_user_requested_before_training',server_script_sha256=sha(run/'train_successmix20_joint13_pi05_server.py'),overlay_manifest_sha256=sha(run/'overlay_manifest.json'),supersedes_request_sha256=old_request_sha,protocol_update='four_x_protocol_update',tolerance_action_loss_multiplier=4.0)
  budget=int(9567313920*5*1.05+(8<<30));assert shutil.disk_usage(run).free>budget
  request['remaining_disk_budget_bytes']=budget
  write(update/'remaining_training_disk_budget.json',dict(free_bytes=shutil.disk_usage(run).free,remaining_required_bytes=budget,full_arrow_cache_already_built=True,five_checkpoint_factor=1.05,reserve_bytes=8<<30))
  write(run/'training_request.json',request)
  module_spec=importlib.util.spec_from_file_location('four_x_normalization_receipt',run/'train_successmix20_joint13_pi05_server.py');new_entry=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(new_entry)
  assert new_entry.normalization_receipt(run,request,verify)==receipt
  old_inputs={k:old_request['config'][k] for k in ('model','data','batch_size','num_workers','seed')};new_inputs={k:request['config'][k] for k in old_inputs}
  new_inputs['data']={k:v for k,v in new_inputs['data'].items() if k!='tolerance_trajectory_action_loss_weight'}
  assert new_inputs==old_inputs
  write(update/'normalization_reuse_verification.json',dict(complete=True,at=now(),only_loss_weight_changed_in_normalization_inputs=True,reason='RunningStats consumes only state/actions; per-sample loss weight is excluded from normalization fingerprint',original_request_sha256=old_request_sha,training_request_sha256=sha(run/'training_request.json'),original_inputs_sha256=receipt['inputs_sha256'],stats_sha256=receipt['stats_sha256'],normalization_receipt_unchanged=True,receipt_sha256=sha(run/'normalization_completion.json')))
  env=os.environ.copy();env.update(PYTHONPATH=str(run/'source_overlay')+':/home/czl/openpi',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',ARROW_DEFAULT_MEMORY_POOL='system',SWANLAB_MODE='cloud')
  write(update/'transition_state.json',dict(stage='preflight_real_dataset_and_weighted_loss',at=now()))
  with (update/'training_preflight.log').open('a') as log:subprocess.run([sys.executable,'-u',str(run/'preflight_successmix20_four_x_server.py'),'--run-dir',str(run)],env=env,cwd='/home/czl/openpi',stdout=log,stderr=subprocess.STDOUT,check=True)
  assert read(update/'training_preflight_verification.json')['complete']
  for name,h in read(run/'repository_source_before.json')['sha256'].items():assert sha(Path('/home/czl/openpi')/name)==h,name
  assert not proc(3312670) or proc(3312670)['state']=='Z'
  assert not proc(3210109) or proc(3210109)['state']=='Z'
  env.pop('JAX_PLATFORMS',None);env.pop('CUDA_VISIBLE_DEVICES',None)
  with (run/'pipeline.log').open('a') as log:
   cmd=[sys.executable,'-u',str(run/'train_successmix20_joint13_pi05_server.py'),'--run-dir',str(run)]
   training=subprocess.Popen(cmd,env=env,cwd='/home/czl/openpi',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launch=dict(pid=training.pid,command=cmd,request_sha256=sha(run/'training_request.json'),at=now(),normalization_reused=True,tolerance_action_loss_multiplier=4.0);write(run/'training_launch.json',launch)
  with (run/'finish_training.log').open('a') as log:
   cmd=[sys.executable,'-u',str(run/'finish_successmix20_joint13_training_server.py'),'--run-dir',str(run)]
   finisher=subprocess.Popen(cmd,env=env,cwd='/home/czl/openpi',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(complete=True,at=now(),training_launch=launch,finisher_pid=finisher.pid,normalization_stats_sha256=receipt['stats_sha256'],preflight_complete=True,training_full10000_pending=True)
  write(update/'transition_completion.json',result);write(update/'transition_state.json',dict(stage='complete',**result));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
 try:main(args.run_dir)
 except BaseException:
  write(args.run_dir/'four_x_protocol_update/transition_state.json',dict(stage='failed',at=now(),traceback=traceback.format_exc()));raise
