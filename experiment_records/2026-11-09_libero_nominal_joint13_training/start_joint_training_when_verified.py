"""Freeze and launch the already authorized training exactly once after both audits."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/media/czl/sata/franka_my_code/openpi');REC=R/'experiment_records/2026-11-09_libero_nominal_joint13_training';D=R/'data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109';RUN='/mnt/SharedData/czl/runs/pi05_libero_nominal_joint13_lora_6gpu_bs72_10k_20261109';SSH=['ssh','-S','/tmp/libero_joint13_20261109.sock','-o','BatchMode=yes','czl@100.73.28.99']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def remote_python(source):
 p=subprocess.run([*SSH,'/home/czl/openpi/.venv/bin/python -'],input=source,text=True,capture_output=True,check=True);return json.loads(p.stdout)
def main():
 with (REC/'training_dispatch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(REC/'training_dispatch_process.json',dict(pid=os.getpid()))
  while True:
   state=read(REC/'upload_state.json')
   if state['stage']=='failed':raise RuntimeError(state)
   if state['stage']=='complete':break
   write(REC/'training_dispatch_state.json',dict(stage='waiting_for_verified_upload',upload_stage=state['stage']));time.sleep(10)
  verify=read(REC/'lerobot_upload_verification.json');assert verify['complete'] and verify['all_uploaded_sha256_equal'] and verify['portable_lerobot_reader_passed'];assert verify['transfer_manifest_sha256']==sha(D/'transfer_manifest.json')
  info=remote_python('''from pathlib import Path
import json,hashlib,shutil,subprocess
r=Path("'''+RUN+'''");d=Path('/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109')
ck=Path('/mnt/SharedData/czl/checkpoints/pi05_libero_reconstructed_nostate_6gpu_bs72/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101/10000')
size=int(subprocess.check_output(['du','-sk',str(ck)],text=True).split()[0])*1024
print(json.dumps(dict(free_bytes=shutil.disk_usage(d).free,dataset_parquet_bytes=sum(p.stat().st_size for p in d.glob('data/**/*.parquet')),old_checkpoint_bytes=size,overlay_manifest_sha256=hashlib.sha256((r/'overlay_manifest.json').read_bytes()).hexdigest(),overlay_preparation=json.loads((r/'overlay_preparation.json').read_text()),server_script_sha256=hashlib.sha256((r/'train_nominal_joint13_pi05_server.py').read_bytes()).hexdigest(),existing_process=json.loads((r/'training_process.json').read_text()) if (r/'training_process.json').exists() else None)))''')
  assert info['overlay_preparation']['complete'] and info['overlay_preparation']['repository_unchanged']
  request=read(REC/'training_request_draft.json');assert request['server_script_sha256']==info['server_script_sha256']
  budget=int(info['dataset_parquet_bytes']*1.03+info['old_checkpoint_bytes']*5*1.05+(8<<30));write(REC/'training_disk_budget.json',dict(**info,remaining_required_bytes=budget,cache_factor=1.03,checkpoint_factor=1.05,reserve_bytes=8<<30))
  assert info['free_bytes']>budget,('Insufficient space; do not delete old files automatically',info['free_bytes'],budget)
  request.update(status='frozen_verified_ready_to_train',overlay_manifest_sha256=info['overlay_manifest_sha256'],remaining_disk_budget_bytes=budget,transfer_manifest_sha256=sha(D/'transfer_manifest.json'),local_dataset_independent_audit_sha256=sha(D/'meta/joint13_independent_audit.json'),upload_verification_sha256=sha(REC/'lerobot_upload_verification.json'),dataset_all_nominal=True,dataset_episodes=1875,dataset_frames=312374,missing_tolerance_labels=0,remote_verifier_sha256=sha(R/'examples/libero/verify_joint13_upload_server.py'))
  write(REC/'training_request.json',request)
  payload=json.dumps(request)
  result=remote_python('''from pathlib import Path
import json,hashlib,subprocess,fcntl,os
r=Path("'''+RUN+'''");request=json.loads('''+repr(payload)+''')
with (r/'training_launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (r/'training_launch.json').exists():
  old=json.loads((r/'training_request.json').read_text());assert old==request,'Existing run request differs'
  print((r/'training_launch.json').read_text())
 else:
  assert not (r/'training_process.json').exists(),'Existing process must be inspected, not relaunched'
  script=r/'train_nominal_joint13_pi05_server.py';assert hashlib.sha256(script.read_bytes()).hexdigest()==request['server_script_sha256']
  v=json.loads((r/'lerobot_upload_verification.json').read_text());assert v['complete'] and v['transfer_manifest_sha256']==request['transfer_manifest_sha256']
  assert hashlib.sha256((r/'lerobot_upload_verification.json').read_bytes()).hexdigest()==request['upload_verification_sha256']
  if (r/'training_request.json').exists():(r/'training_request.json').rename(r/'training_request_draft_before_freeze.json')
  (r/'training_request.json').write_text(json.dumps(request,indent=2)+'\\n')
  cmd=['/home/czl/openpi/.venv/bin/python','-u',str(script),'--run-dir',str(r)]
  with (r/'pipeline.log').open('a') as log:p=subprocess.Popen(cmd,cwd='/home/czl/openpi',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  receipt=dict(pid=p.pid,command=cmd,request_sha256=hashlib.sha256((r/'training_request.json').read_bytes()).hexdigest())
  (r/'training_launch.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))''')
  write(REC/'remote_training_launch.json',result);write(REC/'training_dispatch_state.json',dict(stage='launched',**result,full_training_and_final_audit_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(REC/'training_dispatch_state.json',dict(stage='failed',traceback=traceback.format_exc()));raise
