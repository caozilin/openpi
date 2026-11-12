"""Wait for independent local audit, publish a new dataset, verify remotely; no training."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/media/czl/sata/franka_my_code/openpi');REC=R/'experiment_records/2026-11-09_libero_joint13_successmix20_training';D=R/'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'
REMOTE='/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'
RUN='/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109';SSH=['ssh','-S','/tmp/libero_joint13_20261109.sock','-o','BatchMode=yes','czl@100.73.28.99']
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 with (REC/'upload.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(REC/'upload_process.json',dict(pid=os.getpid()))
  while True:
   state=read(REC/'local_pipeline_state.json')
   if state['stage']=='failed':raise RuntimeError(state)
   if state['stage']=='complete_local_dataset':break
   write(REC/'upload_state.json',dict(stage='waiting_for_independent_local_audit',local_stage=state['stage']));time.sleep(10)
  audit=read(D/'meta/joint13_independent_audit.json');assert audit['complete'] and audit['missing_labels']==0
  paths=sorted([*D.glob('data/**/*.parquet'),*D.glob('meta/**/*')]);paths=[p for p in paths if p.is_file()]
  paths += [D/n for n in ('conversion_request.json','conversion_manifest.json','conversion_completion.json','replacement_request.json')]
  entries=[dict(path=str(p.relative_to(D)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
  manifest=dict(episodes=1875,frames=312374,files=entries,total_bytes=sum(e['bytes'] for e in entries),nominal_episodes=1574,tolerance_episodes=301)
  write(D/'transfer_manifest.json',manifest);filelist=REC/'transfer_files.txt';filelist.write_text(''.join(e['path']+'\n' for e in entries)+'transfer_manifest.json\n')
  write(REC/'upload_state.json',dict(stage='uploading',files=len(entries),bytes=manifest['total_bytes'],manifest_sha256=sha(D/'transfer_manifest.json')))
  subprocess.run([*SSH,f'mkdir -p {REMOTE}'],check=True)
  rsync=['rsync','-rlt','--checksum','--partial','--copy-dest=/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_full_20261101','--stats','--files-from='+str(filelist),'-e','ssh -S /tmp/libero_joint13_20261109.sock -o BatchMode=yes',str(D)+'/',f'czl@100.73.28.99:{REMOTE}/']
  subprocess.run(rsync,check=True)
  subprocess.run(['rsync','-a','-e','ssh -S /tmp/libero_joint13_20261109.sock -o BatchMode=yes',str(R/'examples/libero/verify_successmix20_joint13_upload_server.py'),f'czl@100.73.28.99:{RUN}/'],check=True)
  write(REC/'upload_state.json',dict(stage='remote_verifying',manifest_sha256=sha(D/'transfer_manifest.json')))
  cmd=f'HF_DATASETS_CACHE={RUN}/reader_cache JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/czl/openpi/.venv/bin/python -u {RUN}/verify_successmix20_joint13_upload_server.py'
  subprocess.run([*SSH,cmd],check=True)
  subprocess.run(['rsync','-a','-e','ssh -S /tmp/libero_joint13_20261109.sock -o BatchMode=yes',f'czl@100.73.28.99:{RUN}/lerobot_upload_verification.json',str(REC/'lerobot_upload_verification.json')],check=True)
  v=read(REC/'lerobot_upload_verification.json');assert v['complete'] and v['transfer_manifest_sha256']==sha(D/'transfer_manifest.json')
  write(REC/'upload_state.json',dict(stage='complete',files=len(entries),bytes=manifest['total_bytes'],training_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(REC/'upload_state.json',dict(stage='failed',traceback=traceback.format_exc()));raise
