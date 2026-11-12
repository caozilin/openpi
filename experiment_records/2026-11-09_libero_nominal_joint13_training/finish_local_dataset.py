import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/media/czl/sata/franka_my_code/openpi');REC=R/'experiment_records/2026-11-09_libero_nominal_joint13_training';D=R/'data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109'
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def alive(pid,needle):
 try:return needle in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return False
def main():
 with (REC/'local_pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(REC/'local_pipeline_process.json',dict(pid=os.getpid()))
  launch=read(REC/'conversion_launch.json');pid=launch['pid']
  write(REC/'local_pipeline_state.json',dict(stage='converting',conversion_pid=pid))
  while alive(pid,'enrich_nominal_joint13_dataset.py'):time.sleep(10)
  assert read(D/'conversion_completion.json')['complete']
  with (D/'conversion.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cmd=[str(R/'.venv/bin/python'),'-u',str(R/'examples/libero/audit_nominal_joint13_dataset.py'),'--dataset',str(D)]
  env=dict(os.environ,PYTHONPATH=str(R),JAX_PLATFORMS='cpu',HF_DATASETS_CACHE='/tmp/libero_joint13_audit_hf_20261109',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  with (D/'independent_audit.log').open('a') as log:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
   write(REC/'dataset_audit_process.json',dict(pid=p.pid,command=cmd));write(REC/'local_pipeline_state.json',dict(stage='auditing',audit_pid=p.pid));code=p.wait()
  assert code==0,('Audit failed',code)
  a=read(D/'meta/joint13_independent_audit.json');assert a['complete'] and a['episodes']==1875
  write(REC/'local_pipeline_state.json',dict(stage='complete_local_dataset',episodes=1875,frames=312374,upload_and_training_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(REC/'local_pipeline_state.json',dict(stage='failed',traceback=traceback.format_exc()));raise
