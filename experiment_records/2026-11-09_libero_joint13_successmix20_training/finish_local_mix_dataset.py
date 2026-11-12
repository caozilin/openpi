"""Wait for generated files and explicit disk choice, assemble then independently audit."""
import fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/media/czl/sata/franka_my_code/openpi');REC=R/'experiment_records/2026-11-09_libero_joint13_successmix20_training';D=R/'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109';STAGE=Path('/tmp/libero_joint13_successmix20_replacement_stage_20261109')
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def alive(pid,needle):
 try:return needle in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
 except FileNotFoundError:return False
def main():
 with (REC/'local_pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(REC/'local_pipeline_process.json',dict(pid=os.getpid()))
  pid=read(REC/'stage_launch.json')['pid'];write(REC/'local_pipeline_state.json',dict(stage='converting301_replacements',pid=pid))
  while alive(pid,'build_successmix20_joint13.py'):time.sleep(10)
  assert read(STAGE/'replacement_stage_completion.json')['complete']
  while not (REC/'storage_authorization.json').exists():
   write(REC/'local_pipeline_state.json',dict(stage='waiting_for_user_storage_choice',replacements_ready=301));time.sleep(10)
  choice=read(REC/'storage_authorization.json');assert choice['explicit_user_answer']
  cmd=[str(R/'.venv/bin/python'),'-u',str(REC/'assemble_mixed_dataset.py')]
  if choice['allow_retire_untrained_nominal_conversion']:cmd.append('--retire-untrained-nominal-copy')
  write(REC/'local_pipeline_state.json',dict(stage='assembling1875',retire_untrained_copy=choice['allow_retire_untrained_nominal_conversion']));subprocess.run(cmd,cwd=R,check=True)
  assert read(REC/'assembly_completion.json')['complete']
  cmd=[str(R/'.venv/bin/python'),'-u',str(R/'examples/libero/audit_successmix20_joint13.py'),'--dataset',str(D)]
  env=dict(os.environ,PYTHONPATH=str(R),JAX_PLATFORMS='cpu',HF_DATASETS_CACHE='/tmp/libero_successmix20_audit_hf_20261109',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  with (D/'independent_audit.log').open('a') as log:
   p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);write(REC/'dataset_audit_process.json',dict(pid=p.pid,command=cmd));write(REC/'local_pipeline_state.json',dict(stage='auditing',audit_pid=p.pid));code=p.wait()
  assert code==0,('Audit failed',code);a=read(D/'meta/joint13_independent_audit.json');assert a['complete'] and a['episodes']==1875 and a['tolerance_episodes']==301
  write(REC/'local_pipeline_state.json',dict(stage='complete_local_dataset',episodes=1875,nominal=1574,tolerance=301,frames=312374,upload_training_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(REC/'local_pipeline_state.json',dict(stage='failed',traceback=traceback.format_exc()));raise
