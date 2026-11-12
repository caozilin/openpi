"""Wait for this one authorized job, capture runtime identity, then audit read-only."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path

def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def processes():
 result={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   rest=(d/'stat').read_text().rsplit(')',1)[1].split();cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   result[int(d.name)]=dict(pid=int(d.name),ppid=int(rest[1]),state=rest[0],start_ticks=int(rest[19]),command=cmd)
  except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):continue
 return result

def main(run):
 with (run/'finish_training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(run/'finish_training_process.json',dict(pid=os.getpid(),started_at=now()))
  audit_script=run/'audit_joint13_training_server.py';audit_sha=hashlib.sha256(audit_script.read_bytes()).hexdigest()
  while True:
   state=read(run/'training_state.json') if (run/'training_state.json').exists() else {'stage':'waiting_for_training_launch'}
   if state['stage']=='failed':raise RuntimeError(state)
   if (run/'training_process.json').exists():
    parent=read(run/'training_process.json')['pid'];procs=processes();owned={parent}
    for _ in range(5):owned.update(pid for pid,p in procs.items() if p['ppid'] in owned)
    snapshot=[p for pid,p in procs.items() if pid in owned]
    if any(p['pid']==parent and 'train_nominal_joint13_pi05_server.py' in p['command'] and str(run) in p['command'] for p in snapshot):
     write(run/'training_live_process_snapshot.json',dict(at=now(),stage=state['stage'],processes=snapshot))
     trainers=[p for p in snapshot if '--stage train' in p['command']]
     if trainers:
      trainer=trainers[0];children=[p for p in snapshot if p['ppid']==trainer['pid'] and 'spawn_main' in p['command']]
      env={}
      try:
       for pair in Path(f'/proc/{trainer["pid"]}/environ').read_bytes().split(b'\0'):
        key,_,val=pair.partition(b'=')
        if key in (b'CUDA_VISIBLE_DEVICES',b'JAX_PLATFORMS'):env[key.decode()]=val.decode()
      except FileNotFoundError:pass
      if len(children)==8 and env.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3,4,6':
       write(run/'training_runtime_verified.json',dict(complete=True,at=now(),parent=parent,trainer=trainer,workers=children,num_workers=8,environment=env))
   if state['stage']=='complete' and (run/'training_completion.json').exists():
    completion=read(run/'training_completion.json');assert completion['complete'] and completion['exit_code']==0 and completion['steps']==10000
    own=[p for p in processes().values() if 'train_nominal_joint13_pi05_server.py' in p['command'] and str(run) in p['command'] and p['state']!='Z']
    if not own:break
   write(run/'finish_training_state.json',dict(stage='waiting',training_stage=state['stage'],at=now()));time.sleep(30)
  assert read(run/'training_runtime_verified.json')['complete'],'Must observe the real6GPU trainer and8 workers'
  assert hashlib.sha256(audit_script.read_bytes()).hexdigest()==audit_sha,'Audit code changed while waiting'
  cmd=['/home/czl/openpi/.venv/bin/python','-u',str(audit_script),'--run-dir',str(run)]
  with (run/'independent_training_audit.log').open('a') as log:
   p=subprocess.Popen(cmd,cwd='/home/czl/openpi',stdout=log,stderr=subprocess.STDOUT)
   write(run/'independent_training_audit_process.json',dict(pid=p.pid,command=cmd,audit_script_sha256=audit_sha));write(run/'finish_training_state.json',dict(stage='auditing',audit_pid=p.pid,at=now()));code=p.wait()
  assert code==0,('Independent training audit failed',code)
  final=read(run/'training_independent_final_audit.json');assert final['complete']
  write(run/'task_completion.json',dict(complete=True,finished_at=now(),episodes=1875,frames=312374,all_nominal=True,steps=10000,full_independent_training_audit_sha256=hashlib.sha256((run/'training_independent_final_audit.json').read_bytes()).hexdigest()))
  write(run/'finish_training_state.json',dict(stage='complete',at=now()))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
 try:main(a.run_dir)
 except BaseException:
  write(a.run_dir/'finish_training_state.json',dict(stage='failed',traceback=traceback.format_exc(),at=now()));raise
