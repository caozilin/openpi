"""Six-process orchestration of the shared frozen DSP entry and read-only audit."""
import concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime
ROOT=Path('/media/czl/sata/franka_my_code')
OUT=ROOT/'franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109'
SNAP=OUT/'source_snapshot'; PY=ROOT/'franka_mujoco/.venv/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def now():return datetime.now().astimezone().isoformat()
def environment():return dict(os.environ,PYTHONPATH=str(SNAP)+':'+str(ROOT/'franka_mujoco'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
def protect():
 for name,h in read(OUT/'original_results_protection.json')['sha256'].items():
  assert sha(Path(name))==h,('Original result changed',name)
def worker(s):
 folder=OUT/s['suite']/f"task_{s['task_id']:02d}"/s['demo'];status=folder/'annotation_status.json'
 if status.exists():
  d=read(status)
  if d['complete'] and not d.get('preflight',False):return d
 cmd=[str(PY),'-u',str(SNAP/'annotate_libero_tolerance.py'),'--search-prepared',str(folder/'prepared_input.json'),'--extra','16']
 with (folder/'search.log').open('a') as log:
  p=subprocess.Popen(cmd,cwd=SNAP,env=environment(),stdout=log,stderr=subprocess.STDOUT)
  write(folder/'search_process.json',dict(pid=p.pid,command=cmd,started_at=now()))
  code=p.wait()
 if code:
  d=dict(complete=False,status='adapter_error',suite=s['suite'],task_id=s['task_id'],demo=s['demo'],exit_code=code,log=str(folder/'search.log'))
  write(folder/'adapter_error.json',d);return d
 d=read(status);assert d['complete'] and not d['preflight'];return d
def main():
 with (OUT/'execution.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  request=read(OUT/'annotation_request.json')
  for name,h in request['code_sha256'].items():assert sha(Path(name))==h,name
  protect();sources=read(OUT/'selection.json')['sources'];assert len(sources)==24
  write(OUT/'pipeline_process.json',dict(pid=os.getpid(),command=sys.argv,started_at=now(),request_sha256=sha(OUT/'annotation_request.json')))
  results=[]
  def update(stage):
   summary=dict(processing_complete=len(results)==24,processed=sum(r['complete'] for r in results),planned=24,labeled=sum(r['status']=='labeled' for r in results),no_feasible_label=sum(r['status']=='no_feasible_label' for r in results),baseline_unavailable=sum(r['status']=='baseline_unavailable' for r in results),adapter_errors=sum(r['status']=='adapter_error' for r in results),results=results)
   write(OUT/'annotation_summary.json',summary);write(OUT/'pipeline_state.json',dict(stage=stage,updated_at=now(),**{k:v for k,v in summary.items() if k!='results'}))
  update('searching')
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
   futures=[pool.submit(worker,s) for s in sources]
   for f in concurrent.futures.as_completed(futures):
    results.append(f.result());update('searching');print(now(),len(results),results[-1]['suite'],results[-1]['task_id'],results[-1]['demo'],results[-1]['status'],flush=True)
  if any(not r['complete'] for r in results):raise RuntimeError('Infrastructure errors retained; no automatic retries')
  update('auditing');cmd=[str(PY),'-u','-m','tasktol.data.libero_annotation.audit','--output',str(OUT)]
  with (OUT/'independent_audit.log').open('a') as log:
   p=subprocess.Popen(cmd,cwd=SNAP,env=environment(),stdout=log,stderr=subprocess.STDOUT)
   write(OUT/'audit_process.json',dict(pid=p.pid,command=cmd,started_at=now()));code=p.wait()
  assert code==0,('Audit failed',code)
  audit=read(OUT/'annotation_independent_final_audit.json');assert audit['complete'] and audit['episodes']==24
  protect();write(OUT/'original_results_verification.json',dict(complete=True,files=len(read(OUT/'original_results_protection.json')['sha256']),checked_at=now()))
  update('complete');write(OUT/'task_completion.json',dict(complete=True,scope='24 supplemental annotations only; full joint13 training remains pending',episodes=24,labeled=audit['labeled'],no_feasible_label=audit['no_feasible_label'],baseline_unavailable=audit['baseline_unavailable'],finished_at=now(),audit_sha256=sha(OUT/'annotation_independent_final_audit.json')))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(OUT/'pipeline_failure.json',dict(complete=False,traceback=traceback.format_exc(),at=now()));write(OUT/'pipeline_state.json',dict(stage='failed',at=now()));raise
