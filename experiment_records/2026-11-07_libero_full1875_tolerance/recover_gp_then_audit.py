"""Resume one unfinished GP query sequence; preserve the frozen main batch."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import datetime,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path('/media/czl/sata/franka_my_code/franka_mujoco')
OPENPI=ROOT.parent/'openpi'
RECORD=OPENPI/'experiment_records/2026-11-07_libero_full1875_tolerance'
RUN=ROOT/'analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107'
CASE=RUN/'libero_10/task_02/demo_40'
RECOVERY=RUN/'attempts/0001_gp_fit_recovery'
ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl')
os.environ.update({k:ENV[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','MUJOCO_GL')})
sys.path.insert(0,str(ROOT))
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def state(stage,**extra):
 write(RUN/'recovery_state.json',dict(stage=stage,updated_at=now(),**extra))
 if stage!='waiting_for_original_exit':write(RUN/'pipeline_state.json',dict(stage=stage,updated_at=now(),**extra))
def originals_active():
 for name in ('pipeline_process.json','annotating_process.json'):
  pid=read(RUN/name)['pid'];p=Path('/proc')/str(pid)/'cmdline'
  if p.exists():
   text=p.read_bytes().replace(b'\0',b' ').decode()
   if str(RUN) in text or str(RECORD/'pipeline.py') in text:return True
 for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines():
  if line.startswith(str(ROOT/'.venv/bin/python')+' -u ') and '--search-prepared '+str(RUN)+'/' in line:return True
 return False

def make_fit_fallback(original,trace,save_trace):
 import torch
 from botorch.exceptions.errors import ModelFittingError
 def fit(mll,*args,**kwargs):
  initial=deepcopy(mll.state_dict());rng=torch.get_rng_state().clone()
  try:return original(mll,*args,**kwargs)
  except ModelFittingError as error:
   # Reset to the identical initial model and random stream. Only the
   # L-BFGS-B internal line-search bound changes after the default fails.
   mll.load_state_dict(initial);torch.set_rng_state(rng)
   revised=dict(kwargs);optimizer_kwargs=deepcopy(kwargs.get('optimizer_kwargs',{}));options=optimizer_kwargs.setdefault('options',{})
   assert options['maxiter']==100 and options.get('maxls',20)==20
   options['maxls']=100;revised['optimizer_kwargs']=optimizer_kwargs
   entry=dict(model=type(mll).__name__,default_error=str(error),default_maxls=20,fallback_maxls=100,maxiter=100,default_max_attempts=kwargs.get('max_attempts'),started_at=now(),success=False)
   trace.append(entry);save_trace()
   try:
    result=original(mll,*args,**revised);entry.update(success=True,finished_at=now());save_trace();return result
   except BaseException as error:
    entry.update(fallback_error=repr(error),finished_at=now());save_trace();raise
 return fit

def archive_once(source,destination):
 if source.exists() and not destination.exists():destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)

def main():
 RECOVERY.mkdir(parents=True,exist_ok=True)
 lock=(RUN/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(RUN/'recovery_process.json',dict(pid=os.getpid(),command=sys.argv,started_at=now()))
 try:
  while originals_active():
   if '--wait-parent' not in sys.argv:raise RuntimeError('Original batch still active')
   state('waiting_for_original_exit',pid=os.getpid());time.sleep(20)
  held=[]
  for name in ('pipeline.lock','annotation.lock'):
   stream=(RUN/name).open('r');fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(stream)
  if (RUN/'task_completion.json').exists() and read(RUN/'task_completion.json')['complete']:return
  request=read(RUN/'annotation_request.json');assert request['planned_episodes']==1875
  assert all(sha(Path(p))==digest for p,digest in request['code_sha256'].items()),'Frozen production source changed'
  errors=list(RUN.glob('libero_*/task_*/demo_*/adapter_error.json'));assert errors==[CASE/'adapter_error.json'],f'Unexpected error set: {errors}'
  assert 'ModelFittingError' in (CASE/'search.log').read_text()
  summary=read(RUN/'annotation_summary.json');assert summary['processed']==1875 and summary['adapter_errors']==1
  completed=list(RUN.glob('libero_*/task_*/demo_*/annotation_status.json'));assert len(completed)==1874
  for name in ('annotation_summary.json','pipeline_state.json','pipeline.log','annotating_exit.json'):
   archive_once(RUN/name,RECOVERY/'original'/name)
  for name in ('adapter_error.json','search.log','fit_attempts.json','search_progress.json'):
   archive_once(CASE/name,RECOVERY/'original_case'/name)
  checkpoint=CASE/'subtask_01/checkpoint.json';archive_once(checkpoint,RECOVERY/'original_case/subtask_01/checkpoint.json')
  rawpaths=list(RUN.rglob('trial.json'));protected={str(p):sha(p) for p in rawpaths}
  for p in RUN.glob('libero_*/task_*/demo_*/tolerance_labels.json'):protected[str(p)]=sha(p)
  for p in completed:protected[str(p)]=sha(p)
  protected[str(CASE/'subtask_00/annotation.json')]=sha(CASE/'subtask_00/annotation.json')
  write(RECOVERY/'preserved_files_before.json',protected)
  original=read(checkpoint);observations_before=len(original['state']['observations']);assert observations_before==17 and original['state']['initial_queries']==4
  plan=dict(scope='one unfinished infrastructure-error case, no completed rollout/candidate remeasurement',case='libero_10/02/demo_40',source_episode_index=1555,parent_request_sha256=sha(RUN/'annotation_request.json'),prepared_input_sha256=sha(CASE/'prepared_input.json'),original_checkpoint_sha256=sha(checkpoint),script=str(Path(__file__).resolve()),script_sha256=sha(Path(__file__)),source_code_sha256=request['code_sha256'],original_observations=observations_before,expected_final_observations=20,remaining_queries=3,numerical_difference=dict(policy='only after original fit raises ModelFittingError; restore model and torch RNG then retry with maxls100',lbfgsb_default_maxls=20,lbfgsb_fallback_maxls=100,gp_maxiter=100,max_attempts=5),research_parameters_unchanged=['search_seed','extra16','initial4','GP prior and data','IPOPT20','source actions','authored caps','clearance epsilon0.1mm','cache85'],created_at=now())
  if (RECOVERY/'recovery_request.json').exists():assert read(RECOVERY/'recovery_request.json')==plan,'Recovery request changed'
  else:write(RECOVERY/'recovery_request.json',plan)
  state('recovering_gp_case',case=plan['case'],remaining_queries=3,pid=os.getpid())
  import torch
  from unittest.mock import patch
  import botorch_joint_search as engine
  from tasktol.data.libero_annotation.dsp import search_prepared
  torch.set_num_threads(1);trace=[]
  def save_trace():write(RECOVERY/'numerical_fallback_trace.json',dict(policy=plan['numerical_difference'],events=trace))
  with patch.object(engine,'fit_gpytorch_mll',make_fit_fallback(engine.fit_gpytorch_mll,trace,save_trace)):
   recovered=search_prepared(CASE/'prepared_input.json',extra=16,preflight=False)
  assert recovered['complete'] and recovered['status']=='labeled'
  assert sha(CASE/'prepared_input.json')==plan['prepared_input_sha256']
  assert len(read(checkpoint)['state']['observations'])==20
  changed=[p for p,h in protected.items() if sha(Path(p))!=h];assert not changed,changed
  existing_trial_paths={p for p in protected if p.endswith('/trial.json')};new_trials=[str(p) for p in RUN.rglob('trial.json') if str(p) not in existing_trial_paths];assert len(new_trials)==3,new_trials
  assert not (RECOVERY/'original_case/archived_adapter_error.json').exists()
  shutil.move(CASE/'adapter_error.json',RECOVERY/'original_case/archived_adapter_error.json')
  results=[read(RUN/s['suite']/f"task_{s['task_id']:02d}"/s['demo']/'annotation_status.json') for s in request['sources']]
  assert len(results)==1875 and all(x['complete'] for x in results)
  assert len({(x['suite'],x['task_id'],x['demo']) for x in results})==1875
  counts=Counter(x['status'] for x in results);assert not list(RUN.glob('libero_*/task_*/demo_*/adapter_error.json'))
  merged=dict(complete=True,processing_complete=True,all_labeled=counts['labeled']==1875,processed=1875,labeled=counts['labeled'],no_feasible_label=counts['no_feasible_label'],baseline_unavailable=counts['baseline_unavailable'],adapter_errors=0,results=results,historical_infrastructure_errors=1,recovery_request=str(RECOVERY/'recovery_request.json'))
  write(RUN/'annotation_summary.json',merged)
  write(RECOVERY/'recovery_completion.json',dict(complete=True,preserved_files=len(protected),all_original_file_hashes_unchanged=True,new_trials=new_trials,original_observations=17,final_observations=20,active_adapter_errors=0,historical_infrastructure_errors=1,numerical_fallbacks=len(trace),finished_at=now()))
  finish_audit_and_reports()
 except BaseException:
  state('recovery_failed',error=traceback.format_exc());raise

def finish_audit_and_reports():
 state('auditing_recovered_batch',pid=os.getpid(),episodes=1875)
 command=[str(ROOT/'.venv/bin/python'),'-u','-m','tasktol.data.libero_annotation.audit','--output',str(RUN)]
 with (RUN/'independent_audit.log').open('a') as log:
  child=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 write(RUN/'auditing_process.json',dict(pid=child.pid,command=command,started_at=now()))
 code=child.wait();write(RUN/'auditing_exit.json',dict(pid=child.pid,exit_code=code,finished_at=now()));assert code==0,'Full independent audit failed'
 audit=read(RUN/'annotation_independent_final_audit.json');assert audit['complete'] and audit['episodes']==1875 and audit['adapter_errors']==0
 for source in read(RUN/'selection.json')['sources']:assert sha(Path(source['source_folder'])/'report.json')==source['report_sha256']
 spec=importlib.util.spec_from_file_location('full1875_supervisor',RECORD/'pipeline.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.finish(audit)
 note='\n\n历史基础设施异常1条：Long02/demo40净空GP拟合失败。保留全部旧候选，仅从13/16追加续完3次；GP原拟合失败后使用L-BFGS线搜索maxls100回退（默认20），GP迭代仍100，其他研究参数不变。恢复请求、原始失败、全部保留文件SHA与数值回退日志在`'+str(RECOVERY)+'`。这是一条明确记录的数值恢复，不冒充完全相同数值设置；全量独立审计已通过。\n'
 for p in [RUN/'ANNOTATION_REPORT.md',RECORD/'RESULTS.md']:p.write_text(p.read_text()+note)
 c=read(RUN/'task_completion.json');c.update(historical_infrastructure_errors=1,recovery_complete=True,recovery_request=str(RECOVERY/'recovery_request.json'));write(RUN/'task_completion.json',c)
 module.export_status(RUN,module.VIEW)
 write(RUN/'recovery_state.json',dict(stage='complete',finished_at=now(),independent_audit_complete=True))
 print(json.dumps(c,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
