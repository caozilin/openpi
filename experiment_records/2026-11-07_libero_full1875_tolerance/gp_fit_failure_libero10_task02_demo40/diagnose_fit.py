import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import sys,json,traceback,warnings,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/media/czl/sata/franka_my_code/franka_mujoco');sys.path.insert(0,str(ROOT))
import torch
import botorch_joint_search as engine
from literature_botorch_search import LiteratureJointSearch
from benchmark_paper_cbo import stable_seed
from recover_dsp_random_fit import retry_factory
folder=ROOT/'analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/libero_10/task_02/demo_40/subtask_01';checkpoint=folder/'checkpoint.json';d=json.loads(checkpoint.read_text());s=d['state'];c=s['configuration'];initial=s['observations'][:s['initial_queries']]
torch.set_num_threads(1)
search=LiteratureJointSearch(c['maximum'],c['seed'],initial,state=s,profile='dsp',fit_maxiter=c['fit_maxiter'],sobol_points=c['sobol_points'],restarts=c['restarts'],optimizer_maxiter=c['optimizer_maxiter'],clearance_m=c['clearance_m'],constraint_metric=c['constraint_metric'])
attempts=[];orig=engine.fit_gpytorch_mll;fitnum=[0]
def fitted(mll,*args,**kwargs):
 fitnum[0]+=1;which=fitnum[0];optimizer=kwargs['optimizer']
 def logged(*a,**kw):
  res=optimizer(*a,**kw);attempts.append(dict(model_number=which,status=res.status.name,iterations=int(res.step),fval=float(res.fval),message=str(res.message)));return res
 kwargs['optimizer']=logged
 return retry_factory(orig,[])(mll,*args,**kwargs)
result=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),observations=len(s['observations']),existing_raw_trials_unchanged=True,query_or_rollout_executed=False)
with warnings.catch_warnings(record=True) as ws, torch.random.fork_rng(devices=[]), patch.object(engine,'fit_gpytorch_mll',fitted):
 warnings.simplefilter('always');torch.manual_seed(stable_seed(search.seed,'botorch_fit',len(s['observations'])))
 try:search._fit();result.update(fit_complete=True)
 except Exception as e:result.update(fit_complete=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 result['warnings']=[str(w.message) for w in ws]
result['attempts']=attempts
out=Path('/tmp/libero_gp_fit_diagnosis.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
