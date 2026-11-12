"""Independent full CPU checkpoint/config/metrics audit after joint13 training exits."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',ARROW_DEFAULT_MEMORY_POOL='system',SWANLAB_MODE='cloud')
import argparse,datetime,fcntl,gc,hashlib,json,math,re,sys
from pathlib import Path

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def own_training_processes(run):
 live=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if 'train_nominal_joint13_pi05_server.py' in cmd and str(run) in cmd:live.append(dict(pid=int(proc.name),command=cmd))
 return live

def audit(run):
 # Retain the established dependency import order and do not touch GPUs.
 import torch,pyarrow
 import numpy as np
 import jax
 import orbax.checkpoint as ocp
 lock=(run/'independent_training_audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 request=read(run/'training_request.json');actual=read(run/'actual_config.json');completion=read(run/'training_completion.json');verify=read(run/'lerobot_upload_verification.json');norm=read(run/'normalization_completion.json')
 assert completion['complete'] and completion['exit_code']==0 and completion['steps']==10000
 assert read(run/'training_state.json')['stage']=='complete'
 assert not own_training_processes(run),own_training_processes(run)
 runtime=read(run/'training_runtime_verified.json');assert runtime['complete'] and runtime['num_workers']==8 and runtime['environment']['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,6'
 for proc in read(run/'training_live_process_snapshot.json')['processes']:
  stat=Path(f'/proc/{proc["pid"]}/stat')
  if stat.exists():
   rest=stat.read_text().rsplit(')',1)[1].split()
   assert int(rest[19])!=proc['start_ticks'] or rest[0]=='Z',('Original training process still live',proc)
 with (run/'training.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 spec=request['config'];groups=[['action',0,7,1.0],['rotation_tolerance',7,13,.5]]
 for field in ('name','exp_name','seed','batch_size','num_workers','num_train_steps','ema_decay','fsdp_devices','log_interval','save_interval','keep_period','swanlab_enabled','swanlab_mode','model','lr_schedule','optimizer','weight_loader'):
  if field=='model':
   for k,v in spec[field].items():assert actual[field][k]==v,(field,k,actual[field][k],v)
  else:assert actual[field]==spec[field],(field,actual[field],spec[field])
 assert actual['model']['action_loss_groups']==groups
 assert actual['model']['action_dim']==32 and actual['model']['action_horizon']==10 and not actual['model']['discrete_state_input'] and actual['model']['max_token_len']==48
 assert actual['weight_loader']['params_path']=='gs://openpi-assets/checkpoints/pi05_base/params'
 assert actual['batch_size']==72 and actual['num_train_steps']==10000 and actual['fsdp_devices']==2 and actual['seed']==42 and actual['num_workers']==8
 assert actual['data']['repo_id']=='caozilin/libero_reconstructed_nominal_joint13_20261109'
 assert request['dataset_all_nominal'] and request['dataset_episodes']==1875 and request['dataset_frames']==312374 and request['missing_tolerance_labels']==0
 assert sha(run/'train_nominal_joint13_pi05_server.py')==request['server_script_sha256']
 assert sha(run/'overlay_manifest.json')==request['overlay_manifest_sha256']
 for name,h in read(run/'overlay_manifest.json')['sha256'].items():assert sha(run/'source_overlay'/name)==h,name
 for name,h in read(run/'repository_source_before.json')['sha256'].items():assert sha(Path('/home/czl/openpi')/name)==h,name
 assert verify['complete'] and verify['all_uploaded_sha256_equal'] and verify['all_original_nominal_columns_equal'] and verify['portable_lerobot_reader_passed']
 assert verify['transfer_manifest_sha256']==request['transfer_manifest_sha256']
 assert sha(run/'lerobot_upload_verification.json')==request['upload_verification_sha256']
 dataset=Path(actual['data']['lerobot_dataset_root']);manifest=read(dataset/'transfer_manifest.json')
 assert sha(dataset/'transfer_manifest.json')==request['transfer_manifest_sha256']
 for entry in manifest['files']:
  p=dataset/entry['path'];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],entry['path']
 ns=Path(norm['stats_path']);assert norm['complete'] and sha(ns)==norm['stats_sha256'];stats=read(ns)['norm_stats']
 fingerprint={key:spec[key] for key in ('model','data','batch_size','num_workers','seed')}
 fingerprint['transfer_manifest_sha256']=verify['transfer_manifest_sha256']
 assert hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()==norm['inputs_sha256']
 for key,width in [('state',8),('actions',13)]:
  for values in stats[key].values():assert len(values)==width and np.isfinite(values).all()
  assert np.all(np.asarray(stats[key]['std'])>=0)
 log=(run/'train.log').read_text(errors='replace');metrics={}
 for m in re.finditer(r'Step\s+(\d+):\s*([^\r\n]+)',log):
  step=int(m.group(1));values={k:float(v) for k,v in re.findall(r'([\w./]+)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf)',m.group(2))}
  assert step not in metrics,('Duplicate metric step',step)
  assert values and all(math.isfinite(v) for v in values.values());metrics[step]=values
 expected=list(range(10,10001,10));assert sorted(metrics)==expected
 group_keys=['loss/action','loss/rotation_tolerance','loss_weighted/action','loss_weighted/rotation_tolerance','loss/total']
 for step,v in metrics.items():
  assert all(k in v for k in ['loss','grad_norm','param_norm',*group_keys])
  assert abs(v['loss_weighted/action']-v['loss/action'])<.0002
  assert abs(v['loss_weighted/rotation_tolerance']-.5*v['loss/rotation_tolerance'])<.0002
  assert abs(v['loss/total']-v['loss'])<.0002
  assert abs(v['loss']-v['loss_weighted/action']-v['loss_weighted/rotation_tolerance'])<.0003
 checkpoint=Path(completion['checkpoint_dir']);assert checkpoint==Path(actual['checkpoint_base_dir'])/actual['name']/actual['exp_name']
 results=[]
 for step in range(2000,10001,2000):
  c=checkpoint/str(step);commit=read(c/'_CHECKPOINT_METADATA');assert commit['commit_timestamp_nsecs']>commit['init_timestamp_nsecs']
  asset=c/'assets'/actual['data']['repo_id']/'norm_stats.json';assert sha(asset)==norm['stats_sha256']
  result=dict(step=step,commit_metadata=commit,norm_stats_sha256=sha(asset),items={},projection_shapes={})
  for item in ['params','train_state']:
   h=ocp.PyTreeCheckpointer();meta=h.metadata(c/item);tree=meta.tree if hasattr(meta,'tree') else meta
   args=jax.tree.map(lambda _:ocp.RestoreArgs(restore_type=np.ndarray),tree)
   restored=h.restore(c/item,args=ocp.args.PyTreeRestore(restore_args=args));leaves=jax.tree.leaves(restored);digest=hashlib.sha256();elements=0;nbytes=0
   for a in leaves:
    a=np.asarray(a);assert a.dtype.kind in 'biufc' or str(a.dtype)=='bfloat16';assert np.isfinite(a).all(),(step,item)
    digest.update(str(a.shape).encode());digest.update(str(a.dtype).encode());digest.update(a.tobytes());elements+=a.size;nbytes+=a.nbytes
   if item=='train_state':
    result['actual_train_state_step']=int(np.asarray(restored['step']));assert result['actual_train_state_step']==step
   else:
    for path,a in jax.tree_util.tree_flatten_with_path(restored)[0]:
     name='/'.join(str(getattr(k,'key',getattr(k,'idx',k))) for k in path)
     if ('action_in_proj' in name or 'action_out_proj' in name) and 'kernel' in name:
      shape=list(np.asarray(a).shape);result['projection_shapes'][name]=shape
      assert (shape[0]==32 if 'action_in_proj' in name else shape[-1]==32),(name,shape)
    assert len(result['projection_shapes'])>=2
   result['items'][item]=dict(array_leaves=len(leaves),elements=elements,restored_bytes=nbytes,all_finite=True,values_sha256=digest.hexdigest())
   h.close();del restored,leaves,meta,tree,args,h;gc.collect()
  result['complete']=True;write(run/f'checkpoint_restore_audit_{step}.json',result);results.append(result)
  write(run/'independent_training_audit_state.json',dict(stage='checkpoint_restore',completed_steps=[r['step'] for r in results]));print('CPU restored',step,flush=True)
 assert len({r['items']['params']['values_sha256'] for r in results})==5
 write(run/'checkpoint_final_audit.json',dict(complete=True,checkpoints=results,completed_at=now(),all_arrays_restored_on_cpu=True))
 from swanlab import OpenApi
 expid=(checkpoint/'swanlab_id.txt').read_text().strip();assert expid
 api=OpenApi(log_level='error');exp=api.get_experiment(actual['project_name'],expid,username='caozilin');assert exp.code==200 and exp.data.state=='FINISHED' and exp.data.name==actual['exp_name']
 cloud=api.get_metrics(expid,['loss',*group_keys]);assert cloud.code==200
 frame=cloud.data;assert len(frame)==1000 and sorted(int(i) for i in frame.index)==expected,(len(frame),list(frame.index)[-5:])
 for step in expected:
  for key in ['loss',*group_keys]:
   value=float(frame.loc[step,key]);assert math.isfinite(value) and abs(value-metrics[step][key])<.000061,(step,key,value,metrics[step][key])
 frame[['loss',*group_keys]].to_csv(run/'swanlab_cloud_verified_metrics.csv',index_label='step')
 cloud_receipt=dict(metrics_csv_sha256=sha(run/'swanlab_cloud_verified_metrics.csv'),complete=True,id=expid,url=f'https://swanlab.cn/@caozilin/{actual["project_name"]}/runs/{expid}',cloud_state=exp.data.state,cloud_rows=len(frame),keys=['loss',*group_keys],last_step=10000,all_cloud_values_match_local_logs=True)
 write(run/'swanlab_cloud_final_audit.json',cloud_receipt)
 final=dict(complete=True,completed_at=now(),training_completion=completion,actual_config_matches_request=True,original_repository_unchanged=True,all_source_overlay_sha_verified=True,training_processes_exited=True,training_lock_released=True,logged_steps=1000,train_log_sha256=sha(run/'train.log'),actual_config_sha256=sha(run/'actual_config.json'),uploaded_dataset_sha_rechecked_at_finish=True,last_metrics=metrics[10000],checkpoint_full_cpu_restore_audit=dict(complete=True,checkpoints=results),normalization_dimensions=dict(state=8,actions=13),dataset=dict(episodes=1875,frames=312374,all_nominal=True,missing_labels=0,transfer_manifest_sha256=verify['transfer_manifest_sha256']),swanlab=cloud_receipt)
 write(run/'training_independent_final_audit.json',final);write(run/'independent_training_audit_state.json',dict(stage='complete',completed_at=now()));return final
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();print(audit(a.run_dir),flush=True)
