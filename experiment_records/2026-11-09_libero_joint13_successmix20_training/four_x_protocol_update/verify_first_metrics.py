from pathlib import Path
import subprocess,json,datetime,math,re
rec=Path('/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-09_libero_joint13_successmix20_training')
source='''from pathlib import Path
import json,re,math
from swanlab import OpenApi
r=Path('/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109')
log=(r/'train.log').read_text(errors='replace');matches=list(re.finditer(r'Step\\s+(\\d+):\\s*([^\\r\\n]+)',log))
if not matches:
 print(json.dumps(dict(complete=False,stage='waiting_for_first_real_logged_step')))
else:
 m=matches[0];step=int(m.group(1));v={k:float(x) for k,x in re.findall(r'([\\w./]+)=([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?|nan|inf)',m.group(2))}
 assert all(math.isfinite(x) for x in v.values())
 keys=['loss','loss/action','loss/rotation_tolerance','loss_weighted/action','loss_weighted/rotation_tolerance','loss/total','loss/action_nominal_contribution','loss/action_reweighted_contribution','batch/action_loss_weight_mean','batch/reweighted_action_fraction']
 assert all(k in v for k in keys)
 assert abs(v['loss/action']-v['loss/action_nominal_contribution']-v['loss/action_reweighted_contribution'])<.0003
 assert abs(v['loss_weighted/action']-v['loss/action_nominal_contribution']-4*v['loss/action_reweighted_contribution'])<.0004
 assert abs(v['loss_weighted/rotation_tolerance']-.5*v['loss/rotation_tolerance'])<.0002
 assert abs(v['loss/total']-v['loss'])<.0002
 assert abs(v['loss']-v['loss_weighted/action']-v['loss_weighted/rotation_tolerance'])<.0003
 assert abs(v['batch/action_loss_weight_mean']-1-3*v['batch/reweighted_action_fraction'])<.0003
 api=OpenApi(log_level='error');expid='kl28cdu6bv1mywr8n5wv0';exp=api.get_experiment('openpi',expid,username='caozilin');assert exp.code==200 and exp.data.state=='RUNNING'
 response=api.get_metrics(expid,keys)
 if response.code==200 and step in response.data.index:
  values={key:float(response.data.loc[step,key]) for key in keys}
  assert all(math.isfinite(x) and abs(x-v[key])<.000061 for key,x in values.items())
  print(json.dumps(dict(complete=True,scope='First logged real training step and cloud loss identities; does not prove full10000 steps',first_logged_step=step,metrics=v,swanlab_id=expid,cloud_state=exp.data.state,cloud_values=values,all_four_x_identities_verified=True)))
 else:
  print(json.dumps(dict(complete=False,stage='local_first_metric_verified_waiting_cloud_sync',step=step,metrics=v)))
'''
p=subprocess.run(['ssh','-S','/tmp/libero_joint13_20261109.sock','-o','BatchMode=yes','czl@100.73.28.99','/home/czl/openpi/.venv/bin/python -'],input=source,text=True,capture_output=True,check=True)
v=json.loads(p.stdout);v['observed_at']=datetime.datetime.now().astimezone().isoformat()
if v.get('complete'):
 (rec/'four_x_protocol_update/first_real_metrics_and_cloud_verification.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v))
