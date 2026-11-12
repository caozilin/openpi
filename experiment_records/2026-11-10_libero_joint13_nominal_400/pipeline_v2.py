"""Supervise only the authorized no-tolerance400 evaluation and its audit."""
import base64,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from examples.libero.eval_joint13_nominal_libero_local import DEFAULT as RUN, PYTHON, validate, read, dump
REC=Path(__file__).resolve().parent
SOCKET='/tmp/libero_joint13_nominal400_20261110.sock';HOST='czl@100.73.28.99'
REMOTE='/mnt/SharedData/czl/runs/libero_eval_joint13_successmix20_nominal_400_20261110'
TRAIN='/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109'
SERVER='serve_joint13_libero_eval_server.py'
def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def remote(code):
 p=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes',HOST,'/home/czl/openpi/.venv/bin/python -'],input=code,capture_output=True,text=True,timeout=60)
 if p.returncode:raise RuntimeError(p.stderr[-2500:])
 return json.loads(p.stdout)
def status(stage,**kwargs):dump(RUN/'pipeline_state.json',dict(stage=stage,at=now(),**kwargs))
def start_inference():
 if (RUN/'inference_launch.json').exists():
  old=read(RUN/'inference_launch.json')
  check=remote(f"from pathlib import Path\nimport json\np=Path('/proc/{old['pid']}')\ncmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()\nassert {REMOTE!r} in cmd\nassert int((p/'stat').read_text().rsplit(')',1)[1].split()[19])=={old['start_ticks']}\nprint(json.dumps(dict(reused=True,pid={old['pid']})))")
  dump(RUN/'inference_reuse_verification.json',check);return old
 payload=base64.b64encode((RUN/SERVER).read_bytes()).decode()
 out=remote(f'''import json,subprocess,os,base64,hashlib,time,socket
from pathlib import Path
r=Path({REMOTE!r});t=Path({TRAIN!r});r.mkdir(parents=True,exist_ok=True)
a=json.loads((t/'training_independent_final_audit.json').read_text());assert a['complete']
assert json.loads((t/'task_completion.json').read_text())['complete']
assert not (r/'inference_manager_process.json').exists(),'Do not duplicate a prior inference launch'
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
stats={{int(line.split(',')[0]):[float(v) for v in line.split(',')[1:]] for line in gpu.strip().splitlines()}}
assert all(stats[i][0]<1500 for i in [0,1,2,3,4,6]),gpu
for port in range(18760,18766):
 s=socket.socket();s.bind(('127.0.0.1',port));s.close()
source=base64.b64decode({payload!r});p=r/{SERVER!r};p.write_bytes(source)
with (r/'inference_manager.log').open('a') as log:
 child=subprocess.Popen(['/home/czl/openpi/.venv/bin/python','-u',str(p)],cwd='/home/czl/openpi',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.1)
rest=Path(f'/proc/{{child.pid}}/stat').read_text().rsplit(')',1)[1].split()
o=dict(pid=child.pid,start_ticks=int(rest[19]),command=str(p),server_entry_sha256=hashlib.sha256(source).hexdigest(),gpu_before=gpu,training_final_audit_sha256=hashlib.sha256((t/'training_independent_final_audit.json').read_bytes()).hexdigest())
(r/'inference_launch.json').write_text(json.dumps(o,indent=2)+'\\n');print(json.dumps(o))
''')
 dump(RUN/'inference_launch.json',out);return out

def wait_readiness():
 import urllib.request
 for port in ([] if (RUN/'tunnels.json').exists() else range(18760,18766)):
  args=['ssh','-S',SOCKET,'-O','forward','-L',f'{port}:127.0.0.1:{port}',HOST]
  p=subprocess.run(args,capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
 dump(RUN/'tunnels.json',dict(complete=True,socket=SOCKET,ports=list(range(18760,18766))))
 deadline=time.monotonic()+1800
 while True:
  state=remote(f"from pathlib import Path\nimport json\nr=Path({REMOTE!r})\np=r/'inference_servers_state.json'\nprint(p.read_text() if p.exists() else '{{\"stage\":\"starting\"}}')")
  if state.get('stage')=='failed':raise RuntimeError(state)
  ready=True
  for port in range(18760,18766):
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:assert response.status==200
   except Exception:ready=False;break
  if ready:break
  if time.monotonic()>deadline:raise TimeoutError('Inference loading timed out')
  status('loading_inference',server_state=state);time.sleep(10)
 code=f'''import sys,json
sys.path.insert(0,{str(ROOT/'packages/openpi-client/src')!r})
from openpi_client import msgpack_numpy
import websockets.sync.client
rows=[]
for port in range(18760,18766):
 with websockets.sync.client.connect(f'ws://127.0.0.1:{{port}}',ping_interval=None,compression=None,max_size=None) as ws:
  rows.append(msgpack_numpy.unpackb(ws.recv(timeout=30)))
print(json.dumps(rows))
'''
 p=subprocess.run([str(PYTHON),'-c',code],capture_output=True,text=True,check=True,timeout=120)
 rows=json.loads(p.stdout);req=read(RUN/'evaluation_request.json')
 assert len(rows)==6
 for i,row in enumerate(rows):
  assert row['replica']==i and row['gpu']==req['gpu_ids'][i] and row['eval_identity']==req['eval_identity']
  assert row['effective_output_dimensions']==13 and row['tolerance_units']=='radians'
  assert row['training_request_sha256']==req['training_request_sha256'] and row['norm_stats_sha256']==req['norm_stats_sha256']
 assert len({x['params_values_sha256'] for x in rows})==1
 dump(RUN/'inference_identity_audit.json',dict(complete=True,at=now(),replicas=rows,server_entry_sha256=req['server_entry_sha256'],actual_websocket_handshakes=True))

def stop_inference():
 if not (RUN/'inference_launch.json').exists():return
 expected=read(RUN/'inference_launch.json')
 out=remote(f'''import json,os,signal,time
from pathlib import Path
r=Path({REMOTE!r});pid={expected['pid']};ticks={expected['start_ticks']}
children=[json.loads((r/f'replica_{{i}}_process.json').read_text())['pid'] for i in range(6) if (r/f'replica_{{i}}_process.json').exists()]
def alive(p):
 f=Path(f'/proc/{{p}}/stat');return f.exists() and f.read_text().rsplit(')',1)[1].split()[0]!='Z'
if alive(pid):
 cmd=Path(f'/proc/{{pid}}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert str(r/{SERVER!r}) in cmd
 assert int(Path(f'/proc/{{pid}}/stat').read_text().rsplit(')',1)[1].split()[19])==ticks
 os.kill(pid,signal.SIGTERM)
end=time.monotonic()+45
while any(alive(p) for p in [pid]+children) and time.monotonic()<end:time.sleep(1)
assert not any(alive(p) for p in [pid]+children)
print(json.dumps(dict(complete=True,manager_pid=pid,replica_pids=children,all_exited=True)))
''')
 dump(RUN/'inference_shutdown_verification.json',out)
 if (RUN/'tunnels.json').exists():
  for port in range(18760,18766):
   p=subprocess.run(['ssh','-S',SOCKET,'-O','cancel','-L',f'{port}:127.0.0.1:{port}',HOST],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
 p=subprocess.run(['ssh','-S',SOCKET,'-O','exit',HOST],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
 dump(RUN/'local_cleanup_verification.json',dict(complete=True,own_six_tunnels_and_socket_closed=True,viewer_processes_untouched=True))

def finish_report():
 import csv,collections
 a=read(RUN/'evaluation_independent_final_audit.json');c=read(RUN/'evaluation_completion.json')
 assert a['complete'] and a['episodes']==400 and c['complete'] and c['exit_codes']==[0]*6
 rows=[]
 for x in a['results']:
  folder=RUN/'episodes'/x['suite']/f"task_{x['task_id']:02d}"/f"seed_{x['seed']}"
  for name,h in x['artifact_sha256'].items():assert sha(folder/name)==h
  r=read(folder/'result.json');rows.append(dict(suite=r['suite'],task_id=r['task_id'],seed=r['seed'],success=r['success'],termination=r['termination'],steps=r['steps'],solver_failed_frames=r['total_solver_failures'],source_result=str(folder/'result.json')))
 assert len({(r['suite'],r['task_id'],r['seed']) for r in rows})==400
 suites={s:dict(episodes=sum(x['suite']==s for x in rows),successes=sum(x['suite']==s and x['success'] for x in rows)) for s in ('libero_spatial','libero_object','libero_goal','libero_10')}
 tasks=[dict(suite=s,task_id=t,successes=sum(x['suite']==s and x['task_id']==t and x['success'] for x in rows),episodes=10) for s in suites for t in range(10)]
 report=dict(historical_infrastructure_errors=0,complete=True,episodes=400,successes=sum(r['success'] for r in rows),per_suite=suites,per_task=tasks,terminations=dict(collections.Counter(r['termination'] for r in rows)),results=rows,infrastructure_errors=0)
 dump(RUN/'nominal400_report.json',report)
 with (RUN/'nominal400_results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# joint13无容差：40任务×seed0–9','',f"完整400回合成功{report['successes']}/400（{report['successes']/4:.2f}%），全部独立审计通过。",'','|套件|成功/100|','|---|---:|']+[f"|{s}|{v['successes']}/100|" for s,v in suites.items()]
 lines+=['','|套件|task_id|成功/10|','|---|---:|---:|']+[f"|{v['suite']}|{v['task_id']:02d}|{v['successes']}/10|" for v in tasks]
 lines+=['','本组沿用已验证的同packet同seed同metadata有界重连wrapper，原600秒总deadline；历史容差组的6次中断不计入本组。','模型为successmix20 joint13最终10000步；客户端采用bounds_rad=None的严格无容差分支；模型预测容差仅诊断保存，不进入控制器。连续失败20帧保持参考并终止，首次原生成功后继续20步；最终谓词单独记录。当前共享运行时与历史20261025快照不同，结果不声称为仅模型变化的严格因果对照。','',f'原始本地地址：{RUN}',f'请求：{RUN}/evaluation_request.json',f'审计：{RUN}/evaluation_independent_final_audit.json']
 (RUN/'NOMINAL_400_REPORT.md').write_text('\n'.join(lines)+'\n')
 for n in ['nominal400_report.json','nominal400_results.csv','evaluation_independent_final_audit.json','evaluation_completion.json','NOMINAL_400_REPORT.md']:
  shutil_copy=__import__('shutil').copyfile;shutil_copy(RUN/n,REC/n)
 (REC/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 return {k:v for k,v in report.items() if k!='results'}

def write_paired_comparison():
 from collections import Counter
 req=read(RUN/'evaluation_request.json');baseline=Path(req['paired_predicted_tolerance_run'])
 assert read(baseline/'evaluation_independent_final_audit.json')['complete']
 rows=[]
 for p in sorted(RUN.glob('episodes/*/*/*/result.json')):
  n=read(p);old=baseline/'episodes'/n['suite']/f"task_{n['task_id']:02d}"/f"seed_{n['seed']}"/'result.json';t=read(old)
  assert n['initial_state_sha256']==t['initial_state_sha256']
  assert n['server_metadata']['params_values_sha256']==t['server_metadata']['params_values_sha256']
  outcome=('both_success' if n['success'] and t['success'] else 'nominal_only' if n['success'] else 'tolerance_only' if t['success'] else 'both_failed')
  rows.append(dict(suite=n['suite'],task_id=n['task_id'],seed=n['seed'],nominal_success=n['success'],tolerance_success=t['success'],outcome=outcome,nominal_source=str(p),tolerance_source=str(old)))
 assert len(rows)==400
 dump(RUN/'paired_nominal_vs_predicted_tolerance.json',dict(complete=True,episodes=400,outcomes=dict(Counter(x['outcome'] for x in rows)),results=rows))

def main():
 lock=(RUN/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dump(RUN/'pipeline_process.json',dict(pid=os.getpid(),command=sys.argv,started_at=now()))
 validate(read(RUN/'evaluation_request.json'));children=[]
 amendment=read(REC/'numeric_comparison_audit_amendment.json')
 for path,h in amendment['new_source_sha256'].items():assert sha(Path(path))==h
 for path,h in read(Path(amendment['preserved_committed_manifest'])).items():assert sha(Path(path))==h
 try:
  while True:
   q=remote(f"from pathlib import Path\nimport json\nr=Path({TRAIN!r})\np=r/'task_completion.json'\ns=r/'finish_training_state.json'\nprint(json.dumps(dict(task=json.loads(p.read_text()) if p.exists() else None,finish=json.loads(s.read_text()) if s.exists() else None)))")
   if q['finish'] and q['finish']['stage']=='failed':raise RuntimeError(q)
   if q['task'] and q['task']['complete']:break
   status('waiting_for_training_final_audit',training=q);time.sleep(15)
  start_inference();status('loading_inference');wait_readiness()
  log=(RUN/'evaluation_pipeline.log').open('a');p=subprocess.Popen([str(PYTHON),'-u',str(REC/'run_client.py'),'--run-dir',str(RUN)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);children.append(p)
  dump(RUN/'evaluation_launch.json',dict(pid=p.pid,command='examples.libero.eval_joint13_nominal_libero_local'))
  status('evaluating',evaluation_manager=p.pid)
  initial=None
  initial_verified=(RUN/'initial_client_integration_audit.json').exists() and read(RUN/'initial_client_integration_audit.json')['complete']
  while p.poll() is None:
   if initial is None and not initial_verified and len(list(RUN.glob('episodes/*/task_*/seed_*/result.json')))>=6:
    f=(RUN/'initial_audit.log').open('a');initial=subprocess.Popen([str(PYTHON),'-u',str(REC/'audit_evaluation_v2.py'),'--run-dir',str(RUN),'--initial-sample','6'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);children.append(initial)
   if initial is not None and initial.poll() not in (None,0):raise RuntimeError('Initial integration audit failed; preserve completed results')
   time.sleep(10)
  assert p.returncode==0,'Evaluation infrastructure failure; inspect logs'
  if initial is not None:assert initial.wait()==0
  status('auditing')
  f=(RUN/'independent_audit.log').open('a');audit=subprocess.Popen([str(PYTHON),'-u',str(REC/'audit_evaluation_v2.py'),'--run-dir',str(RUN)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);children.append(audit)
  assert audit.wait()==0,'Full evaluation audit failed'
  result=finish_report();write_paired_comparison();stop_inference()
  dump(RUN/'task_completion.json',dict(result,all_required_audits_complete=True,own_inference_and_tunnels_stopped=True,finished_at=now()))
  status('complete',**result)
  with (REC/'README.md').open('a') as f:f.write(f"\n\n最终：已完成并通过独立审计，{result['successes']}/400。详见RESULTS.md；六副本与隧道已定向结束，viewer保留。\n")
  index=REC.parent/'README.md';text=index.read_text();text=text.replace('预测容差400回合准备/运行中',f"已完成并独立审计通过：{result['successes']}/400");index.write_text(text)
 except BaseException:
  error=traceback.format_exc()
  for child in children:
   if child.poll() is None:child.terminate()
  for child in children:
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  status('failed_or_interrupted',error=error)
  # Keep the precise remote process identity for inspection; never retry rollouts automatically.
  raise
if __name__=='__main__':main()
