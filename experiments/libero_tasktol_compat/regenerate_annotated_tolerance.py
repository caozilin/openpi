"""Dataset orchestration over the existing shared native LIBERO replay worker.

No second controller, physics loop or optimization implementation is defined.
Original physical increments are replayed once with frozen per-frame labels.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2]
SHARED=ROOT.parent/'franka_mujoco'
ANNOTATION=SHARED/'analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107'
DEFAULT=ROOT/'data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1'
RECORD=ROOT/'experiment_records/2026-11-07_libero_annotated_tolerance_dataset'
os.environ['TASKTOL_FROZEN_SHARED_ROOT']=str(SHARED)
os.environ.setdefault('MUJOCO_GL','egl')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.path[:0]=[str(ROOT),str(SHARED),str(ROOT/'third_party/libero')]
from experiments.libero_tasktol_compat import rollout_mujoco_joint_servo as servo
from experiments.libero_tasktol_compat.annotated_tolerance_policy import label_policy
runner=servo.runner
base_freeze=servo.freeze

def read(path):return json.loads(Path(path).read_text())
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def freeze():
 return dict(base_freeze(),**{str(p):runner.sha(p) for p in [Path(__file__),Path(__file__).with_name('annotated_tolerance_policy.py')]})
runner.freeze_sources=freeze

def prepare(out):
 assert not out.exists(),out
 audit=read(ANNOTATION/'annotation_independent_final_audit.json');assert audit['complete'] and audit['episodes']==1875
 nominal=ROOT/'data/libero_tasktol_joint_servo/20261101_full_v1'
 parent=read(nominal/'plan.json');entries=[];excluded=[]
 for item in audit['results']:
  if item['status']!='labeled':excluded.append(item);continue
  folder=ANNOTATION/item['suite']/f"task_{item['task_id']:02d}"/item['demo']
  labels=read(folder/'tolerance_labels.json');assert runner.sha(folder/'tolerance_labels.json')==item['labels_sha256'];label_policy(labels)
  source=Path(labels['source']['folder']);report=read(source/'report.json');assert report['success']
  taskentry=next(t for t in parent['tasks'] if (t['task']['suite'],t['task']['task_id'])==(item['suite'],item['task_id']))
  candidate=next(j for j in taskentry['candidates'] if j['demo']==item['demo'])
  job=dict(candidate,selected_root=None,reference_path=str(source/'state_reference.hdf5'),expected_model_sha256=report['model_sha256'])
  key=f"{item['suite']}/{item['task_id']:02d}/{item['demo']}"
  entries.append(dict(key=key,case=taskentry['case'],task=taskentry['task'],job=job,
    labels=str(folder/'tolerance_labels.json'),labels_sha256=item['labels_sha256'],baseline_folder=str(source),
    baseline_report_sha256=runner.sha(source/'report.json'),reference_sha256=runner.sha(source/'state_reference.hdf5'),
    baseline_actions_sha256=report['actions_sha256'],initial_state_sha256=report['initial_state_sha256'],
    source_frames=len(labels['frames'])))
 assert len(entries)==1851 and len(excluded)==24
 out.mkdir(parents=True);sources=freeze();snapshot=out/'source_snapshot';snapshot.mkdir()
 sm=[]
 for i,(name,digest) in enumerate(sources.items()):
  target=snapshot/f'{i:03d}_{Path(name).name}';shutil.copy2(name,target);sm.append(dict(source=name,snapshot=str(target.relative_to(out)),sha256=digest))
 runner.dump(out/'source_snapshot_manifest.json',sm)
 # Static mask metadata is a standalone copy, so retiring the old media directory cannot remove a code dependency.
 legacy=ROOT/'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1/confirmed_masks.json'
 shutil.copy2(legacy,out/'confirmed_masks.json')
 plan=dict(schema='libero.annotated_tolerance_dataset.1',created_at=now(),annotation_run=str(ANNOTATION),
  audit_sha256=runner.sha(ANNOTATION/'annotation_independent_final_audit.json'),entries=entries,excluded=excluded,
  source_sha256=sources,workers=6,media=dict(format='mp4',codec='libx264',quality=7,cameras=['agentview','robot0_eye_in_hand'],resolution=[256,256],fps=20,rotation180_once=True),
  protocol='One native physical replay per labeled demo; exact original initial model/actions; frozen per-source-frame phases, subtask identity and six signed bounds; same current shared IPOPT20/EMA10 and native joint-reference servo; no VLA or action search; first solver rejection stops and is retained',
  source_protocol_scope='Offline annotation reset each independent block; this is one continuous physical episode with task objects. Label feasibility does not guarantee physical task success.',
  failure_policy='All outcomes retained; no outcome-dependent retries; success index requires full execution and final native goal',
  storage='Two ordinary MP4s, trajectory JSON, states NPZ, solver observations and report. No output image HDF5, no LeRobot conversion, no duplicate reference HDF5, no 500Hz servo samples.',
  replacement_targets=[str(ROOT/'data/libero_tasktol_joint_servo'/n) for n in ('20261031_mask30_confirmed_v1','20261101_release_recovery_long_failures_v1')])
 runner.dump(out/'plan.json',plan)
 for e in entries:
  runner.dump(out/'jobs'/e['key']/'joint_native/plan.json',dict(case=e['case'],task=e['task'],jobs=[e['job']],source_sha256=sources,
   phase_policy=dict(kind='frozen source-frame offline labels',labels=e['labels'],labels_sha256=e['labels_sha256'])))
 summarize(out,plan)
 return plan

def worker(out,key):
 import numpy as np,h5py
 p=read(out/'plan.json');assert freeze()==p['source_sha256']
 e=next(x for x in p['entries'] if x['key']==key);assert runner.sha(e['labels'])==e['labels_sha256']
 assert runner.sha(e['job']['reference_path'])==e['reference_sha256']
 labels=read(e['labels']);policy=label_policy(labels)
 with h5py.File(e['job']['reference_path']) as f:actions=f['actions'][()]
 assert len(actions)==e['source_frames'];assert hashlib.sha256(actions.tobytes()).hexdigest()==e['baseline_actions_sha256']
 started=time.monotonic()
 report=servo.worker(out/'jobs'/key,0,'joint_native',mode='mask30',phase_policy=policy,
   recording_options=dict(paired_video=True,save_servo_physics=False))
 folder=out/'jobs'/key/'joint_native/seed_00/mask30'
 assert report['actions_sha256']==e['baseline_actions_sha256'] and report['initial_state_sha256']==e['initial_state_sha256']
 rows=read(folder/'trajectory.json')
 for i,row in enumerate(rows):
  expected=policy(i,None,None)
  assert row['bounds_rad']==expected['bounds_rad'] and row['subtask_index']==expected['label_subtask_index']
  assert row['phase']==expected['phase'] and row['label_source_frame_index']==i
  np.testing.assert_array_equal(row['action'],actions[i])
 report.update(tolerance='offline_annotated_signed_bounds',label_source=e['labels'],label_source_sha256=e['labels_sha256'],
  media=p['media'],source_frame_policy='original offline source frame/phase/subtask; no live phase reassignment')
 report['tolerance_protocol'].update(bounds_deg=None,per_frame_bounds_rad='trajectory.json/bounds_rad',frame_schedule='frozen offline labels')
 runner.dump(folder/'report.json',report)
 destination=out/'episodes'/key;destination.parent.mkdir(parents=True,exist_ok=True);assert not destination.exists();shutil.move(folder,destination)
 files={f.name:runner.sha(f) for f in destination.iterdir() if f.is_file()}
 assert not list(destination.glob('*.hdf5')) and not (destination/'servo_physics.npz').exists()
 result=dict(key=key,case=e['case'],suite=e['task']['suite'],task_id=e['task']['task_id'],demo=e['job']['demo'],
  success=report['success'],execution_rejected=report['execution_rejected'],executed_frames=report['executed_frames'],
  planned_frames=report['planned_frames'],video_frames=len(rows),folder=str(destination),wall_s=time.monotonic()-started,
  labels_sha256=e['labels_sha256'],artifact_sha256=files)
 runner.dump(out/'records'/(key+'.json'),result);print(json.dumps(result,ensure_ascii=False),flush=True)

def summarize(out,plan):
 records=[read(p) for p in (out/'records').glob('*/*/*.json')];done=[r for r in records if not r.get('infrastructure_error')]
 summary=dict(complete=len(done)==len(plan['entries']),total=len(plan['entries']),completed=len(done),success=sum(r['success'] for r in done),
  failed=sum(not r['success'] for r in done),rejected=sum(r['execution_rejected'] for r in done),infrastructure_errors=len(records)-len(done),
  excluded_unlabeled=len(plan['excluded']),updated_at=now(),suites={s:dict(completed=sum(r['suite']==s for r in done),
  success=sum(r['suite']==s and r['success'] for r in done)) for s in ('libero_spatial','libero_object','libero_goal','libero_10')},results=records)
 runner.dump(out/'summary.json',summary)
 return summary

def run(out,keys=None):
 plan=read(out/'plan.json');assert freeze()==plan['source_sha256']
 selected=[e for e in plan['entries'] if keys is None or e['key'] in keys]
 if keys:assert len(selected)==len(keys)
 def one(e):
  record=out/'records'/(e['key']+'.json')
  if record.exists():
   assert not read(record).get('infrastructure_error'),'Inspect interrupted case before resuming; no silent retries'
   return
  process_record=out/'processes'/(e['key']+'.json');log=out/'logs'/(e['key']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  command=[sys.executable,'-u',str(Path(__file__)),'--output',str(out),'--worker',e['key']]
  with log.open('x') as stream:
   child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
   runner.dump(process_record,dict(pid=child.pid,command=command,started_at=now()))
   code=child.wait()
  runner.dump(process_record.with_name(process_record.stem+'_exit.json'),dict(pid=child.pid,exit_code=code,finished_at=now()))
  if code:runner.dump(record,dict(key=e['key'],infrastructure_error=True,exit_code=code,log=str(log)))
 with ThreadPoolExecutor(max_workers=6) as pool:
  for future in as_completed([pool.submit(one,e) for e in selected]):
   future.result();s=summarize(out,plan);print(json.dumps({k:s[k] for k in ('completed','success','failed','rejected','infrastructure_errors')}),flush=True)
 return summarize(out,plan)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);ap.add_argument('--prepare',action='store_true')
 ap.add_argument('--worker');ap.add_argument('--keys',nargs='+');args=ap.parse_args();out=args.output.resolve()
 if args.worker:worker(out,args.worker);return
 out.mkdir(parents=True,exist_ok=True) if not args.prepare else None
 if args.prepare:prepare(out);return
 lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 runner.dump(out/'process.json',dict(pid=os.getpid(),command=sys.argv,started_at=now()))
 s=run(out,args.keys)
 if s['infrastructure_errors']:raise SystemExit(1)
 if s['complete']:runner.dump(out/'generation_completion.json',dict(complete=True,episodes=1851,finished_at=now(),summary_sha256=runner.sha(out/'summary.json')))

if __name__=='__main__':main()
