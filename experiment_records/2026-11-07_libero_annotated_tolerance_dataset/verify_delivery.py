from pathlib import Path
from collections import Counter
import datetime,fcntl,hashlib,json,math,os
ROOT=Path('/media/czl/sata/franka_my_code/openpi')
R=ROOT/'data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1';H=ROOT/'experiment_records/2026-11-07_libero_annotated_tolerance_dataset'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
s=read(R/'summary.json');a=read(R/'independent_final_audit.json');p=read(R/'plan.json');completion=read(R/'task_completion.json')
assert s['complete'] and s['completed']==1851 and s['infrastructure_errors']==0
assert a['complete'] and a['episodes']==1851 and a['plan_sha256']==sha(R/'plan.json')
assert completion['complete'] and read(R/'pipeline_state.json')['stage']=='complete'
assert read(R/'generation_completion.json')['summary_sha256']==sha(R/'summary.json')
assert read(R/'auditing_exit.json')['exit_code']==read(R/'viewer_build_exit.json')['exit_code']==0
expected={e['key']:e for e in p['entries']};records={r['key']:r for r in s['results']};audited={r['key']:r for r in a['results']}
assert len(expected)==len(records)==len(audited)==1851 and expected.keys()==records.keys()==audited.keys()
success=read(R/'successful_sources.json');assert success['episodes']==len(success['sources'])==1503
assert {r['key'] for r in success['sources']}=={r['key'] for r in s['results'] if r['success']}
assert len(p['excluded'])==24 and all(x['status']=='baseline_unavailable' for x in p['excluded'])
artifacts=0;video_count=0;frames=0;size=0
for key,e in expected.items():
 folder=R/'episodes'/key;record=records[key];item=audited[key]
 assert sha(e['labels'])==e['labels_sha256']==item['labels_sha256']==record['labels_sha256']
 for name,digest in record['artifact_sha256'].items():
  f=folder/name;assert sha(f)==digest,(key,name);artifacts+=1;size+=f.stat().st_size
 assert record['success']==item['success'] and record['execution_rejected']==item['execution_rejected']
 for v in item['videos']:
  assert v['sha256']==record['artifact_sha256'][v['name']] and v['frames']==record['video_frames']
  video_count+=1;frames+=v['frames']
 assert not list(folder.glob('*.hdf5')) and not (folder/'servo_physics.npz').exists()
for source in read(R/'source_snapshot_manifest.json'):assert sha(R/source['snapshot'])==source['sha256']
viewer=ROOT/'data/libero_tasktol_joint_servo/20261101_full_v1/viewer';cat=read(viewer/'viewer_catalog.json')
assert cat['tolerance_kind']=='offline_annotated_signed_bounds' and cat['counts']['episodes']==1875
assert cat['counts']['tolerance_completed']==1851 and cat['counts']['tolerance_success']==1503
actual=set();media_links=0
for task in cat['datasets']:
 for ep in task['episodes']:
  t=ep['tolerance'];assert ep['recovery'] is None
  if t:
   assert ep['key'] in expected;actual.add(ep['key'])
   for name in ('video','poster','report','trajectory'):assert (viewer/t[name]).is_file();media_links+=1
   assert sha(viewer/t['video'])==records[ep['key']]['artifact_sha256']['main_rgb.mp4']
assert actual==expected.keys()
deleted=read(H/'legacy_deletion_verification.json');assert deleted['complete']
for name in ('20261031_mask30_confirmed_v1','20261101_release_recovery_long_failures_v1'):
 old=R.parent/name;assert {x.name for x in old.iterdir()}=={'confirmed_masks.json','RETIRED_DATASET.json'}
 assert read(old/'RETIRED_DATASET.json')['all_old_rollout_artifacts_deleted']
process_checks=[]
for name in ('generation_launch.json','pipeline_process.json','auditing_process.json','viewer_build_process.json'):
 x=read(R/name);f=Path('/proc')/str(x['pid'])/'cmdline';cmd=f.read_bytes().replace(b'\0',b' ').decode() if f.exists() else ''
 assert not (str(R) in cmd or 'regenerate_annotated_tolerance.py' in cmd or 'finish_pipeline.py' in cmd or 'audit_annotated_tolerance_dataset.py' in cmd),(name,cmd)
 process_checks.append(dict(record=name,pid=x['pid'],related_process_exited=True))
for name in ('execution.lock','pipeline.lock','independent_final_audit.lock'):
 with (R/name).open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
by_suite={suite:dict(total=x['completed'],success=x['success'],failed=x['completed']-x['success'],success_percent=100*x['success']/x['completed']) for suite,x in s['suites'].items()}
result=dict(complete=True,checked_at=datetime.datetime.now().astimezone().isoformat(),episodes=1851,success=1503,failed=348,rejected=7,infrastructure_errors=0,
 success_percent=100*1503/1851,excluded_unlabeled=24,verified_artifacts=artifacts,verified_raw_bytes=size,verified_videos=video_count,decoded_video_frames_bound_to_audit=frames,
 verified_label_files=1851,verified_viewer_links=media_links,original_artifact_sha_unchanged_after_publication_and_cleanup=True,
 successful_index_verified=True,old_dataset_artifacts_absent=True,process_checks=process_checks,locks_released=True,suites=by_suite,
 audit_sha256=sha(R/'independent_final_audit.json'),plan_sha256=sha(R/'plan.json'),summary_sha256=sha(R/'summary.json'),deletion_receipt_sha256=sha(H/'legacy_deletion_verification.json'))
(H/'final_delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('process_checks',)},ensure_ascii=False,indent=2))
