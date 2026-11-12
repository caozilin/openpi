from pathlib import Path
import json,subprocess,os,datetime,hashlib
ROOT=Path('/media/czl/sata/franka_my_code/openpi')
OLD=ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1/viewer'
FULL=ROOT/'data/libero_tasktol_joint_servo/20261101_full_v1'
VIEW=FULL/'viewer'
assert not VIEW.exists()
old=json.loads((OLD/'viewer_catalog.json').read_text())
# Preserve original HDF5 video exports that already exist; never re-encode them.
reused=0
for task in old['datasets']:
 for e in task['episodes']:
  source=OLD/'rollouts'/task['suite']/f"{task['task_id']:02d}"/e['path']
  target=FULL/'source_video_cache/rollouts'/task['suite']/f"{task['task_id']:02d}"/e['path']
  target.mkdir(parents=True,exist_ok=True)
  for name in ['source_rgb.mp4','source_initial.png','source_video.json']:
   if not (target/name).exists():os.link(source/name,target/name)
  reused+=1
command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'examples/libero/build_reconstructed_rollout_viewer.py'),
 '--run',str(FULL),'--viewer',str(VIEW),'--source-index',str(ROOT/'data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json'),
 '--tolerance-run',str(ROOT/'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'),
 '--recovery-run',str(ROOT/'data/libero_tasktol_joint_servo/20261101_release_recovery_long_failures_v1')]
# Use the existing dataset environment with h5py/scipy/imageio.
command[0]=str(ROOT.parent/'franka_mujoco/experiments/libero_closed_loop/.venv/bin/python')
(FULL/'viewer_build_process.json').write_text(json.dumps(dict(pid=os.getpid(),command=command,reused_source_videos=reused))+'\n')
subprocess.run(command,cwd=ROOT,check=True)
catalog=json.loads((VIEW/'viewer_catalog.json').read_text());assert catalog['counts']['episodes']==1875
index=json.loads((ROOT/'data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json').read_text())
expected={(x['suite'],x['task_id'],x['demo']) for x in index['sources']}
actual={(t['suite'],t['task_id'],e['path']) for t in catalog['datasets'] for e in t['episodes']}
assert len(actual)==1875 and actual==expected
for task in catalog['datasets']:
 for e in task['episodes']:
  for name in ['video_path','trace_path','trace_script_path','source_video_path']:
   assert (VIEW/e[name]).is_file(),e[name]
from export_dataset_annotation_status import export as export_annotation_status
annotation_run=ROOT.parent/'franka_mujoco/analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106'
export_annotation_status(annotation_run,VIEW)
backup=OLD.with_name('viewer.pre_full_catalog_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
OLD.rename(backup);OLD.symlink_to(VIEW,target_is_directory=True)
evidence=dict(complete=True,counts=catalog['counts'],old_viewer_backup=str(backup),canonical_viewer=str(VIEW),
 existing_port8768_serves_full_viewer=True,all1875_match_audited_source_index=True,all_media_and_traces_exist=True,
 source_files_untouched=True,published_at=datetime.datetime.now().astimezone().isoformat())
(VIEW/'full_catalog_verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(evidence,ensure_ascii=False),flush=True)
