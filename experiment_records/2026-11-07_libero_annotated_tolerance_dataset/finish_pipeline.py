"""Supervise the authorized generation, audit, publication and bounded replacement."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
ROOT=Path('/media/czl/sata/franka_my_code/openpi');HERE=Path(__file__).resolve().parent
RUN=ROOT/'data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1'
NOMINAL=ROOT/'data/libero_tasktol_joint_servo/20261101_full_v1'
ANNOTATION=ROOT.parent/'franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107'
PYTHON=ROOT.parent/'franka_mujoco/experiments/libero_closed_loop/.venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def state(stage,**extra):write(RUN/'pipeline_state.json',dict(stage=stage,updated_at=now(),**extra))
def active(pid,needle):
 p=Path('/proc')/str(pid)/'cmdline'
 return p.exists() and needle in p.read_bytes().replace(b'\0',b' ').decode()
def command(name,args):
 cmd=[str(PYTHON),'-u',*map(str,args)]
 with (RUN/(name+'.log')).open('a') as f:c=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 write(RUN/(name+'_process.json'),dict(pid=c.pid,command=cmd,started_at=now()));code=c.wait()
 write(RUN/(name+'_exit.json'),dict(pid=c.pid,exit_code=code,finished_at=now()));assert code==0,(name,code)
def remove_legacy():
 assert read(RUN/'independent_final_audit.json')['complete']
 expected=[ROOT/'data/libero_tasktol_joint_servo'/n for n in ('20261031_mask30_confirmed_v1','20261101_release_recovery_long_failures_v1')]
 assert read(RUN/'plan.json')['replacement_targets']==list(map(str,expected))
 records=[];archive=HERE/'retired_legacy_metadata';archive.mkdir(exist_ok=True)
 for folder in expected:
  assert folder.parent==ROOT/'data/libero_tasktol_joint_servo' and not folder.is_symlink()
  if (folder/'RETIRED_DATASET.json').exists():continue
  saved=archive/folder.name;saved.mkdir(exist_ok=True)
  for name in ('README.md','summary.json','plan.json','confirmed_masks.json','source_snapshot_manifest.json'):
   if (folder/name).is_file():shutil.copy2(folder/name,saved/name)
  paths=[]
  for p in folder.rglob('*'):
   if p.is_file():paths.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
  write(saved/'deleted_file_manifest.json',paths)
  # Only these explicitly named old tolerance dataset roots are removed.
  shutil.rmtree(folder);folder.mkdir()
  if (saved/'confirmed_masks.json').exists():shutil.copy2(saved/'confirmed_masks.json',folder/'confirmed_masks.json')
  receipt=dict(retired=True,all_old_rollout_artifacts_deleted=True,replacement=str(RUN),metadata_archive=str(saved),deleted_files=len(paths),deleted_logical_bytes=sum(x['bytes'] for x in paths),finished_at=now(),retained='small confirmed_masks rule file and this deletion receipt only')
  write(folder/'RETIRED_DATASET.json',receipt);records.append(receipt)
 # Remove old tolerance/recovery copies from historical viewer exports only.
 viewers=set()
 for parent in (ROOT/'data/libero_tasktol_joint_servo').iterdir():
  if parent.is_dir():
   for v in parent.glob('viewer*'):
    if v.is_dir() and (v/'viewer_catalog.json').exists():viewers.add(v.resolve())
 removed_media=0;stripped_traces=0
 for viewer in sorted(viewers):
  catalog=read(viewer/'viewer_catalog.json')
  if catalog.get('tolerance_kind')=='offline_annotated_signed_bounds':continue
  leaves=[*viewer.glob('rollouts/*/*/*/tolerance'),*viewer.glob('rollouts/*/*/*/recovery')]
  for leaf in leaves:
   assert leaf.name in ('tolerance','recovery') and leaf.is_relative_to(viewer/'rollouts')
   if leaf.is_symlink():leaf.unlink()
   else:shutil.rmtree(leaf)
   removed_media+=1
  for trace in viewer.glob('rollouts/*/*/*/viewer_trace.json'):
   d=read(trace)
   if not d.get('tolerance') and not d.get('recovery'):continue
   d['tolerance']=None;d['recovery']=None
   for frame in d['trajectory']:frame.pop('tolerance',None);frame.pop('recovery',None)
   write(trace,d);script=trace.with_name('viewer_trace.js')
   script.write_text('window.LIBERO_EPISODES=window.LIBERO_EPISODES||{};window.LIBERO_EPISODES['+json.dumps(d['key'])+']='+json.dumps(d,ensure_ascii=False,separators=(',',':'))+';\n')
   stripped_traces+=1
  for task in catalog['datasets']:
   for e in task['episodes']:e['tolerance']=None;e['recovery']=None
  for key in list(catalog.get('counts',{})):
   if key.startswith(('tolerance_','recovery_')):catalog['counts'].pop(key)
  write(viewer/'viewer_catalog.json',catalog)
  (viewer/'assets/catalog.js').write_text('window.LIBERO_CATALOG='+json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+';\n')
 write(HERE/'legacy_deletion_verification.json',dict(complete=True,roots=records,legacy_viewer_media_directories_removed=removed_media,legacy_traces_stripped=stripped_traces,new_dataset_preserved=True,nominal_and_original_hdf5_preserved=True,checked_at=now()))

def publish():
 s=read(RUN/'summary.json');audit=read(RUN/'independent_final_audit.json');assert audit['episodes']==s['completed']==1851 and audit['complete'] and s['infrastructure_errors']==0
 records=s['results'];success=[r for r in records if r['success']]
 write(RUN/'successful_sources.json',dict(schema='libero.annotated_tolerance_success_sources.1',episodes=len(success),sources=success,excluded_unlabeled=24,failed_replays=s['failed'],native_final_success_required=True))
 state('publishing_viewer',completed=1851,success=s['success'])
 cmd=[ROOT/'examples/libero/build_reconstructed_rollout_viewer.py','--run',NOMINAL,'--viewer',NOMINAL/'viewer',
   '--source-index',ROOT/'data/lerobot/caozilin/libero_reconstructed_full_20261101/meta/reconstruction_sources.json','--tolerance-run',RUN]
 command('viewer_build',cmd)
 catalog=read(NOMINAL/'viewer/viewer_catalog.json')
 assert catalog['counts']['episodes']==1875 and catalog['counts']['tolerance_completed']==1851
 assert catalog['counts']['tolerance_success']==s['success'] and catalog['tolerance_kind']=='offline_annotated_signed_bounds'
 sys.path.insert(0,str(ROOT/'examples/libero'))
 from export_dataset_annotation_status import export
 export(ANNOTATION,NOMINAL/'viewer')
 write(RUN/'viewer_publication_verification.json',dict(complete=True,catalog=str(NOMINAL/'viewer/viewer_catalog.json'),counts=catalog['counts'],annotation_status_reexported=True,checked_at=now()))
 state('removing_replaced_legacy_data')
 remove_legacy()
 completion=dict(complete=True,episodes=1851,success=s['success'],failed=s['failed'],rejected=s['rejected'],excluded_unlabeled=24,
  independent_audit_complete=True,legacy_data_deleted=True,viewer_updated=True,finished_at=now(),no_lerobot_conversion=True)
 write(RUN/'task_completion.json',completion);state('complete',**completion)
 names={'libero_spatial':'Spatial','libero_object':'Object','libero_goal':'Goal','libero_10':'Long'}
 table='|套件|已回放|最终原生任务成功|失败|\n|---|---:|---:|---:|\n'
 for suite,name in names.items():
  x=s['suites'][suite];table+=f"|{name}|{x['completed']}|{x['success']}|{x['completed']-x['success']}|\n"
 text=f'''# 最新标注容差：原始LIBERO物理回放数据集

状态：已生成1851条，全部通过独立记录/标签/状态/双MP4及原生谓词审计；旧容差回放已删除并完成viewer替换。

1851次固定来源回放，{s['success']}条完整执行且最终原生任务成功，{s['failed']}条失败（含{s['rejected']}条求解拒绝）；24条无有效标注来源未回放。所有成败结果保留，没有按失败重试。成功索引仅收录满足完整执行+最终原生谓词的样本，不把1851条处理完成冒称全成功。

{table}
沿用同原始重构初态/物理模型/7维物理动作，共享当前IPOPT20/EMA10与原生20Hz/500Hz关节伺服。逐源帧使用已冻结离线标签的阶段、子任务和六方向边界；不是在线语义识别。源离线候选按独立块初始化，本次为连续含任务物体的物理回放，离线100%验收不代表物理100%成功。首个求解拒绝停止协议沿用原数据回放，不套用benchmark的20失败帧缓冲；不因结果改控制器、预算或标签。

双相机256×256、20Hz、旋转180°一次，普通H.264 MP4；同时保留trajectory.json、states.npz、solver_observations.json及report.json。未输出图像HDF5、LeRobot或500Hz伺服采样，未重复复制reference HDF5。未来格式转换应从这些原始记录及其来源引用读取，不将有损MP4冒充无损PNG。

首版只读pilot审计访问了错误的环境包装层属性，原错误日志保留在run/attempts/0001_audit_wrapper_attribute；仅修复审计读取路径后重新检查已有六条，未重跑回合。pilot5成功/1物理失败/基础设施错误0。

旧400固定30°容差及52条恢复复测的原始回放文件已按用户明确授权删除，小型原summary/plan/规则/删除清单保留于本记录retired_legacy_metadata。当前及历史viewer里的旧容差/恢复媒体和嵌入轨迹已清除；名义重构/原始HDF5/离线搜索与标签/benchmark评测不删除。旧目录只留小型规则文件/退役凭据，避免规则导入工具依赖失效。

## 本地源数据地址

- 新原始数据集：`{RUN}`；逐条`episodes/<suite>/<task_id>/demo_N/{{main_rgb.mp4,wrist_rgb.mp4,trajectory.json,states.npz,solver_observations.json,report.json}}`。
- 固定请求及所有来源：`{RUN}/plan.json`；源码：`{RUN}/source_snapshot`和source_snapshot_manifest.json；全量结果：`{RUN}/summary.json`；成功数据索引：`{RUN}/successful_sources.json`。
- 全量独立审计：`{RUN}/independent_final_audit.json`、independent_final_audit_state.json、auditing.log；完成凭据：`{RUN}/task_completion.json`。
- 母集标签：`{ANNOTATION}`及annotation_independent_final_audit.json；各标签SHA在新plan.json绑定。
- 生成/审计入口：`{ROOT}/experiments/libero_tasktol_compat/regenerate_annotated_tolerance.py`、audit_annotated_tolerance_dataset.py；它们调用既有共享执行链，没有复制控制或物理循环。
- 监督/发布：`{HERE}/finish_pipeline.py`；状态日志：`{RUN}/pipeline_state.json`、pipeline.log、generation.log及logs/逐条日志；真实进程：process.json、processes/、pipeline_process.json。
- 替换删除凭据：`{HERE}/legacy_deletion_verification.json`；旧小型元数据：`{HERE}/retired_legacy_metadata`；viewer：`{NOMINAL}/viewer`，发布证据：`{RUN}/viewer_publication_verification.json`。
- 不使用服务器、不训练、不格式转换、不上传。用户viewer服务不终止。
'''
 (HERE/'README.md').write_text(text);(HERE/'RESULTS.md').write_text(text)
 index=HERE.parent/'README.md';v=index.read_text().splitlines()
 for i,line in enumerate(v):
  if '2026-11-07_libero_annotated_tolerance_dataset/README.md' in line:v[i]=f"| 2026-11-07 | [标注容差重建原始LIBERO数据集](2026-11-07_libero_annotated_tolerance_dataset/README.md) | 全1851条生成与独立审计完成，旧组已删除替换 | {s['success']}/1851物理任务成功，24无标签排除；双MP4原始记录、无LeRobot转换、viewer已更新 |"
 index.write_text('\n'.join(v)+'\n')
 write(HERE/'sources.json',dict(run=str(RUN),plan=str(RUN/'plan.json'),summary=str(RUN/'summary.json'),success_sources=str(RUN/'successful_sources.json'),audit=str(RUN/'independent_final_audit.json'),completion=str(RUN/'task_completion.json'),deletion=str(HERE/'legacy_deletion_verification.json')))

def main():
 lock=(RUN/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(RUN/'pipeline_process.json',dict(pid=os.getpid(),command=str(Path(__file__)),started_at=now()))
 try:
  launch=read(RUN/'generation_launch.json');state('generating',generation_pid=launch['pid'])
  while active(launch['pid'],'regenerate_annotated_tolerance.py'):time.sleep(10)
  assert read(RUN/'generation_completion.json')['complete']
  with (RUN/'execution.lock').open('a') as l:fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert read(RUN/'summary.json')['infrastructure_errors']==0
  state('auditing')
  command('auditing',[ROOT/'experiments/libero_tasktol_compat/audit_annotated_tolerance_dataset.py','--output',RUN])
  publish()
 except BaseException:
  state('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
