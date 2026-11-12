"""Compare saved original400 mask30 outcomes against the new annotated group."""
from pathlib import Path
from collections import Counter
import hashlib,json,datetime
H=Path(__file__).resolve().parent
R=H.parents[1]/'data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1'
OLD=H/'retired_legacy_metadata/20261031_mask30_confirmed_v1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a,b=read(OLD/'summary.json'),read(R/'summary.json');pa,pb=read(OLD/'plan.json'),read(R/'plan.json')
ledger={x['path']:x['sha256'] for x in read(OLD/'deleted_file_manifest.json')}
original=R.parent/'20261031_mask30_confirmed_v1'
for name in ('summary.json','plan.json'):assert sha(OLD/name)==ledger[str(original/name)]
assert sha(R/'summary.json')==read(R/'generation_completion.json')['summary_sha256']
assert read(R/'independent_final_audit.json')['complete']
o={r['key']:r for r in a['results']};n={r['key']:r for r in b['results']};common=sorted(o.keys()&n.keys());om={e['key']:e for e in pa['entries']};nm={e['key']:e for e in pb['entries']}
rows=[]
for key in common:
 for f in ('reference_sha256','baseline_actions_sha256','baseline_report_sha256'):assert om[key][f]==nm[key][f]
 assert om[key]['job']['expected_model_sha256']==nm[key]['job']['expected_model_sha256']
 rows.append(dict(key=key,suite=n[key]['suite'],task_id=n[key]['task_id'],demo=n[key]['demo'],old_success=o[key]['success'],new_success=n[key]['success'],old_solver_rejected=o[key]['execution_rejected'],new_solver_rejected=n[key]['execution_rejected'],reference_sha256=nm[key]['reference_sha256'],actions_sha256=nm[key]['baseline_actions_sha256'],baseline_report_sha256=nm[key]['baseline_report_sha256'],model_sha256=nm[key]['job']['expected_model_sha256']))
assert len(rows)==397 and len(o)==400 and len(n)==1851
excluded=sorted(o.keys()-n.keys());unlabeled={f"{x['suite']}/{x['task_id']:02d}/{x['demo']}" for x in pb['excluded']};assert set(excluded)<=unlabeled

def summarize(rs):
 old=sum(r['old_success'] for r in rs);new=sum(r['new_success'] for r in rs);count=len(rs)
 return dict(paired_episodes=count,old_success=old,new_success=new,old_rate_percent=100*old/count,new_rate_percent=100*new/count,improvement_percentage_points=100*(new-old)/count,
  both_success=sum(r['old_success'] and r['new_success']for r in rs),both_failure=sum(not r['old_success'] and not r['new_success']for r in rs),
  recovered=sum(not r['old_success'] and r['new_success']for r in rs),regressed=sum(r['old_success'] and not r['new_success']for r in rs))
changed=[k for k,v in pa['source_sha256'].items() if k in pb['source_sha256'] and pb['source_sha256'][k]!=v]
summary=dict(complete=True,checked_at=datetime.datetime.now().astimezone().isoformat(),comparison='Original400 confirmed-mask +/-30deg physical replay vs latest annotated physical replay; no52 recovery outcomes mixed in',
 old_full=dict(episodes=400,success=286,rate_percent=71.5),new_full=dict(episodes=1851,success=1503,rate_percent=100*1503/1851),
 paired=summarize(rows),suites={s:summarize([r for r in rows if r['suite']==s])for s in ('libero_spatial','libero_object','libero_goal','libero_10')},
 excluded_old_keys=excluded,exclusion_reason='No valid current zero-baseline/label; excluded symmetrically, all3 previously successful',
 source_checks='Reference HDF5 SHA/actions SHA/nominal report SHA/physical model SHA match for all397 paired demos',
 old_summary_and_plan_verified_against_predeletion_ledger=True,
 protocol_differences=['Old current-finger online phase classification plus frozen baseline semantic schedule; new per-source-frame offline annotation phase/subtask/bounds',
 'Old enabled authored axes uniform +/-30deg; new per-demo six-direction search bounds under authored5/15/30deg caps',
 'New shared implementation includes cache85 EMA nominal-advance-then-same-frame sampling; old frozen shared implementation differs'],
 causal_scope='Same-source paired outcome improvement of the complete updated method; not a single-variable Bayesian-search ablation',
 changed_shared_or_runner_source_files=changed,old_summary_sha256=sha(OLD/'summary.json'),old_plan_sha256=sha(OLD/'plan.json'),
 new_summary_sha256=sha(R/'summary.json'),new_plan_sha256=sha(R/'plan.json'),rows=rows)
(H/'legacy30_vs_annotated_paired.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
names={'libero_spatial':'Spatial','libero_object':'Object','libero_goal':'Goal','libero_10':'Long'}
table='|套件|共同demo数|旧30°成功|新标注成功|提升（百分点）|\n|---|---:|---:|---:|---:|\n'
for s,label in names.items():
 t=summary['suites'][s];table+=f"|{label}|{t['paired_episodes']}|{t['old_success']}/{t['paired_episodes']}（{t['old_rate_percent']:.2f}%）|{t['new_success']}/{t['paired_episodes']}（{t['new_rate_percent']:.2f}%）|{t['improvement_percentage_points']:+.2f}|\n"
t=summary['paired'];table+=f"|合计|397|283/397（{t['old_rate_percent']:.2f}%）|325/397（{t['new_rate_percent']:.2f}%）|{t['improvement_percentage_points']:+.2f}|\n"
text=f'''# 旧30°与最新标注容差：同源配对比较

状态：既有结果只读配对统计完成，无新增回放。旧组是最初400条对应语义mask启用轴±30°回放，不是全任务统一111，也不混入52条失败定向姿态恢复复测。

整体旧286/400=71.50%，新1503/1851=81.20%，增加9.70个百分点；总体来源数量不同，因此主比较取共同397条。每条reference HDF5、原动作、名义report与物理model SHA全部一致。

{table}
共同397条：274条两组都成功、63条两组都失败、51条旧失败→新成功、9条旧成功→新失败，净增42成功。Spatial贡献25、Goal5、Long12，Object净成功数不变。未配对的旧3条（{', '.join(excluded)}）在旧组均成功，当前无有效标签，双方配对统计均排除。

这说明提升不只是总体样本范围变化。它衡量的是新版整套方法；旧组使用旧冻结共享实现、在线手指阶段判定和固定语义身份计划、启用轴统一±30°，新组使用新版EMA同轴采样、逐源帧标注阶段/子任务和搜索六方向边界（手工5/15/30°上限）。不能把全部提升单独归因于贝叶斯搜索。未为此启动同实现新对照。

## 本地源数据

- 旧保留摘要/plan：`{OLD}/summary.json`、`{OLD}/plan.json`；已与同目录deleted_file_manifest.json中删除前SHA核对。旧回放已按用户要求删除，本次不声称重新读取或解码旧视频。
- 新原始结果：`{R}`；summary.json、plan.json及independent_final_audit.json。
- 397条配对结果/原动作与参考SHA/套件统计/协议差异：`{H}/legacy30_vs_annotated_paired.json`；只读脚本：`{H}/compare_legacy30.py`。
'''
(H/'LEGACY30_COMPARISON.md').write_text(text)
for name in ('README.md','RESULTS.md'):
 f=H/name;v=f.read_text()
 if 'LEGACY30_COMPARISON.md' not in v:f.write_text(v+'\n## 与旧30°组的同源比较\n\n共同397条：旧283/397（71.28%）→新325/397（81.86%），提升10.58个百分点；51条恢复、9条退化。协议同时有EMA/阶段时序/边界变化，属于整套方案比较。[完整配对报告](LEGACY30_COMPARISON.md)。\n')
index=H.parent/'README.md';v=index.read_text();line=next(x for x in v.splitlines()if '2026-11-07_libero_annotated_tolerance_dataset/README.md' in x)
if '同源397' not in line:index.write_text(v.replace(line,line[:-1].rstrip()+'；同源397较旧30°提高10.58个百分点 |'))
print(json.dumps({k:v for k,v in summary.items()if k in ('paired','suites','excluded_old_keys')},ensure_ascii=False,indent=2))
