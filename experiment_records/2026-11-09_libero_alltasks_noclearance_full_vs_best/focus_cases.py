"""Curate already verified failure and loss cases without changing experiments."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

H=Path(__file__).resolve().parent
rows=json.loads((H/'episodes.json').read_text())
fail=json.loads((H/'full_failure_diagnosis.json').read_text())
paired=[r for r in rows if r['full_complete']]
gain=lambda r,m:100*(r['best_metrics'][m]/r['full_metrics'][m]-1)*(-1 if m in ['velocity','acceleration','jerk','travel'] else 1)
speed=sorted(paired,key=lambda r:-gain(r,'velocity'))
sigma=sorted(paired,key=lambda r:-(r['best_metrics']['sigma_min']/r['full_metrics']['sigma_min']))
my=sorted(paired,key=lambda r:-gain(r,'mY'))
groups=defaultdict(list)
for f in fail['cases']:groups['/'.join(f['key'].split('/')[:2])].append(f)
for group in groups.values():group.sort(key=lambda r:int(r['key'].split('_')[-1]))
def pack(r):
    return dict(key=r['key'],task_name=r['task_name'],stratum=r['stratum'],
        full=r['full_metrics'],best=r['best_metrics'],
        gains_pct={m:gain(r,m) for m in ['velocity','acceleration','mY','sigma_min']},
        block_evidence=[dict(index=b['index'],full_trial=b['full']['path'],
                           full_sha=b['full']['sha256'],best_trial=b['best']['path'],best_sha=b['best']['sha256'])
                        for b in r['blocks']])
doc=dict(failure_groups=groups,velocity_gain_gt10pct=[pack(r) for r in speed if gain(r,'velocity')>10],
         sigma_relative_gain_top5=[pack(r) for r in sigma[:5]],mY_relative_gain_top5=[pack(r) for r in my[:5]],
         definitions='Same verified no-clearance comparison; gains use full as denominator; sample IDs are original demos, not rollout seeds.',
         input_sha256={f:hashlib.sha256((H/f).read_bytes()).hexdigest() for f in ['episodes.json','full_failure_diagnosis.json','verification.json']})
assert sum(len(g) for g in groups.values())==38 and len(doc['velocity_gain_gt10pct'])==14
(H/'focus_cases.json').write_text(json.dumps(doc,indent=2)+'\n')
lines=['# 优先检查的全开失败与运动学损失样本','',
 '状态：2026-11-09，对已完成独立复核的全40任务对照做重点清单整理；无新增搜索或执行。整体失败率38/1875=2.03%，但L08任务失败24/45=53.33%，不能用整体低失败率替代该任务结论。以下编号为原始demo_N，不是闭环评测seed。两组均不筛桌面净空，完整求解及自碰撞条件保留。','',
 '## 全部38条失败，按任务集中程度排列','',
 '|任务|demo编号|失败类别|','|---|---|---|']
for task,group in sorted(groups.items(),key=lambda item:-len(item[1])):
    ids=', '.join(str(int(r['key'].split('_')[-1])) for r in group)
    reasons=', '.join(sorted({r['reason'] for r in group}))
    lines.append(f'|{task}|{ids}|{reasons}|')
lines+=['',
 'Long05/demo15在原块内frame229发生link1/link5碰撞，几何穿透0.572mm；demo43在frame239发生link2/link5碰撞，穿透5.473mm。该位置是原离线候选20Hz帧，非新物理回放证据。其余36条均达到IPOPT20次迭代上限；优先抽查L08/demo0（EXEC，frame249）、demo2（EXEC，frame157）、demo4（DIS，frame192），完整逐条阶段/帧/残差见full_failure_diagnosis.json。','',
 '## 速度代价最大的样本','',
 '筛选搜索相对全开降速>10%的全部14条。降幅=(全开−搜索)/全开，不能把该降幅直接称作以搜索为分母的“全开增加百分比”。正值为改善；速度/加速度越低越好。','',
 '|任务/demo|全开→搜索平均速度rad/s|搜索降速|搜索降低加速度|归一化σmin：全开→搜索|',
 '|---|---:|---:|---:|---:|']
for r in speed:
    if gain(r,'velocity')<=10:continue
    f,b=r['full_metrics'],r['best_metrics']
    lines.append(f"|{r['key']}|{f['velocity']:.6f} → {b['velocity']:.6f}|{gain(r,'velocity'):.2f}%|{gain(r,'acceleration'):.2f}%|{f['sigma_min']:.6f} → {b['sigma_min']:.6f}|")
lines+=['',
 '重点首看L04/demo14（双杯左右分放，速度改善24.96%，加速度改善22.01%）和L05/demo23（书入收纳架，速度改善18.92%，加速度改善55.89%）。','',
 '## 速度收益未覆盖的其他明显代价','',
 '|指标与样本|全开→搜索|同时的速度降幅|',
 '|---|---:|---:|']
for r in sigma[:3]:
    lines.append(f"|归一化σmin，{r['key']}|{r['full_metrics']['sigma_min']:.6f} → {r['best_metrics']['sigma_min']:.6f}|{gain(r,'velocity'):.2f}%|")
for r in my[:3]:
    lines.append(f"|mY，{r['key']}|{r['full_metrics']['mY']:.6f} → {r['best_metrics']['mY']:.6f}（+{gain(r,'mY'):.2f}%）|{gain(r,'velocity'):.2f}%|")
lines+=['',
 'σ比值存在小分母放大，故这里同时列绝对值。L08/demo30的原始未归一化最小σ也从0.000769提高到0.052323；不是仅由归一化表达导致的巨大相对变化。O00/demo47速度只改善0.26%，mY却从0.050321到0.075474，说明速度收益小不能替代所有指标代价小。','',
 '## 精确来源与验证','',
 '- 主记录：`'+str(H/'README.md')+'`；40任务表：`'+str(H/'PER_TASK.md')+'`。',
 '- 本清单及每条full/best原trial绝对路径/SHA：`'+str(H/'focus_cases.json')+'`；入口：`'+str(H/'focus_cases.py')+'`。',
 '- 原始候选、请求、日志、视频来源、两个数据协议分层和完整复核均沿用主记录；全38失败证据：`'+str(H/'full_failure_diagnosis.json')+'`。输入摘要SHA在focus_cases.json绑定；原验证10416份来源SHA、4358份trial、2301块及40任务统计已通过。没有将截断失败轨迹纳入运动学损失比较，也没有改写正式标签。','']
(H/'FOCUS_CASES.md').write_text('\n'.join(lines)+'\n')
print('Curated 38 failures, 14 velocity cases >10%, and sigma/mY outliers.')
