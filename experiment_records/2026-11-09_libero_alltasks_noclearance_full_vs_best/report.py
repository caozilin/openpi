"""Render the complete 40-task analysis from verified paired statistics."""
from collections import Counter
from pathlib import Path
import csv
import json
import math
import os

os.environ.setdefault('MPLCONFIGDIR','/tmp/libero_noclearance_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

H=Path(__file__).resolve().parent
s=json.loads((H/'summary.json').read_text());rows=json.loads((H/'episodes.json').read_text())
verification=json.loads((H/'verification.json').read_text());assert verification['complete']
paired=[r for r in rows if r['full_complete']];N=len(paired)
names={
 'libero_spatial':['盘子与小碗之间取碗','小碗旁取碗','桌中心取碗','饼干盒上取碗','顶抽屉中取碗','小碗上取碗','饼干盒旁取碗','炉子上取碗','盘子旁取碗','木柜上取碗'],
 'libero_object':['字母汤罐进篮','奶油奶酪进篮','沙拉酱进篮','烧烤酱进篮','ketchup进篮','tomato sauce进篮','黄油进篮','牛奶进篮','巧克力布丁进篮','橙汁进篮'],
 'libero_goal':['打开中间抽屉','碗放炉上','酒瓶放柜顶','开顶抽屉并放碗','碗放柜顶','盘子推到炉前','奶酪放碗中','打开炉子','碗放盘子','酒瓶放架上'],
 'libero_10':['汤罐和番茄沙司进篮','奶酪和黄油进篮','开炉并放摩卡壶','碗放底抽屉并关闭','白杯左盘、黄白杯右盘','书放收纳架后格','白杯放盘、布丁放右侧','汤罐和奶酪进篮','双摩卡壶上炉','黄白杯进微波炉并关门']}
prefix={'libero_spatial':'S','libero_object':'O','libero_goal':'G','libero_10':'L'}
suite_name={'libero_spatial':'Spatial','libero_object':'Object','libero_goal':'Goal','libero_10':'Long'}
colors={'libero_spatial':'#3a78ae','libero_object':'#c77c33','libero_goal':'#3a9473','libero_10':'#9b69b0'}
order={k:i for i,k in enumerate(['libero_spatial','libero_object','libero_goal','libero_10'])}
tasks=sorted(s['tasks'],key=lambda t:(order[t['suite']],t['task_id']))
short=lambda t:names[t['suite']][t['task_id']]
code=lambda t:f"{prefix[t['suite']]}{t['task_id']:02d}"
gain=lambda r:100*(1-r['best_metrics']['velocity']/r['full_metrics']['velocity'])
fmt=lambda v:f'{v:.3f}'
pct=lambda n,total=N:f'{100*n/total:.2f}%'

# Extra tail diagnostics retain absolute baseline values to expose ratio effects.
extra={}
for metric in ['sigma_min','sigma_min_full']:
    ranked=sorted(paired,key=lambda r:-(r['best_metrics'][metric]/r['full_metrics'][metric]-1))
    filtered=[r for r in paired if r['full_metrics'][metric]>=.05]
    extra[metric]=dict(top5=[dict(key=r['key'],full=r['full_metrics'][metric],best=r['best_metrics'][metric],
       gain_pct=100*(r['best_metrics'][metric]/r['full_metrics'][metric]-1)) for r in ranked[:5]],
       full_baseline_lt_0_05_count=N-len(filtered),
       ratio_gain_excluding_baseline_lt_0_05_pct=100*(sum(r['best_metrics'][metric] for r in filtered)/sum(r['full_metrics'][metric] for r in filtered)-1))
extra['approx_four_metric_nonregression_1pct_count']=sum(all(
    s['metrics'][m][2]*100*(r['best_metrics'][m]-r['full_metrics'][m])/r['full_metrics'][m]>=-1
    for m in ['velocity','acceleration','mY','sigma_min']) for r in paired)
(H/'tail_diagnostics.json').write_text(json.dumps(extra,indent=2)+'\n')

table=['|任务|简述|配对/来源|全开失败|速度改善均值比|逐条中位数|收益≤1%|加速度改善|mY改善|归一化σmin改善|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
cards=[]
csvrows=[]
for t in tasks:
    v=t['metrics']['velocity'];a=t['metrics']['acceleration'];m=t['metrics']['mY'];sig=t['metrics']['sigma_min']
    n=t['paired_episodes'];q=v['paired_gain_pct'];i=t['initial_vs_adaptive']
    table.append(f"|{code(t)}|{short(t)}|{n}/{t['total_episodes']}|{t['full_incomplete_episodes']}|{v['improvement_pct_of_means']:.3f}%|{q['median']:.3f}%|{v['cumulative_gain_le_pct']['1']['count']}/{n} ({pct(v['cumulative_gain_le_pct']['1']['count'],n)})|{a['improvement_pct_of_means']:+.3f}%|{m['improvement_pct_of_means']:+.3f}%|{sig['improvement_pct_of_means']:+.3f}%|")
    note=[]
    if t['strict_zero_episodes']==t['total_episodes']:
        note.append('该任务语义容差全为0；两组为同一候选，无搜索自由度，收益为0。')
    else:
        if v['improvement_pct_of_means']<=.1:note.append('相对全开的速度收益几乎为零（均值比≤0.1%）。')
        elif v['improvement_pct_of_means']<=1:note.append('速度收益较小（均值比≤1%）。')
        else:note.append('速度收益高于总体均值。')
        if q['median']<1 and v['improvement_pct_of_means']>2:
            note.append('均值主要被少数高收益demo拉高，中位数仍低于1%。')
        if t['full_incomplete_episodes']:
            note.append(f"全开有{t['full_incomplete_episodes']}/{t['total_episodes']}条不完整，不能把当前配对收益推广到这些失败条目。")
        if a['improvement_pct_of_means']<0:note.append(f"速度下降同时平均加速度上升{-a['improvement_pct_of_means']:.3f}%。")
        if sig['improvement_pct_of_means']<0:note.append(f"归一化最小奇异值下降{-sig['improvement_pct_of_means']:.3f}%。")
        if m['improvement_pct_of_means']>10:note.append(f"mY均值提高{m['improvement_pct_of_means']:.2f}%，比速度收益更明显。")
        if i['adaptive_reduction_vs_initial_pct']>.3:
            note.append('该任务追加查询的速度收益高于总体，但仍需看逐demo分布。')
    cards += [f"## {code(t)}：{short(t)}",'',f"原任务名：`{t['task_name']}`。来源{t['total_episodes']}条，配对{n}条，全开失败{t['full_incomplete_episodes']}条；来源分层：{t['stratum_counts']}。",'',
      f"速度均值比改善{v['improvement_pct_of_means']:.3f}%；逐demo改善中位数{q['median']:.3f}%、P90={q['p90']:.3f}%、最大{q['max']:.3f}%。≤0.1%为{v['cumulative_gain_le_pct']['0.1']['count']}/{n}，≤1%为{v['cumulative_gain_le_pct']['1']['count']}/{n}，>5%为{n-v['cumulative_gain_le_pct']['5']['count']}/{n}。",'',
      f"加速度均值改善{a['improvement_pct_of_means']:+.3f}%，其中{a['regressed_gt1pct']}/{n}条恶化>1%；mY改善{m['improvement_pct_of_means']:+.3f}%；归一化σmin改善{sig['improvement_pct_of_means']:+.3f}%，其中{sig['regressed_gt1pct']}/{n}条恶化>1%。",'',
      f"结构化初始化最优相对全开降速{i['initial_reduction_vs_full_pct']:.3f}%；追加查询相对初始化最优再降{i['adaptive_reduction_vs_initial_pct']:.3f}%。该任务贡献全部绝对速度节省的{t['share_of_total_velocity_saving']*100:.2f}%；剔除本任务后总体速度改善{t['overall_velocity_gain_if_task_removed_pct']:.3f}%。",'',
      ' '.join(note),'']
    record=dict(code=code(t),suite=t['suite'],task_id=t['task_id'],task_name=t['task_name'],
       total=t['total_episodes'],paired=n,full_failed=t['full_incomplete_episodes'],
       velocity_gain_pct=v['improvement_pct_of_means'],velocity_median_pct=q['median'],velocity_p90_pct=q['p90'],
       velocity_le_1pct_count=v['cumulative_gain_le_pct']['1']['count'],velocity_le_1pct_fraction=v['cumulative_gain_le_pct']['1']['fraction'],
       acceleration_gain_pct=a['improvement_pct_of_means'],mY_gain_pct=m['improvement_pct_of_means'],
       sigma_min_gain_pct=sig['improvement_pct_of_means'],saving_share=t['share_of_total_velocity_saving'],
       adaptive_gain_pct=i['adaptive_reduction_vs_initial_pct'])
    csvrows.append(record)
with (H/'per_task.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
task_intro=['# 40任务逐项对照','',
 'S=Spatial，O=Object，G=Goal，L=Long，任务编号0起始。全部比较均不筛桌面净空，保留完整非桌面运动学验收。正数为改善：速度/加速度越低越好，mY/归一化σmin越高越好。均值比和逐demo中位数是不同统计。',
 '配对/来源给出本任务运动学比较分母；全开失败单列、不使用截断结果。每方向语义上限仍为原表5°/15°/30°，并非统一111×30°。原1851+补标24的分层差异见主报告。','']
(H/'PER_TASK.md').write_text('\n'.join(task_intro+table+['']+cards))

failure_lines=['# 最大语义容差全开未完整通过的38条','',
 '忽略净空仍保留完整求解与自碰撞验收。搜索最优在1875条均有完整候选；这些全开失败条目不参与四维/七维运动学数值配对。下面原因是执行器终止类别，不是不可达证明。','',
 '|来源|demo|终止原因|原trial绝对路径|','|---|---|---|---|']
for r in s['full_incomplete']:
    for f in r['failures']:
        failure_lines.append(f"|{r['stratum']}|{r['key']}|{f['reason']} / {f['termination']}|`{f['path']}`|")
(H/'FULL_INCOMPLETE.md').write_text('\n'.join(failure_lines)+'\n')

top=sorted(tasks,key=lambda t:-t['velocity_absolute_saving_sum'])[:5]
allm=s['all']['metrics'];v=allm['velocity'];initial=s['all']['initial_vs_adaptive']
main=['# 全40任务：无净空条件下，全开初始化与搜索最优的运动学对照','',
 '状态：2026-11-09，全部只读统计及独立来源/选点/聚合复核完成。未新跑搜索、仿真、训练或物理任务；原始标签、冻结请求、成败候选和用户暂停的训练全部保留。','',
 '优先检查样本见[FOCUS_CASES.md](FOCUS_CASES.md)：全部38条全开失败编号、14条搜索降速>10%的代价样本，以及速度收益较小但σmin/mY变化较大的样本；每条原trial地址和SHA见focus_cases.json。','',
 f"核心结果：1875条全部覆盖，最大语义全开有38条未完整通过；共同完整1837条中，搜索最优较全开平均关节速度降低{v['improvement_pct_of_means']:.3f}%，逐条中位数{v['paired_gain_pct']['median']:.3f}%，{v['cumulative_gain_le_pct']['1']['count']}/{N}={pct(v['cumulative_gain_le_pct']['1']['count'])}的收益≤1%。",'',
 '## 比较对象和分母','',
 '两组都取消桌面相对净空与绝对桌面接触验收，但保留完整运动学可解、自碰撞等非桌面验收。全开组根据prepared_input重建最大点并匹配每块原初始化候选，APP/EXEC各轴使用原手工5°/15°/30°上限、双向全开，严格轴仍0。搜索组在相同已采样候选中按实际观测平均关节速度最小选择，包含初始化和16次贝叶斯追加；其他运动学指标没有参与选点。','',
 f"原1875批次中1851条标签与新补标24条合计1875条、40任务、{s['blocks']}个独立标注块；核对{s['candidate_observations_audit_bound']}条候选元数据与原独立审计SHA，读取验证{s['unique_trials_sha_verified']}份去重trial。补标24采用允许失败零基线几何观测的协议，候选仍为相同IPOPT20/EMA10、cache85正常回放；该协议差异保留为单独分层，不冒称同协议重复实验。原Long02/demo40的GP线搜索数值恢复亦单列敏感性。",'',
 '全开完整1837/1875（97.97%），搜索最优完整1875/1875。全开38条失败分为36条optimizer_infeasible、2条robot_self_collision。36条求解拒绝的原hard_failure_details全部是IPOPT达到20次迭代上限，发生于postgrasp 26条、release 7条、grasp 3条；不能将其视为严格不可达证明。这是已有候选内的执行完整率，不是LIBERO物理任务成功率。尤其L08双摩卡壶上炉有24/45全开失败，只剩21条数值配对。完整失败清单见[FULL_INCOMPLETE.md](FULL_INCOMPLETE.md)，求解状态与逐帧证据见full_failure_diagnosis.json。','',
 '## 全体运动学指标与分布','',
 '每demo内速度/加速度/jerk按独立块N−1/N−2/N−3加权，mY按N加权；最小奇异值取块间最小，关节路程求和。不拼接独立初始化的块、不虚构跨块差分；demo间等权。百分比均值比=两个总体均值之比的改善，逐条分位数来自每条自身的相对改善。正数为改善，负数为退化。','',
 '|指标|全开均值|搜索最优均值|均值比改善|逐条中位数|P05—P95改善|变化绝对值≤1%|恶化>1%|',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
labels={'velocity':'平均关节速度↓ rad/s','acceleration':'平均关节加速度↓ rad/s²','jerk':'平均关节jerk↓ rad/s³',
        'mY':'平均可操作度mY↑','sigma_min':'速度能力归一化最小σ↑','sigma_min_full':'原始几何Jacobian最小σ↑','travel':'关节路程↓ rad'}
for m,z in allm.items():
    q=z['paired_gain_pct']
    main.append(f"|{labels[m]}|{z['full_mean']:.6f}|{z['best_mean']:.6f}|{z['improvement_pct_of_means']:+.4f}%|{q['median']:+.4f}%|{q['p05']:+.3f}%—{q['p95']:+.3f}%|{z['abs_change_le1pct']}/{N} ({pct(z['abs_change_le1pct'])})|{z['regressed_gt1pct']}/{N} ({pct(z['regressed_gt1pct'])})|")
main += ['',
 '速度降低表示相同动作时序下关节运动量减小，不是执行时长变短。搜索只优化速度，因此加速度/jerk可以恶化；平均加速度接近不变，并不意味着每条都接近不变。归一化σ和原始σ对应不同Jacobian尺度，不可混作同一个指标。','',
 '|速度额外收益阈值|样本数/1837|比例|','|---|---:|---:|']
for th in [.01,.1,.5,1,2,5,10]:
    z=v['cumulative_gain_le_pct'][str(th)]
    main.append(f"|≤{th}%|{z['count']}/{N}|{z['fraction']*100:.2f}%|")
main += ['',
 '收益>5%的仅48/1837（2.61%），>10%的14/1837（0.76%）。按绝对关节速度节省排序，前1%样本贡献19.63%，前5%贡献45.89%，前10%贡献61.76%；平均数存在明显长尾。', '',
 '![整体分布与指标权衡](overview.png)','',
 '## 初始化择优与追加搜索的贡献','',
 f"只在结构化初始化中择优，平均速度从{initial['full_mean_velocity']:.6f}降至{initial['initial_best_mean_velocity']:.6f}rad/s，改善{initial['initial_reduction_vs_full_pct']:.4f}%；加上16次追加查询后为{initial['all_best_mean_velocity']:.6f}rad/s，相对初始化最优再降{initial['adaptive_reduction_vs_initial_pct']:.4f}%。",'',
 f"初始化择优已经贡献总绝对速度收益的86.92%。追加查询收益≤0.1%的{initial['adaptive_gain_le_0_1pct']}/{N}（{pct(initial['adaptive_gain_le_0_1pct'])}），≤1%的{initial['adaptive_gain_le_1pct']}/{N}（{pct(initial['adaptive_gain_le_1pct'])}）；逐条追加收益中位数{initial['adaptive_paired_gain_pct']['median']:.8f}%。尽管1277条至少有一块最终选中追加候选，候选来源不能替代真实收益大小。",'',
 '追加查询相对初始化最优收益较高的任务为S09（0.748%）、S06（0.579%）、S07（0.425%）；这些任务也未显示所有样本都需要追加查询。','',
 '## 哪些任务拉高总体差异','',
 '40任务中28个任务的速度均值改善≤1%，32个任务的逐条中位数≤1%。以下按绝对速度节省贡献排序，而非按容易受分母影响的逐条百分比排序：','',
 '|任务|简述|配对/来源|速度均值比改善|逐条中位数|总速度节省贡献|剔除此任务后的整体改善|',
 '|---|---|---:|---:|---:|---:|---:|']
for t in top:
    z=t['metrics']['velocity']
    main.append(f"|{code(t)}|{short(t)}|{t['paired_episodes']}/{t['total_episodes']}|{z['improvement_pct_of_means']:.3f}%|{z['paired_gain_pct']['median']:.3f}%|{t['share_of_total_velocity_saving']*100:.2f}%|{t['overall_velocity_gain_if_task_removed_pct']:.3f}%|")
main += ['',
 '这5个任务贡献43.92%的绝对速度节省；剔除它们后整体改善从0.850%降为0.544%。前3个贡献30.13%，前10个贡献63.05%。任务分组对应约23.08%的逐条百分比方差，其余仍是任务内部差异，不能把全部长尾归因于任务名称。','',
 '特殊情况：L08双摩卡壶上炉的24/45条全开不完整，独占38条全开失败中的24条；其余21条平均降速2.809%，这是有条件的幸存配对统计。L04双杯分放左右盘均值2.774%、中位数0.658%，最高demo_14达到24.961%，明显由少数样本拉高。G00/G05两任务共83条为严格零域，两组完全一致；剔除后1754条仍有74.74%的速度收益≤1%，总体改善0.899%，主要结论保持。','',
 '![40任务速度收益及低收益占比](per_task_velocity.png)','',
 '完整40任务表及每任务独立文字分析见[PER_TASK.md](PER_TASK.md)，包含四项指标、低收益比例、尾部分位数、全开失败、初始化/追加贡献和剔除敏感性。机器可读表见[per_task.csv](per_task.csv)。','',
 '## 分层与数值尾部','',
 '|来源/敏感性|总来源|配对|速度均值改善|','|---|---:|---:|---:|']
for title,z in [('原1851',s['strata']['original1851']),('后续补标24',s['strata']['supplement24']),('合并1875',s['all']),('排除严格零域',s['excluding_strict']),('排除GP恢复1条',s['excluding_gp_recovery'])]:
    main.append(f"|{title}|{z['total_episodes']}|{z['paired_episodes']}|{z['metrics']['velocity']['improvement_pct_of_means']:.4f}%|")
main += ['',
 '归一化最小σ的逐条改善百分比均值11.15%，但中位数仅约0.000016%，两总体均值之比提高3.37%。L08/demo30全开σ=0.00795676，搜索后0.498871，逐条比值达+6169.77%；该百分比受到很小分母放大。排除全开σ<0.05的4条后，均值比仍为+3.346%，说明个别巨大百分比不能当作典型收益，但均值改善也不完全由这4条造成。完整绝对值及分位数见summary.json、tail_diagnostics.json。','',
 '## 验证、范围和本地绝对源地址','',
 f"独立复核检查{verification['source_hashes_checked']}份原文件SHA、重新读取{verification['unique_raw_trials_reparsed']}份raw trial，重建2301个块的全开点及无净空最优，复算全部7项聚合、1837条配对CSV的均值/中位数/阈值/退化计数和40任务速度汇总。详见verification.json及verification.log。",'',
 '这是原净空感知BO已经采样的候选内比较。重新删除净空后从头搜索会改变采样点，本次没有测量；本次也不输出新的物理任务成功率。不同标注块各自初始化，不能当作连续物理长任务验证。旧协议中的全零/非零姿态参考衔接差异仍保留，未通过改SHA或改代码消除。','',
 '上次Spatial00的3.11%是“原净空合格最优→去净空最优”；本报告是“语义最大值全开→去净空最优”，两边都不筛净空，分母与对照组不同。当前S00的此项速度改善为0.442%，没有矛盾。','',
 '- 原始1875请求、候选、标注、原独立审计、source_snapshot与日志根：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107`。逐demo为`<suite>/task_XX/demo_N/{prepared_input.json,tolerance_labels.json,subtask_*/annotation.json,subtask_*/evaluations/<hash>/trial.json}`；配置`annotation_request.json`、`selection.json`、`authored_tolerances_deg.json`；审计`annotation_independent_final_audit.json`；日志`annotation.log`、`pipeline.log`和各demo/search.log。','',
 '- 补标24请求/结果/独立审计/源码/日志根：`/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109`，同样的逐demo结构；`annotation_request.json`保留baseline observation差异，`annotation_independent_final_audit.json`审计24条，`pipeline.log`、`independent_audit.log`与各demo/search.log保留。','',
 '- 名义物理逐回合视频/轨迹/状态原位根：`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261025_success10_v1/episodes`与`/media/czl/sata/franka_my_code/openpi/data/libero_tasktol_joint_servo/20261101_full_v1/episodes`。每条精确source_folder在本目录episodes.json，可核对其`trajectory.json`、`states.npz`、`report.json`及双相机视频；离线候选没有新物理视频，未复制原始数据。','',
 '- 本次分析根：`'+str(H)+'`。`analyze.py`入口、`analysis.log`过程、`episodes.json`逐demo/块/候选路径及SHA/指标、`summary.json`全部统计、`per_demo.csv`逐条7指标、`per_task.csv`40任务、`sources.json`全部读取文件绝对路径和SHA、`verify.py`与`verification.json`/`verification.log`复核、`report.py`文档和图表生成。','',
 '- 未使用远端主机、不运行模型推理、不训练、不上传、不恢复用户暂停的其他工作。','']
(H/'README.md').write_text('\n'.join(main))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
g=np.array([gain(r) for r in paired])
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
ax=axes[0,0];x=np.sort(g);ax.plot(x,np.arange(1,N+1)/N*100,color='#376fa1',lw=2)
for xval in [1,5]:ax.axvline(xval,color='#888888',ls='--',lw=1)
ax.set(xlim=(-.2,25.5),ylim=(0,101),xlabel='Velocity reduction vs full semantic caps (%)',ylabel='Cumulative demos (%)',title='A. Empirical velocity benefit distribution')
thresholds=[.01,.1,.5,1,2,5,float('inf')];previous=0;counts=[]
for threshold in thresholds:
    current=int((g<=threshold).sum());counts.append(current-previous);previous=current
ax=axes[0,1];bars=ax.bar(range(7),np.array(counts)/N*100,color='#447ca9')
ax.set_xticks(range(7),['<=.01','.01-.1','.1-.5','.5-1','1-2','2-5','>5'])
ax.set(xlabel='Velocity benefit bin (%)',ylabel='Demos (%)',title='B. Most samples receive small benefits')
ax.bar_label(bars,labels=[f'{100*n/N:.1f}%' for n in counts],padding=3,fontsize=9)
ax=axes[1,0];ms=['velocity','acceleration','mY','sigma_min'];bars=ax.bar(range(4),[allm[m]['improvement_pct_of_means'] for m in ms],color='#4c8b81')
ax.scatter(range(4),[allm[m]['paired_gain_pct']['median'] for m in ms],color='#a5493e',label='Per-demo median',zorder=3)
ax.set_xticks(range(4),['Velocity','Acceleration','mY','Normalized sigma min']);ax.set(ylabel='Improvement (%)',title='C. Ratios of means and per-demo medians')
ax.bar_label(bars,fmt='%.2f',padding=3);ax.legend(frameon=False)
failed=[t for t in tasks if t['full_incomplete_episodes']]
ax=axes[1,1];ax.bar([code(t) for t in failed],[t['full_incomplete_episodes'] for t in failed],color='#b36152')
ax.set(ylabel='Incomplete full-cap demos',title='D. Full-cap failures kept outside metric pairs')
fig.suptitle('40 LIBERO tasks | both arms ignore table clearance | 1837 complete pairs / 1875 sources',fontsize=14)
fig.savefig(H/'overview.png',dpi=170);plt.close(fig)

ranked=sorted(tasks,key=lambda t:-t['metrics']['velocity']['improvement_pct_of_means'])
fig,axes=plt.subplots(1,2,figsize=(13,14),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained')
y=np.arange(40)
means=[t['metrics']['velocity']['improvement_pct_of_means'] for t in ranked]
medians=[t['metrics']['velocity']['paired_gain_pct']['median'] for t in ranked]
axes[0].barh(y,means,color=[colors[t['suite']] for t in ranked],alpha=.85)
axes[0].scatter(medians,y,marker='|',s=80,color='black',label='Per-demo median',zorder=3)
axes[0].set_yticks(y,[f"{code(t)}  ({t['paired_episodes']}/{t['total_episodes']})" for t in ranked],fontsize=9)
axes[0].invert_yaxis();axes[0].set(xlabel='Velocity reduction (%)',title='Ratio-of-means benefit; black tick = median');axes[0].legend(frameon=False)
fractions=[t['metrics']['velocity']['cumulative_gain_le_pct']['1']['fraction']*100 for t in ranked]
axes[1].barh(y,fractions,color='#7f98a3');axes[1].set_yticks(y,[]);axes[1].invert_yaxis()
axes[1].set(xlim=(0,105),xlabel='Within-task demos with gain <=1% (%)',title='Small-benefit fraction')
for i,value in enumerate(fractions):axes[1].text(min(value+1,102),i,f'{value:.0f}',va='center',fontsize=8)
fig.suptitle('All 40 tasks | paired / source counts shown | S: spatial, O: object, G: goal, L: long',fontsize=13)
fig.savefig(H/'per_task_velocity.png',dpi=170);plt.close(fig)
print('Wrote README.md, PER_TASK.md, FULL_INCOMPLETE.md, CSV tables and two figures.')
