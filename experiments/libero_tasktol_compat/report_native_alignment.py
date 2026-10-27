"""Summarize paired diagnostics and link all retained evidence."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
BASE=ROOT/'experiments/libero_tasktol_compat/results'
OLD=ROOT.parent/'franka_mujoco/experiments/libero_closed_loop/results/native_nominal40'
OUT=BASE/'alignment_report_v1'


def percentile(values,fraction):
    values=sorted(values)
    position=(len(values)-1)*fraction
    left=int(position);right=min(left+1,len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def collect(label,root,native=False):
    reports=sorted(root.glob('*/*/report.json'))
    if len(reports)!=40:raise RuntimeError(f'{label}: expected forty records, found {len(reports)}')
    successes=[];errors=[];elapsed=[];failures=[];cases=[]
    for path in reports:
        report=json.loads(path.read_text())
        if native:
            suite=report['suite'];task_id=report['task_id']
            frames=json.loads(path.with_name('trace.json').read_text())
            errors.extend(r['source_position_error_m']*1000 for r in frames)
            native_context=report.get('reset_rng_matches_source')
        else:
            suite=report['task']['suite'];task_id=report['task']['task_id']
            frames=json.loads(path.with_name('trajectory.json').read_text())['frames']
            errors.extend(r['source_replay_position_error_m']*1000 for r in frames if not r.get('aborted') and r.get('source_replay_position_error_m') is not None)
            elapsed.extend(r['optimizer']['elapsed_ms'] for r in frames)
            native_context=report.get('initialization',{}).get('reset_rng_matches_source')
        success=bool(report['task_success']);successes.append(success)
        kind=None if success else ('optimizer_rejected' if any(r.get('aborted') for r in frames) else 'task_failed_after_execution')
        case=dict(suite=suite,task_id=task_id,success=success,category=kind,report=str(path),
            source_context_verified=native_context,frames=len(frames))
        if not success:failures.append(case)
        cases.append(case)
    result=dict(label=label,success=sum(successes),count=len(successes),rate=sum(successes)/len(successes),
        frames=len(errors),mean_source_position_error_mm=statistics.mean(errors),
        p95_source_position_error_mm=percentile(errors,.95),max_source_position_error_mm=max(errors),
        failures=failures,failure_categories=dict(Counter(x['category'] for x in failures)),cases=cases,
        context_verified_count=sum(x['source_context_verified'] is True for x in cases))
    if elapsed:result['solver_elapsed_ms']=dict(mean=statistics.mean(elapsed),p95=percentile(elapsed,.95),max=max(elapsed),over_50_ms=sum(x>50 for x in elapsed))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    variants=[
        ('旧初始化 / IPOPT / 控制器目标 / kp50',OLD,False),
        ('快照恢复 / 原生 OSC',BASE/'native_osc_paired40_v1',True),
        ('刷新缓存 / 原生 OSC',BASE/'native_osc_paired40_refresh_v2',True),
        ('完整源初始化 / 原生 OSC',BASE/'native_osc_source_history_all40_v5',True),
        ('完整源初始化 / IPOPT / 控制器目标 / kp50',BASE/'ipopt_source_context_kp50_150_all40_v3/kp_50',False),
        ('完整源初始化 / IPOPT / 控制器目标 / kp150',BASE/'ipopt_source_context_kp50_150_all40_v3/kp_150',False),
        ('完整源初始化 / IPOPT / 实际运动 / kp150',BASE/'ipopt_source_context_recorded_kp150_all40_v3/kp_150',False),
    ]
    for version,gains in [('v4',[300,1000]),('v5',[2000,4000])]:
        for gain in gains:
            path=BASE/f'ipopt_source_context_recorded_gain_all40_{version}'/f'kp_{gain}'
            if len(list(path.glob('*/*/report.json')))==40:
                variants.append((f'完整源初始化 / IPOPT / 实际运动 / kp{gain}',path,False))
    summaries=[collect(*variant) for variant in variants]
    masks=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/mask_annotations.json').read_text())['tasks']
    task_evidence={}
    for mask in masks:
        key=f"{mask['suite']}/{mask['task_id']:02d}"
        task_evidence[key]=dict(shared_frozen_mask=mask,observations=[dict(condition=s['label'],**next(c for c in s['cases'] if c['suite']==mask['suite'] and c['task_id']==mask['task_id'])) for s in summaries],
            interpretation_scope='Public evidence and test outcomes; phase labels are inherited from the source and not independently reannotated for each generated trajectory')
    (OUT/'task_evidence_assessment.json').write_text(json.dumps(task_evidence,ensure_ascii=False,indent=2)+'\n')
    payload=dict(scope='Forty fixed selected demonstrations; no VLA checkpoint; no tolerance release in this comparison',
        initialization_fix='Native reset sampler history + RNG + gripper accumulator + controller cache; no extra physics steps',
        source_goal_caution='OSC controller goals need not be achieved by the source robot. Strict IPOPT constraints and observed-motion references evaluate different contracts.',
        variants=summaries)
    (OUT/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# LIBERO 闭环对齐修复与对照','',
        '本轮优先修复初始化和执行对齐。全部记录来自相同 40 条选定示范，未接入 VLA 模型；下表均关闭容差释放。',
        '', '| 条件 | 任务成功 | 末端位置平均差 / mm | 95% 分位 / mm | 最大差 / mm |',
        '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['label']} | {s['success']}/40 | {s['mean_source_position_error_mm']:.3f} | {s['p95_source_position_error_mm']:.3f} | {s['max_source_position_error_mm']:.3f} |")
    lines+=['','位置差对齐源动作执行后的末端位置；只统计实际执行帧，因此各行的误差统计帧数并不完全相同。',
        '', '## 已修复的初始化问题','',
        '- 扁平物理状态不包含固定柜子、炉子等 fixture 的模型基座位置。',
        '- 原生 LIBERO 每次硬重置会追加属性采样器；同一种子或上一条记录的随机状态，未必重建相同布局。按源重置历史重建采样器列表，并逐条核对结束随机状态。',
        '- 恢复快照后使控制器缓存失效；通过原生夹爪格式化函数恢复源十次稳定动作对应的内部累加状态。未增加物理步进。',
        '- 从恢复的源重置上下文获得初始保持姿态，替代首帧实际 FK 的近似。',
        '', '实现位于 `franka_mujoco/tasktol/benchmark/libero_replay_state.py`，现有 `tasktol.benchmark.libero` 默认使用完整初始化。旧初始化可用 `--initialization-mode snapshot_only` 明确复现。原生默认关节 kp=50 保留，kp150 是显式对照配置。',
        '', '## 对剩余问题的证据','',
        '- 改为 kp150 可以修复部分失败，也会产生其他退化；不按任务挑选增益，不将其声称为通用修复。',
        '- 三条原控制器目标下的求解中止，在三种初值、300 次迭代与 5 秒预算的离线诊断中均收敛到局部不可行；候选残余分别约 0.456、3.160、1.077 mm。该证据提示目标可达性边界，不构成全局不可行证明。',
        '- 对应三条源实际运动参考可以完成所有严格 IPOPT 求解，说明“源控制器希望达到的目标”与“源机器人实际走过的轨迹”必须分开评估。',
        '- 对两个开抽屉任务增加了只读接触观测。跟踪实际运动参考时，中层抽屉的指尖/柜体单接触峰值法向力约 5.8 N；原生 OSC 对照约 95.1 N。',
        '- 相同关节 kp1000、相同初始化下，仅更换参考：源控制器目标使中层抽屉最终打开159.7mm并成功，实际运动参考仅打开5.16mm并失败。证据支持参考目标引起的接触加载差异；不能仅用末端位置误差代表接触任务复现。',
        '- 原生对照剩余 libero_10/03 失败，在原 Python 环境按原预处理函数执行时成功。移植完整积分状态、夹爪和控制器目标的单例诊断仍失败，因此尚不能将这一个差异归因于缺少求解器热启动；完整上下文假设未获支持。',
        '- 原生 OSC 对照视频仅为诊断材料，不进入 IPOPT 重录或训练数据。',
        '', '## 留存与验证','',
        '逐条保留原生/IPOPT 视频、初末帧、轨迹和求解日志；源/旧恢复/新恢复首帧并排图在相邻 `source_init_comparison_v1`。',
        '原 TaskTol 主环境回归：37 passed、5 skipped；LIBERO 锁定环境：5 passed、1 deselected。通用 3.10 场景测试在 3.2.3 环境不适用，已在原主环境通过。',
        '', '旋转容差追加诊断：同一实际运动参考、kp4000、四个task0代表，名义执行3/4成功，显式±5°容差为2/4成功；668次IPOPT全部可行。5°未经过数值搜索，阶段仍沿用源标注，因此这不构成容差效果已验证的结论。',
        'kp4000实际运动参考批次求解耗时95%分位约44ms，最大约95ms，84/6544次超过50ms；离线20Hz仿真步进不代表每次墙钟实时期限都已满足。',
        '', '## 各条件失败任务','']
    for s in summaries[3:]:
        lines += [f"### {s['label']}",'']
        lines += [f"- {x['suite']}/{x['task_id']:02d}: {x['category']} — [记录]({__import__('os').path.relpath(x['report'],OUT)})" for x in s['failures']]
        lines += ['']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    headers=''.join(f"<th>{html.escape(s['label'])}</th>" for s in summaries)
    rows=[]
    all_cases=sorted((x['suite'],x['task_id']) for x in summaries[0]['cases'])
    for suite,tid in all_cases:
        cells=[]
        for s in summaries:
            case=next(x for x in s['cases'] if x['suite']==suite and x['task_id']==tid)
            p=Path(case['report']);folder=p.parent
            video=next(folder.glob('*.mp4'))
            rel=Path(__import__('os').path.relpath(video,OUT)).as_posix()
            report_rel=Path(__import__('os').path.relpath(p,OUT)).as_posix()
            status='成功' if case['success'] else case['category']
            process=folder/'process_images.json'
            images_link=f' · <a href="{html.escape(__import__("os").path.relpath(process,OUT))}">过程图索引</a>' if process.exists() else ''
            cells.append(f'<td><b>{html.escape(status)}</b><details><summary>视频与记录</summary><video controls preload="none" src="{html.escape(rel)}"></video><p><a href="{html.escape(report_rel)}">JSON</a>{images_link}</p></details></td>')
        rows.append(f'<tr><th>{suite}/{tid:02d}</th>'+''.join(cells)+'</tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>LIBERO 闭环对齐对照</title><style>body{font:15px system-ui;background:#111827;color:#e5e7eb;padding:24px}a{color:#93c5fd}table{border-collapse:collapse}td,th{border:1px solid #475569;padding:10px;min-width:180px;vertical-align:top}video{width:256px}summary{cursor:pointer}</style><h1>LIBERO 闭环对齐对照</h1><p>40 条固定示范；原生 OSC 为诊断材料；IPOPT 结果与原生结果分列。所有表内对照关闭容差释放。</p><p><a href="RESULTS.md">结论与限制</a> · <a href="summary.json">机器可读指标</a> · <a href="task_evidence_assessment.json">各任务共享mask与证据</a></p><table><tr><th>任务</th>'''+headers+'</tr>'+''.join(rows)+'</table></html>'
    (OUT/'index.html').write_text(page)
    inventory={__import__('os').path.relpath(p,OUT):hashlib.sha256(p.read_bytes()).hexdigest() for _,root,_ in variants for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl','.png','.mp4','.log'} and 'numba_cache' not in p.parts}
    (OUT/'evidence_sha256.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps([{k:s[k] for k in ['label','success','count','mean_source_position_error_mm','p95_source_position_error_mm','max_source_position_error_mm']} for s in summaries],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
