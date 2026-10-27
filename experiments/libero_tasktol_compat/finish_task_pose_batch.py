"""Finish packaging/auditing once the foreground recording experiment completes."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    run, dataset = args.run.resolve(), args.dataset.resolve()
    while True:
        path = run/'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else dict(complete=False, completed=0)
        if summary['complete']:
            break
        if not args.wait:
            raise RuntimeError('Recording experiment has not completed')
        print(json.dumps(dict(stage='recording', completed=summary['completed'], planned=400,
                              checked_at_utc=datetime.now(timezone.utc).isoformat())), flush=True)
        time.sleep(50)
    assert summary['process_errors'] == 0, 'Inspect and resume recorder errors before packaging'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONPATH=str(ROOT)+':'+str(ROOT.parent/'franka_mujoco'))
    steps = [('tracking', 'summarize_task_pose_batch.py', ['--output', str(run)])]
    if not (dataset/'manifest.json').exists():
        steps.append(('packaging', 'build_state_reference_dataset.py', ['--task-refinement', str(run), '--output', str(dataset)]))
    steps.append(('dataset_validation', 'validate_state_dataset.py', ['--dataset', str(dataset)]))
    status = dict(complete=False, run=str(run), dataset=str(dataset), steps=[])
    for label, script, arguments in steps:
        print(json.dumps(dict(stage=label, started=True)), flush=True)
        with (run/f'final_{label}.log').open('w') as stream:
            done = subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *arguments],
                                  env=env, stdout=stream, stderr=subprocess.STDOUT)
        status['steps'].append(dict(stage=label, returncode=done.returncode, log=str(run/f'final_{label}.log')))
        (run/'completion.json').write_text(json.dumps(status, indent=2)+'\n')
        assert done.returncode == 0, status['steps'][-1]
    metrics = json.loads((run/'tracking_summary.json').read_text())
    manifest = json.loads((dataset/'manifest.json').read_text())
    assert manifest['tasks'] == 40 and manifest['attempted_episodes'] == 400
    assert all(task['attempts'] == 10 for task in manifest['task_files'])
    assert manifest['successful_episodes'] == summary['selected_success']
    baseline = metrics['aggregates']['baseline']['all']
    selected = metrics['aggregates']['selected']['all']
    report = ['# LIBERO每任务10条：默认IPOPT原速重构', '',
        '固定随机种子20261024，每任务从全部50条原始示范抽10条，不按已有成功结果筛选；共40任务400条。', '',
        '在线复用原样默认IPOPT和B10完整七关节客户端。默认20次迭代、0.095秒预算，默认目标/约束与共享源码不变。离线投影及评分均不含原机器人/夹爪关节跟踪惩罚；源q只用于FK与数值初值。', '',
        '所有示范使用统一原动作意图投影，完整执行后统一尝试两轮IRLC修正（增益0.5、0.25）。只接受完整可行、保持既有成功且降低末端+物体代价的候选。拒解没有完整轨迹时不做修正，不回退OSC，不换示范补成功。不使用旧候选或CEM。', '',
        '原速20Hz、原夹爪命令、全部等待、零额外settle；逐条原始最后动作因缺少下一state，与先前实验相同不执行。图像、实际state、动作、参考、IPOPT输出、失败前缀和视频全部保留。', '',
        '|指标|首次执行|两轮修正后选择|', '|---|---:|---:|',
        f"|最终成功|{baseline['successes']}/400 ({baseline['successes']/4:.2f}%)|{selected['successes']}/400 ({selected['successes']/4:.2f}%)|",
        f"|拒解|{baseline['rejected']}|{selected['rejected']}|",
        f"|完整执行|{baseline['full_length']}|{selected['full_length']}|", '',
        '跟踪误差按step后实际末端FK对原示范下一state实际FK、同一时间戳统计。完整轨迹和拒解前缀分开，不通过截断低误差来美化结果。不是相对各自含前馈/预载控制参考的误差。', '']
    for title, kind in [('全部完整执行轨迹（含任务失败）', 'full_rollout_error'), ('任务成功轨迹', 'successful_rollout_error')]:
        report += [f'## {title}', '', '|指标|首次执行|修正后选择|', '|---|---:|---:|']
        for label, key, unit in [('平均位置', 'position_mean_mm', 'mm'), ('位置P95', 'position_p95_mm', 'mm'),
                                  ('平均姿态', 'rotation_mean_deg', '°'), ('姿态P95', 'rotation_p95_deg', '°')]:
            a, b = baseline[kind].get(key), selected[kind].get(key)
            report.append(f'|{label}|{a:.4f}{unit}|{b:.4f}{unit}|' if a is not None and b is not None else f'|{label}|无完整轨迹|无完整轨迹|')
        report += ['']
    report += ['## 文件', '', f'- 数据包：`{dataset}`，每任务`dataset.hdf5`保留全部10条attempts，仅成功条目进入data。',
        '- `plan.json`、`policy.json`：冻结抽样清单与完整规则。',
        '- `summary.json`：400条首次/选择结果与全部试录记录；`episodes.csv`逐条结果。',
        '- `tracking_summary.json`、`tracking_episodes.csv`、`tasks.csv`：总量、任务集、逐任务和逐条跟踪及拒解原因。',
        '- 每条`baseline/`、`iteration_1/`、`iteration_2/`保留所有实际试录；`selection.json`说明最终选择和跳过原因。',
        '- 每次录制均通过动作、原夹爪、原始输入hash、FK参考见证、物理步进和视频长度核验；`optimizer_default_audit.json`记录实际默认参数及源码hash。',
        '- 数据包`validation.json`核验观测时序、参考重建、全部候选选择判定、关联文件hash与任务完成记录。本400条批次未宣称逐条独立冷回放。', '',
        '本结果是固定400条离线数据重构结果，不是训练后VLA准确率；此前11条包含历史候选选择和个别CEM搜索，不作为本批成功率。']
    (run/'README.md').write_text('\n'.join(report)+'\n')
    status.update(complete=True, tasks=40, attempted_episodes=400, baseline_success=summary['baseline_success'],
                  selected_success=summary['selected_success'], validation=str(dataset/'validation.json'))
    (run/'completion.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
