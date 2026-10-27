"""Summarize paired native LIBERO compatibility probes and retain media checks."""
from pathlib import Path
import hashlib
import html
import json
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
OPENPI = Path('/media/czl/sata/franka_my_code/openpi')
PAIRS = [('libero_spatial', 'spatial_00'), ('libero_object', 'libero_object_00'),
         ('libero_goal', 'libero_goal_00'), ('libero_10', 'libero_10_00')]


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    checks, rows = [], []

    def check(name, valid):
        checks.append(dict(name=name, passed=bool(valid)))

    for suite, name in PAIRS:
        old_dir = ROOT/'results'/f'baseline_{suite}_00'
        new_dir = ROOT/'results'/f'final_{name}'
        old = json.loads((old_dir/'report.json').read_text())
        new = json.loads((new_dir/'report.json').read_text())
        with np.load(old_dir/'observations.npz') as a, np.load(new_dir/'observations.npz') as b:
            position_error = np.linalg.norm(a['eef_position']-b['eef_position'], axis=1)
            joint_error = np.linalg.norm(a['joint_position']-b['joint_position'], axis=1)
            steps = len(position_error)
            check(suite+':same_replay_clock', np.allclose(a['sim_time'], b['sim_time'], atol=1e-8))
        check(suite+':environment_runs', old['status'] == new['status'] == 'passed')
        check(suite+':all_runtime_stages_pass', all(x['status'] == 'passed' for x in old['stages']+new['stages']))
        for key in ['initial_state_sha256', 'actions_sha256', 'bddl_sha256', 'retained_source_indices', 'seed', 'settling_steps', 'control_freq_hz']:
            check(suite+':same_input:'+key, old['task'][key] == new['task'][key])
        check(suite+':same_original_assets_path', old['libero_paths']['assets'] == new['libero_paths']['assets'] == str(OPENPI/'third_party/libero/libero/libero/assets'))
        fk = next(x for x in new['stages'] if x['name'] == 'native_model_data_and_jacobian')
        check(suite+':tasktol_fk_and_jacobian_match', fk['tasktol_fk_position_error_m'] < 1e-8 and fk['tasktol_jacobian_max_error'] < 1e-8)
        check(suite+':same_final_probe_source', old['probe_sha256'] == new['probe_sha256'] == sha(ROOT/'smoke_test.py'))
        for directory in [old_dir, new_dir]:
            probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,r_frame_rate,nb_read_frames,width,height',
                '-of', 'json', str(directory/'replay.mp4')], text=True))['streams'][0]
            check(directory.name+':video_full_decode', probe['codec_name'] == 'h264'
                  and probe['r_frame_rate'] == '20/1' and int(probe['nb_read_frames']) == steps
                  and [probe['width'], probe['height']] == [256, 256])
        row = dict(suite=suite, task_id=0, task_name=new['task']['name'], demo=new['task']['demo'],
            steps=steps, baseline_success=old['task_success'], candidate_success=new['task_success'],
            max_eef_difference_m=float(position_error.max()), final_eef_difference_m=float(position_error[-1]),
            first_eef_difference_over_1mm=next((int(i) for i,x in enumerate(position_error) if x > .001), None),
            max_joint_vector_difference_rad=float(joint_error.max()),
            tasktol_fk_error_m=fk['tasktol_fk_position_error_m'], tasktol_jacobian_error=fk['tasktol_jacobian_max_error'],
            baseline_report=str((old_dir/'report.json').relative_to(ROOT)),
            candidate_report=str((new_dir/'report.json').relative_to(ROOT)),
            baseline_video=str((old_dir/'replay.mp4').relative_to(ROOT)),
            candidate_video=str((new_dir/'replay.mp4').relative_to(ROOT)),
            baseline_final=str((old_dir/f'after_{steps:04d}.png').relative_to(ROOT)),
            candidate_final=str((new_dir/f'after_{steps:04d}.png').relative_to(ROOT)))
        rows.append(row)

    old_assets = OPENPI/'examples/libero/.venv/lib/python3.8/site-packages/robosuite/models/assets'
    new_assets = ROOT/'.venv/lib/python3.12/site-packages/robosuite/models/assets'
    differences, asset_count = [], 0
    for p in old_assets.rglob('*'):
        if not p.is_file():
            continue
        asset_count += 1
        q = new_assets/p.relative_to(old_assets)
        if not q.is_file() or sha(p) != sha(q):
            differences.append(str(p.relative_to(old_assets)))
    check('robosuite_asset_contents_identical', not differences)
    check('api_bridge_unit_tests', '4 passed' in (ROOT/'results/compatibility_tests.log').read_text())
    check('dependency_consistency', 'All installed packages are compatible' in (ROOT/'results/dependency_check.log').read_text())
    source_reports = [json.loads((ROOT/r['candidate_report']).read_text()) for r in rows]
    for report in source_reports:
        for relative, digest in report['tasktol_source_fingerprints'].items():
            check(report['task']['suite']+':tasktol_source_unchanged:'+relative, sha(ROOT.parents[1]/relative) == digest)
    failed = [c['name'] for c in checks if not c['passed']]
    summary = dict(date='2026-10-20', environment_compatibility='passed_with_local_api_bridge',
        physics_equivalence='not_equivalent_on_tested_long_task', tasktol_controller_integration='not_tested',
        scope='Native LIBERO OSC replay plus reuse of existing TaskTol FK/Jacobian. No TaskTol optimization commands were executed.',
        baseline_success_count=sum(r['baseline_success'] for r in rows),
        candidate_success_count=sum(r['candidate_success'] for r in rows),
        baseline_packages=json.loads((ROOT/rows[0]['baseline_report']).read_text())['packages'],
        candidate_packages=source_reports[0]['packages'], assets_root=source_reports[0]['libero_paths']['assets'],
        robosuite_asset_comparison=dict(files=asset_count, differences=differences),
        paired_results=rows, checks=checks, failed_checks=failed, check_count=len(checks),
        recommendation='Single-process execution is feasible, but keep the current production environments. Investigate physics/control numerical drift before using this upgraded stack as a LIBERO-equivalent tolerance benchmark.')
    (ROOT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    table = '\n'.join(f"| {r['suite']} | {r['steps']} | {'成功' if r['baseline_success'] else '失败'} | {'成功' if r['candidate_success'] else '失败'} | {r['max_eef_difference_m']*1000:.3f} |" for r in rows)
    (ROOT/'RESULTS.md').write_text(f'''# LIBERO × TaskTol 单进程兼容性验证 · 2026-10-20

结论：**增加一个局部 API 适配后，MuJoCo 3.10 / Python 3.12 / NumPy 2.5 环境可以运行原生 LIBERO，并调用原 TaskTol 的运动学代码。物理回放尚不等价。**

| 任务集（各 task 0） | 动作数 | 原 MuJoCo 3.2.3 | 新 MuJoCo 3.10 | 末端最大差异 mm |
|---|---:|---|---|---:|
{table}

每一对使用相同 BDDL、初始状态、动作、种子、10 次 settling 和 20 Hz。原环境 4/4 成功，新环境 3/4 成功；这是各任务集一条示范的诊断结果，不是整个任务集的成功率估计。

长任务从处理帧 52 开始出现大于 1 mm 的末端轨迹差异，最大约 60 mm。主视角结果可见旧环境将目标物放入篮中，新环境未完成目标。新旧 robosuite 的 {asset_count} 个资产文件内容完全一致，LIBERO 资产始终读取原地址。

4 条新环境回放的 TaskTol FK/Jacobian 对齐均通过；8 段视频均完整解码，格式为 H.264、256×256、20 Hz。{len(checks)} 项材料与接口检查，失败项：{failed or '无'}。质量矩阵桥接的 4 项单元测试通过；依赖一致性检查通过。

API 适配只将 robosuite 1.4.1 的 `mj_fullM(model, dst, qM)` 调用转为 MuJoCo 3.10 的 `mj_fullM(model, data, dst)`，继续执行原控制器函数。未修改安装包、LIBERO 源码或 TaskTol 算法。FK 诊断在独立状态副本中 forward，避免把 mj_step 后滞后的派生字段与新 qpos 混用。

本次使用原 OSC 控制器，**没有启用 TaskTol 容差优化或关节目标跟踪**。下一步应先定位新版长任务的回放漂移，或验证保留 LIBERO 原物理版本的共享核心方案；目前不建议把新环境直接替代正式 LIBERO 评测环境。

详见 [逐项数据](summary.json)、[视频与图片](report.html)、[复现说明](README.md)。各次失败及修复记录保留在 `results/attempt_*` 与 `results/unmodified`。
''')
    body = []
    for row in rows:
        body.append(f'<section><h2>{html.escape(row["suite"])} · task 0 · {row["demo"]}</h2><p>{html.escape(row["task_name"])}</p><div class="pair">')
        for kind, label in [('baseline', '原环境 · MuJoCo 3.2.3'), ('candidate', '新环境 · MuJoCo 3.10')]:
            success = row[kind+'_success']
            body.append(f'<article><h3>{label} · {"成功" if success else "未成功"}</h3><video controls preload="metadata" src="{row[kind+"_video"]}"></video><img src="{row[kind+"_final"]}" alt="最终观测"><p><a href="{row[kind+"_report"]}">完整记录</a></p></article>')
        body.append('</div></section>')
    (ROOT/'report.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>LIBERO × TaskTol 环境兼容性验证</title>
<style>body{font:16px/1.65 system-ui;margin:32px auto;max-width:1100px;padding:0 20px;background:#101820;color:#e2ecef}a{color:#70d8d5}h1,h2,h3{line-height:1.3}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}article{padding:16px;background:#1b2932;border:1px solid #38505d}video{width:100%;max-height:360px}img{width:128px;display:block;margin-top:12px}section{margin:40px 0}p{overflow-wrap:anywhere}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style>
<h1>LIBERO × TaskTol 单进程兼容性验证</h1><p>2026-10-20 · 四集各一条原生示范 · 资产按原路径调用</p><p><strong>环境运行已打通；新版物理回放尚不等价。</strong> 原环境四条成功，新环境三条成功。此处仍使用原 OSC 控制器，未启用容差优化。</p><p><a href="RESULTS.md">实验结论</a> · <a href="summary.json">逐项检查</a> · <a href="README.md">复现说明</a></p>'''+''.join(body)+'</html>')
    print(json.dumps(dict(check_count=len(checks), failed_checks=failed,
        baseline_successes=summary['baseline_success_count'], candidate_successes=summary['candidate_success_count']), ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
