"""Build the final selected IPOPT400 catalog and replace the local Studio atomically.

Physics, recordings and frozen mask annotations are read only. Previous Studio
files are retained beside the replacement for rollback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from build_mask_annotation_viewer import PROJECT, ADAPTER, dump, sha, link_or_copy
from mask_viewer.phase_adapter import classify_rows


def pose(value):
    return dict(position_m=value['position_m'], rotation_vector_base_rad=value['rotation_vector_rad'])


def export(job, task, out):
    root = Path(job['selected_root'])
    trial = root/job['suite']/f"{job['task_id']:02d}"/job['demo']
    original = trial/'ipopt_state_B10'
    rows = json.loads((original/'trajectory.json').read_text())
    report = json.loads((original/'report.json').read_text())
    key = f"{job['suite']}/{job['task_id']:02d}/{job['demo']}"
    base = Path('ipopt400')/key
    for name in ['main_rgb.mp4', 'initial.png', 'final.png', 'trajectory.json', 'report.json']:
        link_or_copy(original/name, out/base/name)
    with h5py.File(trial/'state_reference.hdf5') as f:
        source_p = f['source_positions_m'][()]
        source_r = f['source_rotations'][()]
    aperture = [r['pre_action']['finger_qpos_m'][0]-r['pre_action']['finger_qpos_m'][1] for r in rows]
    phases, events, phase_rule = classify_rows(rows)
    frames = []
    for i, row in enumerate(rows):
        before = row['pre_action']
        after = row.get('post_action', before)
        target = row['nominal_target']
        r_after = Rotation.from_rotvec(after['rotation_vector_rad'])
        r_target = Rotation.from_rotvec(target['rotation_vector_rad'])
        rejected = bool(row['aborted'])
        frames.append(dict(index=i, video_frame_index=i, time_s=i/20,
            source_frame=int(row.get('raw_source_index', i)), source_gap_before=0,
            **phases[i], subtask_index=None,
            execution_mask=None,
            action=row['action'], ee_state=pose(before), nominal_target_pose=pose(target),
            source_pose=dict(position_m=source_p[i].tolist(), rotation_vector_base_rad=Rotation.from_matrix(source_r[i]).as_rotvec().tolist()),
            post_action_pose=pose(after),
            joint_state=dict(position_rad=before['joint_position_rad'],
                velocity_rad_s=((np.asarray(before['joint_position_rad'])-np.asarray(rows[i-1]['pre_action']['joint_position_rad']))*20).tolist() if i else [None]*7),
            joint_target_rad=row['joint_target_rad'],
            gripper_state=dict(finger_qpos_m=before['finger_qpos_m'], aperture_m=aperture[i], command=row['action'][6]),
            optimizer=row['optimizer'], aborted=rejected,
            post_target_error_position_mm=None if rejected else ((np.asarray(after['position_m'])-np.asarray(target['position_m']))*1000).tolist(),
            post_target_error_rotation_deg=None if rejected else np.degrees((r_after*r_target.inv()).as_rotvec()).tolist(),
            source_next_position_error_mm=None if rejected else row.get('source_next_position_error_m', 0)*1000,
            source_next_rotation_error_deg=None if rejected else np.degrees(row.get('source_next_rotation_error_rad', 0)),
            observed_gravity_frame=dict(rotation_matrix=None, h=None)))
    assert len(rows) == report['executed_frames'] + int(report['execution_rejected'])
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,r_frame_rate','-of','json',str(original/'main_rgb.mp4')]))['streams'][0]
    assert int(probe['nb_frames']) == len(rows) and probe['r_frame_rate']=='20/1'
    payload = dict(schema_version='libero_ipopt_selected_viewer.1', key=key,
        episode=dict(demo_id=job['demo'], frame_count=len(rows), frequency_hz=20, result=report,
            is_annotation_demo=False, observation_timing='pre_action', terminal_post_action_image_recorded=True),
        trajectory=frames, events=events, phase_rule=phase_rule, stage_annotations=[], metrics=job['selected'],
        selection=dict(trial=root.name, source_hdf5=job['source_hdf5'], source_input_sha256=job['source_input_sha256']),
        source=dict(processed_hdf5=str(original/'recording.hdf5'), selected_root=str(root)),
        media=dict(video=str(base/'main_rgb.mp4'),poster=str(base/'initial.png'),final=str(base/'final.png')),
        timing_note='Video frame i is pre-command state i. Error curves compare post-command state i+1 with command i target. Rejected row is not executed; its post errors are null. Native cached observations may lag the final physics state by one 2ms tick.',
        mask_note='Frozen task-level semantic predictions only; tolerance off; no temporal mask transfer to reconstructed episodes.')
    trace = base/'viewer_trace.json'
    dump(out/trace, payload)
    script = base/'viewer_trace.js'
    (out/script).write_text('window.LIBERO_EPISODES=window.LIBERO_EPISODES||{};window.LIBERO_EPISODES['+json.dumps(key)+']='+json.dumps(payload,ensure_ascii=False,separators=(',', ':'),allow_nan=False)+';\n')
    return dict(path=job['demo'],key=key,frames=len(rows),success=report['success'],representative=False,
        execution_rejected=report['execution_rejected'],selected_trial=root.name,
        video_path=str(base/'main_rgb.mp4'),poster_path=str(base/'initial.png'),final_path=str(base/'final.png'),
        trace_path=str(trace),trace_script_path=str(script),raw_trace_path=str(base/'trajectory.json'),report_path=str(base/'report.json'),
        source_hdf5=job['source_hdf5'], selected_root=str(root), video_sha256=sha(original/'main_rgb.mp4'),
        trace_sha256=sha(out/trace),raw_trace_sha256=sha(original/'trajectory.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=PROJECT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_fixed_v2')
    ap.add_argument('--viewer',type=Path,default=PROJECT/'data/libero40_mask_annotation/20261019_viewer_10demos_v2')
    args = ap.parse_args()
    current = args.viewer.resolve()
    stage = current.with_name(current.name+'.ipopt-staging')
    if stage.exists(): raise ValueError(f'Staging directory already exists: {stage}')
    stage.mkdir()
    old_catalog = json.loads((current/'viewer_catalog.json').read_text())
    summary = json.loads((args.run/'summary.json').read_text())
    jobs = summary['results']
    assert len(jobs)==400 and len({(j['suite'],j['task_id'],j['demo']) for j in jobs})==400
    # Preserve original image evidence and masks, never modify their hard links.
    for item in current.rglob('*'):
        rel = item.relative_to(current)
        if item.is_file() and rel.parts[0] in ['decision_records','keyframes','review']:
            link_or_copy(item,stage/rel)
    for name in ['mask_annotations.json','mask_annotations.sha256','run_manifest.json','reference_25.md','uncertainties.md']:
        if (current/name).is_file(): link_or_copy(current/name,stage/name)
    assets=stage/'assets';assets.mkdir()
    for name in ['mujoco-studio.js','mujoco-studio.css','adapter.css']:
        link_or_copy(current/'assets'/name,assets/name)
    adapter=(ADAPTER/'adapter.js').read_text()
    adapter=adapter.removesuffix('loadDatasets();\n').removesuffix('requestAnimationFrame(animationLoop);\n')
    assert not adapter.rstrip().endswith('loadDatasets();')
    (assets/'adapter.js').write_text(adapter)
    shutil.copy2(ADAPTER/'ipopt_adapter.js',assets/'ipopt_adapter.js')
    catalog = []
    for old in old_catalog['datasets']:
        entry = dict(old)
        entry['episodes']=[];catalog.append(entry)
    by_task={(t['suite'],t['task_id']):t for t in catalog}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(export,j,by_task[(j['suite'],j['task_id'])],stage) for j in jobs]
        for i,future in enumerate(futures,1):
            item=future.result()
            by_task[tuple(item['key'].split('/')[:1])+ (int(item['key'].split('/')[1]),)]['episodes'].append(item)
            if i%40==0: print(f'{i}/400 media and traces checked',flush=True)
    for task in catalog:
        task['episodes'].sort(key=lambda e:int(e['path'][5:]))
        assert len(task['episodes'])==10
        task['success_count']=sum(e['success'] for e in task['episodes'])
    counts=dict(suites=4,tasks=40,episodes=400,success=sum(t['success_count'] for t in catalog),
        failures=sum(not e['success'] for t in catalog for e in t['episodes']),
        solver_rejected=sum(e['execution_rejected'] for t in catalog for e in t['episodes']))
    assert counts['success']==362 and counts['solver_rejected']==13
    data=dict(schema_version='libero40_ipopt_studio.1',datasets=catalog,counts=counts,
        annotation_sha256=old_catalog['annotation_sha256'],run=str(args.run.resolve()),
        information_scope='Selected untouched-default IPOPT400 recordings; frozen task predictions retained as evidence only; tolerance OFF.')
    dump(stage/'viewer_catalog.json',data)
    (assets/'catalog.js').write_text('window.LIBERO_CATALOG='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    document=(current/'index.html').read_text()
    replacements={
        'LIBERO-40 · TaskTol Dataset Studio':'LIBERO-40 · IPOPT 400 · TaskTol Dataset Studio',
        '每任务1个共享预测 / 10个成功示范':'IPOPT重构 · 362成功 / 38失败 · 每任务10条',
        '保留原代表视频，每任务补充9条成功示范。':'查看最终选定的IPOPT重构结果，每任务10条，成功与失败均保留。',
        '共享预测来自原代表示范；新增9条仅补充视频与轨迹，未逐条视觉标注。标称参考和六方向数值容差未提供。':'共享mask及图片依据来自原OSC代表示范；当前阶段由共享MuJoCo分类器因果重算，未迁移旧语义边界。本批容差关闭，mask未参与执行。',
        '共享mask及图片依据来自原OSC代表示范；当前为IPOPT重构视频，未迁移旧阶段边界。本批容差关闭，mask未参与执行。':'共享mask及图片依据来自原OSC代表示范；当前阶段由共享MuJoCo分类器因果重算，未迁移旧语义边界。本批容差关闭，mask未参与执行。',
        '标称参考可用性':'当前命令的累积参考 · world',
        '当前帧的语义标注状态':'动作后相对参考的6维误差',
        '重力对齐 F_obs · 观测代理':'动作后相对源数据集的误差',
        '原始 OSC_POSE 归一化控制 / grip':'重构动作 Δxyz [m] / Δrot [rad] / grip',
        '源帧 / 对齐':'逐帧求解状态',
        'F_obs 参考':'累积参考',
        '<strong>源帧映射</strong>':'<strong>源数据集位置误差</strong>',
        'observed world position · m':'实际位置 / 当前命令参考虚线 · m',
        'observed axis-angle vector · rad':'实际旋转向量 / 当前命令参考虚线 · rad',
        '当前代表轨迹的有效 mask':'求解耗时',
        '二值 xF / yF / zF；补充demo不伪造逐帧标签':'记录的 elapsed_ms · 墙钟时间，非CPU时间',
        'source_frame − processed_frame · filtered-action gaps':'动作后相对源数据集的位置误差 · mm',
        '归一化OSC_POSE输入':'物理位姿增量',
        'LIBERO OSC_POSE controls':'IPOPT 重构动作',
        '原始归一化控制值 · 共享时间游标':'物理增量控制 · 共享时间游标',
        'normalized ΔX / ΔY / ΔZ':'ΔX / ΔY / ΔZ · m',
        'normalized ΔRx / ΔRy / ΔRz':'ΔRx / ΔRy / ΔRz · rad',
        '空间观测轨迹与重力参考':'实际末端轨迹与累积参考',
    }
    for a,b in replacements.items(): document=document.replace(a,b)
    if 'src="assets/ipopt_adapter.js"' not in document:
        document=document.replace('<script src="assets/adapter.js"></script>',
            '<script src="assets/adapter.js"></script><script src="assets/ipopt_adapter.js"></script>')
    if 'id="postPositionErrorChart"' not in document:
        document=document.replace('<div class="chart-card wide action-card">','''<div class="chart-card"><div><strong>动作后位置误差 XYZ</strong><span>实际 − 当前命令参考 · mm</span></div><canvas id="postPositionErrorChart"></canvas></div>
<div class="chart-card"><div><strong>动作后旋转误差 XYZ</strong><span>Log(R实际 R参考ᵀ) · 世界系 · deg</span></div><canvas id="postRotationErrorChart"></canvas></div>
<div class="chart-card wide action-card">''')
    (stage/'index.html').write_text(document);(stage/'report.html').write_text(document)
    shutil.copy2(Path(__file__).with_name('serve_mask_annotation_viewer.py'),stage/'serve.py')
    phase_rule = json.loads((stage/catalog[0]['episodes'][0]['trace_path']).read_text())['phase_rule']
    dump(stage/'viewer_validation.json',dict(complete=True,counts=counts,phase_rule=phase_rule,checks=[
        '400 unique final-selected episodes; 10 per task', 'all MP4 frame counts equal trajectory row counts at 20Hz',
        'videos and raw trajectories SHA256 indexed; hardlinked unchanged originals',
        'rejected rows have no post-action tracking errors', 'frozen mask bytes retained; no temporal label transfer'],
        source_summary_sha256=sha(args.run/'summary.json')))
    (stage/'README.md').write_text('''# LIBERO-40 IPOPT 最终选择结果

4任务集 × 10任务 × 10条；362成功、25任务未完成、13求解拒绝。
沿用原 MuJoCo Dataset Studio 布局、视频与曲线时间游标。

视频第i帧与动作前状态i对齐；虚线为该命令的累积参考。6维误差比较执行后的状态i+1与该参考，拒绝帧无执行后误差。末帧视频不是执行后终态，终态图片单独提供。
源数据误差使用原记录的同期下一状态误差，不做时间平移。原生缓存观测可能落后物理终态一个2ms tick。
动作单位为m/rad，夹爪保持原指令。20Hz、原速、无额外settling。默认IPOPT未改；完整关节参考，容差关闭。
阶段直接复用当前MuJoCo PhaseClassifier，按20Hz每帧动作前开口及当前指令判定，无降采样、无未来指令、无规划标签覆盖。使用默认连续3个稳定样本后转换，稳定阶段锁存至指令改变。LIBERO相反指符号归一为物理开口。
阶段是观测重算，不是历史执行门控记录；旧录制容差关闭。共享mask与图片证据保留原OSC标注，观测子任务编号不映射为原视觉语义子任务，不表示已启用容差。
ipopt400/内为最终选定的视频、原轨迹、报告、首末图片与查看器适配轨迹。文件为本地硬链接或副本，原录制文件不变。
运行 python3 serve.py --port 8768 --background 后打开 http://127.0.0.1:8768/index.html 。
''')
    backup=current.with_name(current.name+'.osc-backup-'+time.strftime('%Y%m%d-%H%M%S'))
    dump(stage/'viewer_manifest.json',dict(run=str(args.run.resolve()),counts=counts,previous_viewer=str(backup),
        builder_sha256=sha(Path(__file__)),adapter_sha256=sha(ADAPTER/'ipopt_adapter.js'),
        phase_adapter_sha256=sha(ADAPTER/'phase_adapter.py'),phase_rule=phase_rule,
        annotation_unchanged=sha(stage/'mask_annotations.json')==data['annotation_sha256']))
    current.rename(backup)
    try: stage.rename(current)
    except BaseException:
        backup.rename(current)
        raise
    print(f'REPLACED {current}\nBACKUP {backup}\n{counts}',flush=True)


if __name__=='__main__':main()
