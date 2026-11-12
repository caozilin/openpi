"""Build a local Studio containing only successful shared-servo rollouts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from build_mask_annotation_viewer import PROJECT, ADAPTER, dump, link_or_copy, sha

RUN=PROJECT/'data/libero_tasktol_joint_servo/20261025_success10_v1'
TEMPLATE=PROJECT/'data/libero40_mask_annotation/20261019_viewer_10demos_v2'


def source_video(source_hdf5,demo,cache,rollout_frames):
    """Encode original saved observations, never OSC regeneration or state replay."""
    import imageio.v2 as imageio
    from PIL import Image
    manifest=cache/'source_video.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        assert saved['source_hdf5']==source_hdf5 and saved['demo']==demo
        assert saved['frame_count']==rollout_frames+1
        assert saved['video_sha256']==sha(cache/'source_rgb.mp4')
        return saved
    with h5py.File(source_hdf5)as f:
        g=f['data'][demo];images=g['obs/agentview_rgb'][()];states=g['states'][()]
    assert len(images)==len(states)==rollout_frames+1
    np.testing.assert_allclose(np.diff(states[:,0]),.05,rtol=0,atol=1e-10)
    digest=hashlib.sha256(images.tobytes()).hexdigest()
    cache.mkdir(parents=True,exist_ok=True)
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        assert saved['source_images_sha256']==digest and saved['frame_count']==len(images)
        assert saved['video_sha256']==sha(cache/'source_rgb.mp4')
        return saved
    # Same display rotation as the reconstructed video. Native image pixels
    # remain untouched in the HDF5, and all original frames are retained here.
    temp=cache/'source_rgb.partial.mp4'
    with imageio.get_writer(temp,fps=20,codec='libx264',quality=8,macro_block_size=1,output_params=['-threads','1'])as video:
        for frame in images:video.append_data(np.ascontiguousarray(frame[::-1,::-1]))
    temp.rename(cache/'source_rgb.mp4')
    Image.fromarray(images[0,::-1,::-1]).save(cache/'source_initial.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(cache/'source_rgb.mp4')]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(images) and probe['r_frame_rate']=='20/1'
    saved=dict(source_hdf5=source_hdf5,demo=demo,image_dataset=f'data/{demo}/obs/agentview_rgb',
        source_images_sha256=digest,video_sha256=sha(cache/'source_rgb.mp4'),frame_count=len(images),
        resolution_hw=list(images.shape[1:3]),frequency_hz=20,display_transform='rotate180',
        provenance='direct original HDF5 stored RGB images; no physics/action replay',
        alignment='original frame i corresponds to reconstructed pre-action frame i; final original frame has no reconstructed command')
    dump(manifest,saved);return saved


def comparison_frame(tr):
    return dict(phase=tr['phase'],semantic_subtask_index=tr.get('semantic_subtask_index'),
        observed_subtask_index=tr['observed_subtask_index'],effective_mask=tr['effective_mask'],
        semantic_mask=tr['semantic_mask'],aborted=tr['aborted'],optimizer=tr['optimizer'],
        source_next_position_error_mm=tr.get('source_next_position_error_mm'),
        source_next_rotation_error_deg=tr.get('source_next_rotation_error_deg'),
        tracking_position_error_mm=tr['tracking_position_error_m']*1000,
        tracking_optimized_rotation_error_deg=float(np.degrees(tr['tracking_rotation_error_rad'])),
        rotation_release_deg=np.degrees(tr.get('rotation_release_rad',[0,0,0])).tolist(),
        actual_rotation_release_deg=np.degrees(tr.get('actual_rotation_release_rad',[0,0,0])).tolist(),
        predicted_violation_deg=np.degrees(tr.get('predicted_tolerance_violation_rad',[0,0,0])).tolist(),
        actual_violation_deg=np.degrees(tr.get('actual_tolerance_violation_rad',[0,0,0])).tolist(),
        joint_target_rad=tr['joint_target_rad'],post_action=tr['post_action'],
        action=tr['action'],base_action=tr.get('base_action',tr['action']),rotation_recovery=tr.get('rotation_recovery'),
        position_error_xyz_mm=((np.asarray(tr['post_action']['position_m'])-tr['nominal_target']['position_m'])*1000).tolist(),
        rotation_error_xyz_deg=np.degrees((Rotation.from_rotvec(tr['post_action']['rotation_vector_rad'])*Rotation.from_rotvec(tr['nominal_target']['rotation_vector_rad']).inv()).as_rotvec()).tolist())

def export(record,out,source_cache,tolerance_run=None,recovery_run=None):
    original=Path(record['folder']);report=json.loads((original/'report.json').read_text())
    rows=json.loads((original/'trajectory.json').read_text())
    assert record['success'] and report['success'] and not report['execution_rejected']
    assert len(rows)==report['executed_frames']==report['planned_frames']
    key=f"{record['suite']}/{record['task_id']:02d}/{record['demo']}";base=Path('rollouts')/key
    for name in ['main_rgb.mp4','initial.png','final.png','trajectory.json','report.json']:
        link_or_copy(original/name,out/base/name)
    with h5py.File(original/'state_reference.hdf5')as f:
        source_hdf5=str(f.attrs['source'])
        source_p=f['source_positions_m'][()];source_r=f['source_rotations'][()]
        actions=f['actions'][()]
    source_info=None
    if not tolerance_run:
        source_info=source_video(source_hdf5,record['demo'],source_cache/base,len(rows))
        for name in ['source_rgb.mp4','source_initial.png','source_video.json']:
            link_or_copy(source_cache/base/name,out/base/name)
    from prepare_rollout_wrist_videos import export_wrist
    export_wrist(original,record['demo'],len(rows),out/base/'wrist_rgb.mp4')
    tolerance=None;tolerance_rows=[];recovery=None;recovery_rows=[]
    if tolerance_run:
        tol_folder=tolerance_run/'episodes'/key
        if (tol_folder/'report.json').is_file():
            tol_report=json.loads((tol_folder/'report.json').read_text())
            tolerance_rows=json.loads((tol_folder/'trajectory.json').read_text())
            assert tol_report['actions_sha256']==report['actions_sha256']
            assert tol_report['initial_state_sha256']==report['initial_state_sha256']
            for name in ['main_rgb.mp4','initial.png','final.png','trajectory.json','report.json']:
                link_or_copy(tol_folder/name,out/base/'tolerance'/name)
            if (tol_folder/'wrist_rgb.mp4').is_file():
                link_or_copy(tol_folder/'wrist_rgb.mp4',out/base/'tolerance/wrist_rgb.mp4')
            tolerance=dict(result=tol_report,folder=str(tol_folder),frames=len(tolerance_rows),
                video=str(base/'tolerance/main_rgb.mp4'),wrist_video=str(base/'tolerance/wrist_rgb.mp4') if (tol_folder/'wrist_rgb.mp4').is_file() else None,poster=str(base/'tolerance/initial.png'),
                report=str(base/'tolerance/report.json'),trajectory=str(base/'tolerance/trajectory.json'))
            if tol_report.get('tolerance') == 'offline_annotated_signed_bounds':
                label_path=Path(tol_report['label_source'])
                assert sha(label_path)==tol_report['label_source_sha256']
                tolerance['label_subtasks']=json.loads(label_path.read_text())['subtasks']
    if recovery_run:
        folder=recovery_run/'episodes'/key
        if (folder/'report.json').is_file():
            r=json.loads((folder/'report.json').read_text());recovery_rows=json.loads((folder/'trajectory.json').read_text())
            assert r['actions_sha256']==report['actions_sha256'] and r['initial_state_sha256']==report['initial_state_sha256']
            for name in ['main_rgb.mp4','initial.png','final.png','trajectory.json','report.json']:
                link_or_copy(folder/name,out/base/'recovery'/name)
            if (folder/'wrist_rgb.mp4').is_file():
                link_or_copy(folder/'wrist_rgb.mp4',out/base/'recovery/wrist_rgb.mp4')
            recovery=dict(result=r,folder=str(folder),frames=len(recovery_rows),
                video=str(base/'recovery/main_rgb.mp4'),poster=str(base/'recovery/initial.png'),
                wrist_video=str(base/'recovery/wrist_rgb.mp4') if (folder/'wrist_rgb.mp4').is_file() else None,
                report=str(base/'recovery/report.json'),trajectory=str(base/'recovery/trajectory.json'))
    frames=[];events=[]
    def pose(p):return dict(position_m=p['position_m'],rotation_vector_base_rad=p['rotation_vector_rad'])
    for i,row in enumerate(rows):
        assert not row['aborted'] and row['physics_steps']==25
        np.testing.assert_array_equal(row['action'],actions[i])
        before=row['pre_action'];after=row['post_action'];target=row['nominal_target']
        phase=row['observed_phase'];cycle=row['observed_subtask_index']
        frame=dict(index=i,video_frame_index=i,time_s=i/20,source_frame=i,source_gap_before=0,
            phase=phase,observed_subtask_index=cycle,phase_source='recorded_online_shared_classifier',
            subtask_index=None,execution_mask='000',action=row['action'],ee_state=pose(before),
            nominal_target_pose=pose(target),post_action_pose=pose(after),
            source_pose=dict(position_m=source_p[i].tolist(),rotation_vector_base_rad=Rotation.from_matrix(source_r[i]).as_rotvec().tolist()),
            joint_state=dict(position_rad=before['joint_position_rad'],
                velocity_rad_s=((np.asarray(before['joint_position_rad'])-np.asarray(rows[i-1]['pre_action']['joint_position_rad']))*20).tolist() if i else [None]*7),
            joint_target_rad=row['joint_target_rad'],gripper_state=dict(finger_qpos_m=before['finger_qpos_m'],
                aperture_m=before['finger_qpos_m'][0]-before['finger_qpos_m'][1],command=row['action'][6]),
            optimizer=row['optimizer'],aborted=False,
            post_target_error_position_mm=((np.asarray(after['position_m'])-target['position_m'])*1000).tolist(),
            post_target_error_rotation_deg=np.degrees((Rotation.from_rotvec(after['rotation_vector_rad'])*Rotation.from_rotvec(target['rotation_vector_rad']).inv()).as_rotvec()).tolist(),
            source_next_position_error_mm=row['source_next_position_error_mm'],source_next_rotation_error_deg=row['source_next_rotation_error_deg'],
            observed_gravity_frame=dict(rotation_matrix=None,h=None))
        if tolerance_rows and i<len(tolerance_rows):
            frame['tolerance']=comparison_frame(tolerance_rows[i])
        if recovery_rows and i<len(recovery_rows):frame['recovery']=comparison_frame(recovery_rows[i])
        frames.append(frame)
        if i==0 or phase!=frames[i-1]['phase'] or cycle!=frames[i-1]['observed_subtask_index']:
            events.append(dict(frame=i,time_s=i/20,phase=phase,observed_subtask_index=cycle))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(original/'main_rgb.mp4')]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(rows) and probe['r_frame_rate']=='20/1'
    payload=dict(schema_version='libero_reconstructed_joint_servo_viewer.1',key=key,
        episode=dict(demo_id=record['demo'],frame_count=len(rows),frequency_hz=20,result=report,observation_timing='pre_action'),
        trajectory=frames,events=events,phase_rule=report['phase_rule'],metrics=record['metrics'],
        selection=dict(trial=f"iteration_{record['selected_iteration']}",source_hdf5=source_hdf5,
            independent_replay_bitwise_identical=record.get('independent_replay_bitwise_identical',False)),
        reconstruction_trials=[{k:t[k] for k in ['iteration','gain','success','accepted','tracking'] if k in t}for t in record['trials'] if not t.get('skipped')],
        source=dict(recording=str(original/'recording.hdf5'),selected_root=str(original)),
        source_comparison=source_info,tolerance=tolerance,recovery=recovery,
        media=dict(video=str(base/'main_rgb.mp4'),poster=str(base/'initial.png'),final=str(base/'final.png'),
            source_video=str(base/'source_rgb.mp4') if source_info else None,
            source_poster=str(base/'source_initial.png') if source_info else None))
    trace=base/'viewer_trace.json';dump(out/trace,payload)
    script=base/'viewer_trace.js';(out/script).write_text('window.LIBERO_EPISODES=window.LIBERO_EPISODES||{};window.LIBERO_EPISODES['+json.dumps(key)+']='+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n')
    return dict(path=record['demo'],key=key,frames=len(rows),success=True,representative=False,execution_rejected=False,
        selected_trial=f"iteration_{record['selected_iteration']}",video_path=str(base/'main_rgb.mp4'),wrist_video_path=str(base/'wrist_rgb.mp4'),poster_path=str(base/'initial.png'),
        final_path=str(base/'final.png'),trace_path=str(trace),trace_script_path=str(script),raw_trace_path=str(base/'trajectory.json'),
        report_path=str(base/'report.json'),selected_root=str(original),video_sha256=sha(original/'main_rgb.mp4'),
        source_video_path=str(base/'source_rgb.mp4') if source_info else None,
        source_poster_path=str(base/'source_initial.png') if source_info else None,source_comparison=source_info,
        tolerance=tolerance,recovery=recovery)


def target_per_task(entry, policy):
    return policy.get('target_successes_per_task') or len(entry['candidates'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--viewer',type=Path)
    p.add_argument('--tolerance-run',type=Path);p.add_argument('--recovery-run',type=Path)
    p.add_argument('--source-index',type=Path,help='Restrict catalog to the exact audited LeRobot source list')
    a=p.parse_args()
    if a.recovery_run:assert a.tolerance_run,'Recovery comparison requires the original tolerance run'
    run=a.run.resolve();current=(a.viewer or run/'viewer').resolve();stage=current.with_name(current.name+'.staging')
    if stage.exists():raise RuntimeError(f'Existing staging directory: {stage}')
    plan=json.loads((run/'plan.json').read_text());all_records=[json.loads(p.read_text())for p in sorted((run/'records').glob('*/*.json'))]
    records=[r for r in all_records if r.get('success')]
    assert records and len({(r['suite'],r['task_id'],r['demo'])for r in records})==len(records)
    if a.source_index:
        source_index=json.loads(a.source_index.read_text())
        expected={(r['suite'],r['task_id'],r['demo']) for r in source_index['sources']}
        assert {(r['suite'],r['task_id'],r['demo']) for r in records}==expected, 'Viewer differs from audited dataset sources'
    target_count=sum(target_per_task(t, plan['policy']) for t in plan['tasks'])
    stage.mkdir(parents=True);assets=stage/'assets';assets.mkdir()
    for name in ['mujoco-studio.js','mujoco-studio.css','adapter.css']:shutil.copy2(TEMPLATE/'assets'/name,assets/name)
    for name in ['adapter.js','ipopt_adapter.js']:
        text=(ADAPTER/name).read_text();text=text.removesuffix('loadDatasets();\n').removesuffix('requestAnimationFrame(animationLoop);\n')
        (assets/name).write_text(text)
    shutil.copy2(ADAPTER/'rollout_adapter.js',assets/'rollout_adapter.js')
    if a.tolerance_run:
        path=assets/'rollout_adapter.js'
        path.write_text(path.read_text().removesuffix('loadDatasets();\n'))
    shutil.copy2(ADAPTER/'rollout_comparison.css',assets/'rollout_comparison.css')
    catalog=[]
    for entry in plan['tasks']:
        attempts=[r for r in all_records if r['case']==entry['case']]
        t=entry['task'];catalog.append(dict(relative_path=f"{t['suite']}/{t['task_id']:02d}",suite=t['suite'],task_id=t['task_id'],
            completed_attempts=len(attempts),failed_attempts=sum(not r.get('success')for r in attempts),target_episode_count=target_per_task(entry, plan['policy']),
            task_name=t['task_name'],display_name=t['language'],frequency_hz=20,episodes=[],episode_count=0,success_count=0,
            shared_annotation=dict(subtasks=[])))
    by_task={(t['suite'],t['task_id']):t for t in catalog}
    with ThreadPoolExecutor(max_workers=4)as pool:
        for record,episode in zip(records,pool.map(lambda r:export(r,stage,run/'source_video_cache',a.tolerance_run,a.recovery_run),records),strict=True):
            by_task[(record['suite'],record['task_id'])]['episodes'].append(episode)
    for t in catalog:
        t['episodes'].sort(key=lambda e:int(e['path'][5:]));t['success_count']=t['episode_count']=len(t['episodes']);assert len(t['episodes'])<=t['target_episode_count']
    counts=dict(suites=4,tasks=40,episodes=len(records),success=len(records),failures=0,solver_rejected=0,
        tasks_complete=sum(t['completed_attempts']==t['target_episode_count'] for t in catalog),target_episodes=target_count)
    if a.tolerance_run:
        confirmed=json.loads((a.tolerance_run/'confirmed_masks.json').read_text())
        for t,m in zip(catalog,confirmed['tasks'],strict=True):t['confirmed_masks']=m['subtasks']
        tolerances=[e['tolerance']for t in catalog for e in t['episodes']if e['tolerance']]
        counts.update(tolerance_completed=len(tolerances),tolerance_success=sum(e['result']['success']for e in tolerances),
            tolerance_failed=sum(not e['result']['success']for e in tolerances))
    if a.recovery_run:
        recovered=[e['recovery']for t in catalog for e in t['episodes']if e['recovery']]
        counts.update(recovery_completed=len(recovered),recovery_success=sum(e['result']['success']for e in recovered),
            recovery_failed=sum(not e['result']['success']for e in recovered),recovery_target=52)
    data=dict(schema_version='libero_reconstructed_rollout_studio.1',datasets=catalog,counts=counts,run=str(run),source_index=str(a.source_index.resolve()) if a.source_index else None,
        information_scope='Successful shared MuJoCo joint-servo reconstructed rollouts only; no historical B10/OSC recordings.')
    annotated = bool(a.tolerance_run and (a.tolerance_run/'plan.json').exists()
                     and json.loads((a.tolerance_run/'plan.json').read_text()).get('schema')=='libero.annotated_tolerance_dataset.1')
    if annotated:
        data['tolerance_kind']='offline_annotated_signed_bounds'
        data['tolerance_target']=1851
    dump(stage/'viewer_catalog.json',data);(assets/'catalog.js').write_text('window.LIBERO_CATALOG='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    document=(TEMPLATE/'index.html').read_text()
    document=document.replace('LIBERO-40 · IPOPT 400 · TaskTol Dataset Studio','LIBERO · 重构 Rollout · TaskTol Dataset Studio')
    document=document.replace('任务集 · 共 400 段视频',f'重构 Rollout · {len(records)} 段成功视频')
    document=document.replace('IPOPT重构 · 362成功 / 38失败 · 每任务10条',f'共享运动生成器 · {len(records)}/{target_count} 成功轨迹')
    document=document.replace('查看最终选定的IPOPT重构结果，每任务10条，成功与失败均保留。','仅查看 MuJoCo 共享关节运动生成器的重构成功轨迹，未完成任务显示实际条数。')
    start=document.index('<section class="panel shared-mask-panel">');end=document.index('<section class="panel video-panel">',start)
    document=document[:start]+'''<section class="panel"><div class="panel-title-row"><div><span class="section-index">00</span><h2>重构与执行记录</h2></div><span id="sharedScope" class="status-chip">共享关节运动生成器</span></div><div id="sharedMaskSummary" class="shared-summary"></div><div id="subtaskTabs" class="subtask-tabs hidden"></div><details id="reasoningDetails"><summary>展开参考修正与候选选择记录</summary><div id="reasoningContent"></div></details></section>\n'''+document[end:]
    document=document.replace('<script src="assets/ipopt_adapter.js"></script>','<script src="assets/ipopt_adapter.js"></script><script src="assets/rollout_adapter.js"></script>')
    document=document.replace('</head>','<link rel="stylesheet" href="assets/rollout_comparison.css">\n</head>')
    document=document.replace('<h2>主视角视频 · 20 Hz</h2>','<h2>重构 / 原始数据集 · 主视角 · 20 Hz</h2>')
    document=document.replace('<div class="video-grid">','<div class="comparison-toolbar"><label><input id="compareSource" type="checkbox" checked> 并排对比原始数据</label><span id="comparisonStatus">同一 demo · 同帧号 · 共用时间轴</span></div><div class="video-grid rollout-comparison">')
    document=document.replace('<figcaption>agentview · 观测先于动作</figcaption></figure>',
        '<figcaption>重构 rollout · 256×256 · 动作前观测</figcaption></figure><figure id="sourceFigure"><video id="sourceVideo" preload="metadata" muted playsinline></video><figcaption>原始数据集 · 128×128 · 保存图像直接导出</figcaption></figure>')
    document=document.replace('<button id="downloadCsv">','<a id="openSourceVideo" target="_blank" rel="noopener">打开原始视频</a><button id="downloadCsv">')
    if a.tolerance_run:
        document=document.replace('<h2>重构 / 原始数据集 · 主视角 · 20 Hz</h2>','<h2>原始 / 无容差 / 30°容差 · 主视角 · 20 Hz</h2>')
        document=document.replace('<div class="video-grid rollout-comparison">','<div class="video-grid rollout-comparison triple-comparison">')
        document=document.replace('<figcaption>重构 rollout · 256×256 · 动作前观测</figcaption>',
            '<figcaption>无容差重构 · 256×256 · 动作前观测</figcaption>')
        document=document.replace('<figure id="sourceFigure">','<figure id="sourceFigure">')
        marker='<figcaption>原始数据集 · 128×128 · 保存图像直接导出</figcaption></figure>'
        document=document.replace(marker,marker+'<figure id="toleranceFigure"><video id="toleranceVideo" preload="metadata" muted playsinline></video><figcaption id="toleranceCaption">30°容差 rollout</figcaption><span id="toleranceVideoState"></span></figure>')
        document=document.replace('<script src="assets/rollout_adapter.js"></script>',
            '<script src="assets/rollout_adapter.js"></script><script src="assets/tolerance_comparison.js"></script>')
        shutil.copy2(ADAPTER/'tolerance_comparison.js',assets/'tolerance_comparison.js')
        if annotated:
            shutil.copy2(ADAPTER/'annotated_tolerance_comparison.js',assets/'annotated_tolerance_comparison.js')
            document=document.replace('</body>','<script src="assets/annotated_tolerance_comparison.js"></script>\n</body>')
            document=document.replace('原始 / 无容差 / 30°容差','原始 / 无容差 / 标注容差')
        document=document.replace('<h2>重构与执行记录</h2>','<h2>任务共享 mask 与容差对比记录</h2>')
    shutil.copy2(ADAPTER/'full_dataset.js', assets/'full_dataset.js')
    shutil.copy2(ADAPTER/'training_selection.js', assets/'training_selection.js')
    document=document.replace('</body>', '<script src="assets/training_selection.js"></script><script src="assets/full_dataset.js"></script>\n</body>')
    (stage/'index.html').write_text(document)
    if a.tolerance_run:
        from install_dataset_release_viewer import install
        install(stage)
        from install_rollout_quad_viewer import install as install_quad
        install_quad(stage)
    shutil.copy2(PROJECT/'examples/libero/serve_mask_annotation_viewer.py',stage/'serve.py')
    dump(stage/'viewer_validation.json',dict(complete=True,counts=counts,all_selected_successful=True,all_video_frames_decoded=True,
        recorded_online_phases=True,source_run=str(run),builder_sha256=sha(Path(__file__)),
        all_sources_direct_original_hdf5=not bool(a.tolerance_run),source_frame_alignment='same demo and frame index,20Hz; nominal and tolerance main/wrist' if a.tolerance_run else 'same demo and frame index,20Hz; original has one extra final frame'))
    if a.tolerance_run and (a.tolerance_run/'validation.json').exists():
        checked=json.loads((a.tolerance_run/'validation.json').read_text())
        assert checked['complete'] and checked['episodes']==counts['tolerance_completed']
        dump(stage/'tolerance_validation.json',checked)
    (stage/'README.md').write_text('仅展示共享MuJoCo关节运动生成器重构后的成功rollout。\n运行 python3 serve.py --port 8768。\n新增成功数据后重新运行 examples/libero/build_reconstructed_rollout_viewer.py 刷新目录。\n')
    if current.exists():current.rename(current.with_name(current.name+'.backup-'+time.strftime('%Y%m%d-%H%M%S')))
    stage.rename(current);print(json.dumps(dict(viewer=str(current),counts=counts),ensure_ascii=False))


if __name__=='__main__':main()
