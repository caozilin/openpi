"""Audit52 recovery rollouts and summarize bounded action changes and outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def rot(v):return Rotation.from_rotvec(v).as_matrix()
def angle(r):return float(np.linalg.norm(Rotation.from_matrix(r).as_rotvec()))

def validate(entry,out,prior,settings):
    folder=out/'episodes'/entry['key'];report=json.loads((folder/'report.json').read_text())
    old=json.loads((prior/'episodes'/entry['key']/'report.json').read_text())
    rows=json.loads((folder/'trajectory.json').read_text());base=json.loads((Path(entry['baseline_folder'])/'trajectory.json').read_text())
    for k in ('actions_sha256','initial_state_sha256','model_sha256','solver_settings','objective_settings','shared_joint_servo'):assert report[k]==old[k]
    with h5py.File(entry['job']['reference_path'])as f:original=f['rotations'][0]
    corrections=[];combined=[];active=0
    for row,b in zip(rows,base):
        assert row['base_action']==b['action']
        np.testing.assert_array_equal(row['action'][:3],b['action'][:3]);assert row['action'][6]==b['action'][6]
        np.testing.assert_allclose(row['nominal_target']['position_m'],b['nominal_target']['position_m'],atol=1e-10,rtol=0)
        original=rot(row['base_action'][3:6])@original
        np.testing.assert_allclose(rot(row['original_nominal_rotation_vector_rad']),original,atol=1e-10,rtol=0)
        d=row['rotation_recovery'];u=rot(row['action'][3:6]);ub=rot(row['base_action'][3:6])
        correction=np.degrees(angle(u@ub.T));corrections.append(float(correction));combined.append(float(np.degrees(angle(u))))
        if d['active']:
            active+=1;assert row['effective_mask']=='000' and row['observed_phase']=='pregrasp' and row['action'][6]<=0
            assert correction<=settings['correction_speed_deg_s']*.05+1e-6
            if np.degrees(angle(ub))<=settings['combined_speed_deg_s']*.05+1e-6:
                assert np.degrees(angle(u))<=settings['combined_speed_deg_s']*.05+1e-6
            else:np.testing.assert_allclose(u,ub,atol=1e-10,rtol=0)
        else:np.testing.assert_array_equal(row['action'],row['base_action'])
        assert row['physics_steps']==(0 if row['aborted'] else 25)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'main_rgb.mp4')]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(rows) and probe['r_frame_rate']=='20/1'
    decoded=subprocess.run(['ffmpeg','-v','error','-i',str(folder/'main_rgb.mp4'),'-f','null','-'],capture_output=True)
    assert decoded.returncode==0 and not decoded.stderr
    recovery=report['rotation_recovery']
    for event in recovery['events']:
        if event['completed']:
            end=rows[event['end_frame']];assert not end['aborted']
            assert event['residual_deg']<=settings['completion_threshold_deg']+1e-6
    return dict(key=entry['key'],task_id=entry['task']['task_id'],success=report['success'],rejected=report['execution_rejected'],
        executed_frames=report['executed_frames'],planned_frames=report['planned_frames'],
        recovery=recovery,active_frames=active,max_added_correction_deg=max(corrections),
        max_combined_action_deg=max(combined),cpu_timeout_frames=report['cpu_timeout_frames'],video_decoded=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    plan=json.loads((out/'plan.json').read_text());prior=Path(plan['prior_run'])
    with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(lambda e:validate(e,out,prior,plan['settings']),plan['entries']))
    events=[e for r in rows for e in r['recovery']['events']];completed=[e for e in events if e['completed']]
    old=json.loads((prior/'summary.json').read_text())
    result=dict(complete=len(rows)==52,tested=len(rows),recovered_success=sum(r['success']for r in rows),
        still_failed=sum(not r['success']for r in rows),rejected=sum(r['rejected']for r in rows),
        cpu_timeout_frames=sum(len(r['cpu_timeout_frames'])for r in rows),all_videos_decoded=True,
        recovery_events=len(events),completed_recovery_events=len(completed),
        no_recovery_window_episodes=[r['key']for r in rows if not r['recovery']['events']],
        completed_recovery_duration_s=dict(mean=float(np.mean([e['elapsed_s']for e in completed])),max=max(e['elapsed_s']for e in completed))if completed else None,
        max_added_angular_speed_deg_s=max(r['max_added_correction_deg']for r in rows)/.05,
        tasks={str(i):dict(tested=sum(r['task_id']==i for r in rows),recovered_success=sum(r['task_id']==i and r['success']for r in rows),
            original_success=sum(r['suite']=='libero_10'and r['task_id']==i and r['success']for r in old['results']))for i in range(10)},
        scope='Only52 previously failed Long episodes re-tested. Original48 successful Long episodes retained, not rerun with recovery.',episodes=rows)
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    text=['# 长任务失败样本：松爪姿态恢复复测','',f"固定52条原容差失败样本，一次回放。新增成功{result['recovered_success']}条，仍失败{result['still_failed']}条；求解拒绝{result['rejected']}条，CPU超时{result['cpu_timeout_frames']}帧。",'',
        '原先成功的48条长任务保留，未在新恢复规则下重新测试；不能把保留原成功加复测成功视为100条统一新策略的独立测评。','',
        '|Long任务ID|原成功/10|本次失败复测|新增成功|保留原成功＋新增/10|','|---|---:|---:|---:|---:|']
    for i,t in result['tasks'].items():text.append(f"|{int(i):02d}|{t['original_success']}|{t['tested']}|{t['recovered_success']}|{t['original_success']+t['recovered_success']}|")
    text+=['',f"恢复事件{len(events)}次，其中参考恢复完成{len(completed)}次。完成耗时：{result['completed_recovery_duration_s']}。无恢复窗口{len(result['no_recovery_window_episodes'])}条。",'',
        '参数：20Hz；附加纠偏45°/s，合成45°/s；原动作已超过合成上限时保留原动作且暂停纠偏；松爪后开口至少70mm并满足共享PRE稳定判定才启动。三次平滑曲线；0.5°参考阈值，2秒仅提示；下一次闭爪停止恢复。恢复期间暂停新增容差，其余阶段mask不变。',
        '', '平移、夹爪、初始状态、物理模型、默认IPOPT设置/目标、共享关节伺服均核对；独立原始名义姿态始终保留。姿态恢复完成只表示参考偏差被修复，不保证放置、抓取或实际动力学误差已经解决。']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='episodes'},ensure_ascii=False))

if __name__=='__main__':main()
