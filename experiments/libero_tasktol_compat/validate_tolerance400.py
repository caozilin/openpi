"""Validate paired execution provenance, media alignment and outcome summaries."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

def stats(values):
    a=np.asarray(values,dtype=float)
    return None if not len(a) else dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))

def validate(entry,out):
    folder=out/'episodes'/entry['key'];baseline=Path(entry['baseline_folder'])
    report=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'trajectory.json').read_text())
    off_report=json.loads((baseline/'report.json').read_text());off_rows=json.loads((baseline/'trajectory.json').read_text())
    assert report['actions_sha256']==off_report['actions_sha256']==entry['baseline_actions_sha256']
    assert report['initial_state_sha256']==off_report['initial_state_sha256']
    assert report['model_sha256']==off_report['model_sha256']
    assert report['solver_settings']==off_report['solver_settings']
    assert report['objective_settings']==off_report['objective_settings']
    assert report['shared_joint_servo']==off_report['shared_joint_servo']
    assert report['planned_frames']==len(off_rows)
    executed=[r for r in rows if not r['aborted']]
    assert len(executed)==report['executed_frames']
    if report['success']:assert len(executed)==len(off_rows) and not report['execution_rejected']
    for r,b in zip(rows,off_rows):
        assert r['action']==b['action'] and r['bounds_rad']==[np.deg2rad(30)]*6
        assert r['physics_steps']==(0 if r['aborted'] else 25)
        if r['phase'] not in ('APP','EXEC'):assert r['effective_mask']=='000'
        assert all(a=='0' or b=='1'for a,b in zip(r['effective_mask'],r['semantic_mask'],strict=True))
        assert np.all(np.isfinite(r['joint_target_rad']))
    with np.load(folder/'states.npz')as f:
        states=np.concatenate([f['pre_action_states'][:len(executed)],f['final_state'][None]])
        np.testing.assert_allclose(np.diff(states[:,0]),.05,atol=1e-9,rtol=0)
        with np.load(baseline/'states.npz')as b:np.testing.assert_array_equal(states[0],b['pre_action_states'][0])
        all_zero=all(s['APP']=='000' and s['EXEC']=='000'for s in entry['task']['semantic_subtasks'])
        exact=None;zero_comparison=None
        if all_zero:
            with np.load(baseline/'states.npz')as b:exact=bool(np.array_equal(f['pre_action_states'],b['pre_action_states'])and np.array_equal(f['final_state'],b['final_state']))
            zero_comparison=dict(max_joint_target_difference_rad=float(np.max(np.abs(np.asarray([r['joint_target_rad']for r in rows])-np.asarray([r['joint_target_rad']for r in off_rows])))),
                max_actual_eef_difference_mm=float(1000*np.max(np.linalg.norm(np.asarray([r['post_action']['position_m']for r in rows])-np.asarray([r['post_action']['position_m']for r in off_rows]),axis=1))))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(folder/'main_rgb.mp4')]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(rows) and probe['r_frame_rate']=='20/1'
    decoded=subprocess.run(['ffmpeg','-v','error','-i',str(folder/'main_rgb.mp4'),'-f','null','-'],capture_output=True)
    assert decoded.returncode==0 and not decoded.stderr,decoded.stderr.decode()
    diagnostics=json.loads((folder/'solver_observations.json').read_text())
    stages=defaultdict(list)
    for r in rows:stages[(r.get('semantic_subtask_index'),r['phase'],r['observed_subtask_index'])].append(r)
    stage_stats=[]
    for (semantic,phase,cycle),group in stages.items():
        coords=np.degrees(np.asarray([r.get('rotation_release_rad',[0,0,0])for r in group]))
        stage_stats.append(dict(semantic_subtask_index=semantic,phase=phase,cycle=cycle,frames=len(group),
            max_absolute_release_xyz_deg=np.max(np.abs(coords),axis=0).tolist(),
            cumulative_absolute_coordinate_change_xyz_deg=np.sum(np.abs(np.diff(np.vstack([np.zeros(3),coords]),axis=0)),axis=0).tolist(),
            final_release_xyz_deg=coords[-1].tolist()))
    def qmetrics(items):
        accepted=[r for r in items if not r['aborted']]
        if not accepted:return {}
        actual=np.asarray([accepted[0]['pre_action']['joint_position_rad']]+[r['post_action']['joint_position_rad']for r in accepted])
        targets=np.asarray([accepted[0]['pre_action']['joint_position_rad']]+[r['joint_target_rad']for r in accepted])
        return dict(actual_mean_speed_l2_rad_s=float(np.mean(np.linalg.norm(np.diff(actual,axis=0)*20,axis=1))),
            target_mean_speed_l2_rad_s=float(np.mean(np.linalg.norm(np.diff(targets,axis=0)*20,axis=1))),
            actual_joint_path_l2_rad=float(np.sum(np.linalg.norm(np.diff(actual,axis=0),axis=1))))
    return dict(key=entry['key'],case=entry['case'],suite=entry['task']['suite'],task_id=entry['task']['task_id'],
        demo=entry['job']['demo'],success=report['success'],rejected=report['execution_rejected'],
        rejected_frame=report['rejected_frame'],executed_frames=len(executed),planned_frames=len(off_rows),
        tracking=report['tracking'],baseline_tracking=off_report['tracking'],
        joint_motion=qmetrics(rows),baseline_joint_motion=qmetrics(off_rows),
        phase_release=stage_stats,solver_status_counts=report['solver_status_counts'],cpu_timeout_frames=report['cpu_timeout_frames'],
        max_ik_wall_ms=max(d['ik_wall_ms']for d in diagnostics),max_ik_cpu_ms=max(d['ik_cpu_ms']for d in diagnostics),
        max_optimized_violation_deg=max([float(np.degrees(max(r.get('predicted_tolerance_violation_rad',[0]))))for r in executed]or[0.]),
        max_actual_violation_deg=max([float(np.degrees(max(r.get('actual_tolerance_violation_rad',[0]))))for r in executed]or[0.]),
        full_video_decoded=True,all_zero_masks=all_zero,zero_mask_states_bitwise_equal=exact,zero_mask_comparison=zero_comparison,
        native_predicates=report['native_predicates'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=a.output.resolve();plan=json.loads((out/'plan.json').read_text())
    with ThreadPoolExecutor(max_workers=a.workers)as pool:rows=list(pool.map(lambda e:validate(e,out),plan['entries']))
    suites={s:dict(total=sum(r['suite']==s for r in rows),success=sum(r['suite']==s and r['success']for r in rows))for s in sorted({r['suite']for r in rows})}
    status=Counter()
    for r in rows:status.update(r['solver_status_counts'])
    result=dict(complete=len(rows)==400,episodes=len(rows),success=sum(r['success']for r in rows),
        failures=sum(not r['success']for r in rows),rejected=sum(r['rejected']for r in rows),suites=suites,
        cpu_timeout_frames=sum(len(r['cpu_timeout_frames'])for r in rows),solver_status_counts=dict(status),
        all_videos_fully_decoded=all(r['full_video_decoded']for r in rows),
        max_ik_wall_ms=max(r['max_ik_wall_ms']for r in rows),max_ik_cpu_ms=max(r['max_ik_cpu_ms']for r in rows),
        max_optimized_violation_deg=max(r['max_optimized_violation_deg']for r in rows),
        max_actual_violation_deg=max(r['max_actual_violation_deg']for r in rows),
        zero_mask_bitwise_checks=[dict(key=r['key'],equal=r['zero_mask_states_bitwise_equal'],metrics=r['zero_mask_comparison'])for r in rows if r['all_zero_masks']],
        source_pose_tracking_mean_of_episode_means=dict(position_mm=stats([r['tracking']['source_position_mm']['mean']for r in rows if r['tracking']['source_position_mm']]),
            rotation_deg=stats([r['tracking']['source_rotation_deg']['mean']for r in rows if r['tracking']['source_rotation_deg']])),
        evaluation='One pass per existing selected demo, fixed actions and original speed. Native final predicate plus full feasible execution; all failures retained. Not VLA policy inference or new data reconstruction.',
        episodes_detail=rows)
    tmp=out/'validation.json.partial';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(out/'validation.json')
    print(json.dumps({k:v for k,v in result.items()if k!='episodes_detail'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
