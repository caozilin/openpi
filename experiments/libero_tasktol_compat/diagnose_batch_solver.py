"""Audit solve termination separately from independent candidate feasibility.

Reads immutable rollout evidence; performs FK/Jacobian queries but no physics
steps, candidate selection or solver parameter changes.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'franka_mujoco')]


def termination(status,code=None):
    if code == -4 or 'Maximum CPU time exceeded' in status:return 'cpu_time_limit'
    if code == -1 or 'Maximum number of iterations exceeded' in status:return 'iteration_limit'
    if code in (0,1,6):return 'converged_or_feasible'
    if 'Algorithm terminated successfully' in status or 'Solved To Acceptable Level' in status:return 'converged_or_feasible'
    return 'other'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--partial',action='store_true');args=p.parse_args()
    import h5py
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tasktol.control.ipopt import IpoptSettings
    run=args.run.resolve();summary=json.loads((run/'summary.json').read_text())
    if not args.partial:assert summary['complete'] and summary['process_errors']==0
    roots=sorted({Path(t['root']) for job in summary['results'] if 'error'not in job for t in job['trials']})
    groups=defaultdict(lambda:dict(trials=0,frames=0,statuses=Counter(),published_statuses=Counter(),rejected_statuses=Counter(),over50=0,over95=0,wall=[]))
    events=[];failures=[];trials=[]
    for root in roots:
        prov=json.loads((root/'provenance.json').read_text());job=prov['plan'][0]
        snaps=json.loads((root/'source_snapshots.json').read_text())
        assert all(hashlib.sha256((root/snaps[k]).read_bytes()).hexdigest()==h for k,h in prov['source_sha256'].items())
        folder=root/job['suite']/f"{job['task_id']:02d}"/job['demo']/'ipopt_state_B10'
        rows=json.loads((folder/'trajectory.json').read_text())
        observation=root/'solver_observations.json'
        observed=json.loads(observation.read_text())['records'] if observation.exists() else [None]*len(rows)
        assert len(observed)==len(rows)
        version=prov['source_sha256'][str(ROOT.parent/'franka_mujoco/tasktol/control/task_space/kinematics.py')]
        g=groups[version];g['trials']+=1;g['frames']+=len(rows)
        needed=[];hist=Counter()
        for i,(row,ob) in enumerate(zip(rows,observed,strict=True)):
            opt=row['optimizer']
            if not opt['solver_called']:continue
            if ob is not None:assert ob['index']==i and ob['published']==(not row['aborted'])
            code=ob['status_code'] if ob is not None else None
            kind=termination(opt['status'],code)
            g['statuses'][kind]+=1;hist[kind]+=1
            g['rejected_statuses' if row['aborted'] else 'published_statuses'][kind]+=1
            wall=opt['elapsed_ms'];g['wall'].append(wall);g['over50']+=wall>50;g['over95']+=wall>95
            if kind in ('cpu_time_limit','iteration_limit') or row['aborted']:
                item=dict(**job,root=str(root),frame=i,version=version,status_code=code,status=opt['status'],termination=kind,
                    published=not row['aborted'],iterations=opt['iterations'],problem_solve_wall_ms=wall,
                    raw_candidate_available=ob is not None,reported_residual_may_describe_held_command=bool(row['aborted']),
                    source_time_s=i*.05)
                if ob is not None:
                    item.update({k:ob[k] for k in ('ik_wall_ms','ik_cpu_ms','adapter_wrapper_wall_ms','adapter_wrapper_cpu_ms','raw_candidate_validation','raw_candidate_q_rad','raw_optimality')})
                    validation=ob['raw_candidate_validation'];settings=IpoptSettings()
                    violations=[]
                    for key,limit in [('position_residual',settings.position_tolerance),('strict_rotation_residual',settings.rotation_tolerance),('inequality_violation',settings.inequality_tolerance)]:
                        if validation[key]>limit:violations.append(dict(component=key,value=validation[key],limit=limit,ratio=validation[key]/limit))
                    if not validation['finite']:violations.append(dict(component='nonfinite'))
                    if not validation['within_joint_bounds']:violations.append(dict(component='joint_bounds'))
                    item['violations']=violations
                events.append(item)
                if row['aborted']:needed.append((i,item))
        trials.append(dict(**job,root=str(root),version=version,termination_counts=dict(hist),rejected=bool(rows[-1]['aborted']),has_raw_candidate=observation.exists()))
        if not needed:continue
        with h5py.File(folder/'recording.hdf5') as f:
            model=mujoco.MjModel.from_xml_string(f['attempts'][job['demo']].attrs['model_file'])
        with h5py.File(folder.parent/'state_reference.hdf5') as f:
            witness=f['goal_joint_witness_rad'][()];sourceq=f['q'][0];anchor=f['rotations'][0]
            projection_residual=f['projection_pose_residual'][()] if 'projection_pose_residual' in f else None
        data=mujoco.MjData(model);site=model.site('gripper0_grip_site').id;hand=model.body('robot0_right_hand').id
        data.qpos[:7]=sourceq;mujoco.mj_kinematics(model,data)
        correction=data.xmat[hand].reshape(3,3).T@anchor
        bounds=np.array([model.joint(f'robot0_joint{k}').range for k in range(1,8)])
        def geometry(q,target_p,target_r):
            q=np.asarray(q)
            if not np.all(np.isfinite(q)):return dict(finite=False)
            data.qpos[:7]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,jp,jr,site)
            sv=np.linalg.svd(np.vstack([jp[:,:7]/.3,jr[:,:7]]),compute_uv=False)
            margin=np.minimum(q-bounds[:,0],bounds[:,1]-q)
            return dict(finite=True,position_max_component_m=float(np.max(np.abs(data.site_xpos[site]-target_p))),
                rotation_max_component_rad=float(np.max(np.abs(Rotation.from_matrix(target_r.T@(data.xmat[hand].reshape(3,3)@correction)).as_rotvec()))),
                sigma_min_length_scaled=float(sv[-1]),condition_length_scaled=float(sv[0]/sv[-1]) if sv[-1]>0 else None,
                min_joint_limit_margin_deg=float(np.degrees(margin.min())),closest_joint=int(np.argmin(margin))+1,
                within_joint_bounds=bool(np.all(margin>=0)))
        for i,item in needed:
            row=rows[i];tp=np.array(row['nominal_target']['position_m']);tr=Rotation.from_rotvec(row['nominal_target']['rotation_vector_rad']).as_matrix()
            item['geometry']=dict(offline_goal_witness=geometry(witness[i],tp,tr),seed=geometry(rows[i-1]['joint_target_rad'] if i else row['pre_action']['joint_position_rad'],tp,tr),held_or_published_command=geometry(row['joint_target_rad'],tp,tr))
            if item['raw_candidate_available']:item['geometry']['raw_candidate']=geometry(item['raw_candidate_q_rad'],tp,tr)
            w=item['geometry']['offline_goal_witness']
            assert w['within_joint_bounds'] and w['position_max_component_m']<1e-9 and w['rotation_max_component_rad']<1e-8
            item['globally_unreachable_goal_supported']=False
            item['seed_to_witness_joint_distance_rad']=float(np.linalg.norm(witness[i]-np.asarray(rows[i-1]['joint_target_rad'] if i else row['pre_action']['joint_position_rad'])))
            if projection_residual is not None:
                item['desired_intent_to_projected_pose_error']=dict(position_norm_m=float(projection_residual[i,0]),rotation_norm_rad=float(projection_residual[i,1]))
            item['geometric_evidence']=dict(
                witness_sigma_below_1e_3=w['sigma_min_length_scaled']<1e-3,
                witness_within_0_01deg_of_limit=w['min_joint_limit_margin_deg']<.01)
            if item['raw_candidate_available']:
                raw_geometry=item['geometry']['raw_candidate']
                if raw_geometry['finite']:
                    item['geometric_evidence'].update(
                        raw_sigma_below_1e_3=raw_geometry['sigma_min_length_scaled']<1e-3,
                        raw_within_0_01deg_of_limit=raw_geometry['min_joint_limit_margin_deg']<.01,
                        raw_closest_joint=raw_geometry['closest_joint'])
            item['interpretation']=('CPU time limit ended the solve; independent feasibility rejected the candidate' if item['termination']=='cpu_time_limit' else 'Iteration budget ended the solve; independent feasibility rejected the candidate' if item['termination']=='iteration_limit' else 'Independent candidate rejection; inspect exact status and residuals')
            if not item['raw_candidate_available']:item['interpretation']+='; historical raw iterate was not retained, held-command residual is not its residual'
            failures.append(item)
    aggregated={}
    for version,g in groups.items():
        a=np.array(g.pop('wall'))
        g['adapter_wall_ms']=dict(calls=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else {}
        aggregated[version]=g
    output=dict(complete=summary['complete'],trials=len(roots),episodes=summary['completed'],groups_by_kinematics_version=aggregated,
        default_settings=asdict(IpoptSettings()),events=events,rejections=failures,trials_index=trials,
        caution='resource-limit status is not itself execution rejection; independently feasible candidates may publish. Wall latency is not CPU timeout. Geometry uses archived XML calibrated once to initial tool FK; length scale .3 m. No causal proof of singularity from a scalar threshold alone.',physics_steps=0)
    target=run/('solver_diagnosis_partial.json' if args.partial else 'solver_diagnosis.json')
    target.write_text(json.dumps(output,indent=2)+'\n')
    selected_roots={job['selected_root'] for job in summary['results'] if 'error'not in job}
    selected_failures=[e for e in failures if e['root']in selected_roots]
    lines=['# 求解终止与闭环拒解诊断','',
        f"完成记录 {summary['completed']}/400，审计 {len(roots)} 次试录；完整批次：{summary['complete']}。各次试录按保存的源码快照核验，不把不同版本合并为单版本性能结果。",'',
        '默认 IPOPT 不变：20 次迭代、95 ms CPU 上限；位置分量阈值 10 μm、姿态分量阈值 1e-4 rad。CPU 超时、迭代上限和候选可行性是三个不同判定；资源终止但独立校验合格的候选可以发布。墙钟超过 50 ms 或 95 ms 本身不等于 CPU 超时。','',
        '|运动学版本|试录|求解帧|CPU超时|其中拒解|迭代上限|其中拒解|墙钟>50ms|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v,g in aggregated.items():
        lines.append(f"|{v[:12]}|{g['trials']}|{g['adapter_wall_ms'].get('calls',0)}|{g['statuses'].get('cpu_time_limit',0)}|{g['rejected_statuses'].get('cpu_time_limit',0)}|{g['statuses'].get('iteration_limit',0)}|{g['rejected_statuses'].get('iteration_limit',0)}|{g['over50']}|")
    lines += ['', '全部 baseline 和 IRLC 候选均计入上表；候选失败不等于最终选择的示范失败。下面仅列最终选择的拒解。', '',
        '|任务/示范|帧|终止|求解wall ms|原始候选位置 μm / 姿态 rad|见证σmin|原始候选距最近关节限位°|',
        '|---|---:|---|---:|---|---:|---:|']
    csv_rows=[]
    for e in selected_failures:
        val=e.get('raw_candidate_validation',{});geom=e['geometry'];w=geom['offline_goal_witness'];raw=geom.get('raw_candidate',{})
        residual=(f"{val['position_residual']*1e6:.3f} / {val['strict_rotation_residual']:.6g}" if val else '历史未保留原始候选')
        margin=raw.get('min_joint_limit_margin_deg');margin_label=f'{margin:.6g}' if margin is not None else '未记录'
        lines.append(f"|{e['case']:02d}/{e['demo']}|{e['frame']}|{e['termination']}|{e['problem_solve_wall_ms']:.3f}|{residual}|{w['sigma_min_length_scaled']:.6g}|{margin_label}|")
        csv_rows.append(dict(case=e['case'],demo=e['demo'],frame=e['frame'],termination=e['termination'],
            solver_wall_ms=e['problem_solve_wall_ms'],raw_position_residual_m=val.get('position_residual'),
            raw_rotation_residual_rad=val.get('strict_rotation_residual'),witness_sigma_min=w['sigma_min_length_scaled'],
            witness_limit_margin_deg=w['min_joint_limit_margin_deg'],raw_limit_margin_deg=margin,
            raw_closest_joint=raw.get('closest_joint'),root=e['root']))
    lines += ['',
        '所有列出的目标均存在满足硬关节限位、FK 与目标一致的离线见证，因此没有证据把这些目标判为运动学全局不可达；这不证明见证无碰撞或存在容易求解的局部路径。', '',
        '几何证据分两类：部分见证处于低 σmin 的近奇异区域；另一些在线候选到达关节硬限位，而离线见证仍在另一合法构型。二者都可能使局部 NLP 在有限迭代内失败，应逐帧结合原始候选、已接受参考历史和见证诊断，不能统一归因于超时。σmin 采用平移行除以 0.3 m 的量纲归一化；阈值只作诊断，未加入约束。', '',
        '新记录的 raw_candidate_validation 才描述被拒绝的原始候选。trajectory 的 joint_target_rad 在拒解帧是保留的上一次已接受命令，其残差不能冒充原始候选残差。历史没有保存原始候选时，明确保留未知。', '',
        'solver_diagnosis*.json 保存全部预算终止/拒解事件、精确残差、CPU/墙钟时间（若有原始记录）、关节限位余量和 FK 见证；本诊断无物理步进、无数据选择、无求解参数修改。']
    (run/('SOLVER_DIAGNOSIS_PARTIAL.md' if args.partial else 'SOLVER_DIAGNOSIS.md')).write_text('\n'.join(lines)+'\n')
    with (run/('selected_rejections_partial.csv' if args.partial else 'selected_rejections.csv')).open('w',newline='') as f:
        if csv_rows:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({k:v for k,v in output.items() if k not in ('events','rejections','trials_index')},indent=2))
    print('events',len(events),'rejections',len(failures),flush=True)
    if not args.partial:
        from diagnose_task_outcomes import diagnose
        diagnose(run)

if __name__=='__main__':main()
