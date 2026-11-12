"""Paired standard four-metric analysis on successful mask30 episodes only.

Read-only FK/Jacobian evaluation; no physics, controller or solver is run.
The full native20Hz optimized joint commands are used, without resampling.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'
FROZEN=ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime'
sys.path[:0]=[str(FROZEN),str(ROOT)]

import h5py
import mujoco
import numpy as np
from tasktol.benchmark.metrics import EvaluationRecorder,standard_kinematic_metrics
from tasktol.config import TELEOP_DT
from tasktol.control.task_space.conditioning import velocity_normalized_sigma_min,sigma_min_normalization_metadata
from tasktol.simulation.robots import PANDA
from experiments.libero_tasktol_compat.annotate_original_closing_vertical import remap_assets

KEYS=('velocity_rad_s','acceleration_rad_s2','mY','sigma_min')
LABELS=('速度 rad/s ↓','加速度 rad/s² ↓','mY ↑','σmin ↑')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(model,data,dof_ids,qpos_ids,site,q):
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    # The shared standard extractor assumes0.1s movement units. Normalize its
    # accumulator to that unit while passing the real0.05s derivative interval.
    recorder=EvaluationRecorder(movement_time_scale_s=TELEOP_DT);recorder.start()
    for i,node in enumerate(q):
        data.qpos[qpos_ids]=node
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        mujoco.mj_jacSite(model,data,jp,jr,site)
        j=np.vstack([jp[:,dof_ids],jr[:,dof_ids]])
        if i==0:
            fast=j.copy();mujoco.mj_forward(model,data);mujoco.mj_jacSite(model,data,jp,jr,site)
            np.testing.assert_allclose(fast,np.vstack([jp[:,dof_ids],jr[:,dof_ids]]),atol=1e-12,rtol=0)
        my=float(np.sqrt(np.clip(np.linalg.det(j@j.T),0.,1.)))
        p=data.site_xpos[site].copy();r=data.site_xmat[site].reshape(3,3).copy()
        recorder.sample(p,r,p,r,node,my,.05,commanded=True,
            sigma_min_full=float(np.linalg.svd(j,compute_uv=False)[-1]),
            sigma_min_velocity_normalized=velocity_normalized_sigma_min(j,PANDA.joint_velocity_limits))
    result=standard_kinematic_metrics(recorder.summary)
    np.testing.assert_allclose(result['velocity_rad_s'],np.mean(np.linalg.norm(np.diff(q,axis=0)/.05,axis=1)),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(result['acceleration_rad_s2'],np.mean(np.linalg.norm(np.diff(q,n=2,axis=0)/.05**2,axis=1)),rtol=1e-12,atol=1e-10)
    return result

def aggregate(rows):
    result=dict(episodes=len(rows),metrics={})
    for key in KEYS:
        b=np.asarray([r['baseline'][key]for r in rows]);t=np.asarray([r['tolerance'][key]for r in rows]);lower=key in KEYS[:2]
        sign=-1 if lower else 1
        result['metrics'][key]=dict(baseline=float(b.mean()),tolerance=float(t.mean()),
            improvement_percent=float(sign*(t.mean()/b.mean()-1)*100),
            improved_episodes=int(np.count_nonzero(sign*(t-b)>1e-9)),
            worsened_episodes=int(np.count_nonzero(sign*(t-b)<-1e-9)),
            tied_episodes=int(np.count_nonzero(np.abs(t-b)<=1e-9)))
    return result

def main():
    assert mujoco.__version__=='3.2.3'
    plan=json.loads((RUN/'plan.json').read_text());summary=json.loads((RUN/'summary.json').read_text())
    successful={r['key']for r in summary['results']if r['success']};assert len(successful)==286
    entries=[e for e in plan['entries']if e['key']in successful]
    robosuite=Path(importlib.util.find_spec('robosuite').origin).parent
    assets=ROOT/'third_party/libero/libero/libero/assets'
    results=[];models=[]
    for case in range(40):
        selected=[e for e in entries if e['case']==case]
        if not selected:continue
        entry=selected[0]
        with h5py.File(entry['job']['source_hdf5'])as f:xml=f['data'][entry['job']['demo']].attrs['model_file']
        tree=ET.fromstring(xml);remap_assets(tree,assets,robosuite)
        model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));data=mujoco.MjData(model)
        ids=np.asarray([model.joint(f'robot0_joint{i}').id for i in range(1,8)])
        qpos=model.jnt_qposadr[ids];dof=model.jnt_dofadr[ids];site=model.site('gripper0_grip_site').id
        model_provenance=dict(case=case,source=entry['job']['source_hdf5'],demo=entry['job']['demo'],
            original_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),control_site='gripper0_grip_site')
        models.append(model_provenance)
        for entry in selected:
            key=entry['key'];baseline=Path(entry['baseline_folder']);tol=RUN/'episodes'/key
            b=json.loads((baseline/'trajectory.json').read_text());t=json.loads((tol/'trajectory.json').read_text())
            assert len(b)==len(t) and all(not r['aborted']for r in t)
            report=json.loads((tol/'report.json').read_text());assert report['success']
            with np.load(baseline/'states.npz')as archive:
                initial=archive['pre_action_states'][0];data.qpos[:]=initial[1:1+model.nq]
            mujoco.mj_kinematics(model,data)
            np.testing.assert_allclose(data.site_xpos[site],b[0]['pre_action']['position_m'],atol=1e-10,rtol=0)
            qb=np.asarray([r['joint_target_rad']for r in b]);qt=np.asarray([r['joint_target_rad']for r in t])
            values={name:metrics(model,data,dof,qpos,site,q)for name,q in [('baseline',qb),('tolerance',qt)]}
            results.append(dict(key=key,case=case,suite=entry['task']['suite'],task_id=entry['task']['task_id'],
                command_nodes=len(qt),derivative_intervals=len(qt)-1,acceleration_intervals=len(qt)-2,**values,
                input_sha256={str(path):sha(path)for path in [baseline/'trajectory.json',tol/'trajectory.json']}))
        print(json.dumps(dict(case=case,paired_completed=len(results))),flush=True)
    result=dict(schema='libero_successful_tolerance_standard_kinematics.1',
        domain='complete optimized20Hz command waypoints only; no actual-state or500Hz samples mixed in',
        selection='286 mask30 successful episodes, paired to their own successful no-tolerance baseline; failed114 excluded by user request',
        control_dt_s=.05,initial_actual_state_prepended=False,startup_derivatives='N-1 speed, N-2 acceleration; no zero padding',
        aggregation='Compute per-episode metrics first, then equal-weight episode means. Percent improvement is ratio of paired group means; lower speed/acceleration and higher mY/sigma_min are better.',
        sigma_min_definition='minimum over each complete episode of sigma_min(Dx^-1 Jg Dq), followed by mean of episode minima; full6x7 geometric Jacobian, no task axes removed',
        normalization=sigma_min_normalization_metadata(PANDA.joint_velocity_limits),
        manipulability='sqrt(clip(det(Jg Jg.T),0,1)); mean over command nodes',
        shared_runtime=str(FROZEN),source_sha256={str(path):sha(path)for path in [Path(__file__),FROZEN/'tasktol/benchmark/metrics.py',FROZEN/'tasktol/control/task_space/conditioning.py']},
        overall=aggregate(results),suites={s:aggregate([r for r in results if r['suite']==s])for s in ['libero_spatial','libero_object','libero_goal','libero_10']},
        tasks={str(case):aggregate([r for r in results if r['case']==case])for case in range(40)if any(r['case']==case for r in results)},
        models=models,episodes=results)
    (RUN/'successful_kinematics_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fields=['key','case','suite','command_nodes']+[f'{side}_{k}'for k in KEYS for side in ['baseline','tolerance']]
    with (RUN/'successful_kinematics_comparison.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:writer.writerow(dict({k:r[k]for k in fields[:4]},**{f'{side}_{k}':r[side][k]for k in KEYS for side in ['baseline','tolerance']}))
    text=['# 成功286条：配对四维运动学比较','',
        '完整20Hz优化关节指令轨迹，同一286条示范配对；先按示范计算，再等权平均。速度/加速度降低为改善，mY/σmin升高为改善。σmin先取每条轨迹最小值，再对示范取均值。未包含失败114条，不能用此表替代全400条成功率。','',
        '|指标|无容差|30°容差|改善|改善条数/286|','|---|---:|---:|---:|---:|']
    for label,key in zip(LABELS,KEYS):
        r=result['overall']['metrics'][key];text.append(f"|{label}|{r['baseline']:.6g}|{r['tolerance']:.6g}|{r['improvement_percent']:+.2f}%|{r['improved_episodes']}|")
    text+=['','|任务集（成功条数）|速度改善|加速度改善|mY改善|σmin改善|','|---|---:|---:|---:|---:|']
    for suite,r in result['suites'].items():text.append('|'+suite+f" ({r['episodes']})|"+'|'.join(f"{r['metrics'][k]['improvement_percent']:+.2f}%"for k in KEYS)+'|')
    (RUN/'successful_kinematics_comparison.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result['overall'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
