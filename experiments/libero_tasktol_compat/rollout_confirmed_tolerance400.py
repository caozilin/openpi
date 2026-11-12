"""Replay the existing 400 frozen reconstructed actions with confirmed masks.

Only dataset orchestration and semantic assignment live here. The existing
LIBERO session, native stepping and shared joint servo execute every action.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1'
DEFAULT=ROOT/'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'
ANNOTATIONS=ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/mask_annotations.json'
os.environ.setdefault('TASKTOL_FROZEN_SHARED_ROOT',str(BASE/'frozen_shared_runtime'))
sys.path[:0]=[os.environ['TASKTOL_FROZEN_SHARED_ROOT'],str(ROOT)]
import tasktol
assert Path(tasktol.__file__).resolve().is_relative_to(Path(os.environ['TASKTOL_FROZEN_SHARED_ROOT']))
from experiments.libero_tasktol_compat import rollout_mujoco_joint_servo as servo
from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner

base_freeze=servo.freeze
def freeze():
    return dict(base_freeze(),**{str(Path(__file__)):runner.sha(Path(__file__))})
runner.freeze_sources=freeze

# These are semantic identities of successive grasp attempts, not phase labels.
# Regrasping the same object retains its task mask. Each entry is inspectable
# against the paired baseline video at the saved cycle boundaries.
REGRASPS={
    (0,'demo_31'):[0,0], (8,'demo_29'):[0,0], (12,'demo_25'):[0,0],
    (30,'demo_39'):[0,1,1], (36,'demo_48'):[0,1,1],
    (36,'demo_43'):[0,1,1], (38,'demo_6'):[0,0,1],
    (38,'demo_32'):[0,1,1],
}

def confirmed_masks(original):
    tasks=[]
    overrides={14:(['011'],['111']),23:(['010'],['000']),25:(['000'],['000']),
        29:(['011'],['000']),32:(['010','010'],['000','000']),
        33:(['010','000'],['000','000']),38:(['010','010'],['000','000']),
        39:(['010','000'],['000','000'])}
    for case,t in enumerate(original['tasks']):
        app,exe=overrides.get(case,([s['APP']['effective_mask']for s in t['subtasks']],
                                  [s['EXEC']['effective_mask']for s in t['subtasks']]))
        labels=[s['object_or_mechanism']for s in t['subtasks']]
        if case==23:labels=['open top drawer, then grasp and place bowl; drawer opening is APP']
        tasks.append(dict(case=case,ordinal=case+1,suite=t['suite'],task_id=t['task_id'],
            subtasks=[dict(index=i,label=labels[i],APP=a,ENG='000',EXEC=e,DIS='000')
                      for i,(a,e) in enumerate(zip(app,exe,strict=True))]))
    return dict(schema='libero40_user_confirmed_semantic_masks.2',confirmed_at='2026-10-31',
        confirmation='User: Goal03 drawer opening through bowl approach is PRE010, POST000; Goal05 remains000; all other proposed edits accepted.',
        mask_order=['xF','yF','zF'],bounds_deg=[30]*6,
        original_annotation_sha256=runner.sha(ANNOTATIONS),tasks=tasks)

def make_phase_policy(subtasks,baseline_cycles,attempt_mapping):
    def resolve(index,observation,action):
        cycle=baseline_cycles[index]
        semantic_index=attempt_mapping[cycle] if cycle<len(attempt_mapping) else None
        phase={'pregrasp':'APP','grasp':'ENG','postgrasp':'EXEC','release':'DIS'}[observation.phase.key]
        if semantic_index is None:phase='outside_subtask';mask='000'
        else:mask=subtasks[semantic_index][phase]
        return dict(phase=phase,semantic_mask=mask,semantic_subtask_index=semantic_index,
            baseline_grasp_cycle=cycle,semantic_assignment_source='frozen paired-baseline semantic schedule',
            mask_prediction=mask)
    return resolve

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    original=json.loads(ANNOTATIONS.read_text());masks=confirmed_masks(original)
    runner.dump(out/'confirmed_masks.json',masks)
    baseline_plan=json.loads((BASE/'plan.json').read_text())
    entries=[]
    for case,entry in enumerate(baseline_plan['tasks']):
        records=[json.loads(p.read_text())for p in (BASE/f'records/case_{case:02d}').glob('*.json')]
        records=sorted((r for r in records if r.get('success')),key=lambda r:int(r['demo'][5:]))
        assert len(records)==10
        for record in records:
            folder=Path(record['folder']);report=json.loads((folder/'report.json').read_text())
            rows=json.loads((folder/'trajectory.json').read_text())
            source=next(j for j in entry['candidates']if j['demo']==record['demo'])
            subtasks=masks['tasks'][case]['subtasks']
            mapping=REGRASPS.get((case,record['demo']),list(range(len(subtasks))))
            actual_cycles=sorted({r['observed_subtask_index']for r in rows if r['observed_phase']=='postgrasp'})
            assert not actual_cycles or max(actual_cycles)<len(mapping),(case,record['demo'],actual_cycles,mapping)
            task=dict(entry['task']);task['masks']={k:subtasks[0][k]for k in ('APP','ENG','EXEC','DIS')}
            task['semantic_subtasks']=subtasks
            job=dict(source,selected_root=None,reference_path=str(folder/'state_reference.hdf5'),
                expected_model_sha256=report['model_sha256'])
            key=f"{task['suite']}/{task['task_id']:02d}/{record['demo']}"
            entries.append(dict(key=key,case=case,task=task,job=job,baseline_folder=str(folder),
                baseline_report_sha256=runner.sha(folder/'report.json'),
                baseline_actions_sha256=report['actions_sha256'],baseline_video_sha256=runner.sha(folder/'main_rgb.mp4'),
                reference_sha256=runner.sha(folder/'state_reference.hdf5'),baseline_cycles=[r['observed_subtask_index']for r in rows],
                attempt_to_semantic_subtask=mapping,phase_policy=dict(
                    semantics='Per-demo task identity schedule frozen from the paired baseline; repeated grasps retain object identity. No future states supplied online.',
                    phase='Live shared finger classifier,20Hz,3 stable frames; ENG/DIS strict. Session rebase follows live observed cycles.',
                    subtasks=subtasks,regrasp_mapping=mapping)))
    sources=freeze()
    plan=dict(schema='libero400_confirmed_tolerance_replay.1',baseline_run=str(BASE),
        frozen_shared_root=os.environ['TASKTOL_FROZEN_SHARED_ROOT'],source_sha256=sources,
        confirmed_masks_sha256=runner.sha(out/'confirmed_masks.json'),entries=entries,
        design='400 fixed selected successful reconstructed action streams; one mask30 rollout each; no action re-optimization, retries, slowdown or gripper changes; all outcomes retained for comparison.',
        bounds_deg=[30]*6,frequency_hz=20,physics_hz=500)
    runner.dump(out/'plan.json',plan)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,path in enumerate(sources):shutil.copy2(path,snapshot/f'{i:03d}_{Path(path).name}')
    for entry in entries:
        runner.dump(out/'jobs'/entry['key']/'joint_native/plan.json',dict(case=entry['case'],task=entry['task'],
            jobs=[entry['job']],source_sha256=sources,phase_policy=entry['phase_policy']))
    runner.dump(out/'summary.json',dict(complete=False,total=400,completed=0,results=[]))
    return plan

def worker(out,key,mode='mask30'):
    import numpy as np
    plan=json.loads((out/'plan.json').read_text());assert freeze()==plan['source_sha256']
    entry=next(e for e in plan['entries']if e['key']==key)
    assert runner.sha(entry['job']['reference_path'])==entry['reference_sha256']
    phase_policy=make_phase_policy(entry['task']['semantic_subtasks'],entry['baseline_cycles'],entry['attempt_to_semantic_subtask'])
    started=time.monotonic()
    report=servo.worker(out/'jobs'/key,0,'joint_native',reference_transform=lambda r,s:r,
        mode=mode,phase_policy=phase_policy if mode=='mask30' else None)
    folder=out/'jobs'/key/f'joint_native/seed_00/{mode}'
    baseline=Path(entry['baseline_folder']);baseline_report=json.loads((baseline/'report.json').read_text())
    assert report['actions_sha256']==entry['baseline_actions_sha256']
    assert report['initial_state_sha256']==baseline_report['initial_state_sha256']
    assert report['solver_settings']==baseline_report['solver_settings']
    assert report['objective_settings']==baseline_report['objective_settings']
    assert report['shared_joint_servo']==baseline_report['shared_joint_servo']
    if mode=='off':
        for name in ('states.npz','servo_physics.npz'):
            with np.load(folder/name)as a,np.load(baseline/name)as b:
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    destination=out/('episodes' if mode=='mask30' else 'off_replay_checks')/key
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(folder),destination)
    result=dict(key=key,case=entry['case'],suite=entry['task']['suite'],task_id=entry['task']['task_id'],
        demo=entry['job']['demo'],success=report['success'],execution_rejected=report['execution_rejected'],
        executed_frames=report['executed_frames'],planned_frames=report['planned_frames'],
        rejected_frame=report['rejected_frame'],cpu_timeout_frames=report['cpu_timeout_frames'],
        solver_status_counts=report['solver_status_counts'],tracking=report['tracking'],
        phase_counts=report['phase_counts'],effective_mask_counts=report['effective_mask_counts'],
        folder=str(destination),baseline_folder=str(baseline),wall_s=time.monotonic()-started,
        video_sha256=runner.sha(destination/'main_rgb.mp4'),same_actions=True,same_initial_state=True,
        same_default_optimizer=True,same_joint_servo=True,off_replay_bitwise_identical=mode=='off')
    runner.dump(out/('records' if mode=='mask30' else 'off_check_records')/(key+'.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def summarize(out,plan):
    records=[json.loads(p.read_text())for p in (out/'records').glob('*/*/*.json')]
    errors=[r for r in records if r.get('infrastructure_error')]
    finished=[r for r in records if not r.get('infrastructure_error')]
    counts={}
    for case in range(40):
        rows=[r for r in finished if r['case']==case]
        counts[str(case)]=dict(completed=len(rows),success=sum(r['success']for r in rows),
            rejected=sum(r['execution_rejected']for r in rows))
    result=dict(complete=len(finished)==400,total=400,completed=len(finished),
        success=sum(r['success']for r in finished),failed=sum(not r['success']for r in finished),
        rejected=sum(r['execution_rejected']for r in finished),infrastructure_errors=len(errors),
        cpu_timeout_frames=sum(len(r['cpu_timeout_frames'])for r in finished),task_counts=counts,
        source_unchanged=freeze()==plan['source_sha256'],results=sorted(records,key=lambda r:(r['case'],int(r['demo'][5:]))),
        updated_at=datetime.now(timezone.utc).isoformat())
    runner.dump(out/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--key');p.add_argument('--keys',nargs='+');p.add_argument('--mode',choices=['off','mask30'],default='mask30')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.worker:worker(out,a.key,a.mode);return
    if a.prepare:plan=prepare(out);print(json.dumps(dict(prepared=len(plan['entries']),output=str(out))));return
    plan=json.loads((out/'plan.json').read_text());assert freeze()==plan['source_sha256']
    selected=[e for e in plan['entries']if not a.keys or e['key']in a.keys]
    if a.keys:assert len(selected)==len(a.keys)
    def run(entry):
        key=entry['key'];record=out/'records'/(key+'.json')
        if record.exists():
            saved=json.loads(record.read_text())
            if not saved.get('infrastructure_error'):return
        log=out/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w')as stream:
            done=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--output',str(out),'--key',key],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:
            runner.dump(record,dict(key=key,case=entry['case'],demo=entry['job']['demo'],infrastructure_error=True,
                returncode=done.returncode,log=str(log)))
    with ThreadPoolExecutor(max_workers=a.workers)as pool:
        pending={pool.submit(run,e):e for e in selected}
        for future in as_completed(pending):
            future.result();result=summarize(out,plan)
            print(json.dumps({k:result[k]for k in ('completed','success','failed','rejected','infrastructure_errors')}),flush=True)
    summarize(out,plan)

if __name__=='__main__':main()
