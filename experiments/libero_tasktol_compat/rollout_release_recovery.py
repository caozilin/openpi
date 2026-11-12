"""Test causal release recovery on all52 previously failed Long episodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'
OUT=ROOT/'data/libero_tasktol_joint_servo/20261101_release_recovery_long_failures_v1'
os.environ.setdefault('TASKTOL_FROZEN_SHARED_ROOT',str(ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime'))
sys.path[:0]=[os.environ['TASKTOL_FROZEN_SHARED_ROOT'],str(ROOT)]
from experiments.libero_tasktol_compat import rollout_confirmed_tolerance400 as previous
from experiments.libero_tasktol_compat.subtask_rotation_recovery import RotationRecovery,RecoverySettings
runner=previous.runner;servo=previous.servo;base_freeze=previous.freeze

def freeze():
    return dict(base_freeze(),**{str(p):runner.sha(p)for p in [Path(__file__),Path(__file__).with_name('subtask_rotation_recovery.py')]})
runner.freeze_sources=freeze

def prepare(out):
    old=json.loads((PRIOR/'plan.json').read_text());summary=json.loads((PRIOR/'summary.json').read_text())
    keys={r['key']for r in summary['results']if r['suite']=='libero_10' and not r['success']}
    assert len(keys)==52
    entries=[e for e in old['entries']if e['key']in keys];out.mkdir(parents=True,exist_ok=False)
    settings=asdict(RecoverySettings())
    plan=dict(schema='libero_release_recovery_failed_long.1',entries=entries,source_sha256=freeze(),settings=settings,
        prior_run=str(PRIOR),prior_summary_sha256=runner.sha(PRIOR/'summary.json'),
        design='One rollout per previously failed Long episode. Original52 failures fixed before testing. Same default optimizer/native servo,20Hz/500Hz,base translations and gripper. Only causal between-release SO(3) additions; no search or retries on task failures.')
    runner.dump(out/'plan.json',plan)
    shutil.copy2(PRIOR/'confirmed_masks.json',out/'confirmed_masks.json')
    snap=out/'source_snapshot';snap.mkdir();manifest=[]
    for i,(name,digest)in enumerate(plan['source_sha256'].items()):
        source=Path(name)
        if source.is_relative_to(runner.SHARED/'tasktol'):source=runner.ACTIVE_SHARED/source.relative_to(runner.SHARED)
        assert runner.sha(source)==digest
        target=snap/f'{i:03d}_{Path(name).name}';shutil.copy2(source,target)
        manifest.append(dict(canonical=name,executed=str(source),snapshot=str(target.relative_to(out)),sha256=digest))
    runner.dump(out/'source_snapshot_manifest.json',manifest)
    for e in entries:
        runner.dump(out/'jobs'/e['key']/'joint_native/plan.json',dict(case=e['case'],task=e['task'],jobs=[e['job']],
            source_sha256=plan['source_sha256'],phase_policy=e['phase_policy'],recovery_policy=settings))
    summarize(out,plan)

def worker(out,key,enabled=True):
    import h5py
    import numpy as np
    plan=json.loads((out/'plan.json').read_text());assert freeze()==plan['source_sha256']
    entry=next(e for e in plan['entries']if e['key']==key)
    with h5py.File(entry['job']['reference_path'])as f:anchor=f['rotations'][0]
    recovery=RotationRecovery(anchor,range(1,len(entry['attempt_to_semantic_subtask'])),RecoverySettings(**plan['settings']))
    policy=previous.make_phase_policy(entry['task']['semantic_subtasks'],entry['baseline_cycles'],entry['attempt_to_semantic_subtask'])
    started=time.monotonic()
    report=servo.worker(out/'jobs'/key,0,'joint_native',reference_transform=lambda r,s:r,
        mode='mask30',phase_policy=policy,control_transform=recovery if enabled else None)
    folder=out/'jobs'/key/'joint_native/seed_00/mask30'
    old=json.loads((PRIOR/'episodes'/key/'report.json').read_text())
    for k in ('actions_sha256','initial_state_sha256','model_sha256','solver_settings','objective_settings','shared_joint_servo'):
        assert report[k]==old[k],k
    rows=json.loads((folder/'trajectory.json').read_text())
    if enabled:
        for row in rows:
            np.testing.assert_array_equal(row['action'][:3],row['base_action'][:3]);assert row['action'][6]==row['base_action'][6]
        report['rotation_recovery']=recovery.report()
    else:
        with np.load(folder/'states.npz')as a,np.load(PRIOR/'episodes'/key/'states.npz')as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        report['control_hook_off_bitwise_identical']=True
    report['base_actions_preserved']=True
    runner.dump(folder/'report.json',report)
    dest=out/('episodes'if enabled else'off_checks')/key;dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folder),dest)
    result=dict(key=key,case=entry['case'],suite=entry['task']['suite'],task_id=entry['task']['task_id'],demo=entry['job']['demo'],
        success=report['success'],previous_success=old['success'],execution_rejected=report['execution_rejected'],
        previous_rejected=old['execution_rejected'],planned_frames=report['planned_frames'],executed_frames=report['executed_frames'],
        rejected_frame=report['rejected_frame'],cpu_timeout_frames=report['cpu_timeout_frames'],
        tracking=report['tracking'],rotation_recovery=report.get('rotation_recovery'),folder=str(dest),
        same_default_optimizer=True,same_native_servo=True,same_base_actions=True,wall_s=time.monotonic()-started)
    runner.dump(out/('records'if enabled else'off_check_records')/(key+'.json'),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def summarize(out,plan):
    records=[json.loads(p.read_text())for p in (out/'records').glob('*/*/*.json')]
    done=[r for r in records if not r.get('infrastructure_error')]
    result=dict(complete=len(done)==52,total=52,completed=len(done),success=sum(r['success']for r in done),
        failed=sum(not r['success']for r in done),rejected=sum(r['execution_rejected']for r in done),
        infrastructure_errors=sum(r.get('infrastructure_error',False)for r in records),
        cpu_timeout_frames=sum(len(r['cpu_timeout_frames'])for r in done),
        tasks={str(i):dict(completed=sum(r['task_id']==i for r in done),recovered_success=sum(r['task_id']==i and r['success']for r in done))for i in range(10)},
        results=sorted(records,key=lambda r:(r['case'],int(r['demo'][5:]))),source_unchanged=freeze()==plan['source_sha256'])
    runner.dump(out/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--key');p.add_argument('--keys',nargs='+')
    p.add_argument('--no-recovery',action='store_true');p.add_argument('--workers',type=int,default=6);a=p.parse_args();out=a.output.resolve()
    if a.prepare:prepare(out);return
    if a.worker:worker(out,a.key,not a.no_recovery);return
    plan=json.loads((out/'plan.json').read_text());assert freeze()==plan['source_sha256']
    entries=[e for e in plan['entries']if not a.keys or e['key']in a.keys]
    def run(e):
        path=out/'records'/(e['key']+'.json')
        if path.exists()and not json.loads(path.read_text()).get('infrastructure_error'):return
        log=out/'logs'/(e['key']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w')as stream:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--output',str(out),'--key',e['key']],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:runner.dump(path,dict(key=e['key'],case=e['case'],demo=e['job']['demo'],infrastructure_error=True,returncode=result.returncode,log=str(log)))
    with ThreadPoolExecutor(max_workers=a.workers)as pool:
        for future in as_completed([pool.submit(run,e)for e in entries]):
            future.result();s=summarize(out,plan);print(json.dumps({k:s[k]for k in ['completed','success','failed','rejected','infrastructure_errors']}),flush=True)

if __name__=='__main__':main()
