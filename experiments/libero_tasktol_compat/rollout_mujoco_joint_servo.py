"""No-tolerance paired trial of direct shared MuJoCo joint tracking in LIBERO.

Reuses the existing audited rollout, native scene restoration, action inputs,
IPOPT client, media recorder and result evaluator. Only actuator transport and
the shared joint tracking calculation are selected by the variant.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'franka_mujoco'),str(ROOT/'third_party/libero'),str(ROOT/'examples/libero')]
from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner

VARIANTS=['b10','joint_native','joint_implicitfast']
base_freeze=runner.freeze_sources
def freeze():
    paths=[Path(__file__),Path(__file__).with_name('mujoco_joint_servo.py'),
        ROOT.parent/'franka_mujoco/assets/mujoco_menagerie/franka_emika_panda/panda.xml']
    return dict(base_freeze(),**{str(p):runner.sha(p) for p in paths})
runner.freeze_sources=freeze


def worker(out,seed,variant,reference_transform=None,observation_callback=None):
    import numpy as np
    import libero.libero.envs as envs
    import tasktol.simulation.libero as client
    from experiments.libero_tasktol_compat.mujoco_joint_servo import MuJoCoJointServoController
    sessions=[]
    native_env=envs.OffScreenRenderEnv;native_session=client.LiberoTaskTolSession
    if variant!='b10':
        def make_env(**kwargs):
            kwargs['controller']='JOINT_POSITION'
            return native_env(**kwargs)
        def make_session(env,**kwargs):
            kwargs.update(joint_control_mode='native',kp=None)
            session=native_session(env,**kwargs)
            controller=MuJoCoJointServoController.from_native(session.controller,session.handles,
                source_integrator=variant=='joint_implicitfast')
            session.controller=controller;env.robots[0].controller=controller
            sessions.append(session)
            return session
        envs.OffScreenRenderEnv=make_env;client.LiberoTaskTolSession=make_session
    try:
        report=runner.worker(out/variant,seed,'off',reference_transform=reference_transform,observation_callback=observation_callback)
    finally:
        envs.OffScreenRenderEnv=native_env;client.LiberoTaskTolSession=native_session
    report['variant']=variant;report['tolerance']='off'
    if sessions:
        controller=sessions[0].controller
        report['shared_joint_servo']=controller.servo_metadata
        report['shared_joint_servo_statistics']=controller.servo_statistics
        assert controller.servo_statistics['calls']==25*report['executed_frames']
        np.savez_compressed(out/variant/f'seed_{seed:02d}/off/servo_physics.npz',samples=np.asarray(controller.reference_samples))
    runner.dump(out/variant/f'seed_{seed:02d}/off/report.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--resume',action='store_true');args=p.parse_args()
    out=args.output.resolve()
    if args.worker:worker(out,args.seed,args.variant);return
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False)
        old=ROOT/'experiments/libero_tasktol_compat/results/spatial00_mask30_10seed_20261025_v2/plan.json'
        plan=json.loads(old.read_text());plan['source_sha256']=freeze();plan['variants']=args.variants
        plan['design']='Same 10 source initial states and frozen selected physical actions; tolerance off. B10 control versus direct shared interpolation/tracker plus original Panda position servos. Native LIBERO physics and gripper; optional source-integrator variant is explicitly separate.'
        snap=out/'source_snapshot';snap.mkdir()
        for i,path in enumerate(plan['source_sha256']):shutil.copy2(path,snap/f'{i:03d}_{Path(path).name}')
        runner.dump(out/'plan.json',plan)
        for variant in VARIANTS:runner.dump(out/variant/'plan.json',plan)
    plan=json.loads((out/'plan.json').read_text());assert freeze()==plan['source_sha256']
    entries=[(seed,variant) for seed in args.seeds for variant in args.variants]
    def run(entry):
        seed,variant=entry;report=out/variant/f'seed_{seed:02d}/off/report.json'
        if args.resume and report.exists():return json.loads(report.read_text())
        with (out/f'seed_{seed:02d}_{variant}.log').open('w') as log:
            done=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--output',str(out),'--seed',str(seed),'--variant',variant],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if done.returncode:return dict(seed=seed,variant=variant,infrastructure_error=True,returncode=done.returncode)
        return json.loads(report.read_text())
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run,entries):
            records.append(r);runner.dump(out/'summary.json',dict(complete=False,results=records))
            print(json.dumps({k:r[k] for k in ['seed','variant','success','execution_rejected','infrastructure_error'] if k in r}),flush=True)
    results=[]
    for variant in VARIANTS:
        for f in sorted((out/variant).glob('seed_*/off/report.json')):results.append(json.loads(f.read_text()))
    expected={(s,v)for s in range(10)for v in args.variants}
    summary=dict(complete=expected<={(r['seed'],r['variant'])for r in results},results=results,source_unchanged=freeze()==plan['source_sha256'],
        counts={v:dict(total=sum(r['variant']==v for r in results),success=sum(r['variant']==v and r['success'] for r in results),
            rejected=sum(r['variant']==v and r['execution_rejected'] for r in results))for v in VARIANTS})
    runner.dump(out/'summary.json',summary);print(json.dumps(summary['counts']),flush=True)


if __name__=='__main__':main()
