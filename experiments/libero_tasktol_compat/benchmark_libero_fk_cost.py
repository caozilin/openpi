"""Read-only FK microbenchmark; never replaces production FK or runs IPOPT."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import h5py
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def bench(output):
    batch=ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_v1'
    diagnostic=ROOT/'experiments/libero_tasktol_compat/results/slow_success_timeout_default_diagnostic_v1'
    pid=json.loads((batch/'batch_host_process.json').read_text())['batch_pid']
    def terminate(sig, frame): raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    os.kill(pid,signal.SIGSTOP)
    output.mkdir(parents=True,exist_ok=False)
    try:
        for _ in range(120):
            children=set()
            for f in Path(f'/proc/{pid}/task').glob('*/children'):children.update(f.read_text().split())
            active=[int(c) for c in children if Path(f'/proc/{c}/stat').exists() and Path(f'/proc/{c}/stat').read_text().split()[2]!='Z']
            if not active:break
            time.sleep(1)
        assert not active
        plan=json.loads((diagnostic/'plan.json').read_text())['jobs']
        jobs=[plan[0],plan[1],next(j for j in plan if j['case']==4 and j['demo']=='demo_7')]
        cases=[]
        for job in jobs:
            root=Path(job['root']);provenance=json.loads((root/'provenance.json').read_text());info=provenance['plan'][0]
            dest=root/info['suite']/f"{info['task_id']:02d}"/info['demo']
            with h5py.File(dest/'ipopt_state_B10/recording.hdf5') as file:
                g=file['attempts'][info['demo']];model=mujoco.MjModel.from_xml_string(g.attrs['model_file'])
                states=g['tasktol/pre_action_states'][()]
            with h5py.File(dest/'state_reference.hdf5') as file:witness=file['goal_joint_witness_rad'][()]
            rows=json.loads((dest/'ipopt_state_B10/trajectory.json').read_text())
            site=model.site('gripper0_grip_site').id;hand=model.body('robot0_right_hand').id
            joint_ids=np.asarray([model.joint(f'robot0_joint{i}').id for i in range(1,8)])
            qids=model.jnt_qposadr[joint_ids];dofs=model.jnt_dofadr[joint_ids]

            def snapshot(data):
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacSite(model,data,jp,jr,site)
                return data.site_xpos[site].copy(),data.xmat[hand].reshape(3,3).copy(),jp[:,dofs],jr[:,dofs]

            def run_fn(data,kind):
                if kind=='full':mujoco.mj_forward(model,data)
                else:mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                return snapshot(data)

            for index in job['indices']:
                for label,q in [('actual',np.asarray(rows[index]['pre_action']['joint_position_rad'])),('witness',witness[index])]:
                    scratch=mujoco.MjData(model);scratch.qpos[qids]=q
                    actual=mujoco.MjData(model);actual.qpos[:]=states[index,1:1+model.nq];actual.qpos[qids]=q
                    fast=mujoco.MjData(model);fast.qpos[:]=scratch.qpos
                    reference=run_fn(scratch,'full');actual_reference=run_fn(actual,'full');got=run_fn(fast,'kinematics')
                    error=max(float(np.max(np.abs(a-b))) for a,b in zip(reference,got,strict=True))
                    state_error=max(float(np.max(np.abs(a-b))) for a,b in zip(reference,actual_reference,strict=True))
                    assert error<1e-12 and state_error<1e-12
                    timings={}
                    for name,data,kind in [('scratch_full_forward',scratch,'full'),('actual_object_positions_full_forward',actual,'full'),('kinematics_comPos_jacSite',fast,'kinematics')]:
                        elapsed=[]
                        for _ in range(100):
                            t=time.perf_counter();run_fn(data,kind);elapsed.append((time.perf_counter()-t)*1e6)
                        timings[name]=dict(median_us=float(np.median(elapsed)),mean_us=float(np.mean(elapsed)),
                            contacts=int(data.ncon),constraint_rows=int(data.nefc),solver_iterations=data.solver_niter.tolist())
                    stages={}
                    for name in ('mj_fwdPosition','mj_fwdVelocity','mj_fwdActuation','mj_fwdAcceleration','mj_fwdConstraint'):
                        values=[]
                        for _ in range(100):
                            t=time.perf_counter();getattr(mujoco,name)(model,scratch);values.append((time.perf_counter()-t)*1e6)
                        stages[name]=float(np.median(values))
                    item=dict(case=job['case'],demo=job['demo'],frame=index,pose=label,
                        max_position_rotation_jacobian_difference=error,max_difference_when_object_state_restored=state_error,
                        timings=timings,stage_median_us=stages,
                        solver=int(model.opt.solver),solver_iteration_limit=int(model.opt.iterations))
                    cases.append(item)
                    print(json.dumps(dict(case=job['case'],demo=job['demo'],frame=index,pose=label,
                        scratch_us=timings['scratch_full_forward']['median_us'],kinematics_us=timings['kinematics_comPos_jacSite']['median_us'],
                        scratch_contacts=scratch.ncon,actual_contacts=actual.ncon)),flush=True)
        result=dict(complete=True,physics_steps=0,optimizer_calls=0,production_kinematics_unchanged=True,
                    method='same archived native model; FK/Jacobian equivalence microbenchmark only; 100 repeats per condition',cases=cases)
        (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    finally:
        os.kill(pid,signal.SIGCONT)
        (output/'batch_resumed.json').write_text(json.dumps(dict(batch_pid=pid,resume_signal_sent=True))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    bench(p.parse_args().output.resolve())
