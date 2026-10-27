"""Re-record a frozen pose action sequence with the untouched library defaults."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

import rerecord_state_reference as runner
from rerecord_state_reference import ROOT,TASKTOL,plain,sha
from tasktol.control.ipopt import IpoptSettings
from tasktol.control.task_space.objective import ObjectiveSettings
from task_pose_object_metrics import evaluate_task_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();parent=args.parent.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((parent/'provenance.json').read_text());job=previous['plan'][0]
    dest=parent/job['suite']/f"{job['task_id']:02d}"/job['demo']
    with h5py.File(dest/'state_reference.hdf5') as f:reference={k:v[()] for k,v in f.items()}
    def reference_factory(states,actions,handles,kin,lookahead,time_scale=1):
        assert lookahead==time_scale==1
        np.testing.assert_array_equal(reference['actions'][:,6],actions[:len(reference['actions']),6])
        return {k:v.copy() for k,v in reference.items()}
    runner.state_reference=reference_factory
    session_type=runner.LiberoTaskTolSession
    observed=[]
    def default_session(env,*,kp,joint_control_mode):
        session=session_type(env,kp=kp,joint_control_mode=joint_control_mode,solver_settings=IpoptSettings())
        assert session.ik.settings==IpoptSettings()
        assert session.ik.objective_settings==ObjectiveSettings.delta_q_squared_conditioning_safeguard()
        observed.append(dict(settings=asdict(session.ik.settings),objective=plain(asdict(session.ik.objective_settings))))
        return session
    runner.LiberoTaskTolSession=default_session
    sources=[Path(__file__),Path(runner.__file__),Path(__file__).with_name('task_pose_object_metrics.py'),
        TASKTOL/'tasktol/control/ipopt/controller.py',TASKTOL/'tasktol/control/ipopt/adapter.py',
        TASKTOL/'tasktol/control/task_space/objective.py',TASKTOL/'tasktol/control/task_space/kinematics.py',
        TASKTOL/'tasktol/simulation/libero.py',TASKTOL/'tasktol/simulation/libero_joint_controller.py']
    (output/'source_snapshot').mkdir()
    for i,path in enumerate(sources):shutil.copyfile(path,output/'source_snapshot'/f'{i}_{path.name}')
    label='frozen_pose_reference_default_library_ipopt_task_only'
    provenance=dict(plan=[job],modes=['ipopt_state_B10'],time_scale=1,lookahead_steps=1,gripper_policy='preserve',
        task_kp=150,null_kp=10,label_method=label,parent=str(parent),solver_settings=asdict(IpoptSettings()),
        optimizer_objective='unmodified library default',robot_joint_tracking_penalty=False,
        reference_actions_sha256=sha(dest/'state_reference.hdf5'),source_sha256={str(p):sha(p) for p in sources})
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'source_snapshots.json').write_text(json.dumps({str(p):f'source_snapshot/{i}_{p.name}' for i,p in enumerate(sources)},indent=2)+'\n')
    result=runner.worker(job,output,1,['ipopt_state_B10'],1,'preserve')
    folder=output/job['suite']/f"{job['task_id']:02d}"/job['demo']/'ipopt_state_B10'
    with h5py.File(folder.parent/'state_reference.hdf5','r+') as f:f.attrs['label_method']=label
    result['results'][0].update(label_method=label,solver_settings=asdict(IpoptSettings()),
        source_state_motion='frozen pose reference; no source joint tracking in offline task cost; unmodified default IPOPT')
    (folder/'report.json').write_text(json.dumps(plain(result['results'][0]),indent=2)+'\n')
    (folder.parent/'summary.json').write_text(json.dumps(plain(result),indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(plain(dict(complete=True,results=[result])),indent=2)+'\n')
    (output/'optimizer_default_audit.json').write_text(json.dumps(dict(defaults=observed,source_sha256=provenance['source_sha256']),indent=2)+'\n')
    print(json.dumps(evaluate_task_run(output)),flush=True)


if __name__=='__main__':main()
