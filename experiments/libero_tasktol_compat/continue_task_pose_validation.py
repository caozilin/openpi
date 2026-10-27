"""Finite fixed-version recording, audit, packaging and independent replay job."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_fixed_v2'
DATASET=ROOT/'data/libero_tasktol_state/20261024_task_pose_default_ipopt400_fixed_v2'
COLD=ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_fixed_v2_cold'

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');args=p.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl',PYTHONPATH=str(ROOT)+':'+str(ROOT.parent/'franka_mujoco'))
    if not args.worker:
        marker=RUN/'validation_host_process.json'
        assert not marker.exists(),'Inspect existing job before launching any duplicate'
        with (RUN/'validation_host.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        metadata=dict(pid=process.pid,run=str(RUN),dataset=str(DATASET),cold=str(COLD),launched_at_utc=datetime.now(timezone.utc).isoformat(),finite_episodes=400)
        marker.write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata));return
    stages=[('recording','rerecord_task_pose_batch.py',['--output',str(RUN),'--resume','--workers','6']),
            ('solver_diagnosis','diagnose_batch_solver.py',['--run',str(RUN)]),
            ('packaging','finish_task_pose_batch.py',['--run',str(RUN),'--dataset',str(DATASET)]),
            ('cold_replay','replay_state_dataset_batch.py',['--dataset',str(DATASET),'--output',str(COLD),'--workers','3','--include-failures'])]
    status=dict(complete=False,stages=[])
    for name,script,arguments in stages:
        status['active_stage']=name
        (RUN/'validation_pipeline.json').write_text(json.dumps(status,indent=2)+'\n')
        with (RUN/f'stage_{name}.log').open('w') as log:
            done=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name(script)),*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status['stages'].append(dict(stage=name,returncode=done.returncode,log=str(RUN/f'stage_{name}.log')))
        (RUN/'validation_pipeline.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if done.returncode:return
    status.update(complete=True,active_stage=None)
    (RUN/'validation_pipeline.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
