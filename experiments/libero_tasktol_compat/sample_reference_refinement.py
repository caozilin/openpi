"""Small CEM pilot over smooth reference corrections, scored by full rollouts.

PhysicsGen (RSS25) motivates demonstration initialization and state costs;
MJPC Predictive Sampling motivates a compact reference parameterization. Here
each candidate uses the unchanged native IPOPT/LIBERO recording, not raw ctrl.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import mujoco
import numpy as np

from closed_loop_state_metrics import evaluate_run, state_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = evaluate_run(parent)
    pro = json.loads((parent/'provenance.json').read_text())
    job = pro['plan'][0]
    dest = parent/job['suite']/f"{job['task_id']:02d}"/job['demo']
    with h5py.File(dest/'ipopt_state_B10/recording.hdf5') as file:
        model = mujoco.MjModel.from_xml_string(file['attempts'][job['demo']].attrs['model_file'])
    with h5py.File(dest/'state_reference.hdf5') as file:
        count = len(file['actions'])
    errors = np.load(parent/'physical_state_errors.npz')['position_tangent']
    _, objects = state_groups(model)
    moving = [o for o in objects if o['translation_dofs']]
    if moving:
        item = max(moving, key=lambda o: np.linalg.norm(errors[-1,o['translation_dofs']]))
        values = np.linalg.norm(errors[:,item['translation_dofs']],axis=1)
        threshold = max(.003,min(.01,float(values[-1])*.1))
        onset = np.flatnonzero(values > threshold)
        center = int(onset[0]) if len(onset) else int(np.argmax(values))
    else:
        center = int(np.argmax(np.linalg.norm(errors[:,:7],axis=1)))
    # Start corrections before the measured divergence; this is automatic,
    # identical across tasks, and does not alter the original command clock.
    left, right = max(0,center-30), min(count-1,center+15)
    if right-left < 10:
        right=min(count-1,left+30)
    knots=np.linspace(left,right,5)
    times=np.arange(count)
    basis=np.column_stack([np.interp(times,knots,np.eye(5)[i],left=0,right=0) for i in (1,2,3)])
    rng=np.random.default_rng(20261024)
    mean=np.zeros((3,6))
    std=np.tile(np.r_[np.full(3,.002),np.full(3,.01)],(3,1))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
    best_root, best_cost = parent, baseline['objective']
    history=[]
    (output/'policy.json').write_text(json.dumps(dict(method='CEM over three 6D reference correction knots',
        parent=str(parent),samples_per_iteration=4,iterations=2,elites=2,seed=20261024,
        full_length_native_rollouts=True,control_hz=20,gripper='unchanged',
        center=center,window=[left,right],cost='fixed closed_loop_state_metrics objective',
        baseline_task_success_must_be_retained=True),indent=2)+'\n')

    def run(round_index,index,parameters):
        correction=basis@parameters
        for start,limit in ((0,.005),(3,.03)):
            values=correction[:,start:start+3]
            values*=np.minimum(1.,limit/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-15))
        proposal=output/f'proposal_{round_index}_{index}.npz'
        np.savez_compressed(proposal,position=correction[:,:3],rotation=correction[:,3:])
        root=output/f'trial_{round_index}_{index}'
        with (output/f'trial_{round_index}_{index}.log').open('w') as stream:
            done=subprocess.run([sys.executable,str(Path(__file__).with_name('refine_reference_iteration.py')),
                '--parent',str(parent),'--output',str(root),'--proposal',str(proposal)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert done.returncode == 0, root
        metric=evaluate_run(root)
        allowed=metric['full_length'] and not metric['rejected'] and (not baseline['success'] or metric['success'])
        return dict(root=str(root),round=round_index,index=index,parameters=parameters.tolist(),
                    objective=metric['objective'],selection_cost=metric['objective'] if allowed else 1e15,
                    success=metric['success'],allowed=allowed)

    for round_index in range(2):
        noise=rng.normal(size=(2,3,6))
        samples=np.concatenate([mean[None]+std*noise,mean[None]-std*noise])
        results=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run,round_index,i,p) for i,p in enumerate(samples)]):
                result=future.result();results.append(result);history.append(result)
                if result['selection_cost'] < best_cost:
                    best_root,best_cost=Path(result['root']),result['selection_cost']
                summary=dict(complete=False,baseline_root=str(parent),baseline_objective=baseline['objective'],
                    selected_root=str(best_root),objective=best_cost,results=history)
                (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(json.dumps(result),flush=True)
        elites=sorted(results,key=lambda r:r['selection_cost'])[:2]
        elite=np.asarray([e['parameters'] for e in elites])
        mean=.5*mean+.5*elite.mean(axis=0)
        std=np.maximum(.6*std+.4*elite.std(axis=0),np.tile(np.r_[np.full(3,.0005),np.full(3,.002)],(3,1)))
    summary['complete']=True
    summary['success']=evaluate_run(best_root)['success']
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
