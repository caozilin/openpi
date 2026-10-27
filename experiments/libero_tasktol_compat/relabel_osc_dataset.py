"""Write new reference-action sidecars for all retained LIBERO episodes.

No physics is stepped. Original HDF5 files and images stay in place; each new
demo links to its original group. This is an OSC-reference diagnostic dataset,
not newly collected IPOPT training rollouts.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL=ROOT.parent/'franka_mujoco'
SELECTION=ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json'


def relabel(case, output):
    output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(output/'numba_cache')
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib')
    sys.path[:0]=[str(ROOT),str(TASKTOL),str(ROOT/'third_party/libero')]
    import h5py
    import mujoco
    import numpy as np
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from tasktol.geometry import rotation_vector
    from tasktol.simulation.libero import libero_handles
    from experiments.libero_tasktol_compat.osc_reference import (
        decode_source_goals, difference_reference_goals, integrate_reference_actions,
    )
    if mujoco.__version__!='3.2.3':
        raise RuntimeError('Use the pinned native LIBERO environment')
    selected=json.loads(SELECTION.read_text())['tasks'][case]
    source=Path(selected['processed_hdf5'])
    bddl=Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')
    env=ControlEnv(bddl_file_name=str(bddl),controller='OSC_POSE',use_camera_obs=False,
                   has_renderer=False,has_offscreen_renderer=False,control_freq=20)
    final=output/'reference_actions.hdf5'
    partial=output/'reference_actions.hdf5.partial'
    episodes=[]
    try:
        env.seed(0);env.reset()
        robot=env.robots[0];ctrl=robot.controller
        if float(robot.initialization_noise['magnitude'])!=0:
            raise RuntimeError('Per-episode reset orientation required when robot initialization noise is enabled')
        if not ctrl.use_delta or ctrl.position_limits is not None or ctrl.orientation_limits is not None:
            raise RuntimeError('Unexpected source OSC action contract')
        handles=libero_handles(env)
        model=env.sim.model._model
        data=mujoco.MjData(model)
        held_rotation=ctrl.goal_ori @ handles.ee_rotation_offset
        with h5py.File(source,'r') as src,h5py.File(partial,'w') as dst:
            source_config=json.loads(src['source_metadata'].attrs['env_args'])['env_kwargs']['controller_configs']
            if source_config['type']!='OSC_POSE' or not np.allclose(source_config['output_max'],ctrl.output_max) or not np.allclose(source_config['output_min'],ctrl.output_min):
                raise RuntimeError('Source OSC scaling differs from native decoder')
            dst.attrs['schema']='libero_osc_independent_reference_actions_v1'
            dst.attrs['source_hdf5']=str(source)
            dst.attrs['action_contract']=json.dumps(dict(
                position='world/base metres, difference of adjacent desired goals',
                rotation='world-frame rotation vector radians: Log(R_goal[t] @ R_goal[t-1].T)',
                first_action='difference from saved explicit actual-pose anchor',gripper='source action unchanged',
                execution='accumulate independent reference, then native absolute OSC; not native normalized delta actions',
                provenance='offline relabel of OSC source data; not IPOPT recollection',
            ))
            dst.attrs['native_initialization_noise_magnitude']=float(robot.initialization_noise['magnitude'])
            dst.attrs['native_reset_held_tool_rotation']=held_rotation
            dst.attrs['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            dst.attrs['reference_module_sha256']=hashlib.sha256((Path(__file__).with_name('osc_reference.py')).read_bytes()).hexdigest()
            group=dst.create_group('data')
            for demo_id in sorted(src['data'],key=lambda x:int(x.split('_')[-1])):
                original=src['data'][demo_id]
                states=original['tasktol/pre_action_states'][()]
                actions=original['actions'][()]
                positions=[];rotations=[]
                for state in states:
                    data.qpos[:]=state[1:1+model.nq]
                    # Full stored qpos -> native forward kinematics only.
                    # No dynamics, controller calls, camera renders or resets.
                    mujoco.mj_kinematics(model,data)
                    positions.append(data.site_xpos[handles.ee_site_id].copy())
                    rotations.append(data.site_xmat[handles.ee_site_id].reshape(3,3) @ handles.ee_rotation_offset)
                positions=np.asarray(positions);rotations=np.asarray(rotations)
                scaled=np.array([ctrl.scale_action(a[:6]) for a in actions])
                goals_p,goals_r=decode_source_goals(positions,rotations,scaled,held_rotation)
                increments=difference_reference_goals(goals_p,goals_r,actions[:,6],positions[0],rotations[0])
                rebuilt_p,rebuilt_r=integrate_reference_actions(increments,positions[0],rotations[0])
                p_error=float(np.max(np.linalg.norm(rebuilt_p-goals_p,axis=1)))
                r_error=float(max(np.linalg.norm(rotation_vector(a@b.T)) for a,b in zip(rebuilt_r,goals_r,strict=True)))
                if p_error>1e-10 or r_error>1e-9:
                    raise RuntimeError(f'Reconstruction failed: {demo_id}')
                demo=group.create_group(demo_id)
                demo.attrs['num_samples']=len(actions)
                demo.attrs['input_state_action_sha256']=hashlib.sha256(states.tobytes()+actions.tobytes()).hexdigest()
                demo.attrs['reconstruction_position_max_m']=p_error
                demo.attrs['reconstruction_rotation_max_rad']=r_error
                demo['source_demo']=h5py.ExternalLink(str(source),original.name)
                for name,value in dict(actions=increments,anchor_position_m=positions[0],anchor_tool_rotation=rotations[0],
                                       goal_positions_m=goals_p,goal_tool_rotations=goals_r).items():
                    demo.create_dataset(name,data=value,compression='gzip')
                episodes.append(dict(demo=demo_id,frames=len(actions),reconstruction_position_max_m=p_error,
                                     reconstruction_rotation_max_rad=r_error))
            dst.attrs['episodes']=len(episodes)
            dst.attrs['frames']=sum(e['frames'] for e in episodes)
        partial.replace(final)
        result=dict(case=case,suite=selected['suite'],task_id=selected['task_id'],source=str(source),
                    output=str(final),episodes=len(episodes),frames=sum(e['frames'] for e in episodes),
                    physics_steps=0,ipopt_calls=0,original_modified=False,episode_checks=episodes)
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',type=int)
    parser.add_argument('--cases',type=int,nargs='+',default=list(range(40)))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.case is not None:
        result=relabel(args.case,args.output)
        print(json.dumps({k:v for k,v in result.items() if k!='episode_checks'}),flush=True)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads(SELECTION.read_text())['tasks']
    def run(case):
        task=selected[case];dest=args.output/task['suite']/f"{task['task_id']:02d}"
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.with_suffix('.log').open('w') as log:
            done=subprocess.run([sys.executable,__file__,'--case',str(case),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT)
        manifest=dest/'manifest.json'
        return json.loads(manifest.read_text()) if manifest.exists() else dict(case=case,error='worker_failed',returncode=done.returncode)
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,c) for c in args.cases]):
            result=future.result();results.append(result)
            (args.output/'summary.json').write_text(json.dumps(dict(complete=len(results)==len(args.cases),results=results),indent=2)+'\n')
            print(f"{len(results)}/{len(args.cases)} "+json.dumps({k:v for k,v in result.items() if k!='episode_checks'}),flush=True)
    if any('error' in r for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
