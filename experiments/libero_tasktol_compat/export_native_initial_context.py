"""Reconstruct source preparation with its original native replay function.

These are diagnostic initialization sidecars, never IPOPT training episodes.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/libero')]
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')


def main():
    import h5py
    import mujoco
    import numpy as np
    from PIL import Image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from examples.libero.regenerate_libero_dataset import _restore_rng,replay_demo
    parser=argparse.ArgumentParser();parser.add_argument('--cases',type=int,nargs='+',default=[33]);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(args.output/'numba_cache')
    selected=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    results=[]
    for case in args.cases:
        t=selected[case];dest=args.output/t['suite']/('%02d'%t['task_id']);dest.mkdir(parents=True)
        with h5py.File(t['processed_hdf5'],'r') as processed:
            order=json.loads(processed.attrs['configuration'])['demo_ids'];idx=order.index(t['demo_id'])
            previous=json.loads(processed['regeneration/episodes'][order[idx-1]].attrs['rng_state']) if idx else None
            expected=processed['data'][t['demo_id']]['tasktol/pre_action_states'][0]
            expected_img=processed['data'][t['demo_id']]['obs/agentview_rgb'][0]
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/t['suite']/(t['task_name']+'.bddl')),camera_heights=256,camera_widths=256)
        try:
            env.env.object_property_initializers *= idx+1
            env.seed(0)
            if previous is not None:_restore_rng(previous)
            original_step=env.step;count=[0];captured={}
            def observed_step(action):
                result=original_step(action);count[0]+=1
                if count[0]==10:
                    state=env.sim.get_state().flatten().copy()
                    spec=mujoco.mjtState.mjSTATE_INTEGRATION
                    integration=np.empty(mujoco.mj_stateSize(env.sim.model._model,spec))
                    mujoco.mj_getState(env.sim.model._model,env.sim.data._data,integration,spec)
                    ctrl=env.robots[0].controller
                    np.savez_compressed(dest/'initial_context.npz',integration_state=integration,
                        body_pos=env.sim.model.body_pos.copy(),body_quat=env.sim.model.body_quat.copy(),
                        goal_pos=ctrl.goal_pos.copy(),goal_ori=ctrl.goal_ori.copy(),initial_joint=ctrl.initial_joint.copy(),
                        gripper_action=env.robots[0].gripper.current_action.copy(),flat_state=state,
                        state_spec=int(spec))
                    captured.update(max_flat_state_difference=float(np.abs(state-expected).max()),
                        pixel_mae=float(np.abs(result[0]['agentview_image'].astype(float)-expected_img).mean()))
                    Image.fromarray(result[0]['agentview_image'][::-1,::-1].copy()).save(dest/'initial.png')
                return result
            env.step=observed_step
            with h5py.File(t['source_hdf5'],'r') as raw:
                replay=replay_demo(env,raw['data'][t['demo_id']])
            result=dict(case=case,suite=t['suite'],task_id=t['task_id'],task_success=bool(replay['success']),
                native_original_environment=True,numpy=np.__version__,**captured)
            (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result);print(json.dumps(result),flush=True)
            obs=env.env._get_observations()
            Image.fromarray(obs['agentview_image'][::-1,::-1].copy()).save(dest/'final.png')
        finally:env.close()
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
