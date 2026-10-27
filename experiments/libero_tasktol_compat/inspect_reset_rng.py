"""Compare source reset RNG consumption under the original and pinned runtimes."""
import json
import os
from pathlib import Path
import sys

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/libero')]
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')


def main():
    import argparse
    import h5py
    import numpy as np
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from examples.libero.regenerate_libero_dataset import _restore_rng,_rng_state
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(args.output/'numba_cache')
    selected=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    results=[]
    for case in [4,23,38]:
        t=selected[case]
        with h5py.File(t['processed_hdf5'],'r') as f:
            order=json.loads(f.attrs['configuration'])['demo_ids'];index=order.index(t['demo_id'])
            previous=order[index-1] if index else None
            before=json.loads(f['regeneration/episodes'][previous].attrs['rng_state']) if previous else None
            expected=json.loads(f['regeneration/episodes'][t['demo_id']].attrs['rng_state'])
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/t['suite']/(t['task_name']+'.bddl')),camera_heights=32,camera_widths=32)
        try:
            env.seed(0)
            if before is not None:_restore_rng(before)
            env.reset()
            actual=_rng_state()
            fixtures={name:dict(position=env.sim.model.body_pos[env.sim.model.body_name2id(obj.root_body)].tolist(),rotation=env.sim.model.body_quat[env.sim.model.body_name2id(obj.root_body)].tolist()) for name,obj in env.env.fixtures_dict.items()}
            item=dict(case=case,numpy=np.__version__,rng_matches_source=actual==expected,
                expected_index=expected['numpy'][2],actual_index=actual['numpy'][2],fixtures=fixtures)
            results.append(item);print(json.dumps(item),flush=True)
        finally:env.close()
    (args.output/'report.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
