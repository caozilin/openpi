"""Fresh-process replay of frozen selected actions, without offline correction."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from experiments.libero_tasktol_compat import reconstruct_joint_servo as reconstruction
from experiments.libero_tasktol_compat import rollout_mujoco_joint_servo as servo
from experiments.libero_tasktol_compat import rollout_mask_tolerance as runner


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args()
    run=a.run.resolve();summary=json.loads((run/'summary.json').read_text())
    plan=json.loads((run/'plan.json').read_text());assert reconstruction.freeze()==plan['source_sha256']
    if a.worker:
        import numpy as np
        selection=next(r['selected'] for r in summary['results']if r['seed']==a.seed)
        ref=reconstruction.load_reference(Path(selection['reference_path']))
        def transform(original,session):
            np.testing.assert_array_equal(original['actions'][:,6],ref['actions'][:,6])
            return ref
        dest=run/'cold_replay'/f'seed_{a.seed:02d}'
        servo.worker(dest,a.seed,'joint_native',reference_transform=transform)
        return
    root=run/'cold_replay';root.mkdir(exist_ok=True);shutil.copy2(__file__,root/Path(__file__).name)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
    def replay(row):
        import numpy as np
        seed=row['seed'];dest=root/f'seed_{seed:02d}'
        cold=dest/'joint_native'/f'seed_{seed:02d}/off';selected=Path(row['selected']['folder'])
        if not (cold/'report.json').exists():
            runner.dump(dest/'plan.json',plan);runner.dump(dest/'joint_native/plan.json',plan)
            with (dest/'worker.log').open('w')as log:
                done=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--run',str(run),'--worker','--seed',str(seed)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            assert done.returncode==0,f'Failed cold replay: {dest}'
        report=json.loads((cold/'report.json').read_text());old=json.loads((selected/'report.json').read_text())
        for key in ['success','planned_frames','executed_frames','execution_rejected','actions_sha256','initial_state_sha256','model_sha256','shared_joint_servo']:
            assert report[key]==old[key]
        assert report['success']
        for file in ['states.npz','servo_physics.npz']:
            with np.load(cold/file)as f,np.load(selected/file)as g:
                assert f.files==g.files
                for k in f.files:np.testing.assert_array_equal(f[k],g[k])
        before=json.loads((selected/'trajectory.json').read_text());after=json.loads((cold/'trajectory.json').read_text())
        assert len(before)==len(after)
        for first,second in zip(before,after,strict=True):
            for key in ['action','pre_action','post_action','nominal_target','joint_target_rad','native_joint_action','phase','success','aborted']:
                assert first[key]==second[key]
        return dict(seed=seed,success=True,executed_frames=report['executed_frames'],
            state_joint_targets_and_physics_servo_samples_bitwise_identical=True,folder=str(cold))
    results=[]
    with ThreadPoolExecutor(max_workers=2)as pool:
        for f in as_completed([pool.submit(replay,r)for r in summary['results']]):
            row=f.result();results.append(row);print(json.dumps(row),flush=True)
            runner.dump(root/'validation.json',dict(complete=len(results)==10,results=results,
                source_unchanged=reconstruction.freeze()==plan['source_sha256'],replay_script_sha256=runner.sha(Path(__file__))))


if __name__=='__main__':main()
