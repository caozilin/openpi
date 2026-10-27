"""Offline feasibility probes, not rollouts or deployable fallback behavior."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL=ROOT.parent/'franka_mujoco'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
sys.path[:0]=[str(TASKTOL),str(ROOT/'third_party/libero')]


def main():
    import h5py
    import numpy as np
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.simulation.libero import libero_handles
    from tasktol.control.ipopt import IpoptIK, IpoptSettings
    from tasktol.control.ipopt.adapter import IpoptAdapter
    from tasktol.control.pose import TargetPose
    from tasktol.geometry import rotation_matrix

    output=ROOT/'experiments/libero_tasktol_compat/results/rejected_targets_probe_v2'
    output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(output/'numba_cache')
    selection=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    results=[]
    for case in [4,22,38]:
        selected=selection[case]
        baseline=TASKTOL/'experiments/libero_closed_loop/results/native_nominal40'/selected['suite']/f"{selected['task_id']:02d}"
        rows=json.loads((baseline/'trajectory.json').read_text())['frames']
        row=rows[-1]
        target=TargetPose(np.array(row['nominal_target']['position_m']),rotation_matrix(np.array(row['nominal_target']['rotation_vector_rad'])))
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/selected['suite']/(selected['task_name']+'.bddl')),controller='JOINT_POSITION',camera_names=['agentview'],camera_heights=32,camera_widths=32,control_freq=20)
        try:
            env.seed(0);env.reset()
            with h5py.File(selected['processed_hdf5'],'r') as f:
                states=f['data'][selected['demo_id']]['tasktol/pre_action_states']
                env.set_init_state(states[0])
                handles=libero_handles(env)
                source_q=states[row['index'],1+handles.qpos_ids]
            settings=IpoptSettings(max_iterations=300,max_cpu_time_s=5.0)
            solver=IpoptIK(env.sim.model._model,handles,settings)
            env.sim.data._data.qpos[handles.qpos_ids]=row['pre_action']['joint_position_rad']
            problem=solver._prepare_single_step_nlp(env.sim.data._data,target,np.ones(6,dtype=bool))
            seeds=dict(last_command=np.array(row['joint_target_rad']),actual_q=np.array(row['pre_action']['joint_position_rad']),source_q=source_q)
            for name,seed in seeds.items():
                result=IpoptAdapter(settings).solve(
                    lambda q:float(problem.objective.evaluate(q)),
                    lambda q:problem.objective.gradient(q,settings.derivative_epsilon,handles.joint_lower,handles.joint_upper),
                    lambda q:problem.constraint_evaluation(q)[0],lambda q,v:problem.constraint_evaluation(q)[1],
                    seed,handles.joint_lower,handles.joint_upper,
                    position_tolerance=settings.position_tolerance,rotation_tolerance=settings.rotation_tolerance,
                    inequality_tolerance=settings.inequality_tolerance,
                )
                item=dict(case=case,suite=selected['suite'],task_id=selected['task_id'],frame=row['index'],seed=name,
                    feasible=bool(result.validation.feasible),status=result.status,status_code=int(result.status_code),
                    position_residual_m=float(result.constraints.position_residual),rotation_residual_rad=float(result.constraints.rotation_residual),
                    iterations=int(result.iterations),elapsed_ms=float(result.elapsed_ms),candidate_q=result.q.tolist(),
                    target_position=target.position.tolist(),target_rotation=target.rotation.tolist(),
                    joint_lower=handles.joint_lower.tolist(),joint_upper=handles.joint_upper.tolist())
                results.append(item)
                (output/'summary.json').write_text(json.dumps(dict(diagnostic_only=True,physics_steps=0,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=results),indent=2)+'\n')
                print(json.dumps(item),flush=True)
        finally:
            env.close()


if __name__=='__main__':main()
