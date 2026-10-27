"""Read-only contact instrumentation around existing native env.step calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
TASKTOL=ROOT.parent/'franka_mujoco'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['ipopt','native_osc'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cases',type=int,nargs='+',default=[20,23])
    parser.add_argument('--kp',type=float,default=1000)
    parser.add_argument('--reference-mode',choices=['recorded_motion','source_goals'],default='recorded_motion')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['NUMBA_CACHE_DIR']=str(args.output/'numba_cache');os.environ['MPLCONFIGDIR']=str(args.output/'matplotlib')
    sys.path[:0]=[str(ROOT),str(TASKTOL),str(ROOT/'third_party/libero')]
    import mujoco
    import numpy as np
    from libero.libero.envs import OffScreenRenderEnv
    from tasktol.benchmark.libero import run_replay
    from experiments.libero_tasktol_compat.native_alignment import replay
    selection=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    original=OffScreenRenderEnv.step
    telemetry=[]
    def observed_step(env,action):
        result=original(env,action)
        model,data=env.sim.model._model,env.sim.data._data
        contacts=[]
        for i,contact in enumerate(data.contact[:data.ncon]):
            force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
            geoms=[int(contact.geom1),int(contact.geom2)]
            names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in geoms]
            bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or '' for g in geoms]
            contacts.append(dict(geoms=names,bodies=bodies,distance_m=float(contact.dist),normal_force_n=float(force[0])))
        objects={name:data.xpos[int(env.env.obj_body_id[name])].tolist() for name in env.env.objects_dict if name in env.env.obj_body_id}
        fixture_joints={name:{joint:float(data.qpos[model.jnt_qposadr[model.joint(joint).id]]) for joint in obj.joints} for name,obj in env.env.fixtures_dict.items()}
        telemetry.append(dict(index=len(telemetry),simulation_time=float(data.time),contacts=contacts,objects=objects,
            fixture_joints=fixture_joints,
            raw_torques=env.robots[0].controller.torques.tolist(),applied_torques=env.robots[0].torques.tolist(),
            fingers=env.robots[0].gripper.current_action.tolist()))
        return result
    OffScreenRenderEnv.step=observed_step
    summaries=[]
    for case in args.cases:
        telemetry.clear();t=selection[case];dest=args.output/t['suite']/f"{t['task_id']:02d}"
        if args.mode=='ipopt':
            report=run_replay(openpi_root=ROOT,output=dest,suite=t['suite'],task_id=t['task_id'],kp=args.kp,reference_mode=args.reference_mode)
        else:
            report=replay(case,dest,refresh_controller=True,source_reset_state=True,source_reset_history=True)
        (dest/'contacts.json').write_text(json.dumps(dict(read_only_instrumentation=True,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frames=telemetry),indent=2)+'\n')
        summaries.append(dict(case=case,success=report['task_success'],contact_frames=len(telemetry)))
        print(json.dumps(summaries[-1]),flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
