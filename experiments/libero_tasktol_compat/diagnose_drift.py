"""Instrument the existing native LIBERO probe; never replace its physics loop.

Profiles change ONLY the named MuJoCo collision options, before any steps and
after every model rebuild. Traces observe the original controller and sim.step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--collision-profile', choices=['default', 'no-native-ccd', 'no-multiccd', 'legacy-collision'], default='default')
    parser.add_argument('--trace-physics', action='store_true')
    parser.add_argument('--disable-islands', action='store_true')
    parser.add_argument('--legacy-forward-warmstart', action='store_true')
    parser.add_argument('--same-state-trace', type=Path)
    parser.add_argument('--same-state-index', type=int, default=1526)
    opts, remaining = parser.parse_known_args()
    if '--output' not in remaining:
        raise ValueError('--output is required')
    out = Path(remaining[remaining.index('--output')+1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import smoke_test
    # Environment imports require these paths/settings before monkeypatching.
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(out/'numba_cache'))
    os.environ.setdefault('MPLCONFIGDIR', str(out/'matplotlib'))
    sys.path[:0] = [str(smoke_test.TASKTOL), str(smoke_test.LIBERO)]
    import mujoco
    import numpy as np
    from libero.libero.envs.env_wrapper import ControlEnv
    from robosuite.utils.binding_utils import MjSim

    context = dict(env=None, frame=None, call=0)
    numeric, contact_rows, models = [], [], []
    original_init, original_step, original_env_step = MjSim.__init__, MjSim.step, ControlEnv.step
    original_forward = MjSim.forward

    def options(model):
        names = ['timestep', 'integrator', 'cone', 'solver', 'jacobian', 'iterations',
                 'ls_iterations', 'noslip_iterations', 'ccd_iterations', 'tolerance',
                 'ls_tolerance', 'noslip_tolerance', 'ccd_tolerance', 'impratio',
                 'gravity', 'density', 'viscosity', 'disableflags', 'enableflags']
        values = {n:np.asarray(getattr(model.opt,n)).tolist() for n in names if hasattr(model.opt,n)}
        for key, disable, enable in [('native_ccd','mjDSBL_NATIVECCD','mjENBL_NATIVECCD'),
                                     ('multiccd','mjDSBL_MULTICCD','mjENBL_MULTICCD'),
                                     ('island','mjDSBL_ISLAND','mjENBL_ISLAND')]:
            if hasattr(mujoco.mjtDisableBit,disable):
                values[key] = not bool(model.opt.disableflags & int(getattr(mujoco.mjtDisableBit,disable)))
            else:
                values[key] = bool(model.opt.enableflags & int(getattr(mujoco.mjtEnableBit,enable)))
        return values

    def construct(self, model):
        original_init(self,model)
        before = options(model)
        if opts.collision_profile != 'default':
            if opts.collision_profile in ['no-native-ccd','legacy-collision']:
                model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            if opts.collision_profile in ['no-multiccd','legacy-collision']:
                model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        if opts.disable_islands:
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
        models.append(dict(before=before,after=options(model)))
        fields = ['body_mass','body_inertia','body_pos','body_quat','geom_size','geom_pos',
                  'geom_quat','geom_friction','geom_solref','geom_solimp','geom_margin','geom_gap',
                  'geom_contype','geom_conaffinity','dof_damping','dof_armature','jnt_stiffness',
                  'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_ctrlrange','actuator_forcerange',
                  'mesh_vert','mesh_vertadr','mesh_vertnum','mesh_pos','mesh_quat','geom_type','geom_dataid','geom_bodyid']
        np.savez_compressed(out/'compiled_model.npz', **{n:np.asarray(getattr(model,n)).copy() for n in fields if hasattr(model,n)})
        names=dict(geoms=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)],
                   bodies=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)],
                   joints=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),
                                qpos=int(model.jnt_qposadr[i]),dof=int(model.jnt_dofadr[i])) for i in range(model.njnt)])
        (out/'model_names.json').write_text(json.dumps(names,indent=2)+'\n')
        if opts.same_state_trace:
            import copy
            data=copy.copy(self.data._data)
            with np.load(opts.same_state_trace) as trace:
                index=opts.same_state_index
                data.qpos[:]=trace['qpos_before'][index]
                data.qvel[:]=trace['qvel_before'][index]
                data.ctrl[:]=trace['ctrl'][index]
                data.time=float(trace['time'][index])
            data.qacc_warmstart[:]=0
            data.qfrc_applied[:]=0
            data.xfrc_applied[:]=0
            mujoco.mj_forward(model,data)
            np.savez_compressed(out/'same_state.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
                qacc=data.qacc.copy(),bias=data.qfrc_bias.copy(),constraint=data.qfrc_constraint.copy(),
                actuator=data.qfrc_actuator.copy())
            contacts=[dict(geoms=[names['geoms'][int(c.geom1)],names['geoms'][int(c.geom2)]],
                           dist=float(c.dist),pos=np.asarray(c.pos).tolist()) for c in data.contact[:data.ncon]]
            (out/'same_state_contacts.json').write_text(json.dumps(contacts,indent=2)+'\n')
        (out/'model_options.json').write_text(json.dumps(dict(mujoco=mujoco.__version__,profile=opts.collision_profile,models=models),indent=2)+'\n')

    def env_step(self,action):
        context.update(env=self,frame=context['call']-10)
        context['call'] += 1
        try:
            return original_env_step(self,action)
        finally:
            context.update(env=None,frame=None)

    def forward(self,*args,**kwargs):
        value=original_forward(self,*args,**kwargs)
        if opts.legacy_forward_warmstart:
            # Documented MuJoCo 3.3.6 migration for clients relying on the old
            # mj_forward side effect. Opt-in diagnostic, not a solver rewrite.
            self.data.qacc_warmstart[:]=self.data.qacc
        return value

    def contacts(sim):
        answer=[]
        for c in sim.data.contact[:sim.data.ncon]:
            names=[sim.model.geom_id2name(int(c.geom1)),sim.model.geom_id2name(int(c.geom2))]
            if not any(n and ('robot0' in n or 'gripper0' in n) for n in names):
                continue
            answer.append(dict(geoms=names,dist=float(c.dist),pos=np.asarray(c.pos).tolist(),
                               dim=int(c.dim),efc_address=int(c.efc_address)))
        return answer

    def step(self,*args,**kwargs):
        if not opts.trace_physics or context['env'] is None:
            return original_step(self,*args,**kwargs)
        controller=context['env'].robots[0].controller
        row=dict(frame=context['frame'],time=float(self.data.time),
                 qpos_before=self.data.qpos.copy(),qvel_before=self.data.qvel.copy(),
                 ctrl=self.data.ctrl.copy(),mass=controller.mass_matrix.copy(),
                 jacobian=controller.J_full.copy(),goal_pos=controller.goal_pos.copy(),
                 goal_ori=controller.goal_ori.copy())
        before=contacts(self)
        value=original_step(self,*args,**kwargs)
        row.update(qpos_after=self.data.qpos.copy(),qvel_after=self.data.qvel.copy(),
                   qacc=self.data.qacc.copy(),constraint_force=self.data.qfrc_constraint.copy(),
                   ncon=int(self.data.ncon))
        numeric.append(row)
        contact_rows.append(dict(index=len(numeric)-1,frame=context['frame'],time=row['time'],before=before,after=contacts(self)))
        return value

    MjSim.__init__, MjSim.step, ControlEnv.step = construct, step, env_step
    MjSim.forward=forward
    sys.argv=[sys.argv[0],*remaining]
    try:
        status=smoke_test.main()
    finally:
        MjSim.__init__, MjSim.step, ControlEnv.step = original_init, original_step, original_env_step
        MjSim.forward=original_forward
        if numeric:
            np.savez_compressed(out/'physics_trace.npz',**{key:np.asarray([r[key] for r in numeric]) for key in numeric[0]})
            with (out/'robot_contacts.jsonl').open('w') as file:
                for row in contact_rows:
                    file.write(json.dumps(row,separators=(',',':'))+'\n')
        (out/'diagnostic.json').write_text(json.dumps(dict(profile=opts.collision_profile,
            disable_islands=opts.disable_islands,legacy_forward_warmstart=opts.legacy_forward_warmstart,
            trace_physics=opts.trace_physics,physics_steps=len(numeric),
            tasktol_optimizer_imported='tasktol.control.ipopt' in sys.modules,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
