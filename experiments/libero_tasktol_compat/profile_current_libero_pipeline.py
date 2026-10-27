"""Observe the maintained cold-replay client; never implements physics stepping.

Camera-off is an ablation of observation overhead, not a deployment setting.
All modes assert replayed state and full joint targets against retained data.
"""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time
from functools import wraps

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/libero_tasktol_compat'), str(ROOT.parent/'franka_mujoco'), str(ROOT/'third_party/libero')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('cameras', 'no_cameras', 'profile'), required=True)
    args = parser.parse_args()
    import numpy as np
    import replay_state_dataset as replay
    from libero.libero.envs.env_wrapper import ControlEnv
    from tasktol.control.ipopt import IpoptIK, IpoptSettings
    from tasktol.control.task_space.kinematics import RobotKinematics
    from tasktol.simulation.libero import LiberoTaskTolSession
    from tasktol.simulation.libero_joint_controller import FactorizedReferenceController
    from robosuite.controllers.osc import OperationalSpaceController
    import tasktol.simulation.libero_joint_controller as factors
    import robosuite.controllers.osc as osc

    args.output.mkdir(parents=True, exist_ok=False)
    times = {}
    restorations = []
    graphics = {}
    original_init = LiberoTaskTolSession.__init__
    def capture_graphics(self, *a, **kw):
        original_init(self, *a, **kw)
        if args.mode != 'no_cameras':
            from OpenGL import GL
            for name, token in [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]:
                value = GL.glGetString(token)
                graphics[name] = None if value is None else value.decode()
    LiberoTaskTolSession.__init__ = capture_graphics
    restorations.append((LiberoTaskTolSession, '__init__', original_init))
    def observe(owner, name, label):
        original = getattr(owner, name)
        times[label] = []
        @wraps(original)
        def measured(*a, **kw):
            started = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                times[label].append((time.perf_counter()-started)*1000)
        setattr(owner, name, measured)
        restorations.append((owner, name, original))

    for owner, name, label in [
        (IpoptIK, 'solve', 'ik_total'),
        (LiberoTaskTolSession, 'step', 'client_step'),
        (ControlEnv, 'step', 'native_env_step'),
        (FactorizedReferenceController, 'run_controller', 'factor_controller'),
        (OperationalSpaceController, 'run_controller', 'original_osc_inside_factor'),
        (factors, 'opspace_matrices', 'factor_opspace_matrices'),
        (osc, 'opspace_matrices', 'native_opspace_matrices'),
        (RobotKinematics, 'evaluate', 'fk'),
    ]:
        observe(owner, name, label)
    # Keep one native backend; changing camera observation flags changes neither
    # step(), control frequency nor the physical model.
    if args.mode == 'no_cameras':
        replay.OffScreenRenderEnv = lambda **kw: ControlEnv(**kw, use_camera_obs=False, has_offscreen_renderer=False)
    profiler = cProfile.Profile() if args.mode == 'profile' else None
    wall = time.perf_counter()
    try:
        if profiler: profiler.enable()
        replay.replay(ROOT/'data/libero_tasktol_state/20261024_task_pose_default400_packaging_pilot',
                      0, 'demo_0', args.output/'replay')
    finally:
        if profiler: profiler.disable()
        for owner, name, original in reversed(restorations): setattr(owner, name, original)
    total = (time.perf_counter()-wall)*1000
    measures = {label: dict(calls=len(t), total_ms=sum(t), median_ms=float(np.median(t)),
                           p95_ms=float(np.percentile(t,95)), max_ms=max(t)) for label,t in times.items()}
    rows = json.loads((args.output/'replay/trajectory.json').read_text())
    raw_sources = json.loads((args.output.parent/'source_sha256.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in raw_sources.items())
    result = dict(mode=args.mode, whole_replay_wall_ms=total, timing=measures, graphics=graphics,
                  frames=len(rows), physics_steps=sum(r['physics_steps'] for r in rows),
                  solver_settings=IpoptSettings().__dict__, source_hashes_match=True,
                  raw_samples_ms=times, profiling_overhead_included=bool(profiler))
    if profiler:
        profiler.dump_stats(str(args.output/'pipeline.prof'))
        stats=pstats.Stats(profiler)
        result['hotspots'] = sorted([dict(file=f,line=l,name=n,calls=c,self_ms=s*1000,cumulative_ms=t*1000)
            for (f,l,n),(p,c,s,t,_) in stats.stats.items()], key=lambda x:x['self_ms'],reverse=True)[:80]
    (args.output/'timings.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw_samples_ms','hotspots')}),flush=True)

if __name__=='__main__': main()
