"""Isolated default-IPOPT replay of slow-success and rejected single frames.

No physics steps and no changes to solver settings, objectives or constraints.
Native models and accepted-command history are restored from retained evidence.
Profiled repeats are separate diagnostics, never dataset candidates.
"""

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pstats
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent/'franka_mujoco'), str(ROOT/'third_party/libero')]


def plain(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def worker(case, demo, root, output, indices):
    import h5py
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from examples.libero import regenerate_libero_dataset as recorder
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from tasktol.control.ipopt import IpoptIK, IpoptSettings
    from tasktol.control.ipopt.adapter import IpoptAdapter, ACCEPTABLE_IPOPT_STATUS_CODES
    from tasktol.control.pose import TargetPose
    from tasktol.control.task_space.objective import ObjectiveSettings
    from tasktol.geometry import rotation_vector
    from tasktol.simulation.libero import libero_handles

    task = json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks'][case]
    dest = root/task['suite']/f"{task['task_id']:02d}"/demo
    rows = json.loads((dest/'ipopt_state_B10/trajectory.json').read_text())
    report = json.loads((dest/'ipopt_state_B10/report.json').read_text())
    with h5py.File(dest/'state_reference.hdf5', 'r') as file:
        witnesses = file['goal_joint_witness_rad'][()]
    with h5py.File(task['processed_hdf5'], 'r') as file:
        config = json.loads(file.attrs['configuration']); order = config['demo_ids']; number = order.index(demo)
        before = json.loads(file['regeneration/episodes'][order[number-1]].attrs['rng_state']) if number else None
        after = json.loads(file['regeneration/episodes'][demo].attrs['rng_state'])
    with h5py.File(task['source_hdf5'], 'r') as file:
        states = file['data'][demo]['states'][()]
    env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task['suite']/(task['task_name']+'.bddl')),
                     controller='OSC_POSE', use_camera_obs=False, has_offscreen_renderer=False, control_freq=20)
    output.mkdir(parents=True, exist_ok=False)
    try:
        env.seed(config['seed']); env.env.object_property_initializers *= number+1
        recorder._restore_rng(before if before is not None else dict(numpy=recorder._rng_state()['numpy'], python=after['python']))
        env.reset(); env.set_init_state(states[0])
        assert plain(recorder._rng_state()) == after
        model = env.sim.model._model; handles = libero_handles(env)
        physical_hash = hashlib.sha256(b''.join(np.asarray(getattr(model,k)).tobytes() for k in
            ['body_pos','body_quat','body_mass','body_inertia','geom_friction','jnt_range'])).hexdigest()
        assert physical_hash == report['model_sha256']
        ik = IpoptIK(model, handles)
        assert ik.settings == IpoptSettings()
        assert ik.objective_settings == ObjectiveSettings.delta_q_squared_conditioning_safeguard()
        initial_q = np.asarray(rows[0]['pre_action']['joint_position_rad'])

        def restore(index):
            trial = IpoptIK(model, handles)
            if index:
                trial._xopt = np.asarray(rows[index-1]['joint_target_rad']).copy()
                trial._previous_state = np.asarray(rows[index-2]['joint_target_rad']).copy() if index >= 2 else initial_q.copy()
                trial._previous_state2 = np.asarray(rows[index-3]['joint_target_rad']).copy() if index >= 3 else initial_q.copy()
            return trial

        results = []
        original_solve = IpoptAdapter.solve
        for index in indices:
            row = rows[index]
            target = TargetPose(np.asarray(row['nominal_target']['position_m']),
                Rotation.from_rotvec(row['nominal_target']['rotation_vector_rad']).as_matrix())
            actual = np.asarray(row['pre_action']['joint_position_rad'])
            data = mujoco.MjData(model); data.qpos[handles.qpos_ids] = actual
            source_pose = ik.kinematics.evaluate(actual, include_manipulability=False, include_link_points=False)
            np.testing.assert_allclose(source_pose.position, row['pre_action']['position_m'], rtol=0, atol=1e-10)
            assert np.linalg.norm(rotation_vector(source_pose.rotation @ Rotation.from_rotvec(row['pre_action']['rotation_vector_rad']).as_matrix().T)) < 1e-10

            def geometry(q):
                q = np.asarray(q)
                pose = ik.kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
                jac = pose.jacobian.copy(); jac[:3] /= .3
                sv = np.linalg.svd(jac, compute_uv=False)
                margins = np.minimum(q-handles.joint_lower, handles.joint_upper-q)
                return dict(q_rad=q, position_residual_m=float(np.max(np.abs(pose.position-target.position))),
                    rotation_residual_rad=float(np.max(np.abs(rotation_vector(target.rotation.T @ pose.rotation)))),
                    sigma_min_length_scaled=float(sv[-1]), condition_length_scaled=float(sv[0]/sv[-1]),
                    min_limit_margin_deg=float(np.degrees(margins.min())), closest_joint=int(np.argmin(margins))+1)

            witness = geometry(witnesses[index])
            assert witness['position_residual_m'] < 1e-10 and witness['rotation_residual_rad'] < 1e-9
            repeats = []
            for repeat in range(3):
                trial = restore(index); raw = []
                def observe(adapter, *args, **kwargs):
                    result = original_solve(adapter, *args, **kwargs); raw.append(result); return result
                IpoptAdapter.solve = observe
                cpu, wall = time.process_time(), time.perf_counter()
                try:
                    q, diagnostics = trial.solve(data, target, np.ones(6, dtype=bool))
                finally:
                    elapsed_wall = (time.perf_counter()-wall)*1000
                    elapsed_cpu = (time.process_time()-cpu)*1000
                    IpoptAdapter.solve = original_solve
                result = raw[0]
                repeats.append(dict(repeat=repeat, adapter_wall_ms=result.elapsed_ms, total_ik_wall_ms=elapsed_wall,
                    total_ik_cpu_ms=elapsed_cpu, iterations=result.iterations, status_code=result.status_code,
                    status=result.status, accepted_status=result.status_code in ACCEPTABLE_IPOPT_STATUS_CODES,
                    candidate_validation=asdict(result.validation), published=bool(diagnostics.feasible),
                    raw_candidate=geometry(result.q), published_command=geometry(q)))
            # Derivative checks are outside all timed solves. They do not alter
            # settings, warm starts, objectives, commands or physical state.
            check_trial = restore(index)
            nlp = check_trial._prepare_single_step_nlp(data, target, np.ones(6,dtype=bool))
            derivative_checks = []
            for name, point in [('seed', nlp.seed), ('offline_witness', witnesses[index]),
                                ('raw_candidate', raw[0].q)]:
                point = np.asarray(point).copy(); epsilon = 1e-6
                _, (analytic_jac, _) = nlp.constraint_evaluation(point)
                analytic_grad = nlp.objective.gradient(point, check_trial.settings.derivative_epsilon,
                    handles.joint_lower, handles.joint_upper, check_trial.settings.expensive_gradient_refresh_rad)
                numeric_jac = np.zeros_like(analytic_jac); numeric_grad = np.zeros_like(point)
                for axis in range(len(point)):
                    shift = np.eye(len(point))[axis]*epsilon
                    numeric_jac[:,axis] = (nlp.constraint_evaluation(point+shift)[0].equality-
                        nlp.constraint_evaluation(point-shift)[0].equality)/(2*epsilon)
                    numeric_grad[axis] = (nlp.objective.evaluate(point+shift)-nlp.objective.evaluate(point-shift))/(2*epsilon)
                jac_error=float(np.max(np.abs(analytic_jac-numeric_jac)))
                grad_error=float(np.max(np.abs(analytic_grad-numeric_grad)))
                derivative_checks.append(dict(point=name,central_difference_epsilon_rad=epsilon,
                    equality_jacobian_max_abs_error=jac_error,objective_gradient_max_abs_error=grad_error,
                    passed=jac_error<1e-5 and grad_error<1e-5))
            # Instrumented repeat only for call counts/hotspots; overhead may
            # alter a CPU-limited solve, so it is not a timing/feasibility trial.
            profile = cProfile.Profile(); trial = restore(index)
            profile.runcall(trial.solve, data, target, np.ones(6, dtype=bool))
            profile.dump_stats(str(output/f'frame_{index:04d}.prof'))
            statistics = pstats.Stats(profile)
            hotspots = []
            for (filename, line, name), (primitive, calls, self_time, cumulative, _) in statistics.stats.items():
                hotspots.append(dict(file=filename, line=line, name=name, calls=calls, primitive_calls=primitive,
                    self_ms=self_time*1000, cumulative_ms=cumulative*1000))
            hotspots.sort(key=lambda item:item['self_ms'], reverse=True)
            calls = {name:sum(item['calls'] for item in hotspots if item['name']==name)
                     for name in ('mj_forward','mj_jacSite','evaluate','gradient','objective','constraints','jacobian','intermediate')}
            seed = initial_q if not index else np.asarray(rows[index-1]['joint_target_rad'])
            result = dict(case=case, demo=demo, frame=index, original_optimizer=row['optimizer'], original_aborted=row['aborted'],
                actual=geometry(actual), seed=geometry(seed), witness=witness,
                target_step_position_mm=float(np.linalg.norm(np.asarray(row['action'][:3]))*1000),
                target_step_rotation_deg=float(np.degrees(np.linalg.norm(row['action'][3:6]))),
                witness_to_seed_joint_distance_rad=float(np.linalg.norm(witnesses[index]-seed)),
                repeats=repeats, derivative_checks=derivative_checks,
                profiled_hotspots=hotspots[:35], profiled_call_counts=calls)
            results.append(result)
            (output/'results.json').write_text(json.dumps(plain(dict(case=case,demo=demo,results=results,
                solver_settings=asdict(IpoptSettings()), objective_settings=asdict(ik.objective_settings),
                model_physical_hash_matches=True, source_fk_matches=True, physics_steps=0)),indent=2)+'\n')
            print(json.dumps(dict(case=case,demo=demo,frame=index,original_aborted=row['aborted'],
                warm_ms=[r['adapter_wall_ms'] for r in repeats],published=[r['published'] for r in repeats])),flush=True)
    finally:
        IpoptAdapter.solve = original_solve if 'original_solve' in locals() else IpoptAdapter.solve
        env.close()


def run(output):
    batch = ROOT/'experiments/libero_tasktol_compat/results/task_pose_default_ipopt400_v1'
    pid = json.loads((batch/'batch_host_process.json').read_text())['batch_pid']
    def resume():
        try: os.kill(pid, signal.SIGCONT)
        except ProcessLookupError: pass
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    output.mkdir(parents=True, exist_ok=False)
    try:
        # Dispatcher is already stopped. Existing candidate workers may finish;
        # children belong to the worker threads that spawned them.
        for _ in range(120):
            children = set()
            for task in Path(f'/proc/{pid}/task').glob('*/children'):
                children.update(task.read_text().split())
            active = []
            for child in children:
                stat = Path(f'/proc/{child}/stat')
                if stat.exists() and stat.read_text().split()[2] != 'Z': active.append(int(child))
            if not active: break
            time.sleep(1)
        assert not active, active
        summary = json.loads((batch/'summary.json').read_text())
        jobs = []
        for result in summary['results']:
            if 'error' in result: continue
            root = Path(result['baseline_root'])
            rows = json.loads((root/result['suite']/f"{result['task_id']:02d}"/result['demo']/'ipopt_state_B10/trajectory.json').read_text())
            if rows[-1]['aborted']:
                jobs.append(dict(case=result['case'],demo=result['demo'],root=str(root),
                                 indices=sorted(set([max(0,len(rows)-2),len(rows)-1]))))
        first = summary['results'][0]
        jobs.insert(0,dict(case=first['case'],demo=first['demo'],root=first['baseline_root'],indices=[0,1,20]))
        (output/'plan.json').write_text(json.dumps(dict(jobs=jobs,default_settings_unchanged=True,
            batch_dispatcher_stopped=True,active_recording_children_at_test=active,
            profile_overhead_excluded_from_timing_repeats=True),indent=2)+'\n')
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',
            MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')
        results = []
        for job in jobs:
            dest=output/f"case{job['case']:02d}_{job['demo']}"
            with dest.with_suffix('.log').open('w') as log:
                done = subprocess.run([sys.executable,str(Path(__file__)),'--worker','--case',str(job['case']),
                    '--demo',job['demo'],'--root',job['root'],'--output',str(dest),'--indices',*[str(i) for i in job['indices']]],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            assert done.returncode == 0, dest
            results.append(json.loads((dest/'results.json').read_text()))
            (output/'summary.json').write_text(json.dumps(dict(complete=len(results)==len(jobs),results=results),indent=2)+'\n')
            print(json.dumps(dict(completed=len(results),planned=len(jobs),case=job['case'],demo=job['demo'])),flush=True)
    finally:
        resume()
        (output/'batch_resumed.json').write_text(json.dumps(dict(batch_pid=pid,resume_signal_sent=True))+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--case',type=int)
    parser.add_argument('--demo');parser.add_argument('--root',type=Path);parser.add_argument('--indices',type=int,nargs='+')
    args=parser.parse_args()
    if args.worker: worker(args.case,args.demo,args.root,args.output,args.indices)
    else: run(args.output.resolve())
