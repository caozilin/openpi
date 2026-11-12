"""Independently verify all400 model-predicted tolerance episodes; no inference."""

import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
sys.path.insert(0,str(ROOT))
RUN = ROOT / "data/libero_evaluation/pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    p = path.with_suffix(".tmp")
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    p.replace(path)


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--initial-sample', type=int, default=0)
    parser.add_argument('--follow', action='store_true', help='Read committed episodes as they appear; certify only after all clients exit')
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    assert args.initial_sample >= 0
    partial = args.initial_sample > 0
    assert not (partial and args.follow)
    os.environ.update(
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    sys.path[:0] = [str(RUN / "frozen_shared_runtime"), str(ROOT), str(ROOT / "third_party/libero")]
    from libero.libero.envs.env_wrapper import ControlEnv
    import mujoco
    import numpy as np
    from examples.libero.success_continuation import audit_record, audit_first_success_state

    from examples.libero.eval_joint13_nominal_libero_local import validate as validate_request, SUITES
    from examples.libero.nominal_joint13_policy import NominalDiagnosticPolicy as PredictedTolerancePolicy, POLICY
    from examples.libero.tasktol_actions import decode_tasktol_action

    lock = (RUN / ("initial_audit.lock" if partial else "independent_audit.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump(RUN / ("initial_audit_process.json" if partial else "independent_audit_process.json"), {"pid": os.getpid(), "command": str(Path(__file__).resolve())})
    request = read(RUN / "evaluation_request.json")
    validate_request(request)
    tolerance_settings = {'tolerance_bounds_policy': POLICY}
    def verify_finished_clients():
        completion = read(RUN / "evaluation_completion.json")
        assert completion["complete"]
        assert completion["exit_codes"] == [0] * 6
        assert completion["infrastructure_errors"] == 0
        for name in ["evaluation_process.json"] + [f"client_{i}_process.json" for i in range(6)]:
            assert not Path(f"/proc/{read(RUN / name)['pid']}").exists(), name
        with (RUN / "evaluation.lock").open("a") as evaluation_lock:
            fcntl.flock(evaluation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return completion
    if not partial and not args.follow:
        completion = verify_finished_clients()
    identities = read(RUN / "inference_identity_audit.json")
    assert identities["complete"]
    assert identities["server_entry_sha256"] == sha(RUN / "serve_joint13_libero_eval_server.py")
    tasks = read(RUN / "official_tasks.json")
    initial = np.load(RUN / "official_initial_states.npz", allow_pickle=False)
    paths = list(RUN.glob("episodes/*/task_*/seed_*/result.json"))
    actual = {(read(p)["suite"], read(p)["task_id"], read(p)["seed"]) for p in paths}
    selected = {(e["suite"], e["task_id"], e["seed"]) for e in request["selected_episodes"]}
    assert len(selected) == 400 and actual <= selected
    if partial:
        selected = set(sorted(actual)[:args.initial_sample])
        assert len(selected) == args.initial_sample
    elif not args.follow:
        assert actual == selected and len(paths) == 400
    verified = []
    def restore_cached_geometry(env, vector):
        env.set_state(vector)
        model, data = env.sim.model._model, env.sim.data._data
        qpos = data.qpos.copy()
        mujoco.mj_integratePos(model, data.qpos, data.qvel, -model.opt.timestep)
        env.sim.forward()
        data.qpos[:] = qpos
    episodes = sorted(request['selected_episodes'], key=lambda e:(e['seed'], SUITES.index(e['suite']), e['task_id'])) if args.follow else request['selected_episodes']
    for episode in episodes:
        suite, tid, seed = (episode[k] for k in ("suite", "task_id", "seed"))
        if (suite, tid, seed) not in selected:
            continue
        folder = RUN / "episodes" / suite / f"task_{tid:02d}" / f"seed_{seed}"
        deadline = time.monotonic() + 7200
        while args.follow and not (folder / 'result.json').exists():
            if list(RUN.glob('episodes/*/task_*/seed_*/infrastructure_error.json')):
                raise RuntimeError('Evaluation infrastructure error while following committed results')
            if time.monotonic() > deadline:
                raise TimeoutError(f'No committed episode: {folder}')
            time.sleep(5)
        result, rows = read(folder / "result.json"), read(folder / "trajectory.json")
        task = next(t for t in tasks if t["suite"] == suite and t["task_id"] == tid)
        assert sha(Path(task["source_init_states"])) == task["source_init_sha256"]
        assert result["complete"]
        assert result["eval_identity"] == request["eval_identity"]
        assert result["tolerance_enabled"] is False
        for key, value in tolerance_settings.items():
            if key == 'tolerance_bounds_policy' and value == 'uniform' and key not in result:
                continue  # Historical uniform-angle records predate this field.
            assert result.get(key) == value, key
        assert not result["state_input_enabled"]
        assert result["solver_failure_policy"] == "hold_last_accepted_reference"
        assert result["max_consecutive_solver_failures"] == 20
        assert result["solver_failure_timeout_seconds"] == 1.0
        assert (result["control_hz"], result["physics_hz"], result["settle_steps"], result["replan_steps"]) == (
            20,
            500,
            10,
            5,
        )
        assert result["max_steps"] == request["max_steps"][suite]
        assert result["gripper_deadband"] == 0.5
        assert all(result["server_metadata"][k] == identities["replicas"][result["server_metadata"]["replica"]][k] for k in result["server_metadata"])
        n = result["steps"]
        assert n == len(rows)
        assert 0 < n <= result["max_steps"] + result.get("post_success_steps", 0)
        assert result["inference_calls"] == (n + 4) // 5
        np.random.seed(seed)
        random.seed(seed)
        env = ControlEnv(
            bddl_file_name=str(
                ROOT / "third_party/libero/libero/libero/bddl_files" / task["problem_folder"] / task["bddl_file"]
            ),
            controller="JOINT_POSITION",
            use_camera_obs=False,
            has_offscreen_renderer=False,
            control_freq=20,
        )
        try:
            env.seed(seed)
            env.reset()
            with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
                actions, states, final = z["actions"], z["pre_action_states"], z["final_state"]
                assert all(np.isfinite(z[k]).all() for k in z.files)
                assert actions.shape == (n, 7)
                assert states.shape[0] == n
                official = initial[f"{suite}_{tid}_{seed}"]
                assert np.array_equal(z["initial_state"], official)
                assert hashlib.sha256(official.tobytes()).hexdigest() == result["initial_state_sha256"]
                baseline = Path(request['paired_predicted_tolerance_run']) / 'episodes' / suite / f'task_{tid:02d}' / f'seed_{seed}'
                baseline_result = read(baseline / 'result.json')
                baseline_rows = read(baseline / 'trajectory.json')
                assert result['server_metadata']['params_values_sha256'] == baseline_result['server_metadata']['params_values_sha256']
                with np.load(baseline / 'trajectory.npz', allow_pickle=False) as old:
                    np.testing.assert_allclose(states[0], old['pre_action_states'][0], atol=1e-12, rtol=0)
                first_chunk_differences = {field: float(np.max(np.abs(
                    np.asarray([x[field] for x in rows[:5]]) - np.asarray([x[field] for x in baseline_rows[:5]])
                ))) for field in ['model_action_raw', 'predicted_bounds_raw_rad']}
                assert all(np.isfinite(v) for v in first_chunk_differences.values())
                joint_indices = np.asarray(env.robots[0]._ref_joint_pos_indexes) + 1
                sys.path.insert(0,'/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-10_libero_joint13_nominal_400')
                from audit_nominal_reference import RecordedNominalReplay
                replay=RecordedNominalReplay(env,official)
                policy = PredictedTolerancePolicy()
                streak = 0
                for i, row in enumerate(rows):
                    finger_indices = np.asarray(env.robots[0]._ref_gripper_joint_pos_indexes) + 1
                    assert np.allclose(states[i, finger_indices], row['pre_action']['finger_qpos_m'], atol=1e-12, rtol=0)
                    np.testing.assert_array_equal(decode_tasktol_action(np.asarray(row['model_action_raw'])), actions[i])
                    assert row['inference_chunk']==i//5 and row['inference_chunk_offset']==i%5
                    selection = policy.resolve(np.asarray(row['pre_action']['finger_qpos_m']), actions[i], row['predicted_bounds_raw_rad'])
                    for field in ('phase','subtask_index','semantic_mask','predicted_bounds_unused'):
                        assert row[field]==selection[field], (suite,tid,seed,i,field)
                    assert row['bounds_rad'] is selection['bounds_rad'] is None
                    replay.verify(actions[i],selection,row)
                    if row['phase'] in ('ENG','DIS'):
                        assert row['semantic_mask'] == '000' 
                    assert row["observation_timing"] == "pre_action"
                    assert row["physics_steps"] == 25
                    failed = not row["optimizer"]["feasible"]
                    streak = streak + 1 if failed else 0
                    assert row["solver_failed"] == row["reference_held"] == failed
                    assert row["nominal_reference_updated"] == (not failed)
                    assert row["consecutive_solver_failures"] == streak
                    assert row["aborted"] == (streak >= 20 and not row["success"])
                    assert not row["aborted"] or i == n - 1
                    if "post_success_seconds" not in result:
                        assert not row["success"] or i == n - 1
                    assert np.array_equal(actions[i], row["action"])
                    after = states[i + 1] if i + 1 < n else final
                    assert np.isclose(after[0] - states[i, 0], 0.05, atol=1e-10)
                    assert np.allclose(
                        states[i, joint_indices], row["pre_action"]["joint_position_rad"], atol=1e-12, rtol=0
                    )
                    assert np.allclose(
                        after[joint_indices], row["post_action"]["joint_position_rad"], atol=1e-12, rtol=0
                    )
                    if failed and i:
                        assert row["nominal_target"] == rows[i - 1]["nominal_target"]
                        assert row["joint_target_rad"] == rows[i - 1]["joint_target_rad"]
                        assert row["applied_gripper_command"] == rows[i - 1]["applied_gripper_command"]
                assert result["total_solver_failures"] == sum(r["solver_failed"] for r in rows)
                assert result["maximum_solver_failure_streak"] == max(r["consecutive_solver_failures"] for r in rows)
                audit_first_success_state(env, states, final, result)
                env.set_state(final)
                env.sim.forward()
                fresh_success = bool(env.check_success())
                model, data = env.sim.model._model, env.sim.data._data
                assert result["servo"]["effective_integrator"] == int(model.opt.integrator) == 0
                qpos, qvel = data.qpos.copy(), data.qvel.copy()
                mujoco.mj_integratePos(model, data.qpos, qvel, -model.opt.timestep)
                roundtrip = data.qpos.copy()
                mujoco.mj_integratePos(model, roundtrip, qvel, model.opt.timestep)
                assert np.allclose(roundtrip, qpos, atol=1e-12, rtol=0)
                env.sim.forward()
                data.qpos[:] = qpos
                native_success = bool(env.check_success())
                audit_record(result, rows, native_success, request=request)
            if result["success"]:
                assert result["termination"] == "task_success"
            elif result["termination"] == "ik_failure_streak":
                assert rows[-1]["aborted"]
                assert streak == 20
            else:
                assert result["termination"] == "step_limit"
                assert n == result["max_steps"]
        finally:
            env.close()
        p = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-threads",
                "1",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(folder / "rollout.mp4"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not p.stderr.strip()
        v = json.loads(p.stdout)["streams"][0]
        assert (v["width"], v["height"], v["r_frame_rate"], int(v["nb_read_frames"])) == (512, 256, "20/1", n)
        verified.append(
            {
                "suite": suite,
                "task_id": tid,
                "seed": seed,
                "steps": n,
                "success": result["success"],
                "final_native_success": native_success,
                "first_chunk_difference_from_tolerance_group": first_chunk_differences,
                "first_success_state_verified": bool(result.get("success") and "post_success_seconds" in result),
                "termination": result["termination"],
                "native_predicate_verified": True,
                "strict_nominal_reference_and_fk_verified": True,
                "official_initial_state_exact": True,
                "fresh_forward_success": fresh_success,
                "failed_frames": result["total_solver_failures"],
                "maximum_failure_streak": result["maximum_solver_failure_streak"],
                "artifact_sha256": {
                    name: sha(folder / name)
                    for name in ("result.json", "trajectory.json", "trajectory.npz", "rollout.mp4")
                },
            }
        )
        dump(
            RUN / ("initial_audit_state.json" if partial else "independent_audit_state.json"),
            {"stage": "auditing", "checked_episodes": len(verified), "expected_episodes": 400},
        )
        print(f"Audited {suite}/{tid:02d}/seed{seed}", flush=True)
    if args.follow:
        deadline = time.monotonic() + 90
        while not (RUN / 'evaluation_completion.json').exists() and time.monotonic() < deadline:
            time.sleep(2)
        completion = verify_finished_clients()
        validate_request(request)
        assert len(verified) == 400
    dump(
        RUN / ("initial_client_integration_audit.json" if partial else "evaluation_independent_final_audit.json"),
        {
            "complete": True,
            "checked_at": datetime.datetime.now().astimezone().isoformat(),
            "scope": f"Initial {len(verified)} committed episodes only; full400 remains unaudited" if partial else "All400 joint13 no-tolerance episodes, seeds0..9, six inference GPUs and hold20 policy",
            "tolerance_settings": tolerance_settings,
            "successes": sum(r["success"] for r in verified),
            "episodes": len(verified),
            "full_evaluation_audited": not partial,
            "infrastructure_errors": 0,
            "every_frame_reference_hold_physics_and_failure_counter_verified": True,
            "all_model_bounds_unused_and_strict_reference_verified": True,
            "no_tolerance_branch_used": True,
            "all_official_initial_states_exact": True,
            "all_videos_decoded": True,
            "native_predicate_sampling": "Original seeded model fixtures; Euler last-step cached geometry with final qpos, inverse integration roundtrip1e-12",
            "audit_source_sha256": sha(Path(__file__)),
            "results": verified,
        },
    )
    dump(
        RUN / ("initial_audit_state.json" if partial else "independent_audit_state.json"),
        {"stage": "complete", "checked_episodes": len(verified), "successes": sum(r["success"] for r in verified)},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        dump(RUN / "independent_audit_state.json", {"stage": "failed", "error": traceback.format_exc()})
        raise
