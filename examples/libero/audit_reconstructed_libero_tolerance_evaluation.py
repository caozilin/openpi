"""Read-only independent audit of the 240 committed local LIBERO rollouts."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/libero_evaluation/pi05_reconstructed_10000_tol30_mask111_40tasks_6seeds_20261102"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
IDENTITY = "libero_reconstructed_step10000_tol30_mask111_40tasks_seed0to5_local_client_20261102"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def video_check(path, steps):
    p = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    assert not p.stderr.strip(), (path, p.stderr)
    stream = json.loads(p.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["r_frame_rate"], int(stream["nb_read_frames"])) == (512, 256, "20/1", steps), (path, stream, steps)
    return stream


def main():
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.path.insert(0, str(ROOT / "third_party/libero"))
    import numpy as np
    import mujoco
    from libero.libero.envs.env_wrapper import ControlEnv

    lock = (RUN / "independent_audit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = RUN / "evaluation_independent_final_audit.json"
    dump(RUN / "independent_audit_process.json", dict(pid=os.getpid(), command=str(Path(__file__).resolve())))
    request = read(RUN / "evaluation_request.json")
    completion = read(RUN / "evaluation_completion.json")
    assert completion["complete"] and completion["exit_codes"] == [0] * 6
    assert completion["infrastructure_errors"] == 0
    assert request["client_location"] == "local" and request["checkpoint_step"] == 10000
    assert request["tolerance_enabled"] and request["semantic_mask"] == "111" and request["tolerance_bounds_deg"] == [30.] * 6
    assert request["replan_steps"] == 5 and request["action_horizon"] == 10 and not request["state_enabled"]
    for path, expected in request["source_sha256"].items():
        assert sha(Path(path)) == expected, path
    assert not list((RUN / "episodes").rglob("infrastructure_error.json"))
    for name in ["evaluation_process.json"] + [f"client_{i}_process.json" for i in range(6)]:
        pid = read(RUN / name)["pid"]
        assert not Path(f"/proc/{pid}").exists(), (name, pid)
    with (RUN / "evaluation.lock").open("a") as evaluation_lock:
        fcntl.flock(evaluation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identities = read(RUN / "inference_final_identity_audit.json")
    assert identities["complete"]
    assert identities["server_entry_sha256"] == sha(ROOT / "examples/libero/serve_reconstructed_libero_eval_server.py")
    tasks = read(RUN / "official_tasks.json")
    initial = np.load(RUN / "official_initial_states.npz", allow_pickle=False)
    paths = sorted((RUN / "episodes").glob("*/task_*/seed_*/result.json"))
    keys = [(read(p)["suite"], read(p)["task_id"], read(p)["seed"]) for p in paths]
    expected = {(s, t, z) for s in SUITES for t in range(10) for z in range(6)}
    assert len(keys) == 240 and set(keys) == expected
    records, video_futures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for task in tasks:
            suite, tid = task["suite"], task["task_id"]
            assert sha(Path(task["source_init_states"])) == task["source_init_sha256"]
            for seed in range(6):
                # Fixture body_pos/body_quat live in mjModel, outside flattened
                # mjData snapshots. Recreate the original seeded hard reset for
                # each episode before restoring its dynamic state.
                np.random.seed(seed)
                random.seed(seed)
                env = ControlEnv(bddl_file_name=str(ROOT / "third_party/libero/libero/libero/bddl_files" / task["problem_folder"] / task["bddl_file"]), controller="JOINT_POSITION", use_camera_obs=False, has_offscreen_renderer=False, control_freq=20)
                try:
                    env.seed(seed)
                    env.reset()
                    joint_indices = np.asarray(env.robots[0]._ref_joint_pos_indexes) + 1
                    folder = RUN / "episodes" / suite / f"task_{tid:02d}" / f"seed_{seed}"
                    result, rows = read(folder / "result.json"), read(folder / "trajectory.json")
                    n = result["steps"]
                    assert result["complete"] and result["eval_identity"] == IDENTITY
                    assert result["language"] == task["language"] and result["initial_state_index"] == seed
                    assert result["max_steps"] == request["max_steps"][suite] and 0 < n <= result["max_steps"]
                    assert (result["control_hz"], result["physics_hz"], result["settle_steps"], result["replan_steps"]) == (20, 500, 10, 5)
                    assert result["gripper_deadband"] == .5 and not result["state_input_enabled"]
                    metadata = result["server_metadata"]
                    assert all(metadata[k] == identities["replicas"][seed][k] for k in metadata)
                    assert len(rows) == n and result["inference_calls"] == (n + 4) // 5
                    with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
                        actions, states, final = z["actions"], z["pre_action_states"], z["final_state"]
                        assert actions.shape == (n, 7) and states.shape[0] == n
                        baseline_folder = Path(request["baseline_run"]) / "episodes" / suite / f"task_{tid:02d}" / f"seed_{seed}"
                        with np.load(baseline_folder / "trajectory.npz", allow_pickle=False) as bz:
                            assert np.array_equal(z["initial_state"], bz["initial_state"])
                            assert np.array_equal(states[0], bz["pre_action_states"][0])
                            k = min(5, n, len(bz["actions"]))
                            first_chunk_max_delta = float(np.max(np.abs(actions[:k] - bz["actions"][:k])))
                            assert first_chunk_max_delta <= 1e-6, (suite, tid, seed, first_chunk_max_delta)
                        assert all(np.isfinite(z[k]).all() for k in z.files)
                        official = initial[f"{suite}_{tid}_{seed}"]
                        assert np.array_equal(z["initial_state"], official)
                        assert hashlib.sha256(official.tobytes()).hexdigest() == result["initial_state_sha256"]
                        for i, row in enumerate(rows):
                            assert row["observation_timing"] == "pre_action"
                            assert row["semantic_mask"] == "111"
                            assert np.allclose(row["bounds_rad"], np.pi / 6, atol=1e-15, rtol=0)
                            if not row["aborted"]:
                                assert np.max(np.abs(row["rotation_release_rad"])) <= np.pi / 6 + 1e-4
                                assert np.max(row["predicted_tolerance_violation_rad"]) <= 1e-4
                            assert np.array_equal(actions[i], row["action"])
                            assert np.allclose(states[i, joint_indices], row["pre_action"]["joint_position_rad"], atol=1e-12, rtol=0)
                            assert actions[i, 6] in (-1, 0, 1)
                            aborted = row["aborted"]
                            assert row["physics_steps"] == (0 if aborted else 25)
                            assert row["optimizer"]["feasible"] == (not aborted)
                            after = states[i + 1] if i + 1 < n else final
                            assert np.isclose(after[0] - states[i, 0], 0 if aborted else .05, atol=1e-10)
                            assert np.allclose(after[joint_indices], row["post_action"]["joint_position_rad"], atol=1e-12, rtol=0)
                            assert not row["success"] or i == n - 1
                            assert not aborted or i == n - 1
                        env.set_state(final)
                        env.sim.forward()
                        fresh_forward_success = bool(env.check_success())
                        # mj_step's Euler position update occurs after its collision
                        # computation. LIBERO queries that cached geometry immediately
                        # after env.step. A fresh forward() changes its sampling time.
                        # Recover the exact pre-integration geometry from saved q/qvel,
                        # then restore final qpos without overwriting the contact cache.
                        # This changes no trajectory, model, threshold or success record.
                        model, data = env.sim.model._model, env.sim.data._data
                        assert result["servo"]["effective_integrator"] == int(model.opt.integrator) == 0
                        assert np.isclose(model.opt.timestep, .002)
                        qpos, qvel = data.qpos.copy(), data.qvel.copy()
                        mujoco.mj_integratePos(model, data.qpos, qvel, -model.opt.timestep)
                        roundtrip = data.qpos.copy()
                        mujoco.mj_integratePos(model, roundtrip, qvel, model.opt.timestep)
                        assert np.allclose(roundtrip, qpos, atol=1e-12, rtol=0)
                        env.sim.forward()
                        data.qpos[:] = qpos
                        native_success = bool(env.check_success())
                        assert native_success == result["success"] == rows[-1]["success"], (suite, tid, seed, native_success)
                    if result["success"]:
                        assert result["termination"] == "task_success" and not rows[-1]["aborted"]
                    elif result["termination"] == "ik_infeasible":
                        assert rows[-1]["aborted"]
                    else:
                        assert result["termination"] == "step_limit" and n == result["max_steps"]
                    future = pool.submit(video_check, folder / "rollout.mp4", n)
                    video_futures.append(future)
                    records.append(dict(suite=suite, task_id=tid, language=task["language"], seed=seed, steps=n, success=native_success, termination=result["termination"], final_native_predicate_restored=True, paired_first_chunk_max_action_delta=first_chunk_max_delta, tolerance_semantic_mask="111", tolerance_directional_bounds_deg=[30.]*6, fresh_forward_success=fresh_forward_success, native_sampling="last Euler step cached geometry with final joint qpos", seeded_fixture_body_poses={k:env.sim.model.body_pos[env.env.obj_body_id[k]].tolist() for k in env.env.fixtures_dict}, artifact_sha256={name:sha(folder / name) for name in ("result.json", "trajectory.json", "trajectory.npz", "rollout.mp4")}))
                finally:
                    env.close()
            print(f"Audited {suite}/task_{tid:02d}: 6 seeds", flush=True)
            dump(RUN / "independent_audit_state.json", dict(stage="auditing", checked_episodes=len(records), expected_episodes=240))
        for future in video_futures:
            future.result()
    per_suite = {s:dict(episodes=60, successes=sum(r["success"] for r in records if r["suite"] == s)) for s in SUITES}
    assert sum(r["success"] for r in records) == completion["successes"]
    assert per_suite == completion["per_suite"]
    per_task = [dict(suite=s, task_id=t, episodes=6, successes=sum(r["success"] for r in records if r["suite"] == s and r["task_id"] == t)) for s in SUITES for t in range(10)]
    assert per_task == completion["per_task"]
    report = dict(complete=True, scope="All 240 tolerance episodes, matched baseline initial states/actions; no new simulation rollout or inference", tolerance_enabled=True, semantic_mask="111", tolerance_bounds_deg=[30.]*6, audit_script_sha256=sha(Path(__file__)), checked_at=datetime.datetime.now().astimezone().isoformat(), successes=sum(r["success"] for r in records), episodes=240, infrastructure_errors=0, per_suite=per_suite, per_task=per_task, fresh_forward_sampling_disagreements=sum(r["fresh_forward_success"] != r["success"] for r in records), native_predicate_sampling="Matches actual LIBERO post-mj_step cached collision geometry; inverse Euler qpos reconstruction verified by roundtrip; final joint qpos preserved", sampling_reference="https://github.com/google-deepmind/mujoco/blob/3.2.3/src/engine/engine_forward.c#L654-L676", source_sha256_verified=len(request["source_sha256"]), native_final_state_predicates_recomputed=240, per_episode_seeded_model_fixtures_restored=True, every_action_state_physics_step_checked=True, all_video_frames_decoded_and_counted=True, client_exit_codes=completion["exit_codes"], local_processes_exited_and_lock_released=True, inference_identity_audit_sha256=sha(RUN / "inference_final_identity_audit.json"), episodes_detail=records)
    dump(report_path, report)
    dump(RUN / "independent_audit_state.json", dict(stage="complete", checked_episodes=240, successes=report["successes"]))
    print(json.dumps({k:v for k,v in report.items() if k != "episodes_detail"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        dump(RUN / "independent_audit_state.json", dict(stage="failed", error=traceback.format_exc()))
        raise
