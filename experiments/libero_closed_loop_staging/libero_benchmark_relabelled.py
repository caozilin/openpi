"""Dataset-reference / IPOPT / native LIBERO collection client.

Physics, torque control, gripper and lifecycle are executed by native LIBERO env.step.
Recorded robot motion supplies independent nominal references, not joint commands.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_replay(
    *,
    openpi_root: Path,
    output: Path,
    suite="libero_spatial",
    task_id=0,
    steps=0,
    kp=None,
    tolerance_deg=None,
    hold_steps=0,
    reference_mode="source_goals",
    initialization_mode="source_context",
    reference_actions_path: Path | None = None,
):
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(output / "numba_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    sys.path.insert(0, str(openpi_root / "third_party/libero"))
    import h5py
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from ..simulation.libero import LiberoTaskTolSession
    from ..geometry import rotation_vector, rotation_matrix
    from ..control.pose import TargetPose
    from .libero_replay_state import processed_reset_context, restore_processed_state

    if mujoco.__version__ != "3.2.3":
        raise RuntimeError("Closed-loop LIBERO baseline is pinned to MuJoCo 3.2.3")
    selection = openpi_root / "data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json"
    annotations = selection.with_name("mask_annotations.json")
    selected = next(
        t for t in json.loads(selection.read_text())["tasks"] if t["suite"] == suite and t["task_id"] == task_id
    )
    annotation = next(
        t for t in json.loads(annotations.read_text())["tasks"] if t["suite"] == suite and t["task_id"] == task_id
    )
    with h5py.File(selected["processed_hdf5"], "r") as f:
        demo = f["data"][selected["demo_id"]]
        initial = demo["tasktol/pre_action_states"][0]
        pre_states = demo["tasktol/pre_action_states"][()]
        final_state = demo["tasktol/final_state"][()]
        source_actions = demo["actions"][()]
        source_indices = demo["tasktol/source_step_indices"][()]
        source_config = json.loads(f["source_metadata"].attrs["env_args"])["env_kwargs"]["controller_configs"]
        reset_context = processed_reset_context(f, selected["demo_id"])
    if (
        source_config.get("type") != "OSC_POSE"
        or not np.allclose(source_config["output_max"], [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        or not np.allclose(source_config["output_min"], [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5])
        or source_config["input_max"] != 1
        or source_config["input_min"] != -1
    ):
        raise ValueError("Unexpected source action contract; explicit decoder adaptation required")
    bddl = Path(get_libero_path("bddl_files")) / suite / (selected["task_name"] + ".bddl")
    env = None
    frames = []
    report = dict(
        status="running",
        execution_backend="native_LIBERO_env.step",
        optimizer_backend="ipopt",
        task=dict(suite=suite, task_id=task_id, name=selected["task_name"], demo=selected["demo_id"]),
        packages={n: metadata.version(n) for n in ("mujoco", "numpy", "cyipopt", "robosuite", "h5py")},
        assets_root=get_libero_path("assets"),
        source_hdf5=selected["processed_hdf5"],
        source_original_hdf5=selected["source_hdf5"],
        initial_state_source="processed tasktol/pre_action_states[0]; already settled snapshot",
        initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        bddl_sha256=sha(bddl),
        annotation_sha256=sha(annotations),
        source_indices=source_indices.tolist(),
        nominal_reference="Independent increments from FK of recorded pre-action robot joints and final robot joints; not normalized OSC commands; never seeded by source joint targets",
        reference_mode=reference_mode,
        initialization_mode=initialization_mode,
        source_controller_config=source_config,
        tolerance=dict(
            mode="off" if tolerance_deg is None else "explicit_diagnostic_bounds_with_frozen_semantic_mask",
            degrees=tolerance_deg,
            numeric_ground_truth=False,
        ),
        hold_steps=hold_steps,
        kp=kp,
    )
    source_root = Path(__file__).resolve().parents[2]
    report["implementation_sha256"] = {
        str(p.relative_to(source_root)): sha(p)
        for p in [
            Path(__file__),
            source_root / "tasktol/simulation/libero.py",
            source_root / "tasktol/simulation/preset_runtime.py",
            source_root / "tasktol/simulation/robot.py",
            source_root / "tasktol/control/ipopt/controller.py",
            source_root / "tasktol/control/task_space/kinematics.py",
            source_root / "tasktol/tolerance/runtime.py",
            source_root / "tasktol/benchmark/libero_replay_state.py",
        ]
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        (output / "trajectory.json").write_text(
            json.dumps(
                dict(frequency_hz=20, observation_timing="pre_action", frames=frames), ensure_ascii=False, indent=2
            )
            + "\n"
        )

    save()
    start = time.monotonic()
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            controller="JOINT_POSITION",
            camera_names=["agentview"],
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        if initialization_mode == "source_context":
            _, report["initialization"] = restore_processed_state(env, initial, reset_context)
        elif initialization_mode == "snapshot_only":
            env.seed(0)
            env.reset()
            env.set_init_state(initial)
            report["initialization"] = dict(mode="snapshot_only", reset_rng_matches_source=False)
        else:
            raise ValueError("Unknown initialization mode")
        session = LiberoTaskTolSession(env, kp=kp)
        relabelled = None
        if reference_mode == "relabelled_actions":
            from .libero_reference_actions import load_reference_actions, verify_reference_anchor

            if reference_actions_path is None:
                raise ValueError("Relabelled replay requires --reference-actions-path")
            relabelled = load_reference_actions(
                reference_actions_path, selected["demo_id"], source_path=selected["processed_hdf5"],
                source_states=pre_states, source_actions=source_actions,
            )
            verify_reference_anchor(relabelled, session.target.position, session.target.rotation)
            session.target.position = relabelled.anchor_position.copy()
            session.target.rotation = relabelled.anchor_rotation.copy()
            session.release.accepted_rotation = relabelled.anchor_rotation.copy()
            report["reference_actions"] = relabelled.metadata
            report["nominal_reference"] = (
                "Physical increments read directly from validated relabelled HDF5; explicit saved initial anchor; "
                "independent nominal accumulation inside LiberoTaskTolSession; no OSC decoding or source joint command replay"
            )
            helper = source_root / "tasktol/benchmark/libero_reference_actions.py"
            report["implementation_sha256"][str(helper.relative_to(source_root))] = sha(helper)
        elif reference_actions_path is not None:
            raise ValueError("Reference-action file requires relabelled_actions mode")
        elif reference_mode not in ("source_goals", "recorded_motion"):
            raise ValueError("Unknown reference mode")
        report["controller"] = type(env.robots[0].controller).__name__
        report["control_dt"] = session.dt
        report["physics_dt"] = float(session.model.opt.timestep)
        report["solver_settings"] = asdict(session.ik.settings)
        report["ee_rotation_offset"] = session.handles.ee_rotation_offset.tolist()
        count = len(source_actions) if not steps else min(steps, len(source_actions))
        qstates = np.vstack(
            (
                pre_states[:count, 1 + session.handles.qpos_ids],
                (
                    pre_states[count, 1 + session.handles.qpos_ids]
                    if count < len(pre_states)
                    else final_state[1 + session.handles.qpos_ids]
                ),
            )
        )
        references = [
            session.ik.kinematics.evaluate(q, include_manipulability=False, include_link_points=False) for q in qstates
        ]
        recorded_references = references
        if reference_mode == "source_goals":
            # Data-side decoding only: OSC deltas are relative to source
            # feedback, not to an independently integrated commanded goal.
            # Freeze that nominal goal stream before executing IPOPT.
            decoded = [references[0]]
            orientation = (
                np.array(report["initialization"]["source_initial_held_site_rotation"]) @ session.handles.ee_rotation_offset
                if initialization_mode == "source_context"
                else references[0].rotation.copy()
            )
            for i in range(count):
                scaled = np.clip(source_actions[i, :6], -1.0, 1.0) * np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
                if np.any(scaled[3:] != 0):
                    orientation = rotation_matrix(scaled[3:]) @ references[i].rotation
                decoded.append(TargetPose(references[i].position + scaled[:3], orientation.copy()))
            references = decoded
            report["nominal_reference"] = (
                "Offline decoded source-feedback-relative OSC goal sequence, frozen before IPOPT; initial held orientation from restored source reset context when available; no live OSC execution"
            )
        labels = {}
        for sub in annotation["subtasks"]:
            for span in sub["phase_spans"]:
                phase = span["phase"]
                mask = sub[phase]["effective_mask"] if phase in ("APP", "EXEC") else "000"
                for i in range(*span["processed_interval"]):
                    labels[i] = (phase, sub["index"], mask)
        obs = session.observation()
        Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()).save(output / "initial.png")
        previous_subtask = 0
        with (
            imageio.get_writer(output / "main_rgb.mp4", fps=20, codec="libx264", quality=8) as video,
            (output / "solver_trace.jsonl").open("w") as trace,
        ):
            for i in range(count + hold_steps):
                hold = i >= count
                if hold:
                    action = np.r_[np.zeros(6), source_actions[count - 1, 6]]
                    phase, subtask, mask = "outside_subtask", previous_subtask, "000"
                else:
                    if relabelled is not None:
                        action = relabelled.actions[i].copy()
                    else:
                        a, b = references[i], references[i + 1]
                        action = np.r_[
                            b.position - a.position, rotation_vector(b.rotation @ a.rotation.T), source_actions[i, 6]
                        ]
                    phase, subtask, mask = labels.get(i, ("outside_subtask", previous_subtask, "000"))
                previous_subtask = subtask
                video.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                obs, row = session.step(
                    action,
                    phase=phase,
                    subtask_index=subtask,
                    semantic_mask=mask,
                    bounds_rad=None if tolerance_deg is None else np.full(6, np.radians(tolerance_deg)),
                )
                row["source_index"] = None if hold else int(source_indices[i])
                row["raw_source_osc_action"] = None if hold else source_actions[i].tolist()
                row["reference_kind"] = "terminal_hold" if hold else reference_mode
                if relabelled is not None and not hold:
                    row["relabelled_reference_position_error_m"] = float(
                        np.linalg.norm(session.target.position - relabelled.goal_positions[i])
                    )
                    row["relabelled_reference_rotation_error_rad"] = float(np.linalg.norm(
                        rotation_vector(session.target.rotation @ relabelled.goal_rotations[i].T)
                    ))
                    if tolerance_deg is None and (
                        row["relabelled_reference_position_error_m"] > 1e-10
                        or row["relabelled_reference_rotation_error_rad"] > 1e-9
                    ):
                        raise RuntimeError("Online nominal reference differs from relabelled goals")
                row["source_replay_position_error_m"] = (
                    None
                    if hold
                    else float(
                        np.linalg.norm(
                            np.asarray(row["post_action"]["position_m"]) - recorded_references[i + 1].position
                        )
                    )
                )
                frames.append(row)
                trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                trace.flush()
                if i % 20 == 0 or row["aborted"]:
                    print(
                        f"{suite}/{task_id:02d} {i + 1}/{count + hold_steps} feasible={row['optimizer']['feasible']} tracking_mm={1000 * row['tracking_position_error_m']:.3f} success={row['success']}",
                        flush=True,
                    )
                if row["aborted"]:
                    break
        Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()).save(output / "final.png")
        report.update(
            status="passed"
            if len(frames) == count + hold_steps and not any(r["aborted"] for r in frames)
            else "failed",
            task_success=bool(env.check_success()),
            frames=len(frames),
            expected_frames=count + hold_steps,
            solver_calls=sum(r["optimizer"]["solver_called"] for r in frames),
            feasible_frames=sum(r["optimizer"]["feasible"] for r in frames),
            tracking_position_max_mm=max(r["tracking_position_error_m"] for r in frames) * 1000,
            tracking_position_mean_mm=np.mean([r["tracking_position_error_m"] for r in frames]) * 1000,
            source_replay_position_max_mm=max(
                r["source_replay_position_error_m"] for r in frames if r["source_replay_position_error_m"] is not None
            )
            * 1000,
            native_env_steps=sum(r["physics_steps"] > 0 for r in frames),
            native_joint_goal_max_error_rad=max(r["native_joint_goal_error_rad"] or 0.0 for r in frames),
            native_controller_kp=session.controller.kp.tolist(),
            physics_steps=sum(r["physics_steps"] for r in frames),
            elapsed_s=time.monotonic() - start,
        )
        if any(any(r["effective_mask"][k] == "1" for k in range(3)) for r in frames):
            report["tolerance"]["ranged_frames"] = sum("1" in r["effective_mask"] for r in frames)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        import traceback

        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        if env is not None:
            env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=Path("/media/czl/sata/franka_my_code/openpi"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"), default="libero_spatial"
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--kp", type=float, default=None, help="Optional native controller gain; omitted preserves LIBERO default"
    )
    parser.add_argument("--reference-mode", choices=("recorded_motion", "source_goals", "relabelled_actions"), default="source_goals")
    parser.add_argument("--reference-actions-path", type=Path, help="Validated physical reference-increment HDF5")
    parser.add_argument("--initialization-mode", choices=("source_context", "snapshot_only"), default="source_context")
    parser.add_argument("--hold-steps", type=int, default=0)
    parser.add_argument("--tolerance-deg", type=float, default=None)
    args = parser.parse_args()
    if args.tolerance_deg is not None and not 0 <= args.tolerance_deg <= 30:
        parser.error("diagnostic tolerance must be between 0 and 30 degrees")
    result = run_replay(**vars(args))
    if result["status"] != "passed" or not result.get("task_success", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
