"""Annotate all saved training layouts using the maintained simulation backend.

Each layout is snapshotted; trials, failures and the aggregate progress survive
interruption. The existing production peak-guided search owns the search policy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_geometry_metric_diagnostics import _run_condition
from tasktol.control.ipopt import IpoptSettings
from tasktol.control.task_space.objective import ObjectiveSettings
from tasktol.counterfactual.runner import CounterfactualTrialResult
from tasktol.counterfactual.search import (
    CounterfactualSearchConfig, EpisodeAnnotationTarget, search_task_tolerance,
)
from tasktol.data.task_sets import training_tasks
from tasktol.simulation.layouts import CUSTOM_LAYOUT_ROOT, load_custom_layout
from tasktol.tolerance.frame import TOLERANCE_FRAME_MODE_REALTIME_NOMINAL


def write_json(path: Path, value: Any) -> None:
    def scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar) + "\n")
    temporary.replace(path)


def discover(layout_root: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(layout_root.glob("*/*.json")):
        layout = load_custom_layout(path)
        for scene, task in training_tasks():
            if layout.scene != scene.runtime_scene:
                continue
            cases.append({
                "scene_id": scene.scene_id, "task_id": task.task_id,
                "runtime_scene": scene.runtime_scene,
                "runtime_task": task.runtime_task,
                "instruction": task.instructions[0], "robot": layout.robot,
                "layout_name": path.stem, "source_path": str(path.resolve()),
                # -1 is the UI's unseeded export; placement is replaced by
                # the saved layout, so use a deterministic simulation seed.
                "seed": max(0, layout.source_seed),
                "layout_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "variant_index": layout.variant_index,
            })
    return cases


def summarize(output: Path, results: list[dict[str, Any]]) -> Path:
    rows = []
    for result in sorted(results, key=lambda r: (r["task_id"], r["layout_name"])):
        row = {k: result[k] for k in ("task_id", "layout_name", "robot", "status", "trial_count")}
        if result["status"] == "complete":
            annotation = json.loads(Path(result["annotation"]).read_text())
            row["bounds_deg"] = annotation["subtask_bounds_deg"]
            row["annotation"] = result["annotation"]
        else:
            row["error"] = result["error"]
            failure = Path(result["search_failure"])
            if failure.is_file():
                row["failure_context"] = json.loads(failure.read_text())["context"]
            row["search_failure"] = str(failure)
        rows.append(row)
    path = output / "summary.json"
    write_json(path, {
        "layout_count": len(rows),
        "completed": sum(r["status"] == "complete" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "unique_simulation_trials": sum(r["trial_count"] for r in rows),
        "results": rows,
    })
    return path


def annotate_case(case: dict[str, Any], output: str, step: int) -> dict[str, Any]:
    root = Path(output) / case["task_id"] / case["layout_name"]
    snapshot = root / "layout.json"
    config = CounterfactualSearchConfig(
        scene=case["scene_id"], task=case["task_id"],
        robot_uid=case["robot"], execution_mode="simulation",
        targets=(EpisodeAnnotationTarget(case["seed"], root / "annotation"),),
        tolerance_step_deg=step, scene_variant_index=case["variant_index"],
        sqp_loss_profile="delta_q_squared_only", sqp_loss_weights=(1.0, 0.0, 0.0),
    )
    context = {
        **case, "backend": "MuJoCoVLABenchmarkEnvironment",
        "custom_layout_path": str(snapshot.resolve()),
        "tolerance_frame_mode": TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
        "robot_table_clearance_failure_m": 0.005,
        "maximum_tolerance_deg": 30, "tolerance_step_deg": step,
        "loss_weights": [1.0, 0.0, 0.0], "ipopt_max_iterations": 20,
        "ipopt_cpu_deadline": "disabled",
    }
    write_json(root / "config.json", context)

    def trial(spec, *, dataset_nominal_frames=None):
        if dataset_nominal_frames is not None:
            raise ValueError("Custom layouts must use their own nominal planner")
        if len(spec.sqp_subtask_stage_rotation_masks) != 1:
            raise ValueError("Custom-layout diagnostic requires a single subtask")
        bounds = tuple(
            tuple(float(x) for x in np.radians(np.column_stack((
                spec.sqp_subtask_stage_negative_limits_deg[0][i],
                spec.sqp_subtask_stage_positive_limits_deg[0][i],
            )).reshape(6))) for i in range(2)
        )
        index = len(list((root / "trials").glob("trial_*.json")))
        payload = _run_condition(
            case["scene_id"], case["runtime_scene"], case["task_id"],
            case["runtime_task"], case["instruction"], case["seed"],
            "release" if any(any(b) for b in bounds) else "strict",
            root / "runtime" / f"trial_{index:03d}", robot_uid=case["robot"],
            oracle_stage_tolerance_bounds_rad=bounds,
            custom_layout_path=snapshot,
            objective_settings_override=replace(
                ObjectiveSettings.delta_q_squared_only(), ema_release_loss_weight=0.0,
            ),
            tolerance_frame_mode=TOLERANCE_FRAME_MODE_REALTIME_NOMINAL,
            record_robot_table_clearance=True,
            robot_table_clearance_failure_m=0.005,
            ipopt_settings=IpoptSettings(max_iterations=20, max_cpu_time_s=1.0e6),
        )
        success = payload["task_success"] and not payload["optimizer_rejected_frames"]
        reason = "optimizer_failure" if payload["optimizer_rejected_frames"] else payload["termination_reason"]
        payload.update(success=bool(success), bounds_deg=[np.degrees(b).tolist() for b in bounds])
        write_json(root / "trials" / f"trial_{index:03d}.json", payload)
        extrema = {
            stage: {
                "negative": payload["release_statistics"][stage]["peak_directional_deg"][0::2],
                "positive": payload["release_statistics"][stage]["peak_directional_deg"][1::2],
            } for stage in ("pregrasp", "postgrasp")
        }
        return CounterfactualTrialResult(
            spec=spec, success=bool(success), task_success=bool(payload["task_success"]),
            target_reached=bool(success), termination_reason=reason,
            target_rotation_error_deg=0.0,
            nominal_target_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            counterfactual_target_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            tolerance_frame_rotation_vector_base_rad=(0.0, 0.0, 0.0),
            evaluation={
                "sqp_subtask_stage_release_extrema_deg": [extrema],
                "declared_subtask_stage_rotation_masks": [payload["declared_stage_rotation_masks"]],
                "executed_frames": payload["frames"],
                "backend": context["backend"],
            },
            hard_failure_reasons=() if success else (reason,),
            trajectory_kinematics=payload["trajectory_metric_domains"],
        )

    try:
        paths = search_task_tolerance(config, trial_runner=trial, trial_context=context)
        result = {**case, "status": "complete", "annotation": str(paths[0])}
    except Exception as error:
        result = {**case, "status": "failed", "error": f"{type(error).__name__}: {error}",
                  "search_failure": str(root / "annotation" / "tolerance_annotation_failure.json")}
    result["trial_count"] = len(list((root / "trials").glob("trial_*.json")))
    write_json(root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, default=CUSTOM_LAYOUT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("analysis_results") / ("custom_layout_annotations_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--step-deg", type=int, default=2)
    args = parser.parse_args()
    if args.workers <= 0 or args.step_deg <= 0 or 30 % args.step_deg:
        parser.error("workers must be positive; step must be a positive divisor of 30")
    cases = discover(args.layout_root)
    results = []
    for case in cases:
        root = args.output / case["task_id"] / case["layout_name"]
        snapshot = root / "layout.json"
        payload = json.loads(Path(case["source_path"]).read_text())
        if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() != hashlib.sha256((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest():
            raise ValueError(f"Existing snapshot differs: {snapshot}")
        write_json(snapshot, payload)
    write_json(args.output / "manifest.json", {"cases": cases, "results": results})
    print(f"Annotating {len(cases)} layouts with {args.workers} workers: {args.output}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(annotate_case, c, str(args.output), args.step_deg) for c in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(args.output / "manifest.json", {"cases": cases, "results": results})
            print(f'{len(results)}/{len(cases)} {result["task_id"]}/{result["layout_name"]}: {result["status"]}, {result["trial_count"]} trials', flush=True)
    print(f"Summary: {summarize(args.output, results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
