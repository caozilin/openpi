"""Annotate rotational tolerance independently for each requested seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tasktol.counterfactual import (
    CounterfactualSearchConfig,
    EpisodeAnnotationTarget,
    search_task_tolerance,
)
from tasktol.data.task_sets import resolve_training_task, training_tasks
from tasktol.simulation.randomization.routing import RandomizationSeedRouting
from tasktol.simulation.robots import ROBOT_CHOICES


DEFAULT_TRAINING_ROBOT = "panda"


def _manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Dataset task has no manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("episodes"), list):
        raise ValueError(f"Dataset manifest has no episode list: {path}")
    return payload


def dataset_targets(
    dataset_dir: Path,
    *,
    selected_seeds: set[int] | None = None,
) -> tuple[EpisodeAnnotationTarget, ...]:
    """Resolve dataset episodes to episode-local annotation targets."""
    manifest = _manifest(dataset_dir)
    probability = float(
        manifest.get("domain_randomization", {}).get("probability", 0.0)
    )
    targets: list[EpisodeAnnotationTarget] = []
    for fallback_index, episode in enumerate(manifest["episodes"]):
        if not isinstance(episode, dict) or "seed" not in episode:
            continue
        seed = int(episode["seed"])
        if selected_seeds is not None and seed not in selected_seeds:
            continue
        episode_index = int(episode.get("episode_index", fallback_index))
        episode_dir = dataset_dir / str(
            episode.get(
                "path", f"episode_{episode_index:04d}_seed_{seed}",
            )
        )
        trajectory_path = episode_dir / "trajectory.json"
        if not trajectory_path.is_file():
            raise ValueError(
                f"Dataset episode is missing trajectory.json: {episode_dir}",
            )
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        routing = (
            trajectory.get("episode", {})
            .get("domain_randomization", {})
            .get("seed_routing", {})
        )
        environment_seed = int(
            routing.get(
                "environment_seed",
                RandomizationSeedRouting(seed, probability).environment_seed,
            )
        )
        targets.append(EpisodeAnnotationTarget(
            seed=seed,
            episode_index=episode_index,
            trajectory_path=trajectory_path,
            output_dir=episode_dir,
            environment_seed=environment_seed,
        ))
    if selected_seeds is not None:
        missing = selected_seeds.difference(target.seed for target in targets)
        if missing:
            raise ValueError(
                f"Requested dataset seed(s) not found: {sorted(missing)}",
            )
    if not targets:
        raise ValueError(f"No dataset episodes found in {dataset_dir}")
    return tuple(targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", default="geometry_plate_box_upright",
        help="training scene ID from tasktol.data.task_sets",
    )
    parser.add_argument(
        "--task", default="geometry_plate_box_upright",
        help="task ID registered under the selected training scene",
    )
    parser.add_argument("--list-training-tasks", action="store_true")
    parser.add_argument("--robot", choices=ROBOT_CHOICES, default="panda")
    parser.add_argument(
        "--execution-mode", choices=("simulation", "kinematic"),
        default="kinematic",
    )
    parser.add_argument(
        "--seed", type=int, action="append",
        help=(
            "episode seed to annotate; repeat for multiple seeds. Without a "
            "dataset, omitted means seed 0. With --dataset-dir, omitted means "
            "all manifest episodes"
        ),
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        help=(
            "one task dataset directory; annotations are written inside each "
            "selected episode directory"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        help=(
            "root for direct-seed tests; each result is written below "
            "seed_<seed>/"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--max-tolerance-deg", type=int, default=30)
    parser.add_argument("--step-deg", type=int, default=5)
    parser.add_argument(
        "--environment-randomization-probability", type=float, default=0.0,
    )
    parser.add_argument("--variant-index", type=int, default=None)
    parser.add_argument("--ipopt-max-iterations", type=int, default=20)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def _direct_targets(
    args: argparse.Namespace,
    training_scene_id: str,
    training_task_id: str,
) -> tuple[EpisodeAnnotationTarget, ...]:
    seeds = tuple(args.seed if args.seed is not None else (0,))
    root = (
        args.output
        if args.output is not None
        else Path("analysis_results") / "tolerance_annotations"
        / args.robot / training_scene_id / training_task_id
    )
    return tuple(
        EpisodeAnnotationTarget(
            seed=seed,
            output_dir=root / f"seed_{seed:02d}",
            environment_seed=RandomizationSeedRouting(
                seed, args.environment_randomization_probability,
            ).environment_seed,
        )
        for seed in seeds
    )


def main() -> None:
    args = parse_args()
    if args.list_training_tasks:
        for scene, task in training_tasks():
            print(
                f"{scene.scene_id} / {task.task_id}  ->  "
                f"{scene.runtime_scene} / {task.runtime_task}",
            )
        return
    training_scene, training_task = resolve_training_task(
        args.scene, args.task,
    )
    targets = (
        dataset_targets(
            args.dataset_dir,
            selected_seeds=(
                None if args.seed is None else set(args.seed)
            ),
        )
        if args.dataset_dir is not None
        else _direct_targets(
            args, training_scene.scene_id, training_task.task_id,
        )
    )
    config = CounterfactualSearchConfig(
        scene=args.scene,
        task=args.task,
        robot_uid=args.robot,
        execution_mode=args.execution_mode,
        targets=targets,
        maximum_frames=args.max_frames,
        maximum_tolerance_deg=args.max_tolerance_deg,
        tolerance_step_deg=args.step_deg,
        environment_randomization_probability=(
            args.environment_randomization_probability
        ),
        scene_variant_index=args.variant_index,
        ipopt_max_iterations=args.ipopt_max_iterations,
        restart=args.restart,
    )
    results = search_task_tolerance(config)
    for path in results:
        print(f"Tolerance annotation complete: {path}")


if __name__ == "__main__":
    main()
