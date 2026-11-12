"""Dataset-backed task catalogue and deterministic benchmark seed plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from ..tolerance.specification import rotation_tolerance_bounds
from ..data.task_sets import (
    MULTI_SUBTASK_SUBCATEGORY,
    tolerance_discrimination_tasks,
    training_tasks,
    unseen_object_tasks,
)


_BENCHMARK_SEED_STREAM = 0x42454E43484D4152  # "BENCHMAR"
_BENCHMARK_CONTROL_HZ = 10.0
_ROLLOUT_TIME_MARGIN_S = 5.0
_BUILTIN_RENDER_WIDTH = 640
_BUILTIN_RENDER_HEIGHT = 480
_BUILTIN_MAXIMUM_FRAMES = 600


@dataclass(frozen=True)
class DatasetTaskSpec:
    directory: Path
    scene: str
    task_id: str
    task_name: str
    instructions: tuple[str, ...]
    width: int
    height: int
    maximum_steps: int
    maximum_demonstration_duration_s: float
    rotation_tolerance_profiles_rad: dict[str, tuple[float, ...]]
    robot_uid: str = "panda"
    benchmark_only: bool = False
    reference_task_directory: str | None = None


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _demonstration_durations_s(
    task_directory: Path,
    metadata: dict,
) -> tuple[float, ...]:
    """Read successful demonstration durations from the task manifest."""
    manifest_path = task_directory / "manifest.json"
    if not manifest_path.is_file():
        return ()
    manifest = _read_json(manifest_path)
    frequency_hz = float(metadata.get("frequency_hz", _BENCHMARK_CONTROL_HZ))
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError(f"Invalid dataset frequency_hz: {task_directory}")
    durations = []
    for episode in manifest.get("episodes", ()):
        if not isinstance(episode, dict) or not bool(episode.get("success", True)):
            continue
        frames = int(episode.get("frames", 0))
        if frames > 0:
            durations.append(frames / frequency_hz)
    return tuple(durations)


def discover_dataset_tasks(dataset_root: Path) -> tuple[DatasetTaskSpec, ...]:
    """Read benchmark tasks exclusively from task-level dataset JSON."""
    records: list[
        tuple[
            Path,
            dict,
            tuple[str, ...],
            int,
            int,
            dict[str, tuple[float, ...]],
            tuple[float, ...],
        ]
    ] = []
    for metadata_path in sorted(dataset_root.rglob("task_metadata.json")):
        metadata = _read_json(metadata_path)
        instruction_path = metadata_path.parent / str(
            metadata.get("instruction_file", "instructions.json")
        )
        instructions_document = _read_json(instruction_path)
        instructions = tuple(
            str(value).strip() for value in instructions_document.get("instructions", ())
            if str(value).strip()
        )
        if not instructions:
            raise ValueError(f"Task contains no language instructions: {instruction_path}")
        video_metadata = metadata.get("video", {})
        # OpenPI-ready dataset videos are letterboxed to 224x224. Benchmarks
        # must still render at the source camera aspect ratio so the policy's
        # own preprocessing sees the same pixels as collection did.
        resolution = video_metadata.get(
            "source_render_resolution",
            video_metadata.get("resolution", (640, 480)),
        )
        raw_profiles = metadata.get("annotation", {}).get(
            "rotation_tolerance_profiles_rad", {},
        )
        profiles: dict[str, tuple[float, ...]] = {}
        for name, value in raw_profiles.items():
            try:
                vector = rotation_tolerance_bounds(value, f"{name!r} tolerance profile")
            except ValueError as error:
                raise ValueError(
                    f"Invalid {name!r} tolerance profile: {metadata_path}"
                ) from error
            profiles[str(name)] = tuple(float(item) for item in vector)
        records.append((
            metadata_path,
            metadata,
            instructions,
            int(resolution[0]),
            int(resolution[1]),
            profiles,
            _demonstration_durations_s(metadata_path.parent, metadata),
        ))

    # A task uses one deadline on every robot. Directory name is part of the
    # dataset's public task identity and distinguishes variants that share a
    # runtime task_id (for example upright and lying geometry tasks).
    grouped_durations: dict[tuple[str, str, str], list[float]] = {}
    for metadata_path, metadata, *_, durations in records:
        key = (
            metadata_path.parent.name,
            str(metadata["scene"]),
            str(metadata["task_id"]),
        )
        grouped_durations.setdefault(key, []).extend(durations)

    tasks: list[DatasetTaskSpec] = []
    for (
        metadata_path,
        metadata,
        instructions,
        width,
        height,
        profiles,
        durations,
    ) in records:
        key = (
            metadata_path.parent.name,
            str(metadata["scene"]),
            str(metadata["task_id"]),
        )
        shared_durations = grouped_durations[key]
        if shared_durations:
            maximum_demo_duration_s = max(shared_durations)
            maximum_steps = int(math.ceil(
                (maximum_demo_duration_s + _ROLLOUT_TIME_MARGIN_S)
                * _BENCHMARK_CONTROL_HZ
                - 1e-12
            ))
        else:
            # Keep metadata-only catalogues usable for development. Released
            # benchmark datasets have manifests, so their normal path always
            # uses the measured demonstration duration above.
            metadata_maximum_steps = int(
                metadata.get("collection", {}).get(
                    "maximum_frames_per_episode", 600,
                )
            )
            maximum_demo_duration_s = (
                metadata_maximum_steps / _BENCHMARK_CONTROL_HZ
            )
            maximum_steps = metadata_maximum_steps + int(round(
                _ROLLOUT_TIME_MARGIN_S * _BENCHMARK_CONTROL_HZ
            ))
        tasks.append(DatasetTaskSpec(
            directory=metadata_path.parent,
            scene=str(metadata["scene"]),
            task_id=str(metadata["task_id"]),
            task_name=str(metadata["task_name"]),
            instructions=instructions,
            width=width,
            height=height,
            maximum_steps=maximum_steps,
            maximum_demonstration_duration_s=maximum_demo_duration_s,
            rotation_tolerance_profiles_rad=profiles,
            robot_uid=str(metadata.get("robot", {}).get("uid", "panda")),
        ))
    return tuple(tasks)


def builtin_training_tasks_for_robot(
    dataset_root: Path,
    robot_uid: str,
) -> tuple[DatasetTaskSpec, ...]:
    """Build benchmark metadata from the repository's curated task registry.

    Panda supports the complete training registry. The other robot models
    expose all 22 short tasks, including the workspace-boundary suite. Synthetic
    directories preserve the public dataset task identity without creating
    files on disk.
    """
    maximum_duration_s = _BUILTIN_MAXIMUM_FRAMES / _BENCHMARK_CONTROL_HZ
    result: list[DatasetTaskSpec] = []
    for scene, task in training_tasks():
        if (
            robot_uid != "panda"
            and scene.subcategory == MULTI_SUBTASK_SUBCATEGORY
        ):
            continue
        result.append(DatasetTaskSpec(
            directory=(
                dataset_root / "builtin" / robot_uid / scene.scene_id / task.task_id
            ),
            scene=scene.runtime_scene,
            task_id=task.runtime_task,
            task_name=task.instructions[0],
            instructions=task.instructions,
            width=_BUILTIN_RENDER_WIDTH,
            height=_BUILTIN_RENDER_HEIGHT,
            maximum_steps=(
                _BUILTIN_MAXIMUM_FRAMES
                + int(round(_ROLLOUT_TIME_MARGIN_S * _BENCHMARK_CONTROL_HZ))
            ),
            maximum_demonstration_duration_s=maximum_duration_s,
            rotation_tolerance_profiles_rad={},
            robot_uid=robot_uid,
        ))
    return tuple(result)


def builtin_tolerance_discrimination_tasks_for_robot(
    dataset_root: Path,
    robot_uid: str,
) -> tuple[DatasetTaskSpec, ...]:
    """Build metadata-only entries for the paired tolerance benchmark."""
    maximum_duration_s = _BUILTIN_MAXIMUM_FRAMES / _BENCHMARK_CONTROL_HZ
    return tuple(
        DatasetTaskSpec(
            directory=(
                dataset_root
                / "builtin_tolerance_discrimination"
                / robot_uid
                / scene.scene_id
                / task.task_id
            ),
            scene=scene.runtime_scene,
            task_id=task.runtime_task,
            task_name=task.instructions[0],
            instructions=task.instructions,
            width=_BUILTIN_RENDER_WIDTH,
            height=_BUILTIN_RENDER_HEIGHT,
            maximum_steps=(
                _BUILTIN_MAXIMUM_FRAMES
                + int(round(_ROLLOUT_TIME_MARGIN_S * _BENCHMARK_CONTROL_HZ))
            ),
            maximum_demonstration_duration_s=maximum_duration_s,
            rotation_tolerance_profiles_rad={},
            robot_uid=robot_uid,
            benchmark_only=True,
        )
        for scene, task in tolerance_discrimination_tasks()
    )


def resolve_dataset_task(
    dataset_root: Path, requested: str, robot_uid: str | None = None,
) -> DatasetTaskSpec:
    selected_robot = robot_uid or "panda"
    discovered = discover_dataset_tasks(dataset_root)
    tasks = (
        *discovered,
        *benchmark_only_tasks_for_robot(dataset_root, selected_robot),
    )
    normalized = requested.strip().casefold()

    def matching(candidates: tuple[DatasetTaskSpec, ...]) -> list[DatasetTaskSpec]:
        return [
            task for task in candidates
            if normalized in {
                task.task_id.casefold(),
                task.task_name.casefold(),
                task.directory.name.casefold(),
            }
        ]

    matches = matching(tasks)
    if not matches:
        # A real dataset remains authoritative when it contains the requested
        # task.  Otherwise the in-repository registry makes single-task
        # benchmarks usable before any demonstrations have been collected.
        builtin = (
            *builtin_training_tasks_for_robot(dataset_root, selected_robot),
            *builtin_tolerance_discrimination_tasks_for_robot(
                dataset_root, selected_robot,
            ),
        )
        if selected_robot != "panda":
            # Single-task execution intentionally permits tasks whose curated
            # definition belongs to Panda when the selected robot can run the
            # same registered MuJoCo scene.
            builtin = (
                *builtin,
                *builtin_training_tasks_for_robot(dataset_root, "panda"),
                *builtin_tolerance_discrimination_tasks_for_robot(
                    dataset_root, "panda",
                ),
            )
        matches = matching(builtin)
        tasks = (*tasks, *builtin)
    if robot_uid is not None:
        robot_matches = [task for task in matches if task.robot_uid == robot_uid]
        # Prefer metadata collected for the execution robot. If it has no such
        # task, permit an explicitly selected task from another robot because
        # the MuJoCo runtime supports the same registered scenes and tasks.
        if robot_matches:
            matches = robot_matches
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(sorted({
        f"{task.directory.name} ({task.task_name}) [{task.robot_uid}]"
        for task in tasks
    })) or "<none>"
    if not matches:
        raise ValueError(f"Unknown dataset task {requested!r}; available: {available}")
    candidates = ", ".join(
        f"{task.directory.name} [{task.robot_uid}]" for task in matches
    )
    raise ValueError(
        f"Ambiguous dataset task {requested!r}; use a task-directory name: "
        f"{candidates}"
    )


def dataset_tasks_for_robot(
    dataset_root: Path,
    robot_uid: str,
) -> tuple[DatasetTaskSpec, ...]:
    """Return real dataset tasks, or the built-in catalogue when none exist."""
    discovered = tuple(
        task for task in discover_dataset_tasks(dataset_root)
        if task.robot_uid == robot_uid
    )
    if discovered:
        return discovered
    return builtin_training_tasks_for_robot(dataset_root, robot_uid)


_UNSEEN_REFERENCE_TASKS = {
    # Prefer the current fruit sequence while retaining deadline inheritance
    # for already-collected v1 datasets with the prior single-fruit tasks.
    "unseen_apple_to_plate": (
        "apple_then_banana_to_tray", "strawberry_to_tray", "pear_to_tray",
    ),
    "unseen_apple_to_bowl": (
        "apple_then_banana_to_tray", "strawberry_to_tray", "pear_to_tray",
    ),
    "unseen_lemon_to_plate": (
        "apple_then_banana_to_tray", "strawberry_to_tray", "pear_to_tray",
    ),
    "unseen_lemon_to_bowl": (
        "apple_then_banana_to_tray", "strawberry_to_tray", "pear_to_tray",
    ),
    "unseen_sports_ball_to_plate": ("geometry_plate_ball",),
    "unseen_upright_box_to_plate": ("geometry_plate_box_upright",),
    "unseen_upright_box_to_region": ("geometry_region_box_upright",),
    "unseen_press_button": ("press_button",),
    "unseen_press_four_slot_power_strip": ("press_power_strip",),
}


def benchmark_only_tasks_for_robot(
    dataset_root: Path,
    robot_uid: str,
) -> tuple[DatasetTaskSpec, ...]:
    """Build metadata-only held-out tasks from their seen-task deadlines."""
    dataset_tasks = dataset_tasks_for_robot(dataset_root, robot_uid)
    by_directory = {task.directory.name: task for task in dataset_tasks}
    result: list[DatasetTaskSpec] = []
    for scene, task in unseen_object_tasks():
        reference_names = _UNSEEN_REFERENCE_TASKS[task.task_id]
        reference = next((
            by_directory[name]
            for name in reference_names
            if name in by_directory
        ), None)
        if reference is None:
            continue
        result.append(DatasetTaskSpec(
            directory=(
                dataset_root / "benchmark_only" / robot_uid / task.task_id
            ),
            scene=scene.runtime_scene,
            task_id=task.runtime_task,
            task_name=task.instructions[0],
            instructions=task.instructions,
            width=reference.width,
            height=reference.height,
            maximum_steps=reference.maximum_steps,
            maximum_demonstration_duration_s=(
                reference.maximum_demonstration_duration_s
            ),
            rotation_tolerance_profiles_rad={},
            robot_uid=robot_uid,
            benchmark_only=True,
            reference_task_directory=reference.directory.name,
        ))
    return tuple(result)


def benchmark_seed_plan(count: int, initial_seed: int) -> tuple[int, ...]:
    """Use the requested seed first, then deterministically derive unique seeds."""
    if count <= 0:
        raise ValueError("Benchmark count must be positive")
    if initial_seed < -1 or initial_seed >= 2**32:
        raise ValueError("Seed must be -1 or an unsigned 32-bit integer")
    result = [int(initial_seed)]
    used = set(result)
    root = 0 if initial_seed == -1 else int(initial_seed)
    rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence((root, _BENCHMARK_SEED_STREAM))
    ))
    while len(result) < count:
        candidate = int(rng.integers(0, 2**32, dtype=np.uint64))
        if candidate not in used:
            used.add(candidate)
            result.append(candidate)
    return tuple(result)
