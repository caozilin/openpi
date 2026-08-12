"""Safely append a fixed set of raw episodes to an existing LeRobot dataset.

The command is a dry-run unless ``--execute`` is passed. Existing Parquet files
are verified against a SHA-256 baseline before any write, and are never
rewritten. Metadata is snapshotted and restored together with removal of newly
created Parquet files if a normal Python exception interrupts the append.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pyarrow.parquet as pq

import convert_franka_mujoco_data_to_lerobot as converter


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_baseline(path: Path) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid sha256sum line") from error
        relative_path = Path(relative.lstrip("*"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{path}:{line_number}: unsafe relative path {relative_path}")
        if relative_path in entries:
            raise ValueError(f"{path}:{line_number}: duplicate path {relative_path}")
        if len(digest) != 64:
            raise ValueError(f"{path}:{line_number}: invalid SHA-256 digest")
        entries[relative_path] = digest.lower()
    return entries


def _verify_existing_dataset(
    dataset_root: Path,
    baseline_path: Path,
    *,
    expected_episodes: int,
    expected_frames: int,
) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("total_episodes") != expected_episodes:
        raise ValueError(
            f"{info_path}: expected {expected_episodes} episodes, got {info.get('total_episodes')}"
        )
    if info.get("total_frames") != expected_frames:
        raise ValueError(f"{info_path}: expected {expected_frames} frames, got {info.get('total_frames')}")

    parquets = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if len(parquets) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} Parquet files, found {len(parquets)}")
    episode_lines = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    stats_lines = (dataset_root / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines()
    if len(episode_lines) != expected_episodes or len(stats_lines) != expected_episodes:
        raise ValueError(
            "Episode metadata count mismatch: "
            f"episodes={len(episode_lines)}, stats={len(stats_lines)}, expected={expected_episodes}"
        )

    row_total = sum(pq.ParquetFile(path).metadata.num_rows for path in parquets)
    if row_total != expected_frames:
        raise ValueError(f"Existing Parquet row total is {row_total}, expected {expected_frames}")

    baseline = _load_baseline(baseline_path)
    if len(baseline) != expected_episodes:
        raise ValueError(f"SHA-256 baseline has {len(baseline)} entries, expected {expected_episodes}")
    for index, (relative_path, expected_digest) in enumerate(baseline.items(), start=1):
        path = dataset_root / relative_path
        if not path.is_file():
            raise ValueError(f"Baseline file is missing: {path}")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise ValueError(f"Baseline hash mismatch: {path}")
        if index % 250 == 0 or index == expected_episodes:
            print(f"Verified existing hashes: {index}/{expected_episodes}")
    return info


def _prepare_append_specs(
    raw_task_dir: Path,
    *,
    append_start: int,
    append_count: int,
    expected_append_frames: int,
) -> list[tuple[Path, dict[str, Any], dict[str, list[float]], str, int, int]]:
    metadata, manifest, robot_uid, width, height = converter._validate_task(raw_task_dir)
    profiles = converter._load_tolerance_annotation(
        raw_task_dir,
        metadata["annotation"]["rotation_tolerance_profiles_rad"],
    )
    entries = list(converter._episode_entries(raw_task_dir, manifest))[append_start : append_start + append_count]
    if len(entries) != append_count:
        raise ValueError(f"Selected {len(entries)} raw episodes, expected {append_count}")

    specs = []
    frame_total = 0
    for entry in entries:
        episode_dir = raw_task_dir / str(entry["path"])
        required = (episode_dir / "trajectory.json", episode_dir / "main_rgb.mp4", episode_dir / "wrist_rgb.mp4")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"{episode_dir}: missing required files {missing}")
        document = json.loads((episode_dir / "trajectory.json").read_text(encoding="utf-8"))
        actual_frames = len(document.get("trajectory", []))
        manifest_frames = int(entry.get("frames", -1))
        if actual_frames != manifest_frames:
            raise ValueError(
                f"{episode_dir}: trajectory has {actual_frames} frames, manifest declares {manifest_frames}"
            )
        frame_total += actual_frames
        specs.append((episode_dir, entry, profiles, robot_uid, width, height))
    if frame_total != expected_append_frames:
        raise ValueError(f"Selected raw episodes contain {frame_total} frames, expected {expected_append_frames}")
    return specs


def _expected_new_paths(dataset: Any, start: int, count: int) -> list[Path]:
    paths = [dataset.root / dataset.meta.get_data_file_path(ep_index=index) for index in range(start, start + count)]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite new episode path: {existing[0]}")
    return paths


def _open_for_append(repo_id: str, dataset_root: Path) -> Any:
    """Open only LeRobot metadata and an empty in-memory table for new rows.

    The normal LeRobotDataset constructor materializes all existing Parquet
    files into a Hugging Face Arrow cache. That is unnecessary for append and
    can require roughly another full-dataset worth of disk space.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.lerobot_dataset import get_safe_default_codec

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset.repo_id = repo_id
    dataset.root = dataset_root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.episode_buffer = dataset.create_episode_buffer()
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = get_safe_default_codec()
    return dataset


def _append(dataset: Any, specs: list[tuple], *, workers: int, progress_interval_seconds: float) -> None:
    import datasets
    import tqdm

    datasets.disable_progress_bars()
    total_frames = sum(int(entry["frames"]) for _, entry, _, _, _, _ in specs)
    progress = tqdm.tqdm(
        total=total_frames,
        desc="Appending",
        unit="frame",
        mininterval=progress_interval_seconds,
        dynamic_ncols=True,
    )
    completed = 0
    episode_iter = iter(specs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}

        def submit_next() -> bool:
            try:
                episode_dir, entry, profiles, robot_uid, width, height = next(episode_iter)
            except StopIteration:
                return False
            future = executor.submit(
                converter._prepare_episode,
                episode_dir,
                entry,
                tolerance_profiles=profiles,
                robot_uid=robot_uid,
                width=width,
                height=height,
            )
            futures[future] = episode_dir
            return True

        for _ in range(min(workers, len(specs))):
            submit_next()
        while futures:
            future = next(as_completed(futures))
            episode_dir = futures.pop(future)
            try:
                frames = future.result()
            except Exception as error:
                raise RuntimeError(f"Failed to prepare {episode_dir}") from error
            converter._write_episode(dataset, frames)
            completed += 1
            progress.set_postfix_str(f"episodes={completed}/{len(specs)}", refresh=False)
            progress.update(len(frames))
            submit_next()
    progress.close()


def main(
    raw_task_dir: Path,
    repo_id: str,
    baseline_path: Path,
    *,
    expected_existing_episodes: int = 4147,
    expected_existing_frames: int = 528803,
    append_start: int = 0,
    append_count: int = 253,
    expected_append_frames: int = 41317,
    workers: int = 12,
    image_writer_threads: int = 2,
    progress_interval_seconds: float = 5.0,
    execute: bool = False,
) -> None:
    """Preflight or execute a protected append to an existing LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if workers <= 0 or image_writer_threads < 0 or progress_interval_seconds <= 0:
        raise ValueError("Invalid worker, image-writer, or progress interval setting")
    raw_task_dir = raw_task_dir.expanduser().resolve()
    baseline_path = baseline_path.expanduser().resolve()
    dataset_root = HF_LEROBOT_HOME / repo_id
    if not raw_task_dir.is_dir() or not dataset_root.is_dir() or not baseline_path.is_file():
        raise FileNotFoundError("Raw task directory, existing dataset, or baseline path does not exist")

    print("Verifying existing dataset and SHA-256 baseline...")
    info = _verify_existing_dataset(
        dataset_root,
        baseline_path,
        expected_episodes=expected_existing_episodes,
        expected_frames=expected_existing_frames,
    )
    print("Validating raw append set...")
    specs = _prepare_append_specs(
        raw_task_dir,
        append_start=append_start,
        append_count=append_count,
        expected_append_frames=expected_append_frames,
    )

    final_episodes = expected_existing_episodes + append_count
    final_frames = expected_existing_frames + expected_append_frames
    print(
        f"Append plan: {append_count} episodes / {expected_append_frames} frames; "
        f"final dataset: {final_episodes} episodes / {final_frames} frames"
    )
    if not execute:
        print("DRY RUN PASSED. No files were written. Re-run with --execute to append.")
        return

    dataset = _open_for_append(repo_id, dataset_root)
    if (
        dataset.meta.total_episodes != expected_existing_episodes
        or dataset.meta.total_frames != expected_existing_frames
    ):
        raise ValueError("Loaded LeRobot dataset counts changed after preflight")
    new_paths = _expected_new_paths(dataset, expected_existing_episodes, append_count)

    rollback_dir = dataset_root.parent / f".{dataset_root.name}.append-rollback-meta"
    if rollback_dir.exists():
        raise FileExistsError(f"Rollback directory already exists: {rollback_dir}")
    shutil.copytree(dataset_root / "meta", rollback_dir)
    try:
        if image_writer_threads:
            dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)
        _append(dataset, specs, workers=workers, progress_interval_seconds=progress_interval_seconds)
        dataset.stop_image_writer()
        if dataset.meta.total_episodes != final_episodes or dataset.meta.total_frames != final_frames:
            raise ValueError(
                "Post-append count mismatch: "
                f"episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}"
            )
        if len(list((dataset_root / "data").rglob("episode_*.parquet"))) != final_episodes:
            raise ValueError("Post-append Parquet count mismatch")

        # Prove the original Parquet files were not changed before discarding
        # the rollback snapshot.
        for index, (relative_path, expected_digest) in enumerate(_load_baseline(baseline_path).items(), start=1):
            if _sha256(dataset_root / relative_path) != expected_digest:
                raise ValueError(f"Old Parquet changed after append: {relative_path}")
            if index % 250 == 0 or index == expected_existing_episodes:
                print(f"Reverified old hashes: {index}/{expected_existing_episodes}")
    except BaseException:
        dataset.stop_image_writer()
        for path in new_paths:
            path.unlink(missing_ok=True)
        shutil.rmtree(dataset_root / "images", ignore_errors=True)
        shutil.rmtree(dataset_root / "meta")
        shutil.copytree(rollback_dir, dataset_root / "meta")
        raise
    else:
        shutil.rmtree(rollback_dir)

    print(f"APPEND COMPLETE: {final_episodes} episodes, {final_frames} frames; all old hashes unchanged.")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
