"""Safely append a fixed set of raw episodes to an existing LeRobot dataset.

The command is a dry-run unless ``--execute`` is passed. Existing Parquet files
are verified against a SHA-256 baseline before any write, and are never
rewritten. Metadata is snapshotted and restored together with removal of newly
created Parquet files if a normal Python exception interrupts the append.
"""

# This entry point intentionally shares the converter's internal transaction helpers.
# ruff: noqa: SLF001

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import convert_franka_mujoco_data_to_lerobot as converter
import pyarrow.parquet as pq


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
    joint_task_tolerance: bool,
) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    converter._validate_dataset_features(info, joint_task_tolerance=joint_task_tolerance)
    if info.get("total_episodes") != expected_episodes:
        raise ValueError(f"{info_path}: expected {expected_episodes} episodes, got {info.get('total_episodes')}")
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
    joint_task_tolerance: bool,
) -> list[converter.EpisodeSpec]:
    _, manifest, robot_uid, width, height = converter._validate_task(raw_task_dir)
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
        document, digest = converter.dataset_format.read_trajectory(episode_dir)
        if joint_task_tolerance:
            converter.dataset_format.load_rotation_tolerance(episode_dir, document, digest)
        actual_frames = len(document.get("trajectory", []))
        manifest_frames = int(entry.get("frames", -1))
        if actual_frames != manifest_frames:
            raise ValueError(
                f"{episode_dir}: trajectory has {actual_frames} frames, manifest declares {manifest_frames}"
            )
        frame_total += actual_frames
        specs.append((episode_dir, entry, robot_uid, width, height))
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
    return converter._open_dataset_for_append(
        repo_id,
        dataset_root,
        image_writer_threads=0,
        image_writer_processes=0,
    )


def _append(
    dataset: Any,
    specs: list[converter.EpisodeSpec],
    *,
    raw_dir: Path,
    joint_task_tolerance: bool,
    workers: int,
    progress_interval_seconds: float,
) -> None:
    import datasets
    import tqdm

    datasets.disable_progress_bars()
    total_frames = sum(int(entry["frames"]) for _, entry, _, _, _ in specs)
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
                spec = next(episode_iter)
                episode_dir, entry, robot_uid, width, height = spec
            except StopIteration:
                return False
            future = executor.submit(
                converter._prepare_episode,
                episode_dir,
                entry,
                joint_task_tolerance=joint_task_tolerance,
                robot_uid=robot_uid,
                width=width,
                height=height,
            )
            futures[future] = spec
            return True

        for _ in range(min(workers, len(specs))):
            submit_next()
        while futures:
            future = next(as_completed(futures))
            spec = futures.pop(future)
            episode_dir = spec[0]
            try:
                frames = future.result()
            except Exception as error:
                raise RuntimeError(f"Failed to prepare {episode_dir}") from error
            converter._append_conversion_state(
                dataset.root,
                dataset.meta.total_episodes,
                converter._source_id(raw_dir, spec),
                converter._source_fingerprint(spec, joint_task_tolerance=joint_task_tolerance),
            )
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
    raw_dir: Path,
    *,
    expected_existing_episodes: int,
    expected_existing_frames: int,
    append_start: int = 0,
    append_count: int,
    expected_append_frames: int,
    joint_task_tolerance: bool = True,
    workers: int = 12,
    image_writer_threads: int = 2,
    progress_interval_seconds: float = 5.0,
    execute: bool = False,
) -> None:
    """Preflight or execute a protected append to an existing LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    if workers <= 0 or image_writer_threads < 0 or progress_interval_seconds <= 0:
        raise ValueError("Invalid worker, image-writer, or progress interval setting")
    raw_task_dir = raw_task_dir.expanduser().resolve()
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_task_dir.is_relative_to(raw_dir):
        raise ValueError("raw_task_dir must be under the original conversion raw_dir")
    baseline_path = baseline_path.expanduser().resolve()
    dataset_root = HF_LEROBOT_HOME / repo_id
    if not raw_task_dir.is_dir() or not dataset_root.is_dir() or not baseline_path.is_file():
        raise FileNotFoundError("Raw task directory, existing dataset, or baseline path does not exist")

    print("Verifying existing dataset and SHA-256 baseline...")
    _verify_existing_dataset(
        dataset_root,
        baseline_path,
        expected_episodes=expected_existing_episodes,
        expected_frames=expected_existing_frames,
        joint_task_tolerance=joint_task_tolerance,
    )
    journal = converter._read_jsonlines(dataset_root / converter.CONVERSION_STATE_PATH)
    if len(journal) != expected_existing_episodes:
        raise ValueError("Current conversion journal must cover every existing episode")
    print("Validating raw append set...")
    specs = _prepare_append_specs(
        raw_task_dir,
        append_start=append_start,
        append_count=append_count,
        expected_append_frames=expected_append_frames,
        joint_task_tolerance=joint_task_tolerance,
    )
    completed_ids = {row["source_id"] for row in journal}
    append_ids = [converter._source_id(raw_dir, spec) for spec in specs]
    if len(set(append_ids)) != len(append_ids) or completed_ids.intersection(append_ids):
        raise ValueError("Append set contains duplicate or already converted source episodes")

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
        _append(
            dataset,
            specs,
            raw_dir=raw_dir,
            joint_task_tolerance=joint_task_tolerance,
            workers=workers,
            progress_interval_seconds=progress_interval_seconds,
        )
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
