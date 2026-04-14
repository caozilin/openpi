"""
Convert vla4desk Franka recordings to a LeRobot dataset.

Expected source layout:

    /path/to/collected/
        task_a/
            epo_1/
                cam1.mp4
                cam2.mp4
                data.json
            epo_2/
                ...
        task_b/
            ...

The resulting dataset stores the same feature names used by the Libero example:
`image`, `wrist_image`, `state`, `actions`.
This lets OpenPI reuse `LeRobotLiberoDataConfig` directly for training.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

import imageio.v2 as imageio
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _iter_episode_dirs(source_dir: pathlib.Path) -> list[pathlib.Path]:
    if (source_dir / "data.json").is_file():
        return [source_dir]

    episode_dirs = []
    for path in source_dir.rglob("data.json"):
        if path.parent.name.startswith("epo_"):
            episode_dirs.append(path.parent)
    return sorted(episode_dirs)


def _infer_fps(episode_dirs: list[pathlib.Path], fps: int | None) -> int:
    if fps is not None:
        return fps

    inferred_fps = None
    for episode_dir in episode_dirs:
        payload = _load_json(episode_dir / "data.json")
        episode_fps = int(round(float(payload.get("collect_hz", 10.0))))
        if inferred_fps is None:
            inferred_fps = episode_fps
        elif inferred_fps != episode_fps:
            raise ValueError(
                f"Mixed collect_hz detected ({inferred_fps} vs {episode_fps}). "
                "Pass --fps explicitly to force a single LeRobot dataset fps."
            )
    if inferred_fps is None:
        raise ValueError("Could not infer fps from source episodes.")
    return inferred_fps


def _get_first_frame_shape(episode_dirs: list[pathlib.Path]) -> tuple[int, int, int]:
    for episode_dir in episode_dirs:
        cam1_path = episode_dir / "cam1.mp4"
        if not cam1_path.is_file():
            continue
        reader = imageio.get_reader(cam1_path)
        try:
            frame = np.asarray(reader.get_data(0))
            if frame.ndim != 3:
                raise ValueError(f"Expected HWC image in {cam1_path}, got shape {frame.shape}")
            return tuple(int(x) for x in frame.shape)
        finally:
            reader.close()
    raise ValueError("Could not read a first frame from any cam1.mp4")


def _decode_action(raw_action: list[float]) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).copy()
    if action.shape != (7,):
        raise ValueError(f"Expected action shape (7,), got {action.shape}")
    return action


def _resolve_task(payload: dict[str, Any], episode_dir: pathlib.Path, prompt: str | None) -> str:
    if prompt:
        return prompt
    json_prompt = str(payload.get("prompt", "")).strip()
    if json_prompt:
        return json_prompt
    task_name = str(payload.get("task_name", "")).strip()
    if task_name:
        return task_name
    return episode_dir.parent.name


def main(
    source_dir: str,
    repo_id: str,
    *,
    fps: int | None = None,
    prompt: str | None = None,
    push_to_hub: bool = False,
    clean_output: bool = True,
):
    source_path = pathlib.Path(source_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")

    episode_dirs = _iter_episode_dirs(source_path)
    if not episode_dirs:
        raise ValueError(
            "No episodes found. Expected either a single episode directory containing "
            "data.json/cam1.mp4/cam2.mp4 or a root directory with task/epo_x subdirectories."
        )

    dataset_fps = _infer_fps(episode_dirs, fps)
    image_shape = _get_first_frame_shape(episode_dirs)
    output_path = HF_LEROBOT_HOME / repo_id

    if clean_output and output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=dataset_fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=2,
    )

    black_frame = np.zeros(image_shape, dtype=np.uint8)

    for episode_dir in episode_dirs:
        payload = _load_json(episode_dir / "data.json")
        samples = payload.get("frames")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Expected non-empty frames list in {episode_dir / 'data.json'}")

        task = _resolve_task(payload, episode_dir, prompt)

        cam1_path = episode_dir / "cam1.mp4"
        cam2_path = episode_dir / "cam2.mp4"
        if not cam1_path.is_file():
            raise FileNotFoundError(f"Missing cam1.mp4 in {episode_dir}")

        cam1_reader = imageio.get_reader(cam1_path)
        cam2_reader = imageio.get_reader(cam2_path) if cam2_path.is_file() else None
        try:
            for idx, sample in enumerate(samples):
                frame1 = np.asarray(cam1_reader.get_data(idx), dtype=np.uint8)
                if frame1.shape != image_shape:
                    raise ValueError(
                        f"Frame shape mismatch in {cam1_path} at index {idx}: "
                        f"expected {image_shape}, got {frame1.shape}"
                    )

                if cam2_reader is not None:
                    frame2 = np.asarray(cam2_reader.get_data(idx), dtype=np.uint8)
                    if frame2.shape != image_shape:
                        raise ValueError(
                            f"Frame shape mismatch in {cam2_path} at index {idx}: "
                            f"expected {image_shape}, got {frame2.shape}"
                        )
                else:
                    frame2 = black_frame

                state = np.asarray(sample["state"], dtype=np.float32)
                if state.shape != (8,):
                    raise ValueError(f"Expected state shape (8,), got {state.shape} in {episode_dir}")

                dataset.add_frame(
                    {
                        "image": frame1,
                        "wrist_image": frame2,
                        "state": state,
                        "actions": _decode_action(sample["action"]),
                        "task": task,
                    }
                )
            dataset.save_episode()
        finally:
            cam1_reader.close()
            if cam2_reader is not None:
                cam2_reader.close()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["franka", "panda", "lerobot", "vla4desk"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
