"""Lossless, resumable LeRobot v2 export of audited reconstructed LIBERO HDF5.

Keeps the recorded 20 Hz, physical delta actions, native pre-action state8 and
all frames. Images rotate 180 degrees exactly once and are embedded losslessly.
No Hub access, action rescaling, frame filtering or simulation is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import shutil

import h5py
import numpy as np

FPS = 20
FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_rows(path, values):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    temp.replace(path)


def source_plan(raw):
    manifest = json.loads((raw / "manifest.json").read_text())
    assert manifest["complete"], "Raw dataset is not complete and validated"
    validation = json.loads((raw / "validation.json").read_text())
    assert validation["complete"] and validation["episodes"] == manifest["episodes"]
    sources = []
    for task in sorted(manifest["tasks"], key=lambda task: task["case"]):
        index = raw / "datasets" / task["suite"] / Path(task["path"]).name
        with h5py.File(index) as file:
            assert file.attrs["control_hz"] == FPS
            language = str(file.attrs["language"])
            demos = sorted(file["data"], key=lambda demo: int(demo.removeprefix("demo_")))
            assert len(demos) == task["episodes"]
            total = 0
            for demo in demos:
                group = file["data"][demo]
                recording = Path(group.file.filename).resolve()
                assert recording.is_relative_to(raw.parent.resolve()), recording
                assert group.attrs["success"] and group["dones"][-1] == 1
                n = int(group.attrs["num_samples"])
                assert group["actions"].shape == (n, 7)
                assert group["obs/ee_states"].shape == (n, 6)
                assert group["obs/gripper_states"].shape == (n, 2)
                for name in ("agentview_rgb", "eye_in_hand_rgb"):
                    assert group["obs"][name].shape == (n, 256, 256, 3)
                    assert group["obs"][name].dtype == np.uint8
                sources.append(dict(episode_index=len(sources), case=task["case"], suite=task["suite"],
                                    task_id=task["task_id"], demo=demo, language=language, frames=n,
                                    recording=str(recording.relative_to(raw.parent.resolve())),
                                    recording_sha256=sha(recording)))
                total += n
            assert total == task["frames"]
    assert len(sources) == manifest["episodes"]
    assert len({(s["case"], s["demo"]) for s in sources}) == len(sources)
    return dict(manifest_sha256=sha(raw / "manifest.json"), sources=sources, fps=FPS,
                state="native actual pre-action ee_states6 + two native gripper qpos",
                actions="unchanged physical xyz metres/world rotvec radians/original gripper",
                images="native RGB rotated 180 degrees once; lossless embedded PNG", version=1)


def open_append(repo_id, output):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=output)
    dataset.repo_id = repo_id
    dataset.root = output
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = None
    dataset.start_image_writer(0, 4)
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def repair_tail(output, plan):
    """Keep only episodes with parquet, episode metadata and statistics commits."""
    info = json.loads((output / "meta/info.json").read_text())
    assert info["fps"] == FPS and info["robot_type"] == "panda_shared_joint_servo"
    assert {k: {**v, "shape": list(v["shape"])} for k, v in FEATURES.items()} == {
        k: info["features"][k] for k in FEATURES}
    episodes = rows(output / "meta/episodes.jsonl")
    stats = rows(output / "meta/episodes_stats.jsonl")
    committed = 0
    for index in range(min(len(episodes), len(stats), len(plan["sources"]))):
        parquet = output / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        if episodes[index]["episode_index"] != index or stats[index]["episode_index"] != index or not parquet.exists():
            break
        assert episodes[index]["length"] == plan["sources"][index]["frames"]
        committed += 1
    write_rows(output / "meta/episodes.jsonl", episodes[:committed])
    write_rows(output / "meta/episodes_stats.jsonl", stats[:committed])
    for path in output.glob("data/chunk-*/episode_*.parquet"):
        if int(path.stem.removeprefix("episode_")) >= committed:
            path.unlink()
    if (output / "images").exists():
        shutil.rmtree(output / "images")
    info.update(total_episodes=committed, total_frames=sum(s["frames"] for s in plan["sources"][:committed]),
                total_tasks=len(rows(output / "meta/tasks.jsonl")), total_videos=0,
                total_chunks=math.ceil(committed / info["chunks_size"]), splits={"train": f"0:{committed}"})
    dump(output / "meta/info.json", info)
    return committed


def read_episode(raw, source):
    with h5py.File(raw.parent / source["recording"]) as file:
        group = file["data"][source["demo"]]
        state = np.concatenate([group["obs/ee_states"][:], group["obs/gripper_states"][:]], axis=1).astype(np.float32)
        actions = group["actions"][:].astype(np.float32)
        assert np.isfinite(state).all() and np.isfinite(actions).all()
        images = group["obs/agentview_rgb"][:][:, ::-1, ::-1].copy()
        wrists = group["obs/eye_in_hand_rgb"][:][:, ::-1, ::-1].copy()
    return state, actions, images, wrists


def audit_episode(raw, output, info, source):
    import pyarrow.parquet as pq
    from PIL import Image

    index = source["episode_index"]
    path = output / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
    table = pq.read_table(path)
    n = source["frames"]
    assert table.num_rows == n
    state, actions, images, wrists = read_episode(raw, source)
    np.testing.assert_array_equal(np.asarray(table["state"].to_pylist(), dtype=np.float32), state)
    np.testing.assert_array_equal(np.asarray(table["actions"].to_pylist(), dtype=np.float32), actions)
    np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(n))
    np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(n, index))
    np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(n) / FPS, atol=1e-5, rtol=0)
    tasks = {r["task_index"]: r["task"] for r in rows(output / "meta/tasks.jsonl")}
    assert {tasks[int(t)] for t in table["task_index"].to_numpy()} == {source["language"]}
    for key, expected in (("image", images), ("wrist_image", wrists)):
        for i, stored in enumerate(table[key].to_pylist()):
            assert stored["bytes"] is not None, "Images must be self-contained in parquet"
            with Image.open(io.BytesIO(stored["bytes"])) as image:
                np.testing.assert_array_equal(np.asarray(image), expected[i])
    return dict(episode_index=index, case=source["case"], suite=source["suite"], task_id=source["task_id"],
                demo=source["demo"], frames=n, all_camera_pixels_equal=True, state_action_equal=True,
                parquet_sha256=sha(path))


def audit(raw, output, plan, repo_id, workers=4):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_episodes"] == len(plan["sources"])
    assert info["total_frames"] == sum(s["frames"] for s in plan["sources"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        checks = list(pool.map(lambda source: audit_episode(raw, output, info, source), plan["sources"]))
    reader = LeRobotDataset(repo_id=repo_id, root=output, episodes=[0], force_cache_sync=False)
    sample = reader[0]
    assert tuple(sample["state"].shape) == (8,) and tuple(sample["actions"].shape) == (7,)
    assert tuple(sample["image"].shape) == (3, 256, 256)
    assert tuple(sample["wrist_image"].shape) == (3, 256, 256)
    suites = {suite: dict(episodes=sum(s["suite"] == suite for s in checks),
                          frames=sum(s["frames"] for s in checks if s["suite"] == suite))
              for suite in sorted({s["suite"] for s in checks})}
    result = dict(complete=True, episodes=len(checks), frames=info["total_frames"], tasks=info["total_tasks"],
                  fps=FPS, all_camera_pixels_equal=True, all_state_action_equal=True,
                  reader_smoke_passed=True, suites=suites, episode_checks=checks,
                  source_manifest_sha256=plan["manifest_sha256"])
    dump(output / "meta/reconstruction_audit.json", result)
    return result


def convert(raw, output, repo_id, resume=False, validate_only=False, max_episodes=None):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    raw, output = raw.resolve(), output.resolve()
    plan = source_plan(raw)
    if max_episodes is not None:
        plan["sources"] = plan["sources"][:max_episodes]
        plan["smoke_only"] = True
    if output.exists():
        assert resume or validate_only, "Output exists; use --resume with the same immutable raw dataset"
        assert json.loads((output / "meta/reconstruction_sources.json").read_text()) == plan, "Source dataset changed"
        if validate_only:
            return audit(raw, output, plan, repo_id)
        committed = repair_tail(output, plan)
        dataset = open_append(repo_id, output)
    else:
        assert not validate_only
        dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=FPS, robot_type="panda_shared_joint_servo",
                                        features=FEATURES, use_videos=False, image_writer_threads=4)
        dump(output / "meta/reconstruction_sources.json", plan)
        committed = 0
    try:
        for source in plan["sources"][committed:]:
            state, actions, images, wrists = read_episode(raw, source)
            dataset.hf_dataset = dataset.create_hf_dataset()
            for i in range(source["frames"]):
                dataset.add_frame(dict(image=images[i], wrist_image=wrists[i], state=state[i],
                                       actions=actions[i], task=source["language"]))
            dataset.save_episode()
            dump(output / "meta/conversion_progress.json", dict(complete=False, completed=dataset.meta.total_episodes,
                                                              total=len(plan["sources"]), frames=dataset.meta.total_frames))
            print(json.dumps(dict(converted=dataset.meta.total_episodes, total=len(plan["sources"]),
                                  case=source["case"], demo=source["demo"])), flush=True)
    finally:
        dataset.stop_image_writer()
    result = audit(raw, output, plan, repo_id)
    dump(output / "meta/conversion_progress.json", dict(complete=True, completed=result["episodes"],
                                                       total=result["episodes"], frames=result["frames"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="caozilin/libero_reconstructed_full_20261101")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    result = convert(args.raw_dir, args.output, args.repo_id, args.resume, args.validate_only, args.max_episodes)
    print(json.dumps({k: v for k, v in result.items() if k != "episode_checks"}), flush=True)
