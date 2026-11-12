"""Reproduce OpenVLA's LIBERO replay preprocessing, with resumable task workers.

Replay semantics follow openvla/openvla at 0ebc5e333ee7a916813bf50db240f1af7ba1d295:
experiments/robot/libero/regenerate_libero_dataset.py. See upstream/OPENVLA_LICENSE.
This entry point runs in examples/libero/.venv (Python 3.8), without OpenVLA models.
Images retain LIBERO's raw orientation; rotate once when exporting training data.
"""

from __future__ import annotations

# Python 3.8 needs datetime.timezone.utc rather than datetime.UTC.
# ruff: noqa: PLR0912, PLR0913, PLR0915, UP017
import argparse
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
from pathlib import Path
import queue
import random
import re
import signal
import sys
import time

UPSTREAM_COMMIT = "0ebc5e333ee7a916813bf50db240f1af7ba1d295"
UPSTREAM_URL = (
    f"https://github.com/openvla/openvla/blob/{UPSTREAM_COMMIT}/experiments/robot/libero/regenerate_libero_dataset.py"
)
SCHEMA_VERSION = 1
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOLUTION = 256
SETTLE_STEPS = 10
NOOP_THRESHOLD = 1e-4
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
_STOP = None
_PROGRESS = None


def _report_progress(job, processed, successful):
    if _PROGRESS is not None:
        _PROGRESS.put((job["output"], processed, successful))


class ReplayInterruptedError(Exception):
    """A stop request arrived before the current episode was committed."""


class ReplayActionRejected(Exception):
    """An optional motion generator rejected a command before physics execution."""


def _numpy():
    import numpy as np

    return np


def _h5py():
    import h5py

    return h5py


def is_noop(action, prev_action=None, threshold=NOOP_THRESHOLD):
    """Match the upstream predicate, including its first-action special case."""
    np = _numpy()
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def _stopping(stop):
    return stop is not None and stop.is_set()


def replay_demo(
    env, demo, *, stop=None, quat_to_axis_angle=None,
    on_settled=None, action_executor=None, retain_failed=False,
    skip_noops=True, settle_steps=SETTLE_STEPS, max_source_steps=None,
    replay_actions=None,
):
    """Return upstream arrays plus actual pre-action states and source indices."""
    np = _numpy()
    if quat_to_axis_angle is None:
        from robosuite.utils.transform_utils import quat2axisangle

        quat_to_axis_angle = quat2axisangle
    original_actions = demo["actions"][()] if replay_actions is None else np.asarray(replay_actions)
    initial_state = demo["states"][0]
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    for _ in range(settle_steps):
        if _stopping(stop):
            raise ReplayInterruptedError
        obs, _, done, _ = env.step(DUMMY_ACTION)
    settled_state = env.sim.get_state().flatten().copy()
    if on_settled is not None:
        on_settled(env, obs, settled_state)
    values = {
        key: []
        for key in (
            "actions",
            "states",
            "robot_states",
            "joint_states",
            "ee_states",
            "gripper_states",
            "agentview_rgb",
            "eye_in_hand_rgb",
            "pre_action_states",
            "pre_action_robot_states",
            "source_step_indices",
        )
    }
    noops = 0
    rejected = None
    rejected_index = None
    for index, action in enumerate(original_actions):
        if max_source_steps is not None and index >= max_source_steps:
            break
        if _stopping(stop):
            raise ReplayInterruptedError
        previous = values["actions"][-1] if values["actions"] else None
        if skip_noops and is_noop(action, previous):
            noops += 1
            continue  # Upstream skips simulation as well as storage for these actions.
        actual_state = env.sim.get_state().flatten().copy()
        actual_robot_state = np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
        # Preserve the original script's intentionally different legacy first frame.
        values["states"].append(initial_state.copy() if not values["actions"] else actual_state)
        values["robot_states"].append(demo["robot_states"][0] if not values["actions"] else actual_robot_state)
        values["pre_action_states"].append(actual_state)
        values["pre_action_robot_states"].append(actual_robot_state)
        values["source_step_indices"].append(index)
        values["actions"].append(action)
        values["gripper_states"].append(obs["robot0_gripper_qpos"].copy())
        values["joint_states"].append(obs["robot0_joint_pos"].copy())
        values["ee_states"].append(
            np.hstack([obs["robot0_eef_pos"], quat_to_axis_angle(obs["robot0_eef_quat"].copy())])
        )
        values["agentview_rgb"].append(obs["agentview_image"].copy())
        values["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"].copy())
        try:
            if action_executor is None:
                obs, _, done, _ = env.step(action.tolist())
            else:
                obs, _, done, _ = action_executor(env, index, action.copy(), obs)
        except ReplayActionRejected as error:
            rejected = str(error)
            rejected_index = index
            break
    count = len(values["actions"])
    success = rejected is None and bool(done) and count > 0
    result = {
        "success": success,
        "source_steps": len(original_actions),
        "retained_steps": count,
        "discarded_noops": noops,
        "initial_state": initial_state,
        "settled_state": settled_state,
        "final_state": env.sim.get_state().flatten().copy(),
        "arrays": {},
        "execution_rejected": rejected is not None,
        "execution_rejection": rejected,
        "rejected_source_index": rejected_index,
    }
    if success or retain_failed:
        result["arrays"] = {key: np.stack(value) for key, value in values.items()}
        result["arrays"]["source_step_indices"] = np.asarray(values["source_step_indices"], dtype=np.int64)
    if not success:
        result["source_step_indices"] = np.asarray(values["source_step_indices"], dtype=np.int64)
    return result


def _atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _flush_file(file):
    file.flush()
    descriptor = file.id.get_vfd_handle()
    if isinstance(descriptor, int):
        os.fsync(descriptor)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rng_state():
    state = _numpy().random.get_state()
    return {
        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        "python": random.getstate(),
    }


def _restore_rng(value):
    np = _numpy()
    state = value["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    state = value["python"]
    random.setstate((state[0], tuple(state[1]), state[2]))


def _demo_ids(data, limit):
    names = list(data)
    if not names:
        raise ValueError("Source data contains no demonstrations")
    if any(re.fullmatch(r"demo_\d+", name) is None for name in names):
        raise ValueError("Expected only demo_N groups in the source data group")
    names.sort(key=lambda name: int(name[5:]))
    return names if limit is None else names[:limit]


def _versions():
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "h5py", "robosuite", "mujoco"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _fingerprint(job, source, ids):
    """Hash replay inputs, excluding unused source image arrays."""
    digest = hashlib.sha256()
    for name in ids:
        demo = source["data"][name]
        digest.update(name.encode())
        for key in ("actions", "states"):
            array = demo[key][()]
            digest.update(str((key, array.shape, str(array.dtype))).encode())
            digest.update(array.tobytes())
        digest.update(demo["robot_states"][0].tobytes())
        digest.update(str(demo.attrs.get("model_file", "")).encode())
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": str(Path(job["source"]).resolve()),
        "source_replay_sha256": digest.hexdigest(),
        "task": job["name"],
        "suite": job["suite"],
        "demo_ids": ids,
        "bddl_sha256": hashlib.sha256(Path(job["bddl"]).read_bytes()).hexdigest(),
        "seed": 0,
        "settle_steps": SETTLE_STEPS,
        "noop_threshold": NOOP_THRESHOLD,
        "resolution": job["resolution"],
        "compression": job["compression"],
        "gzip_level": job["gzip_level"],
        "render_backend": job["render_backend"],
        "blas_threads": job.get("blas_threads", 1),
        "libero_environment_sha256": job.get("libero_environment_sha256"),
        "versions": _versions(),
    }
    encoded = json.dumps(configuration, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


def _dataset(group, name, array, job, *, image=False):
    kwargs = {}
    if job["compression"] != "none":
        kwargs.update(compression=job["compression"], shuffle=True)
        if job["compression"] == "gzip":
            kwargs["compression_opts"] = job["gzip_level"]
        if image:
            kwargs["chunks"] = (1,) + array.shape[1:]
    group.create_dataset(name, data=array, **kwargs)


def _write_demo(group, original, result, job):
    np = _numpy()
    for key, value in original.attrs.items():
        group.attrs[key] = value
    arrays = result["arrays"]
    count = result["retained_steps"]
    group.attrs["num_samples"] = count
    obs = group.create_group("obs")
    for key in ("gripper_states", "joint_states", "ee_states", "agentview_rgb", "eye_in_hand_rgb"):
        _dataset(obs, key, arrays[key], job, image=key.endswith("_rgb"))
    _dataset(obs, "ee_pos", arrays["ee_states"][:, :3], job)
    _dataset(obs, "ee_ori", arrays["ee_states"][:, 3:], job)
    for key in ("actions", "states", "robot_states"):
        _dataset(group, key, arrays[key], job)
    terminal = np.zeros(count, dtype=np.uint8)
    terminal[-1] = 1
    _dataset(group, "rewards", terminal, job)
    _dataset(group, "dones", terminal, job)
    trace = group.create_group("tasktol")
    trace.attrs["state_alignment"] = "pre_action; legacy states[0] is the original reset state"
    for key in ("source_step_indices", "pre_action_states", "pre_action_robot_states"):
        _dataset(trace, key, arrays[key], job)
    for key in ("initial_state", "settled_state", "final_state"):
        _dataset(trace, key, result[key], job)


def _clean_pending(file):
    data = file["data"]
    episodes = file["regeneration/episodes"]
    for group in (data, episodes):
        for name in list(group):
            if name.startswith("__pending_"):
                del group[name]
    for name in list(data):
        if name not in episodes:
            del data[name]  # An interrupted commit never becomes a completed episode.


def _validate_checkpoint(file, ids):
    ledger = file["regeneration/episodes"]
    completed = [name for name in ids if name in ledger]
    if completed != ids[: len(completed)] or set(ledger) != set(completed):
        raise ValueError("Checkpoint does not contain a contiguous source-demo prefix")
    expected = set()
    for name in completed:
        entry = json.loads(ledger[name].attrs["metadata"])
        if not entry["success"]:
            continue
        expected.add(name)
        if name not in file["data"]:
            raise ValueError(f"Checkpoint is missing successful episode {name}")
        group = file["data"][name]
        for key in ("actions", "states", "robot_states", "rewards", "dones"):
            if group[key].shape[0] != entry["retained_steps"]:
                raise ValueError(f"Incomplete {name}/{key} in checkpoint")
        for value in group["obs"].values():
            if value.shape[0] != entry["retained_steps"]:
                raise ValueError(f"Incomplete observations in {name}")
        indices = group["tasktol/source_step_indices"][()].tolist()
        if indices != entry["source_step_indices"]:
            raise ValueError(f"Incorrect source-step mapping in {name}")
    if set(file["data"]) != expected:
        raise ValueError("Checkpoint data and success ledger disagree")
    return completed


def _metadata(file, job, *, complete):
    entries = {}
    for name, episode in file["regeneration/episodes"].items():
        if not name.startswith("__pending_"):
            entries[name] = json.loads(episode.attrs["metadata"])
    successes = sum(x["success"] for x in entries.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "upstream_url": UPSTREAM_URL,
        "suite": job["suite"],
        "task_id": job["task_id"],
        "task_name": job["name"],
        "language": job["language"],
        "source_file": job["source"],
        "fingerprint": file.attrs["fingerprint"],
        "complete": complete,
        "processed_demos": len(entries),
        "successful_demos": successes,
        "failed_demos": len(entries) - successes,
        "discarded_noops": sum(x["discarded_noops"] for x in entries.values()),
        "retained_steps": sum(x["retained_steps"] for x in entries.values() if x["success"]),
        "image_orientation": "raw_libero; rotate 180 degrees once at training export",
        "episodes": entries,
    }


def _create_env(job):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=job["bddl"], camera_heights=job["resolution"], camera_widths=job["resolution"]
    )
    env.seed(0)
    return env


def process_task(job, *, env_factory=None, stop=None, quat_to_axis_angle=None):
    """One task owns one HDF5 and one metadata file; workers never share writers."""
    h5py = _h5py()
    stop = _STOP if stop is None else stop
    final = Path(job["output"])
    partial = final.with_suffix(".hdf5.inprogress")
    sidecar = final.with_suffix(".metainfo.json")
    final.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with final.with_suffix(".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another worker owns {final}") from error
        with h5py.File(job["source"], "r") as source:
            ids = _demo_ids(source["data"], job["max_demos"])
            fingerprint, configuration = _fingerprint(job, source, ids)
            if final.exists():
                if not job["resume"]:
                    raise FileExistsError(f"Output exists; use --resume: {final}")
                with h5py.File(final, "r") as saved:
                    if saved.attrs.get("fingerprint") != fingerprint or not saved.attrs.get("complete", False):
                        raise ValueError(f"Completed output does not match these inputs/settings: {final}")
                    if _validate_checkpoint(saved, ids) != ids:
                        raise ValueError(f"Completed output has missing source episodes: {final}")
                    metadata = _metadata(saved, job, complete=True)
                _atomic_json(sidecar, metadata)
                _report_progress(job, metadata["processed_demos"], metadata["successful_demos"])
                return {**metadata, "status": "skipped", "elapsed_seconds": time.monotonic() - started}
            if partial.exists() and not job["resume"]:
                raise FileExistsError(f"Interrupted output exists; use --resume: {partial}")
            existing = partial.exists()
            env = None
            try:
                with h5py.File(partial, "a") as output:
                    if existing:
                        if output.attrs.get("fingerprint") != fingerprint:
                            raise ValueError(f"Partial output does not match these inputs/settings: {partial}")
                        _clean_pending(output)
                    else:
                        output.attrs.update(fingerprint=fingerprint, configuration=configuration, complete=False)
                        output.create_group("data")
                        output.create_group("regeneration").create_group("episodes")
                        original_metadata = output.create_group("source_metadata")
                        for key, value in source["data"].attrs.items():
                            original_metadata.attrs[key] = value
                    data = output["data"]
                    ledger = output["regeneration/episodes"]
                    # A crash may occur after committing the ledger but before updating counters.
                    data.attrs["num_demos"] = len(data)
                    data.attrs["total"] = sum(group["actions"].shape[0] for group in data.values())
                    completed = _validate_checkpoint(output, ids)
                    _report_progress(job, len(completed), len(data))
                    if len(completed) < len(ids) and not _stopping(stop):
                        env = (env_factory or _create_env)(job)
                        if completed:
                            _restore_rng(json.loads(ledger[completed[-1]].attrs["rng_state"]))
                    for name in ids[len(completed) :]:
                        if _stopping(stop):
                            break
                        try:
                            result = replay_demo(
                                env, source["data"][name], stop=stop, quat_to_axis_angle=quat_to_axis_angle
                            )
                        except ReplayInterruptedError:
                            break
                        pending = "__pending_" + name
                        episode = ledger.create_group(pending)
                        indices = (
                            result["arrays"]["source_step_indices"]
                            if result["success"]
                            else result["source_step_indices"]
                        )
                        entry = {
                            key: result[key]
                            for key in (
                                "success",
                                "source_steps",
                                "retained_steps",
                                "discarded_noops",
                            )
                        }
                        entry.update(
                            demo_id=name,
                            initial_state=result["initial_state"].tolist(),
                            source_step_indices=indices.tolist(),
                        )
                        episode.attrs["metadata"] = json.dumps(entry)
                        episode.attrs["rng_state"] = json.dumps(_rng_state())
                        if result["success"]:
                            _write_demo(data.create_group(pending), source["data"][name], result, job)
                            data.move(pending, name)
                        ledger.move(pending, name)
                        data.attrs["num_demos"] = len(data)
                        data.attrs["total"] = sum(x["actions"].shape[0] for x in data.values())
                        _flush_file(output)
                        metadata = _metadata(output, job, complete=False)
                        _atomic_json(sidecar, metadata)
                        _report_progress(job, metadata["processed_demos"], metadata["successful_demos"])
                        logging.debug(
                            "%s/%s %s: success=%s, frames=%s, noops=%s",
                            job["suite"],
                            job["name"],
                            name,
                            result["success"],
                            result["retained_steps"],
                            result["discarded_noops"],
                        )
                    complete = len(ledger) == len(ids)
                    output.attrs["complete"] = complete
                    metadata = _metadata(output, job, complete=complete)
                    _flush_file(output)
                if complete:
                    os.replace(partial, final)
                    _sync_directory(final.parent)
                _atomic_json(sidecar, metadata)
                return {
                    **metadata,
                    "status": "complete" if complete else "interrupted",
                    "elapsed_seconds": time.monotonic() - started,
                }
            finally:
                if env is not None:
                    env.close()


def _worker_init(stop, backend, threads, progress):
    global _STOP, _PROGRESS  # noqa: PLW0603
    _STOP = stop
    _PROGRESS = progress
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["MUJOCO_GL"] = backend
    _bootstrap_libero()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")


def _bootstrap_libero():
    sys.path.insert(0, str(PROJECT_ROOT / "third_party/libero"))


def _libero_environment_fingerprint():
    """Capture local code edits and the asset inventory once in the parent."""
    root = PROJECT_ROOT / "third_party/libero/libero/libero"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for path in sorted((root / "assets").rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(json.dumps([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def build_jobs(args):
    _bootstrap_libero()
    from libero.libero import benchmark
    from libero.libero import get_libero_path

    raw_root = args.raw_data_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if raw_root == output_root:
        raise ValueError("Use a separate output directory; raw HDF5 must remain unchanged")
    jobs = []
    environment_fingerprint = _libero_environment_fingerprint()
    for suite_name in args.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
        for task_id in ids:
            if task_id < 0 or task_id >= suite.n_tasks:
                raise ValueError(f"Task index {task_id} is outside {suite_name}")
            task = suite.get_task(task_id)
            source = raw_root / suite_name / f"{task.name}_demo.hdf5"
            if not source.is_file():
                raise FileNotFoundError(source)
            jobs.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "name": task.name,
                    "language": task.language,
                    "bddl": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                    "source": str(source),
                    "output": str(output_root / f"{suite_name}_no_noops" / source.name),
                    "max_demos": args.max_demos_per_task,
                    "resume": args.resume,
                    "resolution": RESOLUTION,
                    "compression": args.compression,
                    "gzip_level": args.gzip_level,
                    "render_backend": args.render_backend,
                    "blas_threads": args.blas_threads,
                    "libero_environment_sha256": environment_fingerprint,
                }
            )
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=PROJECT_ROOT / "third_party/libero/libero/datasets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suites", choices=SUITES, nargs="+", default=list(SUITES))
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--max-demos-per-task", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--gzip-level", type=int, choices=range(10), default=1)
    parser.add_argument("--render-backend", choices=("egl", "osmesa", "glfw"), default="egl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the terminal progress bars")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if len(set(args.suites)) != len(args.suites) or (
        args.task_ids is not None and len(set(args.task_ids)) != len(args.task_ids)
    ):
        parser.error("Do not repeat suite names or task indices")
    if (
        args.workers < 1
        or args.blas_threads < 1
        or (args.max_demos_per_task is not None and args.max_demos_per_task < 1)
    ):
        parser.error("workers, blas-threads, and max-demos-per-task must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.blas_threads)
    os.environ["MUJOCO_GL"] = args.render_backend
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    from tqdm import tqdm

    logging.info("Counting selected demonstrations before replay...")
    total_demos = 0
    for job in jobs:
        with _h5py().File(job["source"], "r") as source:
            total_demos += len(_demo_ids(source["data"], job["max_demos"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    progress = context.Queue()
    previous_handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(*_):
        stop.set()
        logging.warning("Stopping replay; completed episodes will be retained for --resume")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    results = []
    started = time.monotonic()
    counts = {}

    def update_progress(message, bar):
        key, processed, successful = message
        old_processed = counts.get(key, (0, 0))[0]
        if processed >= old_processed:
            counts[key] = (processed, successful)
            bar.update(processed - old_processed)
            bar.set_postfix(saved=sum(value[1] for value in counts.values()), refresh=False)

    try:
        with ExitStack() as stack:
            demo_bar = stack.enter_context(
                tqdm(total=total_demos, desc="Demonstrations", unit="demo", disable=args.no_progress)
            )
            task_bar = stack.enter_context(tqdm(total=len(jobs), desc="Tasks", unit="task", disable=args.no_progress))
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=context,
                    initializer=_worker_init,
                    initargs=(stop, args.render_backend, args.blas_threads, progress),
                )
            )
            futures = {executor.submit(process_task, job): job for job in jobs}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        update_progress(progress.get_nowait(), demo_bar)
                    except queue.Empty:
                        break
                for future in finished:
                    job = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:
                        tqdm.write(f"Failed task {job['suite']}/{job['name']}: {error!r}", file=sys.stderr)
                        logging.debug("Worker exception", exc_info=True)
                        result = {
                            "suite": job["suite"],
                            "task_name": job["name"],
                            "status": "failed",
                            "error": repr(error),
                        }
                    if "processed_demos" in result:
                        update_progress(
                            (job["output"], result["processed_demos"], result["successful_demos"]), demo_bar
                        )
                    results.append(result)
                    task_bar.update(1)
                    summary = {
                        "upstream_url": UPSTREAM_URL,
                        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "elapsed_seconds": time.monotonic() - started,
                        "selected_tasks": len(jobs),
                        "finished_tasks": len(results),
                        "workers": args.workers,
                        "results": sorted(results, key=lambda x: (x["suite"], x["task_name"])),
                    }
                    _atomic_json(args.output_dir / "summary.json", summary)
    finally:
        progress.close()
        progress.join_thread()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    logging.info("Finished %s/%s tasks; summary: %s", len(results), len(jobs), args.output_dir / "summary.json")
    if stop.is_set():
        return 130
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
