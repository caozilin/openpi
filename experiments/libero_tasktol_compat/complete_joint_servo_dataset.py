"""Expand the audited success10 run by attempting every remaining source demo.

Uses reconstruct_joint_servo_dataset.attempt unchanged. Existing successes and
failed attempts are reused with provenance; only new successful recordings use
lossless gzip instead of LZF. An exhausted source is different from a successful
episode. Final task HDF5 indexes contain successes only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/libero_tasktol_joint_servo/20261025_success10_v1"
DEFAULT = ROOT / "data/libero_tasktol_joint_servo/20261101_full_v1"
os.environ.setdefault("TASKTOL_FROZEN_SHARED_ROOT", str(BASE / "frozen_shared_runtime"))
sys.path[:0] = [os.environ["TASKTOL_FROZEN_SHARED_ROOT"], str(ROOT)]
import tasktol

assert Path(tasktol.__file__).resolve().is_relative_to(Path(os.environ["TASKTOL_FROZEN_SHARED_ROOT"]))
from experiments.libero_tasktol_compat import reconstruct_joint_servo_dataset as dataset
from experiments.libero_tasktol_compat import finalize_joint_servo_dataset as validation

runner = dataset.runner
base_freeze = dataset.freeze


def freeze():
    return dict(base_freeze(), **{str(Path(__file__)): runner.sha(Path(__file__)),
                                str(Path(validation.__file__)): runner.sha(Path(validation.__file__))})


runner.freeze_sources = freeze


class LosslessObservationWriter(dataset.ObservationWriter):
    def append(self, group, name, value):
        import numpy as np

        a = np.asarray(value)
        if name not in group:
            group.create_dataset(name, shape=(0,) + a.shape, maxshape=(None,) + a.shape,
                                 chunks=(1,) + a.shape, dtype=a.dtype,
                                 compression="gzip", compression_opts=6)
        d = group[name]
        d.resize(self.count + 1, axis=0)
        d[self.count] = a


dataset.ObservationWriter = LosslessObservationWriter


def record_path(out, case, demo):
    return out / "records" / f"case_{case:02d}" / (demo + ".json")


def load_records(out, entry):
    records = {}
    for job in entry["candidates"]:
        path = record_path(out, entry["case"], job["demo"])
        if path.exists():
            row = json.loads(path.read_text())
            assert row["demo"] == job["demo"] and row["case"] == entry["case"]
            if row.get("source_input_sha256"):
                assert row["source_input_sha256"] == job["source_input_sha256"]
            records[job["demo"]] = row
    return records


def collect_summary(out, plan):
    tasks = []
    for entry in plan["tasks"]:
        records = list(load_records(out, entry).values())
        completed = [r for r in records if "error" not in r]
        success = [r for r in completed if r.get("success")]
        tasks.append(dict(case=entry["case"], **{k: entry["task"][k] for k in ("suite", "task_id", "task_name")},
                          source_demos=len(entry["candidates"]), attempted=len(completed),
                          success=len(success), failed=len(completed) - len(success),
                          process_errors=sum("error" in r for r in records),
                          pending=len(entry["candidates"]) - len(completed),
                          frames=sum(r["report"]["executed_frames"] for r in success),
                          successful_demos=[r["demo"] for r in success]))
    result = dict(complete=all(t["pending"] == 0 for t in tasks),
                  total_sources=sum(t["source_demos"] for t in tasks),
                  total_attempted=sum(t["attempted"] for t in tasks),
                  total_success=sum(t["success"] for t in tasks),
                  total_failed=sum(t["failed"] for t in tasks),
                  total_pending=sum(t["pending"] for t in tasks),
                  process_errors=sum(t["process_errors"] for t in tasks),
                  tasks_complete=sum(t["pending"] == 0 for t in tasks),
                  inherited_sources=plan["inherited_sources"], inherited_successes=plan["inherited_successes"],
                  new_successes=sum(t["success"] for t in tasks) - plan["inherited_successes"],
                  updated_at=datetime.now().astimezone().isoformat(), tasks=tasks)
    result["suites"] = {suite: {k: sum(t[k] for t in tasks if t["suite"] == suite)
                                 for k in ("source_demos", "attempted", "success", "failed", "pending", "frames")}
                        for suite in sorted({t["suite"] for t in tasks})}
    runner.dump(out / "summary.json", result)
    return result


def prepare(out, base):
    import h5py

    baseline = json.loads((base / "plan.json").read_text())
    assert json.loads((base / "validation.json").read_text())["complete"]
    # Every source in each HDF5 must occur exactly once in the inherited plan.
    inherited = []
    for entry in baseline["tasks"]:
        jobs = entry["candidates"]
        assert len({j["demo"] for j in jobs}) == len(jobs)
        with h5py.File(jobs[0]["source_hdf5"]) as f:
            assert set(f["data"]) == {j["demo"] for j in jobs}
        for job in jobs:
            path = record_path(base, entry["case"], job["demo"])
            if path.exists():
                row = json.loads(path.read_text())
                assert "error" not in row
                assert row["source_input_sha256"] == job["source_input_sha256"]
                if row["success"]:
                    assert row["independent_replay_bitwise_identical"]
                    assert (Path(row["folder"]) / "recording.hdf5").is_file()
                inherited.append((entry["case"], row))
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(baseline, source_sha256=freeze(), baseline_run=str(base),
                baseline_plan_sha256=runner.sha(base / "plan.json"),
                inherited_sources=len(inherited), inherited_successes=sum(r["success"] for _, r in inherited),
                schema="libero_all_source_reconstruction.1")
    plan["policy"] = dict(baseline["policy"], target_successes_per_task=None,
                          completion="all unique source demos attempted, irrespective of task success",
                          observation_compression="lossless gzip level 6 for new episodes; original LZF retained",
                          reuse="all prior completed source attempts, including failures; no repeat of a source")
    runner.dump(out / "plan.json", plan)
    for case, row in inherited:
        runner.dump(record_path(out, case, row["demo"]), dict(row, inherited_from=str(base)))
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for i, path in enumerate(plan["source_sha256"]):
        source = Path(path)
        if not source.exists() or runner.sha(source) != plan["source_sha256"][path]:
            source = Path(os.environ["TASKTOL_FROZEN_SHARED_ROOT"]) / source.relative_to(ROOT.parent / "franka_mujoco")
        assert runner.sha(source) == plan["source_sha256"][path]
        shutil.copy2(source, snapshot / f"{i:03d}_{source.name}")
    collect_summary(out, plan)
    return plan


def worker(out, case, index):
    plan = json.loads((out / "plan.json").read_text())
    assert freeze() == plan["source_sha256"], "Frozen reconstruction inputs changed"
    job = plan["tasks"][case]["candidates"][index]
    path = record_path(out, case, job["demo"])
    if path.exists() and "error" not in json.loads(path.read_text()):
        return
    destination = out / "episodes" / job["suite"] / f"{job['task_id']:02d}" / job["demo"]
    # Preserve files from a host interruption before retrying the same source.
    if destination.exists():
        quarantine = out / "interrupted" / f"{time.time_ns()}_case_{case:02d}_{job['demo']}"
        quarantine.parent.mkdir(exist_ok=True)
        shutil.move(str(destination), quarantine)
    original_dump = runner.dump

    def durable_commit(target, value):
        # Scratch may be on tmpfs. shutil.move then copies across filesystems;
        # synchronize the published copies before committing their success row.
        if Path(target) == path and value.get("success"):
            folder = Path(value["folder"])
            for artifact in folder.iterdir():
                if artifact.is_file():
                    with artifact.open("rb") as stream:
                        os.fsync(stream.fileno())
            directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        original_dump(target, value)

    runner.dump = durable_commit
    dataset.attempt(out, case, index)


def finalize(out, plan, allow_partial=False):
    import h5py

    summary = collect_summary(out, plan)
    assert allow_partial or summary["complete"], "Unattempted sources remain"
    assert freeze() == plan["source_sha256"]
    indexes = []
    for entry in plan["tasks"]:
        successes = [r for r in load_records(out, entry).values() if r.get("success") and "error" not in r]
        task = entry["task"]
        index = out / "datasets" / task["suite"] / (task["task_name"] + "_demo.hdf5")
        index.parent.mkdir(parents=True, exist_ok=True)
        temp = index.with_suffix(".hdf5.partial")
        frames = 0
        with h5py.File(temp, "w") as f:
            group = f.create_group("data")
            for record in successes:
                assert record["independent_replay_bitwise_identical"]
                audit = validation.verify_episode(out, record, json.loads(dataset.SELECTION.read_text())["tasks"][entry["case"]])
                frames += audit["frames"]
                target = Path(record["folder"]) / "recording.hdf5"
                group[record["demo"]] = h5py.ExternalLink(os.path.relpath(target, index.parent), "/data/" + record["demo"])
            group.attrs["total"] = frames
            f.attrs.update(language=task["language"], source_plan=str(out / "plan.json"),
                           successful_episodes=len(successes), control_hz=20, physics_hz=500,
                           image_orientation="native LIBERO; rotate once at training export",
                           action_units="metres and world rotvec radians plus original gripper")
        temp.replace(index)
        indexes.append(dict(case=entry["case"], suite=task["suite"], task_id=task["task_id"],
                            path=str(index), episodes=len(successes), frames=frames))
        print(json.dumps(dict(verified_task=entry["case"], episodes=len(successes))), flush=True)
    manifest = dict(complete=summary["complete"], episodes=summary["total_success"], tasks=indexes,
                    baseline_run=plan["baseline_run"], baseline_required=True,
                    policy=plan["policy"], source_unchanged=freeze() == plan["source_sha256"],
                    schema="native_LIBERO_HDF5_pre_action_dual_camera_physical_reference_actions")
    runner.dump(out / "manifest.json", manifest)
    runner.dump(out / "validation.json", dict(manifest, independently_replayed=True,
                full_image_decode=True, servo_bitwise_verified=True))
    lines = ["# LIBERO 全量来源重构数据集", "",
             f"已完成：{summary['total_attempted']} / {summary['total_sources']} 个来源，成功保存 {summary['total_success']} 条，未成功 {summary['total_failed']} 条。",
             f"复用原有 {plan['inherited_successes']} 条成功数据，新增 {summary['new_successes']} 条成功数据。", "",
             "|任务集|来源总数|已处理|成功|未成功|执行帧|", "|---|---:|---:|---:|---:|---:|"]
    for suite, counts in summary["suites"].items():
        lines.append(f"|{suite}|{counts['source_demos']}|{counts['attempted']}|{counts['success']}|{counts['failed']}|{counts['frames']}|")
    lines += ["", "datasets/ 中每任务一个 HDF5 索引，仅引用完整执行且原生任务判定成功的数据。",
              "新增录制采用无损 gzip，保留原始 uint8 双相机；原有 400 条保持原址，通过相对 HDF5 外部链接复用。",
              f"**搬迁时必须同时保留此目录和 {plan['baseline_run']}。**",
              "全部成功动作均经独立回放逐位核对物理状态和伺服样本，最终校验完整读取双相机图像。",
              "逐任务统计见 summary.json；失败诊断见 records/。来源恢复率是离线重构收集统计，不是策略成功率。", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("complete", "total_attempted", "total_success", "total_failed", "total_pending")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", type=int)
    parser.add_argument("--index", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        worker(out, args.case, args.index)
        return
    plan = json.loads((out / "plan.json").read_text()) if args.resume else prepare(out, args.base.resolve())
    assert freeze() == plan["source_sha256"], "Sources changed since preparation"
    lock = (out / "execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.prepare_only:
        return
    if args.finalize_only:
        finalize(out, plan, args.allow_partial)
        return
    jobs = [(entry["case"], index, job) for entry in plan["tasks"]
            for index, job in enumerate(entry["candidates"])
            if not record_path(out, entry["case"], job["demo"]).exists()
            or "error" in json.loads(record_path(out, entry["case"], job["demo"]).read_text())]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_CACHE_DIR="/tmp/libero_full_numba",
               MPLCONFIGDIR="/tmp/libero_full_mpl")

    def run(item):
        case, index, job = item
        if shutil.disk_usage(out).free < args.min_free_gb * 1e9:
            raise RuntimeError("Disk reserve reached; completed sources are safe. Free space and --resume.")
        logfile = out / "logs" / f"case_{case:02d}_{job['demo']}.log"
        logfile.parent.mkdir(exist_ok=True)
        with logfile.open("w") as stream:
            result = subprocess.run([sys.executable, "-u", str(Path(__file__)), "--worker", "--output", str(out),
                                     "--case", str(case), "--index", str(index)], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Worker exited {result.returncode}: {logfile}")
        row = json.loads(record_path(out, case, job["demo"]).read_text())
        print(json.dumps(dict(case=case, demo=job["demo"], success=row["success"], wall_s=row["wall_s"])), flush=True)

    # Bound submissions so a disk stop never launches the rest of the dataset.
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for _ in range(args.workers):
            item = next(iterator, None)
            if item is not None:
                pending[pool.submit(run, item)] = item
        try:
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    pending.pop(future)
                    item = next(iterator, None)
                    if item is not None:
                        pending[pool.submit(run, item)] = item
                summary = collect_summary(out, plan)
                print(json.dumps({k: summary[k] for k in ("total_attempted", "total_success", "total_failed", "total_pending")}), flush=True)
        finally:
            collect_summary(out, plan)
    if args.limit is None:
        finalize(out, plan)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
