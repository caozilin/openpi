"""Precompute the unchanged strict episode verifier's cache in parallel.

The running finalizer still owns final publication and all completion gates.
Each worker uses a private cache and atomically installs only a fully verified
result, avoiding any collision with the serial finalizer's .partial files.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_tasktol_compat import complete_joint_servo_dataset as full


def verify(job):
    run, record, task = job
    run = Path(run)
    private = run / ".parallel_validation" / f"{os.getpid()}_{record['case']:02d}_{record['demo']}"
    audit = full.validation.verify_episode(private, record, task)
    assert audit["full_decode"] and audit["servo_bitwise_verified"]
    assert audit["frames"] == record["report"]["executed_frames"]
    relative = Path("episode_validation") / f"case_{record['case']:02d}" / (record["demo"] + ".json")
    source, target = private / relative, run / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.verified")
    with source.open("rb") as source_file, temporary.open("wb") as output:
        shutil.copyfileobj(source_file, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    shutil.rmtree(private)
    return dict(case=record["case"], demo=record["demo"], frames=audit["frames"])


def main():
    run = full.DEFAULT
    summary = json.loads((run / "summary.json").read_text())
    assert summary["complete"] and summary["total_attempted"] == 2000 and summary["process_errors"] == 0
    selection = json.loads(full.dataset.SELECTION.read_text())["tasks"]
    plan = json.loads((run / "plan.json").read_text())
    assert full.freeze() == plan["source_sha256"]
    jobs = []
    for entry in reversed(plan["tasks"]):
        for record in full.load_records(run, entry).values():
            cache = run / "episode_validation" / f"case_{entry['case']:02d}" / (record["demo"] + ".json")
            if record.get("success") and not cache.exists():
                jobs.append((str(run), record, selection[entry["case"]]))
    state = dict(pid=os.getpid(), complete=False, total=len(jobs), verified=0, frames=0)
    full.runner.dump(run / "parallel_validation_status.json", state)
    with ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(verify, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            state.update(verified=state["verified"] + 1, frames=state["frames"] + result["frames"], latest=result)
            full.runner.dump(run / "parallel_validation_status.json", state)
            print(json.dumps(result), flush=True)
    state["complete"] = True
    full.runner.dump(run / "parallel_validation_status.json", state)


if __name__ == "__main__":
    main()
