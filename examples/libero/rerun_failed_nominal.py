"""Reevaluate only committed failures, preserving the original LIBERO seeds."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "data/libero_evaluation/pi05_reconstructed_10000_nominal_failed_retry_hold20_20261104"
PYTHON = ROOT.parent / "franka_mujoco/experiments/libero_closed_loop/.venv/bin/python"
FROZEN = ROOT / "data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime"


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def failed_selection(baseline):
    rows = [read(p) for p in sorted((baseline / "episodes").glob("*/task_*/seed_*/result.json"))]
    keys = {(r["suite"], r["task_id"], r["seed"]) for r in rows}
    expected = {(s, t, z) for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for t in range(10) for z in range(6)}
    if len(rows) != 240 or keys != expected or not all(r["complete"] for r in rows):
        raise ValueError("The baseline must contain exactly 240 complete unique episodes")
    return [{k:r[k] for k in ("suite", "task_id", "seed", "termination", "steps")} for r in rows if not r["success"]]


def validate_request(request):
    baseline = Path(request["baseline_run"])
    selection = failed_selection(baseline)
    if selection != request["selected_episodes"] or len(selection) != request["expected_episodes"]:
        raise ValueError("Selected episodes must be exactly the original failed episodes")
    if request["tolerance_enabled"] or request["max_consecutive_solver_failures"] != 20:
        raise ValueError("This retry requires tolerance off and a 20-frame failure limit")
    for name, key in (("evaluation_request.json", "baseline_request_sha256"), ("evaluation_completion.json", "baseline_completion_sha256")):
        if hashlib.sha256((baseline / name).read_bytes()).hexdigest() != request[key]:
            raise ValueError(f"Baseline evidence changed: {name}")
    for name, expected in request["baseline_result_sha256"].items():
        if hashlib.sha256((baseline / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Baseline result changed: {name}")
    for path, expected in request["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {path}")


def summarize(run, request):
    expected = {(r["suite"], r["task_id"], r["seed"]) for r in request["selected_episodes"]}
    rows = [read(p) for p in sorted((run / "episodes").glob("*/task_*/seed_*/result.json"))]
    keys = {(r["suite"], r["task_id"], r["seed"]) for r in rows}
    if len(rows) != len(keys) or not keys <= expected:
        raise ValueError("Unexpected or duplicate retry episode")
    if any(not r["complete"] or r["eval_identity"] != request["eval_identity"] for r in rows):
        raise ValueError("Uncommitted or mismatched retry result")
    errors = list((run / "episodes").glob("*/task_*/seed_*/infrastructure_error.json"))
    value = dict(complete=keys == expected and not errors, expected_episodes=len(expected), completed_episodes=len(rows), successes=sum(r["success"] for r in rows), infrastructure_errors=len(errors), episodes=rows, updated_at=datetime.datetime.now().astimezone().isoformat())
    dump(run / "summary.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT)
    parser.add_argument("--worker", type=int, choices=range(6))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    request = read(run / "evaluation_request.json")
    sys.path.insert(0, str(ROOT))
    from examples.libero.success_continuation import validate_request as validate_success_request
    validate_success_request(request)
    validate_request(request)
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [str(FROZEN), str(ROOT), str(ROOT / "third_party/libero"), str(ROOT / "packages/openpi-client/src")]
    from examples.libero.solver_failure_hold import validate_failure_policy_request
    validate_failure_policy_request(request, 20)
    if args.worker is not None:
        from examples.libero.eval_reconstructed_libero_local import worker
        worker(run, args.worker, 20)
        return
    lock = (run / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump(run / "evaluation_process.json", dict(pid=os.getpid(), command=sys.argv))
    dump(run / "evaluation_state.json", dict(stage="waiting_for_inference", expected_episodes=request["expected_episodes"], client_location="local"))
    import urllib.request
    deadline = time.monotonic() + 1800
    for port in request["ports"]:
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError("Unhealthy inference server")
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError("Inference readiness timed out") from None
                time.sleep(5)
    children = []
    try:
        for seed, port in enumerate(request["ports"]):
            with (run / f"client_{seed}.log").open("a") as log:
                p = subprocess.Popen([str(PYTHON), "-u", str(Path(__file__).resolve()), "--run-dir", str(run), "--worker", str(seed)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append(p)
            dump(run / f"client_{seed}_process.json", dict(pid=p.pid, seed=seed, port=port))
        dump(run / "evaluation_state.json", dict(stage="evaluating", expected_episodes=request["expected_episodes"], client_location="local", pids=[p.pid for p in children]))
        while any(p.poll() is None for p in children):
            summarize(run, request)
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("Retry client failed; inspect infrastructure_error.json")
            time.sleep(5)
        summary = summarize(run, request)
        if not summary["complete"] or any(p.returncode != 0 for p in children):
            raise RuntimeError("Retry did not complete all selected episodes")
        dump(run / "evaluation_completion.json", dict(summary, exit_codes=[p.returncode for p in children], eval_identity=request["eval_identity"]))
        dump(run / "evaluation_state.json", dict(stage="complete", episodes=summary["completed_episodes"], successes=summary["successes"]))
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
