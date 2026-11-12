"""Wait for audited full reconstruction, verify/upload, convert on the server.

No credentials are stored. Uses the user-authorized SSH control connection.
The same standalone script also runs the server verification/conversion stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/libero_tasktol_joint_servo/20261101_full_v1"
HOST = "czl@100.73.28.99"
SOCKET = "/tmp/libero_full_upload_20261101.sock"
REMOTE_RAW = Path("/mnt/SharedData/czl/raw/libero_tasktol_joint_servo")
REMOTE_RUN = Path("/mnt/SharedData/czl/runs/libero_reconstructed_full_20261101")
REMOTE_OUTPUT = Path("/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_full_20261101")
REMOTE_PYTHON = "/home/czl/openpi/.venv/bin/python"
REPO_ID = "caozilin/libero_reconstructed_full_20261101"


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status(path, stage, **details):
    previous = read(path) if path.exists() else {}
    dump(path, dict(previous, stage=stage, updated_at=datetime.now().astimezone().isoformat(), **details))
    print(stage, flush=True)


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def local_audit(run):
    import h5py

    summary, manifest, validation, plan = [read(run / (name + ".json"))
                                           for name in ("summary", "manifest", "validation", "plan")]
    assert summary["complete"] and summary["total_attempted"] == summary["total_sources"] == 2000
    assert summary["total_pending"] == summary["process_errors"] == 0
    assert summary["total_success"] + summary["total_failed"] == 2000
    assert len(manifest["tasks"]) == len(plan["tasks"]) == 40
    assert manifest["complete"] and validation["complete"]
    assert manifest["source_unchanged"] and validation["servo_bitwise_verified"]
    assert validation["independently_replayed"] and validation["full_image_decode"]
    assert summary["total_success"] == manifest["episodes"] == validation["episodes"]
    checks = []
    all_keys = set()
    for entry, index in zip(plan["tasks"], manifest["tasks"], strict=True):
        case = entry["case"]
        assert case == index["case"]
        records = [read(p) for p in (run / "records" / f"case_{case:02d}").glob("*.json")]
        assert len(records) == 50 and len({r["demo"] for r in records}) == 50
        jobs = {j["demo"]: j for j in entry["candidates"]}
        successes = {r["demo"]: r for r in records if r.get("success")}
        for record in records:
            assert "error" not in record
            assert record["source_input_sha256"] == jobs[record["demo"]]["source_input_sha256"]
            assert (case, record["demo"]) not in all_keys
            all_keys.add((case, record["demo"]))
        frames = 0
        with h5py.File(index["path"]) as file:
            assert set(file["data"]) == set(successes)
            for demo, record in successes.items():
                group = file["data"][demo]
                n = int(group.attrs["num_samples"])
                assert group.attrs["success"] and group["dones"][-1] == 1
                assert record["independent_replay_bitwise_identical"]
                evidence = read(run / "episode_validation" / f"case_{case:02d}" / (demo + ".json"))["audit"]
                assert evidence["frames"] == n and evidence["full_decode"] and evidence["servo_bitwise_verified"]
                assert group["actions"].shape == (n, 7)
                for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                    assert group["obs"][camera].shape == (n, 256, 256, 3)
                frames += n
        assert len(successes) == index["episodes"] and frames == index["frames"]
        checks.append(dict(case=case, suite=index["suite"], task_id=index["task_id"],
                           attempted=50, success=len(successes), failed=50 - len(successes), frames=frames))
    assert len(all_keys) == 2000 and sum(t["success"] for t in checks) == manifest["episodes"]
    result = dict(complete=True, sources=2000, episodes=manifest["episodes"],
                  frames=sum(t["frames"] for t in checks), manifest_sha256=sha(run / "manifest.json"),
                  tasks=checks, suites=summary["suites"])
    dump(run / "completion_audit.json", result)
    (run / "success_counts.csv").write_text("suite,task_id,attempted,success,failed,frames\n" + "".join(
        f"{r['suite']},{r['task_id']},50,{r['success']},{r['failed']},{r['frames']}\n" for r in checks))
    return result


def transfer_plan(run):
    parent = run.parent
    selected = set()
    for directory in ("datasets", "episodes", "records", "episode_validation", "source_snapshot", "batch_versions"):
        selected.update(p for p in (run / directory).rglob("*") if p.is_file() and not p.name.endswith(".partial"))
    for name in ("plan.json", "summary.json", "manifest.json", "validation.json", "README.md", "completion_audit.json",
                 "success_counts.csv", "preflight.json", "SERVER_HANDOFF.md"):
        if (run / name).is_file():
            selected.add(run / name)
    baseline = Path(read(run / "plan.json")["baseline_run"])
    assert baseline.parent == parent
    selected.update(p for p in (baseline / "episodes").rglob("*") if p.is_file())
    for name in ("plan.json", "summary.json", "manifest.json", "validation.json", "collection_statistics.json", "README.md"):
        if (baseline / name).is_file():
            selected.add(baseline / name)
    files = [dict(path=str(p.relative_to(parent)), size=p.stat().st_size, sha256=sha(p)) for p in sorted(selected)]
    result = dict(schema=1, files=files, bytes=sum(f["size"] for f in files), file_count=len(files),
                  raw_manifest_sha256=sha(run / "manifest.json"))
    dump(run / "transfer_manifest.json", result)
    (run / "transfer_files.txt").write_text("".join(f["path"] + "\n" for f in files))
    return result


def ssh(command):
    return subprocess.run(["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, command],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout


def rsync(source, destination, *, files_from=None, logfile):
    command = ["rsync", "-rt", "--protect-args", "--partial", "--partial-dir=.rsync-partial", "--info=progress2",
               "-e", shlex.join(["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"])]
    if files_from is not None:
        command += ["--files-from", str(files_from)]
    command += [str(source), HOST + ":" + str(destination)]
    with logfile.open("a") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def start_remote():
    # Reuse a live process; verify exact script identity rather than PID alone.
    code = f"""import json,os,subprocess
from pathlib import Path
root=Path({str(REMOTE_RUN)!r})
state=root/'remote_state.json'
if state.exists() and json.loads(state.read_text()).get('stage')=='complete':
 print('Already complete');raise SystemExit(0)
pidfile=root/'remote_process.json'
if pidfile.exists():
 pid=json.loads(pidfile.read_text())['pid']; proc=Path(f'/proc/{{pid}}')
 if proc.exists() and (proc/'stat').read_text().split()[2]!='Z' and b'publish_full_joint_servo_dataset.py' in (proc/'cmdline').read_bytes():
  print('Already running',pid);raise SystemExit(0)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(root/'hf'),HF_DATASETS_CACHE=str(root/'hf_datasets'))
with (root/'server_pipeline.log').open('a') as log:
 p=subprocess.Popen([{REMOTE_PYTHON!r},'-u',str(root/'publish_full_joint_servo_dataset.py'),'--server'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=root)
pidfile.write_text(json.dumps(dict(pid=p.pid))+'\\n');print('Started',p.pid)
"""
    return ssh(shlex.join(["python3", "-c", code]))


def server_main():
    state = REMOTE_RUN / "remote_state.json"
    lock = (REMOTE_RUN / "server_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status(state, "verifying_upload", pid=os.getpid())
    manifest = read(REMOTE_RUN / "transfer_manifest.json")
    assert len(manifest["files"]) == manifest["file_count"]
    for index, item in enumerate(manifest["files"]):
        path = REMOTE_RAW / item["path"]
        assert path.is_file() and path.stat().st_size == item["size"], path
        assert sha(path) == item["sha256"], path
        if index % 500 == 0:
            dump(REMOTE_RUN / "upload_verification_progress.json", dict(verified=index, total=len(manifest["files"])))
    dump(REMOTE_RUN / "upload_verification.json", dict(complete=True, files=len(manifest["files"]),
                                                       bytes=manifest["bytes"], transfer_manifest_sha256=sha(REMOTE_RUN / "transfer_manifest.json")))
    raw = REMOTE_RAW / "20261101_full_v1"
    assert sha(raw / "manifest.json") == manifest["raw_manifest_sha256"]
    status(state, "converting_lerobot")
    command = [REMOTE_PYTHON, "-u", str(REMOTE_RUN / "convert_reconstructed_hdf5_to_lerobot.py"),
               "--raw-dir", str(raw), "--output", str(REMOTE_OUTPUT), "--repo-id", REPO_ID, "--resume"]
    with (REMOTE_RUN / "conversion.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = read(REMOTE_OUTPUT / "meta/reconstruction_audit.json")
    expected = read(raw / "completion_audit.json")
    assert audit["complete"] and audit["reader_smoke_passed"]
    assert audit["all_camera_pixels_equal"] and audit["all_state_action_equal"]
    assert audit["episodes"] == expected["episodes"] and audit["frames"] == expected["frames"]
    assert audit["tasks"] == 40 and len(audit["suites"]) == 4
    for suite, counts in expected["suites"].items():
        assert audit["suites"][suite]["episodes"] == counts["success"]
        assert audit["suites"][suite]["frames"] == counts["frames"]
    result = dict(complete=True, raw=str(raw), lerobot=str(REMOTE_OUTPUT), repo_id=REPO_ID,
                  upload_sha256_verified=True, conversion_exit_code=0, audit=audit,
                  source_manifest_sha256=manifest["raw_manifest_sha256"])
    dump(REMOTE_RUN / "server_completion.json", result)
    status(state, "complete", episodes=audit["episodes"], frames=audit["frames"])


def local_main(run):
    state = run / "pipeline_state.json"
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status(state, "waiting_for_reconstruction", pid=os.getpid(), heartbeat_id="libero-lerobot",
           remote_raw=str(REMOTE_RAW), remote_output=str(REMOTE_OUTPUT), remote_run=str(REMOTE_RUN))
    while not all((run / name).exists() for name in ("validation.json", "manifest.json", "README.md")) or not all(
        read(run / name).get("complete") for name in ("validation.json", "manifest.json")
    ):
        process = read(run / "process.json")
        if not alive(process["pid"]):
            raise RuntimeError("Reconstruction process exited without final validation; inspect and resume same output")
        time.sleep(60)
    status(state, "auditing_local")
    audit = local_audit(run)
    status(state, "hashing_transfer", episodes=audit["episodes"], frames=audit["frames"])
    manifest = transfer_plan(run)
    status(state, "uploading", files=manifest["file_count"], bytes=manifest["bytes"])
    ssh(shlex.join(["mkdir", "-p", str(REMOTE_RAW), str(REMOTE_RUN), str(REMOTE_OUTPUT.parent)]))
    available = int(ssh("df -B1 --output=avail /mnt/SharedData | tail -1").strip())
    assert available > 2.5 * manifest["bytes"] + 20e9, "Insufficient space for raw and lossless LeRobot export"
    rsync(str(run.parent) + "/", str(REMOTE_RAW) + "/", files_from=run / "transfer_files.txt", logfile=run / "rsync.log")
    for source in (Path(__file__), ROOT / "examples/libero/convert_reconstructed_hdf5_to_lerobot.py", run / "transfer_manifest.json"):
        rsync(source, REMOTE_RUN / source.name, logfile=run / "rsync.log")
    status(state, "server_processing", remote_start=start_remote())
    failures = 0
    while True:
        try:
            remote = json.loads(ssh(shlex.join(["cat", str(REMOTE_RUN / "remote_state.json")])))
            failures = 0
            if remote["stage"] == "failed":
                raise RuntimeError("Server pipeline failed: " + str(remote))
            if remote["stage"] == "complete":
                result = json.loads(ssh(shlex.join(["cat", str(REMOTE_RUN / "server_completion.json")])))
                assert result["complete"] and result["source_manifest_sha256"] == audit["manifest_sha256"]
                dump(run / "server_completion.json", result)
                status(state, "complete", episodes=result["audit"]["episodes"], frames=result["audit"]["frames"])
                return
        except subprocess.CalledProcessError:
            failures += 1
            if failures >= 10:
                raise
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    state = REMOTE_RUN / "remote_state.json" if args.server else args.run / "pipeline_state.json"
    try:
        if args.server:
            server_main()
        else:
            local_main(args.run.resolve())
    except Exception as error:
        status(state, "failed", error=repr(error))
        traceback.print_exc()
        sys.exit(1)
