"""Publish only the locally converted and audited LeRobot dataset.

Raw-HDF5 transfer and server conversion remain paused. The server verifies all
uploaded file hashes and reads the portable dataset without needing raw HDF5.
"""
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
OUTPUT = ROOT / "data/lerobot/caozilin/libero_reconstructed_full_20261101"
HOST = "czl@100.73.28.99"
SOCKET = "/tmp/libero_full_upload_20261101.sock"
REMOTE_OUTPUT = Path("/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_full_20261101")
REMOTE_RUN = Path("/mnt/SharedData/czl/runs/libero_reconstructed_full_20261101")
REMOTE_PYTHON = "/home/czl/openpi/.venv/bin/python"
REPO_ID = "caozilin/libero_reconstructed_full_20261101"


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status(stage, **details):
    path = RUN / "lerobot_upload_state.json"
    previous = read(path) if path.exists() else {}
    dump(path, dict(previous, stage=stage, updated_at=datetime.now().astimezone().isoformat(), **details))
    print(stage, flush=True)


def ssh(command):
    return subprocess.run(["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, command],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout


def rsync(source, destination, files_from=None):
    command = ["rsync", "-rt", "--protect-args", "--partial", "--partial-dir=.rsync-partial", "--info=progress2",
               "-e", shlex.join(["ssh", "-S", SOCKET, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"])]
    if files_from is not None:
        command += ["--files-from", str(files_from)]
    command += [str(source), HOST + ":" + str(destination)]
    with (RUN / "lerobot_rsync.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def validate_audited_parquet_hashes(info, audit, files):
    hashes = {item["path"]: item["sha256"] for item in files}
    assert len(audit["episode_checks"]) == info["total_episodes"]
    for check in audit["episode_checks"]:
        index = check["episode_index"]
        path = info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        assert hashes.get(path) == check["parquet_sha256"], f"Audited parquet changed: {path}"


def remote_verify():
    import numpy as np
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    lock = (REMOTE_RUN / "lerobot_verification.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(REMOTE_RUN / "lerobot_transfer_manifest.json")
    assert manifest["episodes"] == 1875 and manifest["frames"] == 312374
    for i, item in enumerate(manifest["files"]):
        path = REMOTE_OUTPUT / item["path"]
        assert path.is_relative_to(REMOTE_OUTPUT) and not path.is_symlink()
        assert path.is_file() and path.stat().st_size == item["size"], path
        assert sha(path) == item["sha256"], path
        if i % 100 == 0:
            dump(REMOTE_RUN / "lerobot_verification_progress.json", dict(verified=i, total=len(manifest["files"])))
    info = read(REMOTE_OUTPUT / "meta/info.json")
    audit = read(REMOTE_OUTPUT / "meta/reconstruction_audit.json")
    validate_audited_parquet_hashes(info, audit, manifest["files"])
    sources = read(REMOTE_OUTPUT / "meta/reconstruction_sources.json")["sources"]
    assert info["total_episodes"] == audit["episodes"] == len(sources) == 1875
    assert info["total_frames"] == audit["frames"] == sum(s["frames"] for s in sources) == 312374
    assert info["total_tasks"] == 40 and info["fps"] == 20
    assert info["features"]["state"]["shape"] == [8] and info["features"]["actions"]["shape"] == [7]
    assert audit["complete"] and audit["reader_smoke_passed"]
    assert audit["all_camera_pixels_equal"] and audit["all_state_action_equal"]
    assert len({(s["case"], s["demo"]) for s in sources}) == 1875
    assert len(list(REMOTE_OUTPUT.glob("data/chunk-*/episode_*.parquet"))) == 1875
    selected = [next(s["episode_index"] for s in sources if s["suite"] == suite)
                for suite in sorted({s["suite"] for s in sources})]
    dataset = LeRobotDataset(repo_id=REPO_ID, root=REMOTE_OUTPUT, episodes=selected, force_cache_sync=False)
    offset = 0
    samples = []
    for episode in selected:
        sample = dataset[offset]
        assert tuple(sample["state"].shape) == (8,) and tuple(sample["actions"].shape) == (7,)
        assert tuple(sample["image"].shape) == tuple(sample["wrist_image"].shape) == (3, 256, 256)
        assert np.isfinite(np.asarray(sample["state"])).all() and np.isfinite(np.asarray(sample["actions"])).all()
        samples.append(episode)
        offset += sources[episode]["frames"]
    result = dict(complete=True, location="server", dataset=str(REMOTE_OUTPUT), repo_id=REPO_ID,
                  episodes=1875, frames=312374, tasks=40, fps=20, suites=audit["suites"],
                  all_uploaded_sha256_equal=True, verified_files=len(manifest["files"]), bytes=manifest["bytes"],
                  portable_lerobot_reader_passed=True, reader_sample_episodes=samples,
                  local_pixel_state_action_audit_preserved=True,
                  transfer_manifest_sha256=sha(REMOTE_RUN / "lerobot_transfer_manifest.json"))
    dump(REMOTE_RUN / "lerobot_upload_verification.json", result)
    return result


def local_main():
    lock = (RUN / "lerobot_upload.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status("waiting_for_local_conversion", pid=os.getpid(), raw_upload_paused=True,
           local_output=str(OUTPUT), remote_output=str(REMOTE_OUTPUT))
    while not (RUN / "local_completion.json").exists() or read(RUN / "local_conversion_state.json")["stage"] != "complete":
        state = read(RUN / "local_conversion_state.json")
        if state["stage"] in ("failed", "paused_by_user"):
            raise RuntimeError("Local conversion needs attention: " + str(state))
        time.sleep(60)
    completed = read(RUN / "local_completion.json")
    assert completed["complete"] and completed["converter_exit_code"] == 0
    assert completed["audit"]["episodes"] == 1875 and completed["audit"]["frames"] == 312374
    assert read(RUN / "local_conversion_state.json")["stage"] == "complete"
    assert read(RUN / "pipeline_state.json")["stage"] == "paused_by_user"
    status("hashing_lerobot")
    files = []
    for path in sorted(OUTPUT.rglob("*")):
        assert not path.is_symlink(), path
        if path.is_file():
            assert path.relative_to(OUTPUT).parts[0] in ("meta", "data"), path
            files.append(dict(path=str(path.relative_to(OUTPUT)), size=path.stat().st_size, sha256=sha(path)))
    manifest = dict(episodes=1875, frames=312374, files=files, bytes=sum(f["size"] for f in files))
    validate_audited_parquet_hashes(read(OUTPUT / "meta/info.json"), completed["audit"], files)
    dump(RUN / "lerobot_transfer_manifest.json", manifest)
    file_list = RUN / "lerobot_transfer_files.txt"
    file_list.write_text("".join(f["path"] + "\n" for f in files))
    ssh(shlex.join(["mkdir", "-p", str(REMOTE_OUTPUT), str(REMOTE_RUN)]))
    available = int(ssh("df -B1 --output=avail /mnt/SharedData | tail -1").strip())
    assert available > manifest["bytes"] + 15e9, "Server has insufficient space for LeRobot dataset"
    status("uploading_lerobot", bytes=manifest["bytes"], files=len(files))
    rsync(str(OUTPUT) + "/", str(REMOTE_OUTPUT) + "/", file_list)
    rsync(Path(__file__), REMOTE_RUN / Path(__file__).name)
    rsync(RUN / "lerobot_transfer_manifest.json", REMOTE_RUN / "lerobot_transfer_manifest.json")
    status("verifying_server")
    env = ["env", "PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
           "CUDA_VISIBLE_DEVICES=", "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1",
           "HF_HOME=" + str(REMOTE_RUN / "lerobot_hf"),
           "HF_DATASETS_CACHE=" + str(REMOTE_RUN / "lerobot_hf_datasets")]
    command = env + [REMOTE_PYTHON, str(REMOTE_RUN / Path(__file__).name), "--remote-verify"]
    with (RUN / "lerobot_server_verify.log").open("a") as log:
        subprocess.run(["ssh", "-S", SOCKET, "-o", "BatchMode=yes", HOST, shlex.join(command)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(ssh(shlex.join(["cat", str(REMOTE_RUN / "lerobot_upload_verification.json")])))
    assert result["complete"] and result["all_uploaded_sha256_equal"] and result["portable_lerobot_reader_passed"]
    assert result["transfer_manifest_sha256"] == sha(RUN / "lerobot_transfer_manifest.json")
    dump(RUN / "lerobot_upload_completion.json", result)
    status("complete", episodes=1875, frames=312374, raw_upload_paused=True)


if __name__ == "__main__":
    try:
        if "--remote-verify" in sys.argv:
            print(json.dumps(remote_verify(), ensure_ascii=False), flush=True)
        else:
            local_main()
    except Exception as error:
        if "--remote-verify" not in sys.argv:
            status("failed", error=repr(error))
        traceback.print_exc()
        sys.exit(1)
