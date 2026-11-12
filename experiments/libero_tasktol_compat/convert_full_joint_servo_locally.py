"""Convert the completed LIBERO reconstruction locally, without any upload."""
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/libero_tasktol_joint_servo/20261101_full_v1"
OUTPUT = ROOT / "data/lerobot/caozilin/libero_reconstructed_full_20261101"
REPO_ID = "caozilin/libero_reconstructed_full_20261101"
STATE = RUN / "local_conversion_state.json"


def read(path):
    return json.loads(path.read_text())


def update(stage, **details):
    previous = read(STATE) if STATE.exists() else {}
    value = dict(previous, stage=stage, updated_at=datetime.now().astimezone().isoformat(), **details)
    temp = STATE.with_suffix(".json.partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(STATE)
    print(stage, flush=True)


def main():
    lock = (RUN / "local_conversion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = read(RUN / "completion_audit.json")
    assert expected["complete"] and expected["sources"] == 2000
    assert read(RUN / "validation.json")["complete"]
    assert read(RUN / "pipeline_state.json")["stage"] == "paused_by_user"
    estimate = read(RUN / "local_conversion_storage_estimate.json")
    assert estimate["free_bytes"] > estimate["conservative_with_30pct_margin_bytes"] + 5e9
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HOME=str(RUN / "local_conversion_cache/hf"),
               HF_DATASETS_CACHE=str(RUN / "local_conversion_cache/hf_datasets"))
    command = [sys.executable, "-u", str(ROOT / "examples/libero/convert_reconstructed_hdf5_to_lerobot.py"),
               "--raw-dir", str(RUN), "--output", str(OUTPUT), "--repo-id", REPO_ID, "--resume"]
    update("hashing_and_converting", pid=os.getpid(), output=str(OUTPUT), repo_id=REPO_ID,
           expected_episodes=expected["episodes"], expected_frames=expected["frames"], upload_paused=True)
    with (RUN / "local_conversion.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        update("hashing_and_converting", converter_pid=child.pid)

        def pause(signum, frame):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=30)
            update("paused_by_user", converter_exit_code=child.returncode, upload_paused=True)
            raise SystemExit(130)

        signal.signal(signal.SIGINT, pause)
        signal.signal(signal.SIGTERM, pause)
        code = child.wait()
    update("checking_completion", converter_exit_code=code)
    assert code == 0, f"Local converter exited {code}; inspect local_conversion.log and --resume same output"
    audit = read(OUTPUT / "meta/reconstruction_audit.json")
    assert audit["complete"] and audit["reader_smoke_passed"]
    assert audit["all_camera_pixels_equal"] and audit["all_state_action_equal"]
    assert audit["episodes"] == expected["episodes"] and audit["frames"] == expected["frames"]
    assert audit["tasks"] == 40 and len(audit["suites"]) == 4
    assert audit["source_manifest_sha256"] == expected["manifest_sha256"]
    for suite, counts in expected["suites"].items():
        assert audit["suites"][suite]["episodes"] == counts["success"]
        assert audit["suites"][suite]["frames"] == counts["frames"]
    completion = dict(complete=True, location="local", output=str(OUTPUT), repo_id=REPO_ID,
                      converter_exit_code=0, upload_paused=True, audit=audit)
    target = RUN / "local_completion.json"
    target.write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n")
    lines = ["# LIBERO 本地 LeRobot 转换完成", "",
             f"来源2000条，成功{audit['episodes']}条，未成功{2000-audit['episodes']}条，执行帧{audit['frames']}。", "",
             f"LeRobot：`{OUTPUT}`", "", "20Hz、state8、actions7，256×256双相机旋转180度一次并无损嵌入PNG。",
             "所有episode的state/action、时间戳、任务和双相机全部像素均与重构HDF5一致，LeRobot读取测试通过，转换退出码0。", "",
             "上传及服务器转换保持用户暂停状态；本次没有恢复上传。", "",
             "|任务集|成功轨迹|执行帧|", "|---|---:|---:|"]
    lines += [f"|{suite}|{counts['episodes']}|{counts['frames']}|" for suite, counts in audit["suites"].items()]
    (RUN / "LOCAL_LEROBOT_REPORT.md").write_text("\n".join(lines) + "\n")
    update("complete", episodes=audit["episodes"], frames=audit["frames"], converter_exit_code=0, upload_paused=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update("failed", error=repr(error), upload_paused=True)
        traceback.print_exc()
        sys.exit(1)
