"""Observe this authorized mixed-dataset pipeline without starting or restarting jobs."""

import datetime
import json
from pathlib import Path
import subprocess


RECORD = Path(__file__).resolve().parent
REMOTE_RUN = "/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109"
REMOTE_DATA = "/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109"


def main():
    result = {"observed_at": datetime.datetime.now().astimezone().isoformat(), "local": {}}
    pids = []
    for name in ("upload", "training_dispatch"):
        result["local"][name] = json.loads((RECORD / f"{name}_state.json").read_text())
        pid = json.loads((RECORD / f"{name}_process.json").read_text())["pid"]
        pids.append(pid)
        proc = Path(f"/proc/{pid}")
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            result["local"][f"{name}_process"] = {
                "pid": pid,
                "matches_this_run": str(RECORD) in command and f"{name if name == 'upload' else 'start_joint_training_when_verified'}" in command,
            }
        except FileNotFoundError:
            result["local"][f"{name}_process"] = {"pid": pid, "exists": False}
    result["local"]["process_snapshot"] = subprocess.run(
        ["ps", "-p", ",".join(map(str, pids)), "-o", "pid,ppid,stat,etime,pcpu"],
        capture_output=True, text=True, check=False,
    ).stdout
    result["local"]["upload_children"] = subprocess.run(
        ["ps", "--ppid", str(pids[0]), "-o", "pid,ppid,stat,etime,comm"],
        capture_output=True, text=True, check=False,
    ).stdout
    source = f"""
from pathlib import Path
import json, shutil, re
d = Path({REMOTE_DATA!r})
r = Path({REMOTE_RUN!r})
files = list(d.glob('data/**/*.parquet'))
v = dict(parquets=len(files), parquet_bytes=sum(p.stat().st_size for p in files), free_gib=shutil.disk_usage(r).free / 2**30)
compact_files = [p for p in (r / 'exact_transport_bundle').rglob('*') if p.is_file()]
v['compact_present_files'] = len(compact_files)
v['compact_present_file_bytes'] = sum(p.stat().st_size for p in compact_files)
for filename, key in (('normalization.log', 'normalization_progress'), ('train.log', 'latest_training_metric')):
    p = r / filename
    v[key] = None
    if p.exists():
        with p.open('rb') as stream:
            stream.seek(max(0, p.stat().st_size - 12000))
            tail = stream.read().decode(errors='replace')
        pattern = r'(?:Generating train split|Computing stats)[^\\r\\n]*' if filename == 'normalization.log' else r'Step\\s+\\d+:[^\\r\\n]*'
        matches = list(re.finditer(pattern, tail))
        if matches:
            v[key] = matches[-1].group(0)
for name in ('exact_transport_reconstruction_state', 'upload_verification_state', 'training_state', 'normalization_completion', 'finish_training_state', 'independent_training_audit_state', 'task_completion'):
    p = r / (name + '.json')
    v[name] = json.loads(p.read_text()) if p.exists() else None
v['four_x_protocol_update'] = {{}}
for key in ('transition_state', 'transition_completion', 'normalization_reuse_verification', 'training_preflight_verification'):
    p = r / 'four_x_protocol_update' / (key + '.json')
    v['four_x_protocol_update'][key] = json.loads(p.read_text()) if p.exists() else None
v['live_owned_processes'] = []
for name in ('exact_transport_reconstruction_process', 'training_process', 'finish_training_process', 'independent_training_audit_process', 'four_x_protocol_update/transition_process'):
    p = r / (name + '.json')
    if not p.exists():
        continue
    pid = json.loads(p.read_text())['pid']
    try:
        cmd = Path(f'/proc/{{pid}}/cmdline').read_bytes().replace(b'\\0', b' ').decode()
        rest = Path(f'/proc/{{pid}}/stat').read_text().rsplit(')', 1)[1].split()
        v['live_owned_processes'].append(dict(role=name, pid=pid, matches_this_run=str(r) in cmd, process_state=rest[0], ppid=int(rest[1]), start_ticks=int(rest[19])))
    except FileNotFoundError:
        v['live_owned_processes'].append(dict(role=name, pid=pid, exists=False))
print(json.dumps(v))
"""
    try:
        remote = subprocess.run(
            ["ssh", "-S", "/tmp/libero_joint13_20261109.sock", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "czl@100.73.28.99", "/home/czl/openpi/.venv/bin/python -"],
            input=source, text=True, capture_output=True, timeout=45, check=False,
        )
        result["remote"] = json.loads(remote.stdout) if remote.returncode == 0 else {
            "probe_error": remote.stderr[-500:], "returncode": remote.returncode,
        }
    except subprocess.TimeoutExpired:
        result["remote"] = {"probe_timeout": True, "terminal_state_proven": False}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
