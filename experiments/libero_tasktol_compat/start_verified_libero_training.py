"""Start the authorized formal run once both local data publication gates pass."""

import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'data/libero_tasktol_joint_servo/20261101_full_v1'
HOST = 'czl@100.73.28.99'
SOCKET = '/tmp/libero_full_upload_20261101.sock'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def status(stage, **details):
    write(RUN / 'training_dispatch_state.json', dict(
        stage=stage, pid=os.getpid(),
        updated_at=datetime.datetime.now().astimezone().isoformat(), **details))


def ready():
    for name in ('local_conversion_state.json', 'lerobot_upload_state.json'):
        if read(RUN / name)['stage'] != 'complete':
            return False
    local = read(RUN / 'local_completion.json')
    upload = read(RUN / 'lerobot_upload_completion.json')
    assert local['complete'] and local['converter_exit_code'] == 0
    audit = local['audit']
    assert (audit['episodes'], audit['frames'], audit['tasks'], audit['fps']) == (1875, 312374, 40, 20)
    assert audit['all_camera_pixels_equal'] and audit['all_state_action_equal'] and audit['reader_smoke_passed']
    assert upload['complete'] and upload['all_uploaded_sha256_equal'] and upload['portable_lerobot_reader_passed']
    assert upload['transfer_manifest_sha256'] == hashlib.sha256((RUN / 'lerobot_transfer_manifest.json').read_bytes()).hexdigest()
    request = read(RUN / 'training_request.json')
    assert request['authorized'] and request['cancelled_all_benchmarks']
    assert hashlib.sha256(Path(request['server_script']).read_bytes()).hexdigest() == request['server_script_sha256']
    return True


REMOTE = r'''
import fcntl, hashlib, json, pathlib, subprocess, os
run = pathlib.Path('/mnt/SharedData/czl/runs/pi05_libero_reconstructed_lora_6gpu_bs72_10k_20261101')
script = run / 'train_reconstructed_pi05_server.py'
request = json.loads((run / 'training_request.json').read_text())
assert request['authorized'] and request['cancelled_all_benchmarks']
assert request['server_script_sha256'] == hashlib.sha256(script.read_bytes()).hexdigest()
verification = json.loads(pathlib.Path('/mnt/SharedData/czl/runs/libero_reconstructed_full_20261101/lerobot_upload_verification.json').read_text())
assert verification['complete'] and verification['all_uploaded_sha256_equal'] and verification['portable_lerobot_reader_passed']
assert (verification['episodes'], verification['frames'], verification['tasks']) == (1875,312374,40)
active = []
for process in pathlib.Path('/proc').iterdir():
    if process.name.isdigit():
        try:
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if str(script).encode() in args and (process / 'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z':
                active.append(int(process.name))
        except (OSError, ProcessLookupError):
            pass
if active:
    print(json.dumps({'stage':'reused_active_training','pids':active}))
elif (run / 'training_completion.json').exists():
    done = json.loads((run / 'training_completion.json').read_text())
    assert done['complete'] and done['exit_code'] == 0 and done['steps'] == 10000
    print(json.dumps({'stage':'already_complete','completion':done}))
else:
    lock = (run / 'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Release before launch; the child takes this same lock. One local dispatcher owns its own lock.
    fcntl.flock(lock, fcntl.LOCK_UN)
    with (run / 'pipeline.log').open('a') as log:
        process = subprocess.Popen(['/home/czl/openpi/.venv/bin/python','-u',str(script),'--run-dir',str(run)],
            cwd='/home/czl/openpi', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    receipt = {'stage':'launched','pid':process.pid,'run_dir':str(run)}
    (run / 'training_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
'''


def main():
    lock = (RUN / 'training_dispatch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(RUN / 'training_dispatch_process.json', dict(pid=os.getpid(), script=str(Path(__file__).resolve())))
    while not ready():
        status('waiting_for_verified_upload')
        time.sleep(60)
    status('launching_verified_training')
    command = ['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=20',HOST,
               '/home/czl/openpi/.venv/bin/python','-']
    # A lost SSH response may occur after launch; the remote process check makes retries safe.
    for attempt in range(10):
        try:
            result = subprocess.run(command, input=REMOTE, text=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            status('dispatch_retry', attempt=attempt + 1, error='SSH observation timed out; recheck same remote process before any launch')
            time.sleep(60)
            continue
        if result.returncode == 0:
            receipt = json.loads(result.stdout)
            write(RUN / 'training_dispatch_completion.json', receipt)
            status('dispatched', remote=receipt)
            return
        status('dispatch_retry', attempt=attempt + 1, error=result.stderr[-4000:])
        time.sleep(60)
    raise RuntimeError('Unable to dispatch verified formal training; inspect dispatch state')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        traceback.print_exc()
        raise
