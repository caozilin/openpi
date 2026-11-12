"""Host-only launch/readiness checks for this explicitly authorized run."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
sys.path.insert(0, str(ROOT))
from examples.libero.eval_nominal50 import RUN, REMOTE, IDENTITY, PYTHON, read, dump
from examples.libero.nominal50_pipeline import HOST, SOCKET


def ssh_python(code):
    return subprocess.run(['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', HOST,
        'python3 -c ' + shlex.quote(code)], check=True, capture_output=True, text=True, timeout=120)


def launch_server():
    request = read(RUN / 'evaluation_request.json')
    code = f'''from pathlib import Path
import subprocess,hashlib,json,os
root=Path({REMOTE!r})
entry=root/'serve_nominal50_server.py'
assert hashlib.sha256(entry.read_bytes()).hexdigest()=={request['server_entry_sha256']!r}
record=root/'inference_manager_process.json'
if record.exists():
 old=json.loads(record.read_text())
 assert not Path(f"/proc/{{old['pid']}}").exists(),'Existing manager PID must be inspected; refusing a duplicate'
with (root/'inference_pipeline.log').open('a') as log:
 process=subprocess.Popen(['/home/czl/openpi/.venv/bin/python','-u',str(entry)],cwd='/home/czl/openpi',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=process.pid,entry=str(entry))))
'''
    result = json.loads(ssh_python(code).stdout)
    dump(RUN / 'inference_launch.json', result)
    forwards = []
    for port in range(18760, 18766):
        p = subprocess.run(['ssh', '-S', SOCKET, '-O', 'forward', '-L', f'{port}:127.0.0.1:{port}', HOST],
            check=True, capture_output=True, text=True, timeout=20)
        forwards.append(dict(port=port, exit_code=p.returncode))
    dump(RUN / 'inference_tunnels.json', dict(complete=True,socket=SOCKET,forwards=forwards))
    print(result, flush=True)


def ready_and_launch():
    deadline = time.monotonic() + 1800
    pending = set(range(18760, 18766))
    while pending:
        for port in sorted(pending):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=3) as response:
                    assert response.status == 200
                pending.remove(port)
                print(f'Inference ready: {port}',flush=True)
            except Exception:
                pass
        if time.monotonic() > deadline:
            raise TimeoutError(f'Inference not ready: {pending}')
        if pending:
            time.sleep(5)
    code = f'''from pathlib import Path
import hashlib,json
root=Path({REMOTE!r})
manager=json.loads((root/'inference_manager_process.json').read_text())
records=[json.loads((root/f'replica_{{i}}_ready.json').read_text()) for i in range(6)]
for r in [manager]+records:
 pid=r['pid'];command=Path(f'/proc/{{pid}}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert str(root/'serve_nominal50_server.py') in command
for r in records:
 assert r['eval_identity']=={IDENTITY!r}
 env=Path(f"/proc/{{r['pid']}}/environ").read_bytes().split(b'\\0')
 assert f"CUDA_VISIBLE_DEVICES={{r['gpu']}}".encode() in env
print(json.dumps(dict(manager=manager,records=records,server_entry_sha256=hashlib.sha256((root/'serve_nominal50_server.py').read_bytes()).hexdigest())))
'''
    snapshot = json.loads(ssh_python(code).stdout)
    sys.path.insert(0, str(ROOT / 'packages/openpi-client/src'))
    from openpi_client import msgpack_numpy
    import websockets.sync.client
    replicas = []
    for index in range(6):
        with websockets.sync.client.connect(f'ws://127.0.0.1:{18760 + index}',
            compression=None,max_size=None,ping_interval=None,ping_timeout=None,open_timeout=30) as socket:
            metadata = msgpack_numpy.unpackb(socket.recv(timeout=30))
        assert metadata['eval_identity'] == IDENTITY and metadata['replica'] == index
        assert metadata['gpu'] == [0,1,2,3,4,6][index]
        assert metadata['checkpoint'].endswith('/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101/10000')
        assert not metadata['state_enabled'] and metadata['model_action_horizon'] == 10
        replicas.append(metadata)
    request = read(RUN / 'evaluation_request.json')
    assert snapshot['server_entry_sha256'] == request['server_entry_sha256']
    dump(RUN / 'inference_identity_audit.json', dict(complete=True,replicas=replicas,
        server_entry_sha256=snapshot['server_entry_sha256'],real_process_snapshot=snapshot))
    if (RUN / 'pipeline_process.json').exists():
        old=read(RUN / 'pipeline_process.json')
        assert not Path(f"/proc/{old['pid']}").exists(),'Refusing an existing pipeline'
    with (RUN / 'pipeline.log').open('a') as log:
        process = subprocess.Popen([str(PYTHON),'-u','-m','examples.libero.nominal50_pipeline'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(RUN / 'pipeline_launch.json',dict(pid=process.pid,command='examples.libero.nominal50_pipeline'))
    print(dict(pipeline_pid=process.pid,identity_audit_complete=True),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['server','clients'])
    args=parser.parse_args()
    launch_server() if args.action=='server' else ready_and_launch()
