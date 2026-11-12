"""Supervise only the authorized nominal1600 extension, audit, merge and cleanup."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

from examples.libero.eval_nominal50 import ROOT, RUN, REMOTE, IDENTITY, PYTHON, read, dump, validate
from examples.libero.nominal50_protocol import finalize

SOCKET = '/tmp/libero_nominal50_20261105.sock'
HOST = 'czl@100.73.28.99'


def remote_python(code):
    command = 'python3 -c ' + shlex.quote(code)
    return subprocess.run(['ssh', '-S', SOCKET, '-o', 'BatchMode=yes', HOST, command],
        check=True, text=True, capture_output=True, timeout=120)


def stop_inference(run):
    # PID is accepted only if both its recorded identity and actual command
    # name this exact run. Never signal a port owner or a GPU-wide process set.
    code = f'''import json,os,signal,time
from pathlib import Path
root=Path({REMOTE!r})
record=json.loads((root/'inference_manager_process.json').read_text())
pid=record['pid']
children=[json.loads((root/f'replica_{{i}}_process.json').read_text())['pid'] for i in range(6)]
def alive(p):
 path=Path(f'/proc/{{p}}/stat')
 return path.exists() and path.read_text().split()[2]!='Z'
if alive(pid):
 command=Path(f'/proc/{{pid}}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert str(root/'serve_nominal50_server.py') in command,command
 os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+90
while any(alive(p) for p in [pid]+children) and time.monotonic()<deadline:time.sleep(1)
assert not any(alive(p) for p in [pid]+children)
print(json.dumps(dict(complete=True,manager_pid=pid,replica_pids=children,all_exited=True)))
'''
    output = remote_python(code)
    evidence = json.loads(output.stdout)
    dump(run / 'inference_shutdown_verification.json', evidence)
    cancel = []
    for port in range(18760, 18766):
        p = subprocess.run(['ssh', '-S', SOCKET, '-O', 'cancel', '-L', f'{port}:127.0.0.1:{port}', HOST],
            capture_output=True, text=True, timeout=20)
        cancel.append(dict(port=port, exit_code=p.returncode, stderr=p.stderr.strip()))
    assert all(p['exit_code'] == 0 for p in cancel), cancel
    p = subprocess.run(['ssh', '-S', SOCKET, '-O', 'exit', HOST], capture_output=True, text=True, timeout=20)
    assert p.returncode == 0
    dump(run / 'local_cleanup_verification.json', dict(complete=True,forward_cancellations=cancel,
        dedicated_control_socket_closed=True,viewer_processes_untouched=True))


def backup_reports(run):
    names = ['NOMINAL_50SEED_REPORT.md', 'combined_formal_2000.json', 'combined_formal_2000.csv',
        'evaluation_independent_final_audit.json', 'preserved_formal400.json', 'evaluation_completion.json']
    p = subprocess.run(['scp', '-o', f'ControlPath={SOCKET}', '-o', 'BatchMode=yes',
        *[str(run / n) for n in names], f'{HOST}:{REMOTE}/'], capture_output=True, text=True, timeout=120)
    dump(run / 'report_backup.json', dict(complete=p.returncode == 0, exit_code=p.returncode,
        files=names, stderr=p.stderr[-2000:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = read(run / 'evaluation_request.json')
    validate(request)
    assert request['server_eval_identity'] == IDENTITY
    if (run / 'task_completion.json').exists():
        assert read(run / 'task_completion.json')['complete']
        print('Already complete.', flush=True)
        return
    dump(run / 'pipeline_process.json', dict(pid=os.getpid(), command=sys.argv))
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        with (run / 'evaluation_pipeline.log').open('a') as log:
            client = subprocess.Popen([str(PYTHON), '-u', '-m', 'examples.libero.eval_nominal50',
                '--run-dir', str(run)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append(client)
        with (run / 'independent_audit.log').open('a') as log:
            audit = subprocess.Popen([str(PYTHON), '-u', '-m', 'examples.libero.audit_formal_nominal',
                '--run-dir', str(run), '--follow'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append(audit)
        dump(run / 'independent_audit_launch.json', dict(pid=audit.pid,follow=True))
        dump(run / 'pipeline_state.json', dict(stage='evaluating_and_auditing',client_manager=client.pid,audit=audit.pid))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Evaluation or independent audit failed; completed rollouts remain immutable')
            time.sleep(10)
        assert all(p.returncode == 0 for p in children)
        assert read(run / 'evaluation_independent_final_audit.json')['complete']
        result = finalize(run)
        backup_reports(run)
        stop_inference(run)
        dump(run / 'task_completion.json', dict(result,all_required_audits_complete=True,
            own_inference_and_tunnels_stopped=True,viewer_processes_untouched=True))
        dump(run / 'pipeline_state.json', dict(stage='complete',**result))
        print(result, flush=True)
    except BaseException:
        failure = traceback.format_exc()
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        cleanup_error = None
        try:
            stop_inference(run)
        except Exception:
            cleanup_error = traceback.format_exc()
        dump(run / 'pipeline_state.json', dict(stage='failed_or_interrupted',error=failure,cleanup_error=cleanup_error))
        raise


if __name__ == '__main__':
    main()
