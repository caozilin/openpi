"""Bounded transport-only diagnostic; no model, no evaluation, no remote files."""
import subprocess,time,json,os,concurrent.futures,datetime,pathlib
BASE=['ssh','-S','/tmp/libero_timing_investigation_20261104.sock','-o','BatchMode=yes','czl@100.73.28.99']
payload=os.urandom(393520)
def call(command,data=None):
 t=time.perf_counter();r=subprocess.run(BASE+[command],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=40,check=True);return dict(seconds=time.perf_counter()-t,response_bytes=len(r.stdout))
handshakes=[call('true') for _ in range(3)]
upload=[call("python3 -c 'import sys; print(len(sys.stdin.buffer.read()))'",payload) for _ in range(3)]
download=call("head -c 393520 /dev/urandom")
t=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:parallel=list(pool.map(lambda _:call("python3 -c 'import sys; print(len(sys.stdin.buffer.read()))'",payload),range(6)))
elapsed=time.perf_counter()-t
r=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),packet_bytes=len(payload),ssh_command_round_trips=handshakes,serial_upload=upload,download=download,six_parallel_upload=parallel,six_parallel_wall_seconds=elapsed,six_parallel_aggregate_mbps=len(payload)*6*8/elapsed/1e6,scope='Current SSH transport-only test, includes channel setup and command launch; not historical GPU inference timing')
pathlib.Path('analysis/inference_timing_20261104/transport.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
