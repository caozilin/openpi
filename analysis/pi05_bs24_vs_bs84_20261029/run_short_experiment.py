"""Supervise the unchanged train.py, keeping its exact optimization/data loop."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

ROOT = Path('/home/czl/openpi')
RUN = Path('/mnt/SharedData/czl/runs/pi05_state_action7_bs24_6gpu_15min_20261029')
BASE_RUN = Path('/mnt/SharedData/czl/runs/pi05_state_action7_bs84_15k_20261028')
CONFIG = 'pi05_franka_mujoco_state_action7_lora_bs24_6gpu'
EXPERIMENT = 'Pi05_nominal100_state_action7_LoRA_6gpu_bs24_15min_20261029'
DURATION = 900


def write(name, value):
    target = RUN / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = Path('/mnt/SharedData/czl/checkpoints') / CONFIG / EXPERIMENT
        assert not checkpoint.exists(), f'Existing experiment: {checkpoint}'
        gpu = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], text=True)
        assert all(int(row.split(',')[1]) < 500 for row in gpu.strip().splitlines()[:6]), gpu
        plan = json.loads((BASE_RUN / 'training_command.json').read_text())
        for relative, digest in plan['source_sha256'].items():
            if relative == 'src/openpi/training/config.py':
                continue
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
        norm = Path('/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7/caozilin/franka_100_nominal/norm_stats.json')
        assert hashlib.sha256(norm.read_bytes()).hexdigest() == plan['norm_sha256']
        command = [str(ROOT / '.venv/bin/python'), '-u', 'scripts/train.py', CONFIG,
                   '--exp-name', EXPERIMENT,
                   '--data.lerobot-dataset-root', '/mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal',
                   '--data.hf-datasets-cache-dir', '/mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal',
                   '--data.assets.assets-dir', '/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7',
                   '--checkpoint-base-dir', '/mnt/SharedData/czl/checkpoints',
                   '--swanlab-log-dir', '/mnt/SharedData/czl/swanlog']
        env = os.environ.copy()
        env.pop('JAX_PLATFORMS', None)
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5', XLA_PYTHON_CLIENT_MEM_FRACTION='0.95',
                   PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                   HF_ENDPOINT='https://hf-mirror.com',
                   HF_DATASETS_CACHE='/mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal')
        start = time.monotonic()
        state = {'state': 'starting', 'launch_unix': time.time(), 'duration_after_first_progress_seconds': DURATION,
                 'first_progress_unix': None, 'last_step': 0, 'stop_requested': False,
                 'config': CONFIG, 'experiment': EXPERIMENT, 'checkpoint_dir': str(checkpoint)}
        write('command.json', {'argv': command, 'global_batch': 24, 'num_gpus': 6, 'per_gpu_batch': 4,
                              'unchanged_from': str(BASE_RUN), 'norm_sha256': plan['norm_sha256'],
                              'timer_policy': '900 seconds after first compiled progress event; stop at next 10-step metric boundary'})
        (RUN / 'config_source.py').write_bytes((ROOT / 'src/openpi/training/config.py').read_bytes())
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        state.update(pid=process.pid, supervisor_pid=os.getpid())
        write('status.json', state)
        stop_monitor = threading.Event()

        def monitor():
            with (RUN / 'resources.jsonl').open('w') as output:
                while not stop_monitor.is_set():
                    try:
                        gpus = subprocess.check_output([
                            'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw',
                            '--format=csv,noheader,nounits',
                        ], text=True, timeout=8)
                        output.write(json.dumps({'unix': time.time(), 'gpus': gpus,
                                                 'load_average': os.getloadavg()}) + '\n')
                        output.flush()
                        # Safety deadline if the training process stops emitting logs.
                        first = state['first_progress_unix']
                        deadline = (time.time() - first > DURATION + 120) if first else (time.monotonic() - start > 1200)
                        if deadline and process.poll() is None:
                            state['stop_requested'] = True
                            state['stop_reason'] = 'supervisor_safety_timeout'
                            os.killpg(process.pid, signal.SIGINT)
                    except Exception as error:
                        output.write(json.dumps({'unix': time.time(), 'monitor_error': repr(error)}) + '\n')
                        output.flush()
                    stop_monitor.wait(10)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        with (RUN / 'train.log').open('w') as log, (RUN / 'events.jsonl').open('w') as events:
            for line in process.stdout:
                now = time.time()
                log.write(line)
                log.flush()
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                if 'Progress on:' in clean and state['first_progress_unix'] is None:
                    state.update(first_progress_unix=now, state='training')
                    write('status.json', state)
                metric = re.search(r'Step\s+(\d+): ([^\n]+)', clean)
                if metric:
                    step = int(metric.group(1))
                    values = {key: float(value) for key, value in re.findall(r'([\w/]+)=([-+0-9.eE]+)', metric.group(2))}
                    state['last_step'] = step
                    event = {'unix': now, 'step': step, 'samples': step * 24, 'metrics': values}
                    events.write(json.dumps(event) + '\n')
                    events.flush()
                    if state['first_progress_unix'] is None:
                        state['first_progress_unix'] = now
                    elapsed = now - state['first_progress_unix']
                    state['postcompile_elapsed_seconds'] = elapsed
                    if elapsed >= DURATION and not state['stop_requested']:
                        state.update(stop_requested=True, stop_reason='requested_15_minute_training_budget',
                                     stop_unix=now)
                        os.killpg(process.pid, signal.SIGINT)
                    write('status.json', state)
                if 'https://swanlab.cn/' in clean:
                    events.write(json.dumps({'unix': now, 'tracking_line': clean.strip()}) + '\n')
                    events.flush()
        returncode = process.wait()
        stop_monitor.set()
        thread.join(timeout=12)
        state.update(state='completed_short_run' if state['stop_requested'] else 'exited',
                     returncode=returncode, finished_unix=time.time(),
                     wall_seconds=time.monotonic() - start,
                     finished_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
        write('status.json', state)
        write('exit_status.json', state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    run()
