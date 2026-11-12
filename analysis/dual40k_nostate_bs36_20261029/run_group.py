"""Detached supervisor for one of the two authorized 40k runs; no time limit."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

ROOT = Path('/home/czl/openpi')
PARENT = Path('/mnt/SharedData/czl/runs/dual40k_nostate_bs36_20261029')
GROUPS = {
    'nominal': {
        'config': 'pi05_franka_nominal100_action7_nostate_3gpu_bs36_40k',
        'experiment': 'Pi05_nominal100_action7_NoState_LoRA_3gpu_bs36_40k_20261029',
        'repo': 'caozilin/franka_100_nominal',
        'assets': '/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7',
        'gpu_ids': '0,1,2',
    },
    'joint': {
        'config': 'pi05_tasktol_mix20_joint32_nostate_4xloss_3gpu_bs36_40k',
        'experiment': 'TaskTol_mix20_joint13_4xActionLoss_NoState_LoRA_3gpu_bs36_40k_20261029',
        'repo': 'caozilin/franka_nominal80_tolerance20_potpost001_dsp16_20261027',
        'assets': '/mnt/SharedData/czl/assets/pi05_franka_mujoco_joint13',
        'gpu_ids': '4,5,6',
    },
}


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('group', choices=GROUPS)
    args.add_argument('--start-at', type=float, default=0.)
    args = args.parse_args()
    group = GROUPS[args.group]
    run = PARENT / args.group
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = json.loads((PARENT / 'preflight_result.json').read_text())
        assert preflight['passed']
        checked = preflight['groups'][args.group]
        assert checked['config'] == group['config']
        norm = Path(group['assets']) / group['repo'] / 'norm_stats.json'
        assert hashlib.sha256(norm.read_bytes()).hexdigest() == checked['norm_sha256']
        manifest = json.loads((PARENT / 'deployment_manifest.json').read_text())
        for relative, digest in manifest['files'].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
        checkpoint = Path('/mnt/SharedData/czl/checkpoints') / group['config'] / group['experiment']
        assert not checkpoint.exists(), f'Existing checkpoint directory: {checkpoint}'
        assert shutil.disk_usage('/mnt/SharedData').free > 200 * 1024**3
        for relative in ('src/openpi/training/config.py', 'src/openpi/training/optimizer.py',
                         'src/openpi/training/weight_loaders.py', 'src/openpi/training/data_loader.py',
                         'src/openpi/training/experiment_tracking.py', 'src/openpi/training/sharding.py',
                         'src/openpi/models/pi0.py', 'src/openpi/models/pi0_config.py',
                         'src/openpi/models/tokenizer.py', 'src/openpi/policies/franka_mujoco_policy.py',
                         'src/openpi/transforms.py', 'scripts/train.py'):
            target = run / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        shutil.copy2(norm, run / 'norm_stats.json')
        cache = str(Path('/mnt/SharedData/czl/hf_datasets') / group['repo'])
        command = [str(ROOT / '.venv/bin/python'), '-u', 'scripts/train.py', group['config'],
                   '--exp-name', group['experiment'],
                   '--data.lerobot-dataset-root', str(Path('/mnt/SharedData/czl/lerobot') / group['repo']),
                   '--data.hf-datasets-cache-dir', cache,
                   '--data.assets.assets-dir', group['assets'],
                   '--checkpoint-base-dir', '/mnt/SharedData/czl/checkpoints',
                   '--swanlab-log-dir', '/mnt/SharedData/czl/swanlog']
        env = os.environ.copy()
        env.pop('JAX_PLATFORMS', None)
        env.update(CUDA_VISIBLE_DEVICES=group['gpu_ids'], XLA_PYTHON_CLIENT_MEM_FRACTION='0.95',
                   PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                   HF_ENDPOINT='https://hf-mirror.com', HF_DATASETS_CACHE=cache)
        cfg = checked['resolved_config']
        cfg['exp_name'] = group['experiment']
        cfg['checkpoint_base_dir'] = '/mnt/SharedData/czl/checkpoints'
        cfg['swanlab_log_dir'] = '/mnt/SharedData/czl/swanlog'
        write(run / 'actual_config.json', cfg)
        write(run / 'training_command.json', {
            'argv': command, 'group': args.group, 'gpu_ids': group['gpu_ids'],
            'num_gpus': 3, 'global_batch': 36, 'per_gpu_batch': 12, 'num_train_steps': 40000,
            'save_interval': 5000, 'keep_period': 5000, 'resume': False,
            'norm_sha256': checked['norm_sha256'], 'checkpoint_dir': str(checkpoint),
            'init_source': 'gs://openpi-assets/checkpoints/pi05_base/params',
            'common_start_requested_unix': args.start_at,
            'local_shutdown_safe': True, 'time_limit_seconds': None,
        })
        while time.time() < args.start_at:
            time.sleep(min(1., max(0., args.start_at - time.time())))
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        readings = {int(line.split(',')[0]): [int(v) for v in line.split(',')[1:]] for line in gpu.strip().splitlines()}
        assert all(readings[int(i)][0] < 1000 and readings[int(i)][1] < 10 for i in group['gpu_ids'].split(',')), gpu
        started = time.time()
        state = {'state': 'starting', 'group': args.group, 'launch_unix': started,
                 'started_shanghai': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                 'first_completed_step_unix': None, 'last_logged_step': 0,
                 'checkpoint_dir': str(checkpoint), 'supervisor_pid': os.getpid()}
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        state['pid'] = process.pid
        write(run / 'status.json', state)
        stop_monitor = threading.Event()

        def monitor():
            with (run / 'resources.jsonl').open('w') as output:
                while not stop_monitor.is_set():
                    try:
                        gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw',
                                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
                        output.write(json.dumps({'unix': time.time(), 'gpu_readings': gpus,
                                                 'load_average': os.getloadavg(),
                                                 'shared_disk_free_bytes': shutil.disk_usage('/mnt/SharedData').free}) + '\n')
                        output.flush()
                    except Exception as error:
                        output.write(json.dumps({'unix': time.time(), 'monitor_error': repr(error)}) + '\n')
                        output.flush()
                    stop_monitor.wait(60)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        with (run / 'train.log').open('w') as log, (run / 'events.jsonl').open('w') as events:
            for line in process.stdout:
                now = time.time()
                log.write(line)
                log.flush()
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                progress = re.search(r'Progress on:\s+([0-9.]+)(k?)it/', clean)
                if progress and state['first_completed_step_unix'] is None:
                    state.update(state='training', first_completed_step_unix=now)
                    write(run / 'status.json', state)
                metric = re.search(r'Step\s+(\d+): ([^\n]+)', clean)
                if metric:
                    step = int(metric.group(1))
                    values = {k: float(v) for k, v in re.findall(r'([\w/]+)=([-+0-9.eE]+)', metric.group(2))}
                    state.update(last_logged_step=step, last_metrics=values, last_metric_unix=now)
                    events.write(json.dumps({'unix': now, 'step': step, 'samples': step * 36, 'metrics': values}) + '\n')
                    events.flush()
                    write(run / 'status.json', state)
                if 'https://swanlab.cn/' in clean:
                    events.write(json.dumps({'unix': now, 'tracking': clean.strip()}) + '\n')
                    events.flush()
        result = process.wait()
        stop_monitor.set()
        thread.join(timeout=12)
        state.update(state='completed' if result == 0 else 'failed', returncode=result,
                     finished_unix=time.time(), wall_seconds=time.time() - started)
        write(run / 'status.json', state)
        write(run / 'exit_status.json', state)
        print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
