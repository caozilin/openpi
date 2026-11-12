"""Launch the authorized 7-GPU run after both preflight checks succeed."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/home/czl/openpi')
RUN=Path('/mnt/SharedData/czl/runs/pi05_state_action7_bs84_15k_20261028')
NAME='pi05_franka_mujoco_state_action7_lora'
EXP='Pi05_nominal100_state_action7_LoRA_7gpu_bs84_15k_lr5e-5_20261028'
CKPT=Path('/mnt/SharedData/czl/checkpoints')/NAME/EXP

def write(name,data):
    p=RUN/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

def main():
    with (RUN/'.launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for check in ('preflight_result.json','data_preflight_result.json'):
            result=json.loads((RUN/check).read_text())
            assert result['passed'],(check,result)
        assert not CKPT.exists(),f'Checkpoint directory already exists: {CKPT}'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        rows=[row.split(',') for row in gpu.strip().splitlines()]
        assert len(rows)==7 and all(int(row[1])<500 for row in rows),gpu
        assert shutil.disk_usage('/mnt/SharedData').free>200*1024**3
        command=[str(ROOT/'.venv/bin/python'),'-u','scripts/train.py',NAME,
            '--exp-name',EXP,
            '--data.lerobot-dataset-root','/mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal',
            '--data.hf-datasets-cache-dir','/mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal',
            '--data.assets.assets-dir','/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7',
            '--checkpoint-base-dir','/mnt/SharedData/czl/checkpoints',
            '--swanlab-log-dir','/mnt/SharedData/czl/swanlog']
        env=os.environ.copy()
        env.pop('JAX_PLATFORMS',None)
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6',XLA_PYTHON_CLIENT_MEM_FRACTION='0.95',
            PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
            HF_ENDPOINT='https://hf-mirror.com',HF_DATASETS_CACHE='/mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal')
        files=['src/openpi/training/config.py','src/openpi/training/optimizer.py',
            'src/openpi/training/weight_loaders.py','src/openpi/training/data_loader.py',
            'src/openpi/training/experiment_tracking.py','src/openpi/models/pi0.py',
            'src/openpi/models/pi0_config.py','src/openpi/models/tokenizer.py',
            'src/openpi/policies/franka_mujoco_policy.py','src/openpi/transforms.py','scripts/train.py']
        hashes={}
        for f in files:
            p=RUN/'sources'/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,p)
            hashes[f]=hashlib.sha256(p.read_bytes()).hexdigest()
        norm=Path('/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7/caozilin/franka_100_nominal/norm_stats.json')
        shutil.copy2(norm,RUN/'norm_stats.json')
        plan={'argv':command,'config':NAME,'experiment':EXP,'checkpoint_dir':str(CKPT),
            'global_batch':84,'per_gpu_batch':12,'num_gpus':7,'num_train_steps':15000,
            'state':True,'action_dim':32,'effective_loss_dims':7,'max_token_len':200,'action_horizon':16,
            'learning_rate':{'warmup_steps':1000,'peak_lr':5e-5,'decay_steps':15000,'decay_lr':5e-6},
            'save_interval':1500,'keep_period':1500,'log_interval':10,'project':'openpi-tasktol',
            'resume':False,'norm_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
            'source_sha256':hashes,'gpu_before':gpu}
        write('training_command.json',plan)
        with (RUN/'train.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write('process.json',{'pid':process.pid,'supervisor_pid':os.getpid(),'started_unix':time.time(),
                'started_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()})
            write('status.json',{'state':'running','pid':process.pid,'checkpoint_dir':str(CKPT)})
            result=process.wait()
        write('exit_status.json',{'returncode':result,'finished_unix':time.time()})
        write('status.json',{'state':'complete' if result==0 else 'failed','returncode':result,'checkpoint_dir':str(CKPT)})
        sys.exit(result)

if __name__=='__main__':main()
