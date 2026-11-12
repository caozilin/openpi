"""Six independent inference replicas for local LIBERO evaluation clients."""
import argparse,dataclasses,datetime,fcntl,hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
RUN=Path('/mnt/SharedData/czl/runs/libero_eval_pi05_reconstructed_10000_40tasks_6seeds_20261102')
TRAIN=Path('/mnt/SharedData/czl/runs/pi05_libero_reconstructed_lora_6gpu_bs72_10k_20261101')
CHECKPOINT=Path('/mnt/SharedData/czl/checkpoints/pi05_libero_reconstructed_nostate_6gpu_bs72/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101/10000')
GPUS=[0,1,2,3,4,6];PORTS=list(range(18760,18766));IDENTITY='libero_reconstructed_step10000_40tasks_seed0to5_local_client_20261102'
def dump(name,obj):
 p=RUN/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2,default=str)+'\n');q.replace(p)
def worker(index):
 import torch,pyarrow
 import jax,numpy as np
 from openpi.models import pi0_config
 from openpi.training import config
 from openpi.policies import policy_config
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 spec=json.loads((TRAIN/'training_request.json').read_text())['config']
 model=pi0_config.Pi0Config(**spec['model'])
 cfg=dataclasses.replace(config.get_config('pi05_libero'),name=spec['name'],model=model,freeze_filter=model.get_freeze_filter(),data=config.LeRobotLiberoDataConfig(repo_id=spec['data']['repo_id'],base_config=config.DataConfig(prompt_from_task=True),extra_delta_transform=False),swanlab_enabled=False)
 assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
 policy=policy_config.create_trained_policy(cfg,CHECKPOINT)
 class SeedPolicy:
  def infer(self,obs):
   seed=obs.pop('_libero_eval_noise_seed')
   if type(seed) is not int or not 0<=seed<2**32:raise ValueError('invalid deterministic inference seed')
   noise=np.random.default_rng(seed).standard_normal((10,32),dtype=np.float32)
   return policy.infer(obs,noise=noise)
 metadata={'eval_identity':IDENTITY,'checkpoint':str(CHECKPOINT),'gpu':GPUS[index],'replica':index,'action_units':'xyz metres/world rotvec radians/native gripper','state_enabled':False,'model_action_horizon':10,'replan_steps':5,'noise_seed_protocol':'_libero_eval_noise_seed.v1'}
 dump(f'replica_{index}_ready.json',dict(metadata,pid=os.getpid(),ready_at=datetime.datetime.now().astimezone().isoformat()))
 print('REPLICA READY',metadata,flush=True)
 WebsocketPolicyServer(SeedPolicy(),host='127.0.0.1',port=PORTS[index],metadata=metadata).serve_forever()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);a=ap.parse_args()
 if a.worker is not None:worker(a.worker);return
 RUN.mkdir(exist_ok=True)
 lock=open(RUN/'inference_servers.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((TRAIN/'training_independent_final_audit.json').read_text())['complete']
 for port in PORTS:
  s=socket.socket();s.bind(('127.0.0.1',port));s.close()
 dump('inference_manager_process.json',{'pid':os.getpid(),'command':sys.argv})
 processes=[]
 def stop(signum,frame):
  for p in processes:
   if p.poll() is None:p.terminate()
  for p in processes:
   try:p.wait(timeout=25)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  dump('inference_servers_state.json',{'stage':'stopped','exit_codes':[p.returncode for p in processes]});sys.exit(0)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 for i,gpu in enumerate(GPUS):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.85',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',ARROW_DEFAULT_MEMORY_POOL='system',SWANLAB_MODE='disabled',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH='/home/czl/openpi/src:/home/czl/openpi/packages/openpi-client/src:/home/czl/openpi')
  log=open(RUN/f'replica_{i}.log','a')
  p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(i)],cwd='/home/czl/openpi',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  processes.append(p)
  dump(f'replica_{i}_process.json',{'pid':p.pid,'gpu':gpu,'port':PORTS[i],'command':str(Path(__file__).resolve())})
 dump('inference_servers_state.json',{'stage':'loading','pids':[p.pid for p in processes],'gpus':GPUS,'ports':PORTS,'client_location':'local'})
 while True:
  for i,p in enumerate(processes):
   if p.poll() is not None:
    dump('inference_servers_state.json',{'stage':'failed','replica':i,'exit_code':p.returncode});raise RuntimeError(f'replica {i} exited {p.returncode}')
  if all((RUN/f'replica_{i}_ready.json').exists() for i in range(6)):
   dump('inference_servers_state.json',{'stage':'serving','pids':[p.pid for p in processes],'gpus':GPUS,'ports':PORTS,'client_location':'local'})
  time.sleep(10)
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();sys.exit(1)
