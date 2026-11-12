"""Check the exact training model at per-device batch 12, without saving updates."""
import dataclasses
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from openpi.models import model
from openpi.training import config, sharding
from scripts import train

RUN = Path('/mnt/SharedData/czl/runs/pi05_state_action7_bs84_15k_20261028')

def write(name, data):
    (RUN/name).write_text(json.dumps(data, indent=2)+'\n')

def main():
    cfg=dataclasses.replace(config.get_config('pi05_franka_mujoco_state_action7_lora'),
        batch_size=12, exp_name='preflight_no_checkpoint', swanlab_enabled=False)
    assert jax.device_count()==1
    mesh=sharding.make_mesh(cfg.fsdp_devices)
    repl=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    data_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rng, init_rng=jax.random.split(jax.random.key(cfg.seed))
    print('Initializing official pretrained weights',flush=True)
    started=time.perf_counter()
    with sharding.set_mesh(mesh):
        state,state_sharding=train.init_train_state(cfg,init_rng,mesh,resume=False)
    jax.block_until_ready(state)
    init_seconds=time.perf_counter()-started
    print('Initialization complete',init_seconds,flush=True)
    write('preflight_status.json',{'stage':'compiling_train_step','init_seconds':init_seconds})
    b=cfg.batch_size
    obs=model.Observation(
        images={k:jnp.zeros((b,224,224,3),jnp.float32) for k in model.IMAGE_KEYS},
        image_masks={k:jnp.full((b,),k!='right_wrist_0_rgb',bool) for k in model.IMAGE_KEYS},
        state=jnp.zeros((b,32),jnp.float32),
        tokenized_prompt=jnp.ones((b,200),jnp.int32),
        tokenized_prompt_mask=jnp.ones((b,200),bool),
        action_loss_weight=jnp.ones((b,),jnp.float32),
    )
    batch=(obs,jnp.zeros((b,16,32),jnp.float32))
    batch=jax.device_put(batch,data_sharding)
    step=jax.jit(functools.partial(train.train_step,cfg),
        in_shardings=(repl,state_sharding,data_sharding),
        out_shardings=(state_sharding,repl),donate_argnums=(1,))
    rows=[]
    for index in range(4):
        t=time.perf_counter()
        with sharding.set_mesh(mesh):state,info=step(rng,state,batch)
        jax.block_until_ready((state,info))
        row={'step':index+1,'seconds':time.perf_counter()-t,'loss':float(info['loss'])}
        rows.append(row);print(json.dumps(row),flush=True)
    memory=jax.local_devices()[0].memory_stats()
    result={'passed':True,'batch_size':12,'action_dim':32,'max_token_len':200,
        'discrete_state_input':True,'action_horizon':16,'loss_groups':cfg.model.action_loss_groups,
        'pretrained_weights_loaded':True,'init_seconds':init_seconds,'steps':rows,
        'memory_bytes':{k:int(memory[k]) for k in ('bytes_in_use','peak_bytes_in_use','bytes_limit') if k in memory},
        'note':'Synthetic observations, real pretrained initialization and train_step; no checkpoint written.'}
    write('preflight_result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('preflight_result.json',{'passed':False,'error':repr(exc)})
        raise
