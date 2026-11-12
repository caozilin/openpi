"""Build the shared dataset cache and check real transformed training samples."""
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
import numpy as np
from openpi.training import config, data_loader

ROOT=Path('/home/czl/openpi')
RUN=Path('/mnt/SharedData/czl/runs/pi05_state_action7_bs84_15k_20261028')
DATASET='/mnt/SharedData/czl/lerobot/caozilin/franka_100_nominal'
CACHE='/mnt/SharedData/czl/hf_datasets/caozilin/franka_100_nominal'
ASSETS='/mnt/SharedData/czl/assets/pi05_franka_mujoco_state_action7'

def main():
    logging.basicConfig(level=logging.INFO)
    base=config.get_config('pi05_franka_mujoco_state_action7_lora')
    data=dataclasses.replace(base.data,lerobot_dataset_root=DATASET,
        hf_datasets_cache_dir=CACHE,assets=config.AssetsConfig(assets_dir=ASSETS))
    dc=data.create(Path(base.assets_base_dir),base.model)
    assert dc.norm_stats is not None
    for key in ('state','actions'):
        for field in ('mean','std','q01','q99'):
            value=np.asarray(getattr(dc.norm_stats[key],field))
            assert value.shape==(7,) and np.isfinite(value).all(),(key,field)
    start=time.perf_counter()
    raw=data_loader.create_torch_dataset(dc,base.model.action_horizon,base.model)
    seconds=time.perf_counter()-start
    dataset=data_loader.transform_dataset(raw,dc)
    checks=[]
    for index in (0,len(dataset)//4,len(dataset)//2,len(dataset)-1):
        item=dataset[index]
        assert item['actions'].shape==(16,32)
        assert item['state'].shape==(32,)
        assert item['tokenized_prompt'].shape==(200,)
        np.testing.assert_array_equal(item['actions'][:,7:],0)
        assert float(item['action_loss_weight'])==1.
        assert np.isfinite(item['state']).all() and np.isfinite(item['actions']).all()
        count=int(np.asarray(item['tokenized_prompt_mask']).sum())
        assert 0<count<200, count
        checks.append({'index':index,'tokens':count})
    norm=Path(ASSETS)/base.data.repo_id/'norm_stats.json'
    result={'passed':True,'frames':len(dataset),'dataset_cache_seconds':seconds,
        'sample_checks':checks,'norm_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
        'cache_dir':CACHE,'dataset_root':DATASET,'norm_path':str(norm)}
    (RUN/'data_preflight_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (RUN/'data_preflight_result.json').write_text(json.dumps({'passed':False,'error':repr(exc)}))
        raise
