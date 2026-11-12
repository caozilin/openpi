"""Read actual LIBERO examples through the new input/normalization/model observation path on CPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',ARROW_DEFAULT_MEMORY_POOL='system')
import argparse,importlib.util,json,hashlib,sys,dataclasses
from pathlib import Path

def main(run):
 sys.path[:0]=[str(run/'source_overlay'),'/home/czl/openpi']
 spec=importlib.util.spec_from_file_location('four_x_training_entry',run/'train_successmix20_joint13_pi05_server.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
 cfg=entry.child(run,'config')
 import torch,pyarrow,numpy as np,jax
 from openpi import transforms
 from openpi.training import data_loader
 from openpi.models import model,pi0
 from openpi.policies.libero_joint_policy import LiberoJointInputs
 dc=cfg.data.create(cfg.assets_dirs,cfg.model)
 assert dc.norm_stats['state'].mean.shape==(8,) and dc.norm_stats['actions'].mean.shape==(13,)
 # Use the already built Arrow cache directly for this read-only check.
 dc=dataclasses.replace(dc,lerobot_arrow_cache_dir=dc.hf_datasets_cache_dir)
 raw=data_loader.create_torch_dataset(dc,10,cfg.model);transformed=data_loader.transform_dataset(raw,dc)
 episodes=[json.loads(x) for x in (entry.DATASET/'meta/episodes.jsonl').read_text().splitlines()]
 offsets=[];at=0
 for episode in episodes:offsets.append(at);at+=episode['length']
 checks=[];samples=[]
 # The frozen selection proves0 nominal and2 tolerance; check two frames per type.
 for episode in (0,2):
  for delta in (0,5):
   index=offsets[episode]+delta;original=raw[index];flag=int(np.asarray(original['trajectory_is_tolerance']).reshape(-1)[0]);expected=1.0+3*flag
   repacked=transforms.compose(dc.repack_transforms.inputs)(original)
   before=LiberoJointInputs(cfg.model.model_type,1.0)(repacked)
   after=LiberoJointInputs(cfg.model.model_type,4.0)(repacked)
   for key in ('state','actions'):np.testing.assert_array_equal(before[key],after[key])
   item=transformed[index];assert float(item['action_loss_weight'])==expected
   assert item['actions'].shape==(10,32) and np.isfinite(item['actions']).all() and np.all(item['actions'][:,13:]==0)
   samples.append(item);checks.append(dict(episode=episode,frame=delta,flag=flag,weight=float(item['action_loss_weight']),normalized_action_sha256=hashlib.sha256(item['actions'].tobytes()).hexdigest()))
 batch=jax.tree.map(lambda *items:np.stack(items),*samples);obs=model.Observation.from_dict(batch)
 assert set(np.asarray(obs.action_loss_weight).tolist())=={1.0,4.0}
 errors=jax.numpy.ones((4,10,32));errors=errors.at[...,13:].set(1e6)
 values,parts=jax.jit(lambda e:pi0._grouped_action_loss_with_breakdown(e,cfg.model.action_loss_groups,obs.action_loss_weight))(errors)
 assert np.allclose(np.asarray(values)[:,0],[1.5,1.5,4.5,4.5])
 parts={key:float(np.asarray(value).mean()) for key,value in parts.items()}
 assert parts['loss_weighted/action']==2.5 and parts['loss_weighted/rotation_tolerance']==0.5
 assert parts['loss/action_nominal_contribution']==0.5 and parts['loss/action_reweighted_contribution']==0.5
 request=json.loads((run/'training_request.json').read_text());assert cfg.swanlab_enabled and cfg.swanlab_mode=='online'
 result=dict(complete=True,scope='CPU actual LeRobot horizon10/repack/state8/target13/normalize/pad32/Observation/JIT loss; no optimizer step',checks=checks,nominal_actions_and_labels_unchanged=True,normalization_targets_equal_before_after=True,swanlab_enabled=True,swanlab_mode='online',config_weight=cfg.data.tolerance_trajectory_action_loss_weight,metrics=parts,request_sha256=hashlib.sha256((run/'training_request.json').read_bytes()).hexdigest())
 p=run/'four_x_protocol_update/training_preflight_verification.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);main(p.parse_args().run_dir)
