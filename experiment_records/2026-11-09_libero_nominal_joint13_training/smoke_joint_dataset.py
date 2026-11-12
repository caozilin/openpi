from pathlib import Path
import copy,json,shutil
import numpy as np
import pyarrow.parquet as pq
from examples.libero.enrich_nominal_joint13_dataset import read,append_columns,write,feature_stats
from openpi.shared.libero_joint_dataset import SOURCE_SCHEMA,TOLERANCE_ORDER,validate_info
from openpi.policies.libero_joint_policy import LiberoJointInputs
from openpi.models.model import ModelType

def main():
 root=Path('/media/czl/sata/franka_my_code/openpi');src=root/'data/lerobot/caozilin/libero_reconstructed_full_20261101'
 out=Path('/tmp/libero_joint13_real_reader_smoke_20261109');out.mkdir(exist_ok=True);(out/'meta').mkdir(exist_ok=True)
 labels=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107/libero_spatial/task_00/demo_0/tolerance_labels.json')
 source=read(src/'meta/reconstruction_sources.json')['sources'][0];table=pq.read_table(src/'data/chunk-000/episode_000000.parquet');enriched,v=append_columns(table,source,read(labels))
 (out/'data/chunk-000').mkdir(parents=True,exist_ok=True);pq.write_table(enriched,out/'data/chunk-000/episode_000000.parquet',compression='snappy')
 for name in ('episodes.jsonl','tasks.jsonl','episodes_stats.jsonl'):shutil.copyfile(src/'meta'/name,out/'meta'/name)
 info=read(src/'meta/info.json');info['source_schema']=SOURCE_SCHEMA;info['features'].update(rotation_tolerance=dict(dtype='float32',shape=[6],names=list(TOLERANCE_ORDER)),trajectory_is_tolerance=dict(dtype='int64',shape=[1],names=['trajectory_is_tolerance']))
 info['joint13_contract']=dict(rotation_tolerance_units='rad',rotation_tolerance_order=list(TOLERANCE_ORDER),trajectory_types={'0':'nominal','1':'tolerance'});validate_info(info);write(out/'meta/info.json',info)
 # Import torch before Arrow/LeRobot to retain the established allocator order.
 import torch
 from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
 ds=LeRobotDataset('caozilin/libero_joint13_smoke',root=out,episodes=[0],delta_timestamps={k:[i/20 for i in range(10)] for k in ['actions','rotation_tolerance']})
 for i in (0,40,96):
  row=ds[i];data={'observation/image':row['image'],'observation/wrist_image':row['wrist_image'],'observation/state':row['state'],'actions':row['actions'],'rotation_tolerance':row['rotation_tolerance'],'trajectory_is_tolerance':row['trajectory_is_tolerance'],'prompt':'pick up bowl'}
  packed=LiberoJointInputs(ModelType.PI05)(data);assert packed['actions'].shape==(10,13)
  np.testing.assert_allclose(packed['actions'][0,7:],v[i],atol=0,rtol=0)
  expected=np.array(table['actions'].to_pylist()[i],dtype=np.float32);np.testing.assert_array_equal(packed['actions'][0,:7],expected)
  assert row['image'].shape==(3,256,256)
  assert int(row['trajectory_is_tolerance'].item())==0
 write(root/'experiment_records/2026-11-09_libero_nominal_joint13_training/real_reader_smoke.json',dict(complete=True,episode=0,frames=len(ds),checked_indices=[0,40,96],action_horizon=10,state_dim=8,effective_target_dim=13,source_png_bytes_unchanged=True,cache=str(out),all_nominal=True))
 print('Real LeRobot read + joint13 transform passed',flush=True)
if __name__=='__main__':main()
