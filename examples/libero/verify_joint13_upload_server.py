"""Read-only remote verification: manifest, original nominal columns, actual LeRobot."""
import hashlib,json,os,sys
from pathlib import Path
RUN=Path('/mnt/SharedData/czl/runs/pi05_libero_nominal_joint13_lora_6gpu_bs72_10k_20261109')
NEW=Path('/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109')
OLD=Path('/mnt/SharedData/czl/lerobot/caozilin/libero_reconstructed_full_20261101')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 import torch
 import pyarrow.parquet as pq
 import numpy as np
 sys.path.insert(0,str(RUN/'source_overlay'))
 from openpi.shared.libero_joint_dataset import validate_info
 manifest=read(NEW/'transfer_manifest.json');assert manifest['episodes']==1875 and manifest['frames']==312374
 for i,e in enumerate(manifest['files']):
  p=NEW/e['path'];assert p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],e['path']
  if i%100==0:write(RUN/'upload_verification_state.json',dict(stage='sha256',checked=i,total=len(manifest['files'])))
 info=read(NEW/'meta/info.json');validate_info(info);audit=read(NEW/'meta/joint13_independent_audit.json');assert audit['complete'] and audit['episodes']==1875 and audit['all_nominal'] and audit['missing_labels']==0
 sources=read(NEW/'meta/reconstruction_sources.json')['sources'];frames=0
 for i,s in enumerate(sources):
  rel=info['data_path'].format(episode_chunk=i//1000,episode_index=i);new=pq.read_table(NEW/rel);old=pq.read_table(OLD/rel)
  assert new.num_rows==old.num_rows==s['frames']
  for col in old.column_names:assert new[col].equals(old[col]),(i,col)
  values=np.asarray(new['rotation_tolerance'].to_pylist(),np.float32);assert values.shape==(s['frames'],6) and np.isfinite(values).all() and (values>=0).all() and (values<=np.float32(np.pi/6)+1e-7).all()
  assert new['trajectory_is_tolerance'].to_pylist()==[0]*s['frames'];frames+=s['frames']
  if i%100==0:write(RUN/'upload_verification_state.json',dict(stage='original_nominal_columns',checked=i,total=1875))
 assert frames==312374
 from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
 # LeRobotv2.1 builds compact offsets for selected episodes but indexes them
 # with original episode IDs. Map IDs only in this read-only audit subclass.
 class SelectedEpisodeReader(LeRobotDataset):
  def _get_query_indices(self, idx, ep_idx):
   return super()._get_query_indices(idx, self.episodes.index(ep_idx))
 episodes=[];suites=set()
 for s in sources:
  if s['suite'] not in suites:episodes.append(s['episode_index']);suites.add(s['suite'])
 ds=SelectedEpisodeReader('caozilin/libero_reconstructed_nominal_joint13_20261109',root=NEW,episodes=episodes,delta_timestamps={k:[i/20 for i in range(10)] for k in ('actions','rotation_tolerance')})
 for j,ep in enumerate(episodes):
  offset=int(ds.episode_data_index['from'][j]);row=ds[offset];assert row['actions'].shape==(10,7) and row['rotation_tolerance'].shape==(10,6) and row['state'].shape==(8,) and row['image'].shape==(3,256,256);assert int(row['episode_index'])==ep and int(row['trajectory_is_tolerance'])==0
 result=dict(complete=True,episodes=1875,frames=312374,tasks=40,fps=20,files=len(manifest['files']),all_uploaded_sha256_equal=True,all_original_nominal_columns_equal=True,portable_lerobot_reader_passed=True,all_nominal=True,missing_labels=0,reader_episodes=episodes,transfer_manifest_sha256=sha(NEW/'transfer_manifest.json'),local_independent_audit_sha256=sha(NEW/'meta/joint13_independent_audit.json'))
 write(RUN/'lerobot_upload_verification.json',result);write(RUN/'upload_verification_state.json',dict(stage='complete',**result));print(json.dumps(result))
if __name__=='__main__':main()
