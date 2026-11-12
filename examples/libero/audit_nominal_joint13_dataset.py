"""Independent byte/label/source audit and actual LeRobot reads for enriched nominal data."""
import argparse,hashlib,io,json,os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openpi.shared.libero_joint_dataset import validate_info,TOLERANCE_ORDER

def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def audit(root):
 info=read(root/'meta/info.json');validate_info(info);request=read(root/'conversion_request.json');src=Path(request['source_root']);completion=read(root/'conversion_completion.json');assert completion['complete']
 assert completion['request_sha256']==sha(root/'conversion_request.json')
 assert sha(src/'meta/reconstruction_sources.json')==request['source_index_sha256'];assert sha(src/'meta/info.json')==request['source_info_sha256']
 assert len(request['episodes'])==1875;total=0;selected={};audits_checked={}
 for entry in request['episodes']:
  s=entry['source'];i=s['episode_index'];relative=info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i);new=pq.read_table(root/relative);old=pq.read_table(src/relative)
  receipt=read(root/'receipts'/f'{i:06d}.json');assert sha(src/relative)==receipt['source_sha256'];assert sha(root/relative)==receipt['output_sha256']
  assert new.column_names==old.column_names+['rotation_tolerance','trajectory_is_tolerance']
  for name in old.column_names:assert new[name].equals(old[name]),(i,name)
  label_path=Path(entry['label']['path']);assert sha(label_path)==entry['label']['sha256']==receipt['label_sha256'];label=read(label_path)
  for key in ('episode_index','suite','task_id','demo','frames','recording_sha256'):assert label['source'][key]==s[key],(i,key)
  ap=Path(entry['label']['audit'])
  if str(ap) not in audits_checked:
   assert sha(ap)==entry['label']['audit_sha256'];a=read(ap);assert a['complete'];audits_checked[str(ap)]=a
  a=audits_checked[str(ap)];ar=next(r for r in a['results'] if (r['suite'],r['task_id'],r['demo'])==(s['suite'],s['task_id'],s['demo']));assert ar['status']=='labeled' and ar['labels_sha256']==sha(label_path)
  assert label['units']=='rad' and label['order']==list(TOLERANCE_ORDER) and label['control_hz']==20
  expected=np.asarray([f['rotation_tolerance_rad'] for f in label['frames']],np.float32);actual=np.array(new['rotation_tolerance'].to_pylist(),np.float32)
  np.testing.assert_array_equal(actual,expected);assert expected.shape==(s['frames'],6);assert np.isfinite(expected).all() and (expected>=0).all() and (expected<=np.float32(np.pi/6)+1e-7).all()
  assert [f['source_frame_index'] for f in label['frames']]==list(range(s['frames']))
  for f,v in zip(label['frames'],expected,strict=True):
   if f['phase'] in ('grasp','release'):assert not np.any(v)
  assert new['trajectory_is_tolerance'].to_pylist()==[0]*s['frames'];assert entry['trajectory_is_tolerance']==0
  assert new.num_rows==old.num_rows==s['frames'];total+=new.num_rows;selected.setdefault((s['suite'],s['task_id']),i)
  if (i+1)%50==0:write(root/'independent_audit_state.json',dict(stage='byte_and_label_audit',processed=i+1,total=1875));print(f'Audited {i+1}/1875',flush=True)
 assert total==312374 and len(selected)==40
 for name in ('episodes.jsonl','tasks.jsonl','reconstruction_sources.json'):assert sha(root/'meta'/name)==sha(src/'meta'/name)
 write(root/'independent_audit_state.json',dict(stage='actual_lerobot_reads',processed=1875,total=1875))
 import torch
 from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
 # LeRobotv2.1 builds compact offsets for selected episodes but indexes them
 # with original episode IDs. Map IDs only in this read-only audit subclass.
 class SelectedEpisodeReader(LeRobotDataset):
  def _get_query_indices(self, idx, ep_idx):
   return super()._get_query_indices(idx, self.episodes.index(ep_idx))
 ds=SelectedEpisodeReader('caozilin/libero_reconstructed_nominal_joint13_20261109',root=root,episodes=list(selected.values()),delta_timestamps={k:[j/20 for j in range(10)] for k in ('actions','rotation_tolerance')})
 from openpi.policies.libero_joint_policy import LiberoJointInputs
 from openpi.models.model import ModelType
 checks=[]
 for j,i in enumerate(selected.values()):
  entry=request['episodes'][i];s=entry['source'];start=int(ds.episode_data_index['from'][j]);end=int(ds.episode_data_index['to'][j]);assert end-start==s['frames']
  rel=info['data_path'].format(episode_chunk=i//1000,episode_index=i);orig=pq.read_table(src/rel);lab=read(Path(entry['label']['path']))
  for k in (0,(end-start)//2,end-start-1):
   row=ds[start+k];assert int(row['episode_index'])==i
   packed=LiberoJointInputs(ModelType.PI05)({'observation/image':row['image'],'observation/wrist_image':row['wrist_image'],'observation/state':row['state'],'actions':row['actions'],'rotation_tolerance':row['rotation_tolerance'],'trajectory_is_tolerance':row['trajectory_is_tolerance'],'prompt':'audit'})
   assert packed['actions'].shape==(10,13);np.testing.assert_array_equal(packed['actions'][0,:7],np.array(orig['actions'][k].as_py(),np.float32));np.testing.assert_array_equal(packed['actions'][0,7:],np.array(lab['frames'][k]['rotation_tolerance_rad'],np.float32))
   for camera in ('image','wrist_image'):
    pixels=np.asarray(Image.open(io.BytesIO(orig[camera][k].as_py()['bytes'])))
    np.testing.assert_array_equal(np.rint(row[camera].numpy().transpose(1,2,0)*255).astype(np.uint8),pixels)
  checks.append(dict(episode_index=i,suite=s['suite'],task_id=s['task_id'],indices=[0,(end-start)//2,end-start-1]))
 result=dict(complete=True,episodes=1875,frames=total,tasks=40,fps=20,all_original_columns_identical=True,all_embedded_png_bytes_identical=True,all_labels_equal_audited_search_outputs=True,all_nominal=True,portable_lerobot_reader_passed=True,reader_checks=checks,request_sha256=sha(root/'conversion_request.json'),manifest_sha256=sha(root/'conversion_manifest.json'),missing_labels=0)
 write(root/'meta/joint13_independent_audit.json',result);write(root/'independent_audit_state.json',dict(stage='complete',processed=1875,total=1875));return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();print(audit(a.dataset),flush=True)
