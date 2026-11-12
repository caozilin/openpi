from pathlib import Path
import json,io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import imageio.v2 as imageio
r=Path('/media/czl/sata/franka_my_code/openpi');d=r/'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109';req=json.loads((d/'replacement_request.json').read_text());e=req['entries'][0];i=e['episode_index'];rel=f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet';table=pq.read_table(d/rel);old=pq.read_table(r/'data/lerobot/caozilin/libero_reconstructed_full_20261101'/rel);folder=Path(e['replay']['folder']);trace=json.loads((folder/'trajectory.json').read_text());n=e['source']['frames']
for name in ['actions','timestamp','frame_index','episode_index','index','task_index']:assert table[name].equals(old[name])
assert table['trajectory_is_tolerance'].to_pylist()==[1]*n
expected=np.array([t['pre_action']['position_m']+t['pre_action']['rotation_vector_rad']+t['pre_action']['finger_qpos_m'] for t in trace],np.float32);np.testing.assert_array_equal(np.array(table['state'].to_pylist(),np.float32),expected)
for camera,video in [('image','main_rgb.mp4'),('wrist_image','wrist_rgb.mp4')]:
 reader=imageio.get_reader(str(folder/video),format='ffmpeg');count=0
 try:
  for j,pixels in enumerate(reader):np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(table[camera][j].as_py()['bytes']))),pixels);count+=1
 finally:reader.close()
 assert count==n
v=dict(complete=True,episode_index=i,frames=n,actions_unchanged=True,identity_time_language_indices_unchanged=True,preaction_states_equal=True,all_dual_video_frames_equal_decoded_mp4=True,rotation_not_applied_twice=True,trajectory_is_tolerance=1,scope='one independent pilot; full301/1875 audit still required')
(r/'experiment_records/2026-11-09_libero_joint13_successmix20_training/pilot_independent_audit.json').write_text(json.dumps(v,indent=2)+'\n');print(v)
