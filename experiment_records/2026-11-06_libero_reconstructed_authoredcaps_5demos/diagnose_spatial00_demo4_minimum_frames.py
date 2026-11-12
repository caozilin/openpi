from pathlib import Path
import sys,json,datetime,hashlib
import numpy as np
import mujoco
from unittest.mock import patch
ROOT=Path('/media/czl/sata/franka_my_code/franka_mujoco');sys.path.insert(0,str(ROOT))
from tasktol.tasks.evaluation.predicates.contact import RobotTableClearanceRecorder
RUN=ROOT/'analysis_results/libero_reconstructed_authoredcaps_dsp16_first5pertask_20261106'
FOLDER=RUN/'libero_spatial/task_00/demo_4'
read=lambda p:json.loads(Path(p).read_text())
doc=read(FOLDER/'prepared_input.json');rows=read(Path(doc['source']['folder'])/'trajectory.json')
model=mujoco.MjModel.from_xml_string(doc['metadata']['robot_xml']);data=mujoco.MjData(model)
recorder=RobotTableClearanceRecorder(model,robot_root_name='robot0_link0',table_body_name='work_surface')
series={}
with patch('mujoco.mj_step',side_effect=AssertionError('Read-only FK; physics forbidden')):
 for kind in ['pre_action','post_action','recorded_joint_target']:
  values=[]
  for i,row in enumerate(rows):
   pose=row['pre_action'] if kind=='pre_action' else row['post_action']
   q=row['joint_target_rad'] if kind=='recorded_joint_target' else pose['joint_position_rad']
   # The recorded commanded joint target is evaluated with the pre-action finger state,
   # matching the offline annotator's frozen finger sampling.
   fingers=row['pre_action']['finger_qpos_m'] if kind=='recorded_joint_target' else pose['finger_qpos_m']
   for j,v in enumerate(q,1):data.qpos[model.joint(f'robot0_joint{j}').qposadr[0]]=v
   for j,v in enumerate(fingers,1):data.qpos[model.joint(f'gripper0_finger_joint{j}').qposadr[0]]=v
   mujoco.mj_forward(model,data)
   distance,robot_geom,table_geom=recorder.minimum(data)
   values.append(dict(source_action_index=i,video_frame_index=i if kind=='pre_action' else i+1 if kind=='post_action' else None,
     time_s=i/20 if kind=='pre_action' else (i+1)/20,clearance_mm=distance*1000,
     robot_geom=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,robot_geom),
     robot_body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[robot_geom])),
     phase=row['observed_phase']))
  series[kind]=values
annotations=read(FOLDER/'subtask_00/annotation.json')['observations']
zero=next(x for x in annotations if not any(x['actual_pre_caps_deg']+x['actual_post_caps_deg']))
raw_path=Path(zero['raw_trial_path']);assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==zero['raw_trial_sha256']
zero_min=read(raw_path)['result']['evaluation']['robot_table_clearance']['minimum']
local=zero_min['frame_index'];source_index=doc['provenance']['source_segmentation'][0]['source_frames'][local]
result=dict(complete=True,checked_at=datetime.datetime.now().astimezone().isoformat(),physics_steps=0,
 robot_xml_sha256=hashlib.sha256(doc['metadata']['robot_xml'].encode()).hexdigest(),tabletop=doc['metadata']['tabletop'],
 source_folder=doc['source']['folder'],source_frames=len(rows),frame_indexing='zero-based; video samples pre-action',
 minima={kind:min(values,key=lambda x:x['clearance_mm']) for kind,values in series.items()},
 offline_zero_caps_ik_minimum=dict(source_action_index=source_index,action_start_s=source_index/20,action_end_s=(source_index+1)/20,
   clearance_mm=zero_min['distance_m']*1000,robot_body=zero_min['robot_body'],robot_geom=zero_min['robot_geom'],phase=zero_min['phase'],
   raw_trial=str(raw_path),raw_trial_sha256=zero['raw_trial_sha256'],raw_time_s_invalid_for_physical_timeline=zero_min['time_s']),
 actual_at_offline_reference_minimum={kind:values[source_index] for kind,values in series.items()},per_frame=series)
output=Path('/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-06_libero_reconstructed_authoredcaps_5demos/spatial00_demo4_clearance_frames.json')
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['per_frame','tabletop']},ensure_ascii=False,indent=2))
