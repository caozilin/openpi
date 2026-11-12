"""Independent all-frame audit for per-task successful-pool20% replacement data."""
import argparse,collections,hashlib,io,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from examples.libero.enrich_nominal_joint13_dataset import read,sha,write
from openpi.shared.libero_joint_dataset import validate_info

def audit(output):
    import imageio.v2 as imageio
    req=read(output/'replacement_request.json');info=read(output/'meta/info.json');validate_info(info)
    replay=Path(req['replay_root']);assert sha(replay/'successful_sources.json')==req['replay_success_index_sha256'];assert sha(replay/'independent_final_audit.json')==req['replay_audit_sha256']
    replay_audit=read(replay/'independent_final_audit.json');assert replay_audit['complete'] and replay_audit['success']==1503
    assert read(output/'conversion_completion.json')['complete']
    base=Path(req['base']);nominal_root=Path('/media/czl/sata/franka_my_code/openpi/data/lerobot/caozilin/libero_reconstructed_full_20261101')
    sources=read(nominal_root/'meta/reconstruction_sources.json')['sources'];selected={e['episode_index']:e for e in req['entries']}
    assert len(selected)==301 and len(sources)==1875 and req['nominal_episodes']==1574
    from examples.libero.build_successmix20_joint13 import select_per_task
    pool=read(Path(req['replay_root'])/'successful_sources.json');picked,tasks=select_per_task(pool['sources'],req['random_seed']);assert tasks==req['tasks']
    assert {(r['suite'],r['task_id'],r['demo']) for r in picked}=={(e['source']['suite'],e['source']['task_id'],e['source']['demo']) for e in req['entries']}
    conversion=read(output/'conversion_manifest.json');byid={r['episode_index']:r for r in conversion['episodes']}
    tolerance_sources=read(output/'meta/tolerance_sources.json')['episodes'];checks=[];frames=0;reader_selection={};label_audits={}
    for s in sources:
        i=s['episode_index'];rel=f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet';p=output/rel;t=pq.read_table(p);nominal=pq.read_table(nominal_root/rel);flag=1 if i in selected else 0
        assert sha(p)==byid[i]['output_sha256'];assert t.num_rows==nominal.num_rows==s['frames']
        for name in ['actions','timestamp','frame_index','episode_index','index','task_index']:assert t[name].equals(nominal[name]),(i,name)
        assert t['trajectory_is_tolerance'].to_pylist()==[flag]*s['frames']
        assert tolerance_sources[i]['trajectory_is_tolerance']==flag
        labelinfo=tolerance_sources[i]['label'];lp=Path(labelinfo['path']);assert sha(lp)==labelinfo['sha256'];lab=read(lp);values=np.asarray([f['rotation_tolerance_rad'] for f in lab['frames']],np.float32)
        ap=Path(labelinfo['audit'])
        if str(ap) not in label_audits:
            assert sha(ap)==labelinfo['audit_sha256'];a=read(ap);assert a['complete'];label_audits[str(ap)]={(r['suite'],r['task_id'],r['demo']):r for r in a['results']}
        a=label_audits[str(ap)][s['suite'],s['task_id'],s['demo']];assert a['status']=='labeled' and a['labels_sha256']==labelinfo['sha256']
        np.testing.assert_array_equal(np.asarray(t['rotation_tolerance'].to_pylist(),np.float32),values)
        if not flag:
            for name in ['image','wrist_image','state']:assert t[name].equals(nominal[name]),(i,name)
        else:
            e=selected[i];folder=Path(e['replay']['folder']);r=read(folder/'report.json');assert r['success'] and r['native_goal_at_stop'] and not r['execution_rejected']
            for name,h in e['replay']['artifact_sha256'].items():assert sha(folder/name)==h,(i,name)
            assert e['label']['sha256']==labelinfo['sha256']==e['replay']['labels_sha256']
            trace=read(folder/'trajectory.json');assert len(trace)==s['frames'];expected=[]
            actions=np.array(nominal['actions'].to_pylist(),np.float32)
            for j,row in enumerate(trace):
                assert row['observation_timing']=='pre_action' and row['index']==j and row['label_source_frame_index']==j and row['physics_steps']==25 and not row['aborted']
                np.testing.assert_array_equal(actions[j],np.asarray(row['action'],np.float32));pre=row['pre_action'];expected.append(pre['position_m']+pre['rotation_vector_rad']+pre['finger_qpos_m'])
                np.testing.assert_allclose(row['bounds_rad'],lab['frames'][j]['rotation_tolerance_rad'],atol=1e-14,rtol=0)
            np.testing.assert_array_equal(np.array(t['state'].to_pylist(),np.float32),np.array(expected,np.float32))
            for camera,video in [('image','main_rgb.mp4'),('wrist_image','wrist_rgb.mp4')]:
                reader=imageio.get_reader(str(folder/video),format='ffmpeg');count=0
                try:
                    for j,pixels in enumerate(reader):
                        encoded=t[camera][j].as_py()['bytes'];stored=np.array(Image.open(io.BytesIO(encoded)));np.testing.assert_array_equal(stored,pixels);count+=1
                finally:reader.close()
                assert count==s['frames']
        frames+=s['frames'];reader_selection.setdefault((s['suite'],s['task_id'],flag),i)
        if (i+1)%25==0:write(output/'independent_audit_state.json',dict(stage='all_frame_verification',episodes=i+1,total=1875));print('Audited',i+1,'/1875',flush=True)
    assert frames==312374 and len(reader_selection)==80
    for name in ('episodes.jsonl','tasks.jsonl','reconstruction_sources.json'):assert sha(output/'meta'/name)==sha(nominal_root/'meta'/name)
    write(output/'independent_audit_state.json',dict(stage='actual_lerobot_reader',episodes=1875,total=1875))
    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies.libero_joint_policy import LiberoJointInputs
    from openpi.models.model import ModelType
    class SelectedEpisodeReader(LeRobotDataset):
        def _get_query_indices(self,idx,ep_idx):return super()._get_query_indices(idx,self.episodes.index(ep_idx))
    ids=sorted(reader_selection.values());ds=SelectedEpisodeReader('caozilin/libero_reconstructed_joint13_successmix20_20261109',root=output,episodes=ids,delta_timestamps={k:[j/20 for j in range(10)] for k in ['actions','rotation_tolerance']})
    for j,i in enumerate(ids):
        start=int(ds.episode_data_index['from'][j]);end=int(ds.episode_data_index['to'][j]);table=pq.read_table(output/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet')
        for f in (0,(end-start)//2,end-start-1):
            row=ds[start+f];assert int(row['episode_index'])==i and int(row['trajectory_is_tolerance'])==int(i in selected)
            packed=LiberoJointInputs(ModelType.PI05)({'observation/image':row['image'],'observation/wrist_image':row['wrist_image'],'observation/state':row['state'],'actions':row['actions'],'rotation_tolerance':row['rotation_tolerance'],'trajectory_is_tolerance':row['trajectory_is_tolerance'],'prompt':'audit'})
            assert packed['actions'].shape==(10,13)
            for h in range(10):
                k=min(f+h,end-start-1);expected=np.array(table['actions'][k].as_py()+table['rotation_tolerance'][k].as_py(),np.float32);np.testing.assert_array_equal(packed['actions'][h],expected)
        checks.append(dict(episode_index=i,trajectory_is_tolerance=int(i in selected)))
    result=dict(complete=True,episodes=1875,frames=312374,tasks=40,fps=20,nominal_episodes=1574,tolerance_episodes=301,success_pool=1503,per_task_sampling_verified=True,all_nominal_actions_unchanged=True,all_source_identities_preserved=True,all_nominal_observations_identical=True,all_tolerance_decoded_video_pixels_equal=True,all_tolerance_preaction_states_equal=True,all_labels_equal_audited_search_outputs=True,all_trajectory_flags_verified=True,missing_labels=0,portable_lerobot_reader_passed=True,reader_checks=checks,request_sha256=sha(output/'replacement_request.json'),manifest_sha256=sha(output/'conversion_manifest.json'))
    write(output/'meta/joint13_independent_audit.json',result);write(output/'independent_audit_state.json',dict(stage='complete',episodes=1875));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();print(audit(a.dataset),flush=True)
