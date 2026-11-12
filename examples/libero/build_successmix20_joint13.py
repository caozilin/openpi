"""Replace a reproducible per-task20% of successful tolerance replays, retaining nominal actions."""
from __future__ import annotations
import argparse,collections,concurrent.futures,copy,hashlib,io,json,os,random,shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from examples.libero.enrich_nominal_joint13_dataset import read,sha,write,feature_stats

ROOT=Path('/media/czl/sata/franka_my_code/openpi')
BASE=ROOT/'data/lerobot/caozilin/libero_reconstructed_nominal_joint13_20261109'
REPLAY=ROOT/'data/libero_tasktol_joint_servo/20261107_annotated_tolerance_v1'
DEFAULT_OUT=ROOT/'data/lerobot/caozilin/libero_reconstructed_joint13_successmix20_20261109'

def select_per_task(pool,seed):
    groups=collections.defaultdict(list)
    for row in pool:
        assert row['success'] and not row['execution_rejected']
        groups[row['suite'],row['task_id']].append(row)
    if len(groups)!=40:raise ValueError('Require all40 task success pools')
    chosen=[];tasks=[]
    for (suite,task),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:int(r['demo'].removeprefix('demo_')))
        if len({r['demo'] for r in rows})!=len(rows):raise ValueError('Duplicate source in success pool')
        sub_seed=int.from_bytes(hashlib.sha256(f'{seed}:{suite}:{task}'.encode()).digest()[:8],'big')
        count=(len(rows)+2)//5  # nearest integer of N/5, no floating-point rounding.
        sampled=random.Random(sub_seed).sample(rows,count);sampled.sort(key=lambda r:int(r['demo'].removeprefix('demo_')))
        chosen.extend(sampled);tasks.append(dict(suite=suite,task_id=task,candidates=len(rows),selected=count,seed=sub_seed,candidate_demos=[r['demo'] for r in rows],selected_demos=[r['demo'] for r in sampled]))
    return chosen,tasks

def prepare(output,seed=42):
    if (output/'replacement_request.json').exists():
        d=read(output/'replacement_request.json');assert d['random_seed']==seed;return d
    audit=read(REPLAY/'independent_final_audit.json');assert audit['complete'] and audit['episodes']==1851
    success=read(REPLAY/'successful_sources.json');assert success['episodes']==1503
    chosen,tasks=select_per_task(success['sources'],seed);assert len(chosen)==301
    nominal=read(BASE/'conversion_request.json');assert len(nominal['episodes'])==1875
    bykey={(r['source']['suite'],r['source']['task_id'],r['source']['demo']):r for r in nominal['episodes']}
    auditrows={r['key']:r for r in audit['results']};entries=[]
    for row in chosen:
        e=bykey[row['suite'],row['task_id'],row['demo']];a=auditrows[row['key']]
        assert a['success'] and a['native_success'] and not a['execution_rejected'] and a['original_actions_initial_state_verified'] and a['per_frame_labels_verified']
        assert row['executed_frames']==row['planned_frames']==e['source']['frames'];assert e['label']['sha256']==row['labels_sha256']==a['labels_sha256']
        entries.append(dict(episode_index=e['source']['episode_index'],source=e['source'],label=e['label'],replay=row))
    entries.sort(key=lambda e:e['episode_index']);output.mkdir(parents=True,exist_ok=True)
    req=dict(schema='libero_joint13_success_pool20/1.0',base=str(BASE),replay_root=str(REPLAY),random_seed=seed,selection='per-task round20percent of successful tolerance replay candidates; without replacement; retain matching source identity',planned_episodes=1875,frames=312374,tolerance_episodes=301,nominal_episodes=1574,success_pool=1503,tasks=tasks,entries=entries,nominal_manifest_sha256=sha(BASE/'conversion_manifest.json'),nominal_audit_sha256=sha(BASE/'meta/joint13_independent_audit.json'),replay_success_index_sha256=sha(REPLAY/'successful_sources.json'),replay_audit_sha256=sha(REPLAY/'independent_final_audit.json'),labels_version='original1851 plus audited24 supplement; user confirmed no edits',images='decode existing preaction H264MP4; no extra rotation; embed PNG of decoded pixels; original rendering was lossy',actions='retain exact corresponding nominal actions7, not optimized joint controls')
    write(output/'replacement_request.json',req);return req

def decode_pngs(path,expected):
    import imageio.v2 as imageio
    result=[];stats=[]
    reader=imageio.get_reader(str(path),format='ffmpeg')
    try:
        for frame in reader:
            if frame.shape!=(256,256,3) or frame.dtype!=np.uint8:raise ValueError(('Invalid MP4 frame',frame.shape,frame.dtype))
            out=io.BytesIO();Image.fromarray(frame).save(out,format='PNG',compress_level=3)
            result.append(dict(bytes=out.getvalue(),path=None));stats.append(frame)
    finally:reader.close()
    if len(result)!=expected:raise ValueError(('Video frame count mismatch',path,len(result),expected))
    pixels=np.stack(stats).astype(np.float64)/255.
    stat={k:f(pixels,axis=(0,1,2)).reshape(3,1,1).tolist() for k,f in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]};stat['count']=[expected]
    return result,stat

def replace_episode(entry,output):
    source=entry['source'];i=entry['episode_index'];rel=f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet';dest=output/rel;receipt=output/'replacement_receipts'/f'{i:06d}.json'
    if receipt.exists():
        d=read(receipt);assert sha(dest)==d['output_sha256'];return d
    base=BASE/rel;table=pq.read_table(base);n=source['frames'];assert table.num_rows==n
    replay=entry['replay'];folder=Path(replay['folder'])
    for name,h in replay['artifact_sha256'].items():assert sha(folder/name)==h,(i,name)
    label=read(Path(entry['label']['path']));assert sha(Path(entry['label']['path']))==entry['label']['sha256']==replay['labels_sha256']
    report=read(folder/'report.json');assert report['success'] and report['native_goal_at_stop'] and not report['execution_rejected'] and report['executed_frames']==report['planned_frames']==n
    assert report['tolerance']=='offline_annotated_signed_bounds';assert report['tolerance_protocol']['semantic_mapping']['labels_sha256']==entry['label']['sha256']
    trace=read(folder/'trajectory.json');assert len(trace)==n
    actions=np.asarray(table['actions'].to_pylist(),np.float32);states=[]
    for j,row in enumerate(trace):
        assert row['index']==j and row['label_source_frame_index']==j and row['observation_timing']=='pre_action' and row['physics_steps']==25 and not row['aborted']
        np.testing.assert_array_equal(np.asarray(row['action'],np.float32),actions[j])
        np.testing.assert_allclose(row['bounds_rad'],label['frames'][j]['rotation_tolerance_rad'],atol=1e-14,rtol=0)
        pre=row['pre_action'];states.append(pre['position_m']+pre['rotation_vector_rad']+pre['finger_qpos_m'])
    states=np.asarray(states,np.float32);assert states.shape==(n,8) and np.isfinite(states).all()
    data=table
    for camera,name in [('image','main_rgb.mp4'),('wrist_image','wrist_rgb.mp4')]:
        images,stats=decode_pngs(folder/name,n);data=data.set_column(data.schema.get_field_index(camera),data.schema.field(camera),pa.array(images,type=table[camera].type))
        if camera=='image':image_stats=stats
        else:wrist_stats=stats
    data=data.set_column(data.schema.get_field_index('state'),data.schema.field('state'),pa.array(states.tolist(),type=table['state'].type))
    data=data.set_column(data.schema.get_field_index('trajectory_is_tolerance'),data.schema.field('trajectory_is_tolerance'),pa.array(np.ones(n,np.int64)))
    for name in table.column_names:
        if name not in ('image','wrist_image','state','trajectory_is_tolerance'):assert data[name].equals(table[name]),(i,name)
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp.parquet');pq.write_table(data,tmp,compression='snappy');restored=pq.read_table(tmp);assert data.equals(restored,check_metadata=True);tmp.replace(dest)
    result=dict(episode_index=i,suite=source['suite'],task_id=source['task_id'],demo=source['demo'],frames=n,trajectory_is_tolerance=1,base_sha256=sha(base),output_sha256=sha(dest),bytes=dest.stat().st_size,replay_source=replay,label=entry['label'],state_stats=feature_stats(states),image_stats=image_stats,wrist_image_stats=wrist_stats,actions_unchanged=True)
    write(receipt,result);return result

def build(output,workers,pilot,replacements_only=False):
    import fcntl
    output.mkdir(parents=True,exist_ok=True)
    with (output/'conversion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);request=prepare(output);entries=request['entries'];selected={e['episode_index'] for e in entries}
        if pilot:entries=entries[:1]
        elif not replacements_only:
            # Link immutable unaffected tables only. Replaced tables are always independent files.
            for path in BASE.glob('data/**/*.parquet'):
                i=int(path.stem.split('_')[1]);dest=output/path.relative_to(BASE)
                if i not in selected and not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);os.link(path,dest)
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(replace_episode,e,output) for e in entries]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result());write(output/'conversion_state.json',dict(stage='pilot' if pilot else 'replacing',processed=len(results),total=len(entries)));print('Replaced',len(results),'/',len(entries),flush=True)
        if pilot:return
        assert len(results)==301
        if replacements_only:
            write(output/'replacement_stage_completion.json',dict(complete=True,replacements=301,scope='replacement files only; not a complete training dataset',request_sha256=sha(output/'replacement_request.json')))
            write(output/'conversion_state.json',dict(stage='replacement_stage_complete',processed=301,total=301))
            return
        meta=output/'meta';meta.mkdir(exist_ok=True)
        for name in ('episodes.jsonl','tasks.jsonl','reconstruction_sources.json','tolerance_sources.json','annotation_protocols.json','source_reconstruction_audit.json'):shutil.copyfile(BASE/'meta'/name,meta/name)
        info=read(BASE/'meta/info.json');info['joint13_contract']['trajectory_counts']={'nominal':1574,'tolerance':301};info['joint13_contract']['replacement_protocol']='per-task rounded20% of successful tolerance replay pool';info['joint13_contract']['tolerance_images']='PNG encoded decoded H264MP4 frames, already rotated180 once';write(meta/'info.json',info)
        stats=[json.loads(line) for line in (BASE/'meta/episodes_stats.jsonl').read_text().splitlines()]
        for row in results:
            s=stats[row['episode_index']]['stats'];s['state']=row['state_stats'];s['image']=row['image_stats'];s['wrist_image']=row['wrist_image_stats'];s['trajectory_is_tolerance']=feature_stats(np.ones((row['frames'],1),np.int64))
        (meta/'episodes_stats.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in stats))
        write(meta/'replacement_sources.json',request)
        write(output/'conversion_completion.json',dict(complete=True,episodes=1875,frames=312374,nominal_episodes=1574,tolerance_episodes=301,request_sha256=sha(output/'replacement_request.json'),independent_audit_pending=True))
        write(output/'conversion_state.json',dict(stage='complete',processed=301,total=301))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUT);p.add_argument('--workers',type=int,default=6);p.add_argument('--pilot',action='store_true');p.add_argument('--replacements-only',action='store_true');a=p.parse_args();assert 1<=a.workers<=6;build(a.output,a.workers,a.pilot,a.replacements_only)
