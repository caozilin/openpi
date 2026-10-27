"""Extend a frozen LIBERO-40 annotation run to 10 demos/task and Dataset Studio.

Read-only HDF5 adapter. The annotation masks are copied verbatim, never inferred
or revised from the numeric robot traces exported for the new viewer.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
ADAPTER = Path(__file__).resolve().parent / 'mask_viewer'
FPS = 20
PHASES = {'APP': 'pregrasp', 'ENG': 'grasp', 'EXEC': 'postgrasp', 'DIS': 'release'}

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(2**20), b''): h.update(block)
    return h.hexdigest()

def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_size != target.stat().st_size or sha(source) != sha(target):
            raise ValueError(f'Refusing to replace different snapshot file: {target}')
        return
    try: os.link(source, target)
    except OSError: shutil.copy2(source, target)

def numeric(array):
    values = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(values)): raise ValueError('Nonfinite observed robot trace')
    return np.round(values, 8).tolist()

def quaternion_matrix(q):
    x,y,z,w = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def observed_gravity_frame(q):
    # Calibration: data quaternion is right_hand, whose +Y is physical closure.
    # This is an observed diagnostic proxy, NOT independently integrated nominal.
    r = quaternion_matrix(q)
    c = r[:,1]
    projected = c.copy(); projected[2] = 0
    h = float(np.linalg.norm(projected))
    if h < 1e-6: return None, h
    y = projected/h; z=np.array([0.,0.,1.]); x=np.cross(y,z)
    return numeric(np.column_stack((x,y,z))), h

def gripper_events(commands, aperture):
    commands = np.asarray(commands)>0
    n=len(commands); changes=np.flatnonzero(commands[1:]!=commands[:-1])+1
    if commands[0]: changes=np.r_[0,changes]
    events=[]
    for j,raw in enumerate(changes):
        t=int(raw); stop=int(changes[j+1]) if j+1<len(changes) else n
        response=next((i for i in range(t+1,stop) if abs(aperture[i]-aperture[t])>.0005),None)
        stable=None
        if response is not None:
            stable=next((i for i in range(max(t+5,response+4),stop)
                if np.max(np.abs(np.diff(aperture[i-4:i+1:2])))<=.0002
                and np.max(np.abs(np.diff(aperture[i-4:i+1])))<=.0002),None)
        events.append(dict(kind='close' if commands[t] else 'open',command_frame=t,
                           first_aperture_response=response,stable_frame=stable,
                           status='aperture_confirmed' if stable is not None else 'unconfirmed',
                           role='gripper_candidate_not_semantic_subtask'))
    candidates=['gripper_closed_stable' if commands[0] else 'gripper_open_stable']*n
    for j,e in enumerate(events):
        stop=events[j+1]['command_frame'] if j+1<len(events) else n
        for i in range(e['command_frame'],stop):
            candidates[i]=('gripper_closing' if e['kind']=='close' else 'gripper_opening') if e['stable_frame'] is None or i<e['stable_frame'] else ('gripper_closed_stable' if e['kind']=='close' else 'gripper_open_stable')
    return events,candidates

def choose_episodes(task, metadata, seed, count=10):
    successful=sorted((d for d,e in metadata['episodes'].items() if e['success']), key=lambda d:int(d[5:]))
    representative=task['demo_id']
    if len(successful)<count or representative not in successful: raise ValueError(task['task_name'])
    key=f'{seed}:{task["suite"]}:{task["task_id"]}'
    rng=random.Random(key)
    extra=sorted(rng.sample([d for d in successful if d!=representative], count-1), key=lambda d:int(d[5:]))
    return [representative,*extra]

def export_episode(job):
    task, selected, episode_meta, demo, order, source_run, output = job
    suite=task['suite']; tid=task['task_id']; name=task['task_name']
    key=f'{suite}/{tid:02d}/{demo}'
    video_rel=f'videos/{suite}/{name}/{demo}.mp4'; video=output/video_rel
    poster_rel=f'posters/{suite}/{tid:02d}/{demo}.jpg'
    trace_rel=f'trajectories/{suite}/{tid:02d}/{demo}.json'
    script_rel=f'trajectories/{suite}/{tid:02d}/{demo}.js'
    representative=demo==task['demo_id']
    with h5py.File(selected['processed_hdf5'],'r') as file:
        group=file['data'][demo]
        actions=np.asarray(group['actions']); ee=np.asarray(group['obs/ee_states'])
        joints=np.asarray(group['obs/joint_states']); fingers=np.asarray(group['obs/gripper_states'])
        robot=np.asarray(group['tasktol/pre_action_robot_states'])
        source=np.asarray(group['tasktol/source_step_indices']).astype(int)
        n=len(actions)
        assert ee.shape==(n,6) and joints.shape==(n,7) and fingers.shape==(n,2) and robot.shape==(n,9)
        assert n==episode_meta['retained_steps'] and source.tolist()==episode_meta['source_step_indices']
        assert np.allclose(ee[:,:3],robot[:,2:5],atol=1e-7)
        qnorm=np.linalg.norm(robot[:,5:9],axis=1)
        assert np.allclose(qnorm,1,atol=1e-4)
        aperture=fingers[:,0]-fingers[:,1]; events, candidates=gripper_events(actions[:,-1],aperture)
        velocity=np.diff(joints,axis=0)*FPS
        labels={}
        if representative:
            for sub in task['subtasks']:
                for span in sub['phase_spans']:
                    p=span['phase']; start,end=span['processed_interval']
                    mask=sub[p]['effective_mask'] if p in ['APP','EXEC'] else '000'
                    for i in range(start,end): labels[i]=(PHASES[p],sub['index'],mask)
        frames=[]
        for i in range(n):
            phase,subtask,mask=labels.get(i,('outside_subtask',None,None)) if representative else (candidates[i],None,None)
            frame_matrix,h=observed_gravity_frame(robot[i,5:9])
            frames.append(dict(index=i,video_frame_index=i,time_s=i/FPS,reset_relative_time_s=.5+i/FPS,
                source_frame=int(source[i]),source_gap_before=int(source[i]-source[i-1]-1) if i else int(source[i]),
                phase=phase,phase_source='reviewed_representative_keyframes' if representative else 'gripper_event_candidate_only',
                subtask_index=subtask,execution_mask=mask,
                action=numeric(actions[i]),nominal_target_pose=None,
                ee_state=dict(position_m=numeric(ee[i,:3]),rotation_vector_base_rad=numeric(ee[i,3:]),quaternion_xyzw=numeric(robot[i,5:9]),reference_frame='world',observation_timing='pre_action'),
                joint_state=dict(position_rad=numeric(joints[i]),velocity_rad_s=numeric(velocity[i-1]) if i else [None]*7),
                gripper_state=dict(finger_qpos_m=numeric(fingers[i]),aperture_m=round(float(aperture[i]),8),command=float(actions[i,6])),
                observed_gravity_frame=dict(rotation_matrix=frame_matrix,h=round(h,10),degenerate_proxy=h<1e-6,
                    reference='observed right_hand quaternion with calibrated physical closure; not action-integrated nominal'),
                rotation_tolerance_rad=None))
        payload=dict(schema_version='libero_observed_viewer.1',key=key,
            episode=dict(demo_id=demo,instruction=task['language'],frame_count=n,frequency_hz=FPS,
                is_annotation_demo=representative,annotation_demo_id=task['demo_id'],result=dict(success=True,source='regeneration_success_ledger'),
                stage_scope='reviewed_selected_mode' if representative else 'task_shared_prediction_unreviewed_mode',
                observation_timing='pre_action',terminal_post_action_image_recorded=False),
            trajectory=frames,stage_annotations=[],events=events,
            visual_phase_spans=task['subtasks'] if representative else None,
            media=dict(video=video_rel,poster=poster_rel),
            coordinate_note='Position is grip_site world position; data rotation is right_hand world orientation. Paper tool Y matches physical closure after calibration. F here uses observed orientation only.',
            action_note='Raw normalized OSC_POSE controls; input limits[-1,1], output max[.05,.05,.05,.5,.5,.5]. Raw commands are not measured metres/radians; nominal goal/integrated reference not exported.',
            source=dict(processed_hdf5=selected['processed_hdf5'],source_hdf5=selected['source_hdf5'],source_step_indices=source.tolist(),state_field='tasktol/pre_action_robot_states; not legacy states[0]'),
            metrics=dict(duration_s=n/FPS,retained_source_gaps=int(np.sum(np.maximum(np.diff(source)-1,0))),
                observed_joint_path_length_rad=float(np.sum(np.linalg.norm(np.diff(joints,axis=0),axis=1))),
                observed_ee_path_length_m=float(np.sum(np.linalg.norm(np.diff(ee[:,:3],axis=0),axis=1))),
                mean_observed_joint_speed_rad_s=float(np.mean(np.linalg.norm(velocity,axis=1))) if n>1 else None,
                metrics_scope='differences between retained observations at20Hz; not optimizer command-trajectory benchmark metrics'))
        dump(output/trace_rel,payload)
        script=output/script_rel
        script.write_text('window.LIBERO_EPISODES=window.LIBERO_EPISODES||{};window.LIBERO_EPISODES['+json.dumps(key)+']='+json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+';\n')
        images=group['obs/agentview_rgb']; poster=output/poster_rel;poster.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(images[0][::-1,::-1].copy()).save(poster,quality=90)
        if representative:
            link_or_copy(source_run/task['video_path'],video)
        elif not video.exists():
            video.parent.mkdir(parents=True,exist_ok=True); temporary=video.with_name(video.stem+'.partial.mp4')
            encoder=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','256x292','-framerate',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
            try:
                for i in range(n):
                    canvas=Image.new('RGB',(256,292),(15,18,24));canvas.paste(Image.fromarray(images[i][::-1,::-1].copy()))
                    draw=ImageDraw.Draw(canvas);draw.text((3,257),f'p={i} src={source[i]} t={i/FPS:.2f}s\npre-action | agentview | {demo}',font=font,fill='white')
                    encoder.stdin.write(canvas.tobytes())
                encoder.stdin.close()
                if encoder.wait()!=0: raise RuntimeError(str(video))
                temporary.replace(video)
            except BaseException:
                encoder.kill();encoder.wait();raise
    return dict(path=demo,key=key,frames=n,success=True,representative=representative,display_order=order,
                video_path=video_rel,poster_path=poster_rel,trace_path=trace_rel,trace_script_path=script_rel,
                video_sha256=sha(video),trace_sha256=sha(output/trace_rel),
                source_hdf5=selected['source_hdf5'],processed_hdf5=selected['processed_hdf5'],
                source_step_indices=source.tolist(),selection='preserved annotation representative' if representative else 'seeded arbitrary choice among remaining saved successful demos; no semantic screening')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run',type=Path,default=PROJECT/'data/libero40_mask_annotation/20261019_mainview_v1')
    ap.add_argument('--output',type=Path,default=PROJECT/'data/libero40_mask_annotation/20261019_viewer_10demos_v2')
    ap.add_argument('--mujoco-viewer',type=Path,default=Path('/media/czl/sata/franka_my_code/franka_mujoco/tasktol/data/viewer/static'))
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seed',type=int,default=20261019)
    args=ap.parse_args();start=time.time();src=args.source_run.resolve();out=args.output.resolve()
    if src==out:raise ValueError('Frozen annotation run must remain unchanged')
    out.mkdir(parents=True,exist_ok=True)
    annotation_bytes=(src/'mask_annotations.json').read_bytes();frozen_hash=hashlib.sha256(annotation_bytes).hexdigest()
    annotation=json.loads(annotation_bytes);selection=json.loads((src/'selection_manifest.json').read_text())
    selected={(t['suite'],t['task_id']):t for t in selection['tasks']}
    assert len(annotation['tasks'])==40
    # Preserve the full visual evidence and per-task decision files, maintaining relative links.
    for p in src.rglob('*'):
        if not p.is_file() or any(x in p.relative_to(src).parts for x in ['videos','tools','__pycache__']):continue
        if p.suffix in ['.png','.json','.md'] or p.name=='mask_annotations.sha256':link_or_copy(p,out/p.relative_to(src))
    catalog=[];jobs=[]
    for task in annotation['tasks']:
        s=selected[(task['suite'],task['task_id'])]
        meta=json.loads(Path(s['processed_hdf5']).with_suffix('.metainfo.json').read_text())
        assert meta['complete']
        demos=choose_episodes(task,meta,args.seed)
        catalog.append(dict(relative_path=f'{task["suite"]}/{task["task_id"]:02d}',source_type='libero',robot='panda',scene=task['suite'],
            suite=task['suite'],task_id=task['task_id'],task_name=task['task_name'],display_name=task['language'],frequency_hz=FPS,
            episode_count=10,representative_demo_id=task['demo_id'],shared_annotation=task,episodes=[]))
        for order,demo in enumerate(demos):jobs.append((task,s,meta['episodes'][demo],demo,order,src,out))
    records={}
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(export_episode,job):job for job in jobs}
        for i,f in enumerate(as_completed(futures),1):
            record=f.result();records[record['key']]=record
            if i%10==0 or i==1:print(f'{i}/400 videos + trajectories exported ({time.time()-start:.1f}s)',flush=True)
    for task in catalog:
        task['episodes']=sorted([r for k,r in records.items() if k.startswith(task['relative_path']+'/')],key=lambda r:r['display_order'])
        assert len(task['episodes'])==10
    assets=out/'assets';assets.mkdir(exist_ok=True)
    upstream_app=(args.mujoco_viewer/'app.js').read_text()
    # Keep the original rendering/interaction implementation, defer boot to the LIBERO adapter.
    upstream_app=upstream_app.replace('const wristVideo = $("#wristVideo");','const wristVideo = {currentTime:0, playbackRate:1, pause(){}, load(){}, play(){return Promise.resolve();}};')
    upstream_app=upstream_app.replace('requestAnimationFrame(animationLoop);\nloadDatasets();','// LIBERO adapter starts the viewer after loading its local catalog.')
    upstream_app=upstream_app.replace('.addEventListener("click",togglePlayback)', '.addEventListener("click",()=>togglePlayback())').replace('.addEventListener("ended",pauseVideos)', '.addEventListener("ended",()=>pauseVideos())').replace('.addEventListener("click",loadDatasets)', '.addEventListener("click",()=>loadDatasets())')
    (assets/'mujoco-studio.js').write_text(upstream_app)
    shutil.copy2(args.mujoco_viewer/'styles.css',assets/'mujoco-studio.css')
    for name in ['adapter.js','adapter.css']:shutil.copy2(ADAPTER/name,assets/name)
    shutil.copy2(Path(__file__).with_name('serve_mask_annotation_viewer.py'),out/'serve.py')
    shutil.copy2(ADAPTER/'README.md',out/'README.md')
    from mask_viewer.layout import adapt_layout
    document=adapt_layout((args.mujoco_viewer/'index.html').read_text())
    (out/'index.html').write_text(document);(out/'report.html').write_text(document)
    data=dict(schema_version='libero40_studio.2',datasets=catalog,annotation_sha256=frozen_hash,
        counts=dict(suites=4,tasks=40,episodes=400,extra_episodes=360),
        information_scope='Frozen visual predictions reused unchanged. Robot numeric traces newly authorized only for viewer details; additional demos not visually annotated.',
        source_run=str(src))
    dump(out/'viewer_catalog.json',data)
    (assets/'catalog.js').write_text('window.LIBERO_CATALOG='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    dump(out/'viewer_manifest.json',dict(schema_version='2.0',created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_run=str(src),annotation_sha256=frozen_hash,
        annotation_unchanged=annotation_bytes==(src/'mask_annotations.json').read_bytes(),selection_seed=args.seed,counts=data['counts'],
        original_viewer=dict(path=str(args.mujoco_viewer),files=[dict(name=n,sha256=sha(args.mujoco_viewer/n)) for n in ['index.html','app.js','styles.css']],
            reused='Dataset Studio layout, styling, canvas chart engine, orbit/projection math, transport and synchronized cursor. Adapter replaces data catalog/episode IO and mismatched semantics.'),
        build_sources=[dict(path=str(p),sha256=sha(p)) for p in [Path(__file__),ADAPTER/'layout.py',ADAPTER/'adapter.js',ADAPTER/'adapter.css',ADAPTER/'README.md',Path(__file__).with_name('serve_mask_annotation_viewer.py')]],
        robot_trace_scope=data['information_scope'],format=dict(frequency_hz=20,view='agentview only',image_transform='180deg once',observations='pre_action',
            nominal_pose='not_available; never replaced with actual as if nominal',numeric_tolerance_bounds=None,action_units='raw normalized OSC_POSE commands',
            gravity_frame='observed diagnostic proxy; calibrated physical closure, no alternate axis on degeneracy'),
        access_allowlist=['actions','obs/agentview_rgb','obs/ee_states','obs/joint_states','obs/gripper_states','tasktol/pre_action_robot_states','tasktol/source_step_indices'],
        elapsed_seconds=time.time()-start))
    print(f'VIEWER_READY {out}/index.html',flush=True)

if __name__=='__main__':main()
