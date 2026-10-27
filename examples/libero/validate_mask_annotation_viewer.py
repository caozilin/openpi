"""Validate all400 MP4s and observation-aligned traces without revising masks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import numpy as np
import h5py

PROJECT=Path(__file__).resolve().parents[2]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,default=PROJECT/'data/libero40_mask_annotation/20261019_viewer_10demos_v2')
    args=parser.parse_args();root=args.viewer.resolve();manifest=json.loads((root/'viewer_manifest.json').read_text());catalog=json.loads((root/'viewer_catalog.json').read_text());source=Path(manifest['source_run'])
    frozen=json.loads((source/'mask_annotations.json').read_text());bykey={(t['suite'],t['task_id']):t for t in frozen['tasks']}
    checks=[]
    def check(name,result,detail=None):
        checks.append(dict(name=name,passed=bool(result),detail=detail))
    check('frozen_annotation_hash_unchanged',sha(source/'mask_annotations.json')==manifest['annotation_sha256']==sha(root/'mask_annotations.json'))
    check('exactly40_tasks',len(catalog['datasets'])==40)
    check('four_suites10tasks_each',all(sum(d['suite']==s for d in catalog['datasets'])==10 for s in ['libero_spatial','libero_object','libero_goal','libero_10']))
    check('exactly400_mp4_files',len(list((root/'videos').rglob('*.mp4')))==400)
    references=[];total_frames=0;episodes=[]
    for d in catalog['datasets']:
        key=(d['suite'],d['task_id']);original=bykey[key]
        check(f'{key}:task_shared_prediction_identical',d['shared_annotation']==original)
        check(f'{key}:ten_unique_demos',len(d['episodes'])==10 and len({e['path'] for e in d['episodes']})==10)
        check(f'{key}:original_demo_preserved_first',d['episodes'][0]['path']==original['demo_id'] and sum(e['representative'] for e in d['episodes'])==1)
        processed=Path(d['episodes'][0]['processed_hdf5']);meta=json.loads(processed.with_suffix('.metainfo.json').read_text())
        check(f'{key}:success_ledger',meta['complete'] and all(meta['episodes'][e['path']]['success'] for e in d['episodes']))
        phases={}
        for s in original['subtasks']:
            for span in s['phase_spans']:
                p=span['phase'];mask=s[p]['effective_mask'] if p in ['APP','EXEC'] else '000'
                for i in range(*span['processed_interval']):phases[i]=(s['index'],mask)
            for f in s['keyframes']:references.append(f['path'])
        with h5py.File(processed,'r') as h5:
            for episode in d['episodes']:
                episodes.append(episode)
                for name in ['video_path','poster_path','trace_path','trace_script_path']:references.append(episode[name])
                payload=json.loads((root/episode['trace_path']).read_text());frames=payload['trajectory'];n=episode['frames'];total_frames+=n
                group=h5['data'][episode['path']];ee=np.asarray(group['obs/ee_states']);q=np.asarray(group['obs/joint_states']);fingers=np.asarray(group['obs/gripper_states']);actions=np.asarray(group['actions']);idx=np.asarray(group['tasktol/source_step_indices'])
                check(episode['key']+':frame_count',len(frames)==n==len(ee))
                check(episode['key']+':mapping_and_clock',all(f['index']==i and f['video_frame_index']==i and f['source_frame']==int(idx[i]) and f['time_s']==i/20 and f['reset_relative_time_s']==.5+i/20 for i,f in enumerate(frames)))
                check(episode['key']+':observations_align',np.allclose([f['ee_state']['position_m'] for f in frames],ee[:,:3],atol=1e-7) and np.allclose([f['ee_state']['rotation_vector_base_rad'] for f in frames],ee[:,3:],atol=1e-7) and np.allclose([f['joint_state']['position_rad'] for f in frames],q,atol=1e-7) and np.allclose([f['action'] for f in frames],actions,atol=1e-7))
                check(episode['key']+':physical_aperture_difference',np.allclose([f['gripper_state']['aperture_m'] for f in frames],fingers[:,0]-fingers[:,1],atol=1e-7))
                check(episode['key']+':observed_velocity_at20Hz',frames[0]['joint_state']['velocity_rad_s']==[None]*7 and np.allclose([f['joint_state']['velocity_rad_s'] for f in frames[1:]],np.diff(q,axis=0)*20,atol=1e-7))
                check(episode['key']+':no_fabricated_nominal_or_numeric_bounds',all(f['nominal_target_pose'] is None and f['rotation_tolerance_rad'] is None for f in frames) and payload['stage_annotations']==[])
                if episode['representative']:
                    check(episode['key']+':reviewed_labels_preserved',all((f['subtask_index'],f['execution_mask'])==phases.get(i,(None,None)) for i,f in enumerate(frames)))
                    check(episode['key']+':original_mp4_bit_identical',sha(root/episode['video_path'])==sha(source/original['video_path']))
                else:
                    check(episode['key']+':extra_demo_semantics_unassigned',all(f['subtask_index'] is None and f['execution_mask'] is None and f['phase_source']=='gripper_event_candidate_only' for f in frames))
                matrices=[f['observed_gravity_frame']['rotation_matrix'] for f in frames if f['observed_gravity_frame']['rotation_matrix'] is not None]
                check(episode['key']+':gravity_proxy_frames_valid',all(np.allclose(np.array(m).T@np.array(m),np.eye(3),atol=1e-6) and np.allclose(np.array(m)[:,2],[0,0,1]) and np.linalg.det(m)>0 for m in matrices))
                check(episode['key']+':trace_hash',sha(root/episode['trace_path'])==episode['trace_sha256'])
    def probe(e):
        p=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,r_frame_rate,nb_read_frames,width,height','-of','json',str(root/e['video_path'])],text=True))['streams'][0]
        valid=p['codec_name']=='h264' and p['r_frame_rate']=='20/1' and int(p['nb_read_frames'])==e['frames'] and p['width']==256 and p['height']==292
        return e['key'],valid,p,sha(root/e['video_path'])==e['video_sha256']
    probes=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(key,valid,p,hash_ok) in enumerate(pool.map(probe,episodes),1):
            check(key+':h264_full_decoded_framecount',valid,p);check(key+':video_hash',hash_ok);probes.append(dict(key=key,**p))
            if i%100==0:print(f'validated {i}/400 decoded videos',flush=True)
    class Document(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.refs=[]
        def handle_starttag(self,tag,attrs):
            v=dict(attrs)
            if 'id'in v:self.ids.append(v['id'])
            for field in ['src','href']:
                if v.get(field):self.refs.append(v[field])
    doc=Document();doc.feed((root/'index.html').read_text());references.extend(doc.refs)
    missing=[r for r in set(references) if not r.startswith(('#','http:','https:')) and not (root/r).is_file()]
    check('all_relative_html_media_evidence_paths_exist',not missing,missing)
    check('unique_dom_ids',len(doc.ids)==len(set(doc.ids)))
    check('main_view_only','mainVideo' in doc.ids and 'wristVideo' not in doc.ids)
    check('all_interactive_panels_present',all(i in doc.ids for i in ['folderSelect','datasetList','episodeList','spatialCanvas','positionChart','rotationChart','toleranceChart','jointChart','jointVelocityChart','apertureChart','sourceChart','actionTranslationChart','actionRotationChart','actionGripperChart','reasoningContent','subtaskTabs']))
    for js in (root/'assets').glob('*.js'):
        if js.name=='catalog.js':continue
        result=subprocess.run(['node','--check',str(js)],capture_output=True,text=True);check('javascript_syntax:'+js.name,result.returncode==0,result.stderr or None)
    failed=[c['name'] for c in checks if not c['passed']]
    payload=dict(status='passed' if not failed else 'failed',checks=checks,check_count=len(checks),failed_checks=failed,
        counts=dict(suites=4,tasks=40,episodes=len(episodes),additional_episodes=360,frames=total_frames),
        annotation_sha256=manifest['annotation_sha256'],video_probes=probes,
        note='These checks validate media, time alignment, adapter semantics and frozen-label integrity; they do not establish cross-demonstration mask accuracy.')
    (root/'viewer_validation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in payload.items() if k not in ['checks','video_probes']},ensure_ascii=False,indent=2))
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
