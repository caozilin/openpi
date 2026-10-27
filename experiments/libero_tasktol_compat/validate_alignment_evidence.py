"""Validate saved native videos and export explicit observable process evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import subprocess

BASE=Path('/media/czl/sata/franka_my_code/openpi/experiments/libero_tasktol_compat/results')


def inspect(report_path):
    report=json.loads(report_path.read_text());folder=report_path.parent
    native='diagnostic_only' in report
    frames=json.loads((folder/('trace.json' if native else 'trajectory.json')).read_text())
    if not native:frames=frames['frames']
    video=next(folder.glob('*.mp4'))
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(video)],capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)['streams'][0]
    decoded=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    checks=dict(video_full_decode=decoded.returncode==0 and not decoded.stderr,
        video_frames_match=int(stream['nb_read_frames'])==len(frames),resolution=stream['width']==256 and stream['height']==256,
        frequency=stream['r_frame_rate']=='20/1')
    if not native:
        checks.update(
            ipopt_called_every_frame=all(r['optimizer']['solver_called'] for r in frames),
            native_steps=all(r['physics_steps']==(0 if r['aborted'] else 25) for r in frames),
            native_goal_matches=all(r['native_joint_goal_error_rad'] is None or r['native_joint_goal_error_rad']<1e-10 for r in frames),
            context_verified=report.get('initialization',{}).get('reset_rng_matches_source') is True,
            no_extra_settling=report.get('initialization',{}).get('extra_physics_steps')==0,
        )
        candidates={0,len(frames)-1}
        candidates.update(i for i in range(1,len(frames)) if (frames[i]['phase'],frames[i]['subtask_index'])!=(frames[i-1]['phase'],frames[i-1]['subtask_index']))
        eligible=[(i,r['source_replay_position_error_m']) for i,r in enumerate(frames) if not r['aborted']]
        if eligible:
            worst=max(eligible,key=lambda x:x[1])[0]
            candidates.update([worst,min(worst+1,len(frames)-1)])
        candidates=sorted(candidates)
        images=folder/'process_images';images.mkdir(exist_ok=True)
        pattern=str(images/'frame_%03d.png')
        selection='+'.join(f'eq(n\\,{i})' for i in candidates)
        extraction=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-vf',f'select={selection}','-vsync','0','-y',pattern],capture_output=True,text=True)
        checks['process_images']=extraction.returncode==0 and len(list(images.glob('frame_*.png')))==len(candidates)
        manifest=[dict(video_frame=i,phase=frames[i]['phase'],subtask=frames[i]['subtask_index'],path=str(images/f'frame_{j:03d}.png')) for j,i in enumerate(candidates,1)]
        (folder/'process_images.json').write_text(json.dumps(dict(observation_timing='pre_action',images=manifest),indent=2)+'\n')
    return dict(report=str(report_path),checks=checks,passed=all(checks.values()),decode_error=decoded.stderr)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    paths=[p for name in args.batch for p in sorted((BASE/name).rglob('report.json'))]
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(inspect,p) for p in paths]):
            results.append(future.result())
    args.output.write_text(json.dumps(dict(records=len(results),passed_records=sum(r['passed'] for r in results),checks=sum(len(r['checks']) for r in results),passed_checks=sum(sum(r['checks'].values()) for r in results),results=results),indent=2)+'\n')
    print('records',len(results),'passed',sum(r['passed'] for r in results))
    if not all(r['passed'] for r in results):
        print(json.dumps([r for r in results if not r['passed']],indent=2))
        raise SystemExit(1)


if __name__=='__main__':main()
