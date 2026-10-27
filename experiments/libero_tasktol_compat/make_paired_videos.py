"""Pair existing, verified native OSC / IPOPT recordings without new rollouts."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import h5py
import numpy as np

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
sys.path.insert(0,str(ROOT))
BASE = ROOT/'experiments/libero_tasktol_compat/results'
FONT = 'Noto Sans CJK SC'
FPS = 20
TAIL = 40
HEIGHT = 880


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(path)],
        capture_output=True,text=True,check=True)
    return json.loads(result.stdout)['streams'][0]


def stamp(seconds):
    total = round(seconds*100)
    return f'{total//360000}:{total//6000%60:02d}:{total//100%60:02d}.{total%100:02d}'


def escaped(value):
    return str(value).replace('\\','/').replace('{','(').replace('}',')').replace('\n',r'\N')


def captions(path, task, osc, ipopt, left, right, count, metrics):
    lines = [
        '[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 1024', f'PlayResY: {HEIGHT}',
        'ScaledBorderAndShadow: yes', 'WrapStyle: 2', '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
        f'Style: Default,{FONT},22,&H00FFFFFF,&H00FFFFFF,&H00181818,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1',
        '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    ]

    def add(start, end, x, y, text, size=22, color='FFFFFF'):
        tags = r'{\an7\pos('+f'{x},{y}'+r')\fs'+str(size)+r'\c&H'+color+r'&}'
        lines.append(f'Dialogue: 0,{stamp(start)},{stamp(end)},Default,,0,0,0,,{tags}{escaped(text)}')

    duration = (count+TAIL)/FPS
    title = f"{task['suite']}/{task['task_id']:02d}   |   {task['demo_id']}   |   seed=0"
    add(0,duration,16,8,title,25)
    wrapped = textwrap.wrap(task['task_name'].replace('_',' '),width=100)
    add(0,duration,16,43,'\n'.join(wrapped),17)
    add(0,duration,16,84,'左：原版 OSC',24)
    add(0,duration,528,84,'右：IPOPT + 原生关节控制',24)
    add(0,duration,16,631,'OSC：实际 − 自身参考（世界轴）',22,'FFFF88')
    add(0,duration,528,631,'IPOPT：实际 − 自身参考（世界轴）',22,'FFFF88')
    add(0,duration,16,771,'同一重构 action · kp150 · 无容差 · ΔR 为世界轴旋转向量（°）',21,'DDDDDD')
    aborted = next((i for i,r in enumerate(right) if r['aborted']),None)
    for index in range(count+TAIL):
        start,end=index/FPS,(index+1)/FPS
        metric=metrics[index]
        for side,frames,report in [(0,left,osc),(1,right,ipopt)]:
            x = 16 if side==0 else 528
            key='left' if side==0 else 'right'
            error=metric[f'{key}_tracking_error_6d']
            color='66BFFF' if side==1 and not metric['synchronized'] else 'FFFF88'
            add(start,end,x,667,f'Δx {error[0]:+.3f}   Δy {error[1]:+.3f}   Δz {error[2]:+.3f} mm',20,color)
            add(start,end,x,701,f'ΔRx {error[3]:+.3f}   ΔRy {error[4]:+.3f}   ΔRz {error[5]:+.3f} °',20,color)
            ref_index=metric[f'{key}_reference_index']
            ref_text='初始参考锚点' if ref_index is None else f'已下发参考 #{ref_index+1:03d}'
            add(start,end,x,735,f"{ref_text} · 状态 t={metric[f'{key}_pose_time_s']:.2f}s",18,'DDDDDD')
            if side==1 and aborted is not None and index>=aborted:
                add(start,end,x,807,f'求解中止于动作 {aborted+1}/{count}',21,'66BFFF')
                add(start,end,x,843,'画面及已下发参考保持',18,'66BFFF')
            elif index>=count:
                success=bool(report['task_success'])
                add(start,end,x,807,'实际终态：'+('成功' if success else '未成功'),23,'88FF88' if success else '88AAFF')
                add(start,end,x,843,'最后一次动作执行后的画面',18,'DDDDDD')
            else:
                # Source video frame i is the observation BEFORE action i.
                success=bool(frames[index-1]['success']) if index else None
                state='起始' if success is None else ('当前成功' if success else '当前未成功')
                add(start,end,x,807,f'画面帧 {index+1:03d}/{count}',21)
                add(start,end,x,843,state,18,'88FF88' if success else 'DDDDDD')
    path.write_text('\n'.join(lines)+'\n')


def make(task, output):
    rel=Path(task['suite'])/f"{task['task_id']:02d}"
    osc_dir=BASE/'osc_reference_hdf5_for_pair40_v1'/rel
    ipopt_dir=BASE/'ipopt_relabelled_kp150_all40_v1/kp_150'/rel
    osc=json.loads((osc_dir/'report.json').read_text())
    ipopt=json.loads((ipopt_dir/'report.json').read_text())
    left=json.loads((osc_dir/'trace.json').read_text())
    right=json.loads((ipopt_dir/'trajectory.json').read_text())['frames']
    hdf5=ROOT/'data/libero_reference_actions/20261020_osc_reference_v1'/rel/'reference_actions.hdf5'
    with h5py.File(hdf5,'r') as file:
        labels=file['data'][task['demo_id']]
        actions=labels['actions'][()]
        anchor_position=labels['anchor_position_m'][()]
        anchor_rotation=labels['anchor_tool_rotation'][()]
    count=len(actions)
    source_left=osc_dir/'diagnostic_native_osc.mp4'
    source_right=ipopt_dir/'main_rgb.mp4'
    probes=[probe(p) for p in (source_left,source_right)]
    checks=dict(
        same_demo=osc['demo']==ipopt['task']['demo']==task['demo_id'],
        same_seed=osc['seed']==ipopt['initialization']['source_seed']==0,
        same_initial_state=osc['initial_state_sha256']==ipopt['initial_state_sha256'],
        same_fixture_poses=osc['fixtures']==ipopt['initialization']['fixtures'],
        same_gripper_initialization=osc['initial_gripper_action']==ipopt['initialization']['initial_gripper_action'],
        reset_rng_verified=osc['reset_rng_matches_source'] and ipopt['initialization']['reset_rng_matches_source'],
        same_initial_pixels=sha(osc_dir/'initial.png')==sha(ipopt_dir/'initial.png'),
        osc_accumulates_relabelled_actions=osc['osc_reference_mode']=='accumulated_reference',
        osc_uses_same_relabelled_hdf5_actions=np.array_equal(np.asarray([r['reference_delta_action'] for r in left]),actions),
        ipopt_uses_same_relabelled_hdf5_actions=np.array_equal(np.asarray([r['action'] for r in right]),actions[:len(right)]),
        same_reference_source=hdf5.resolve()==Path(ipopt['reference_actions']['path']).resolve() and sha(hdf5)==ipopt['reference_actions']['sha256'],
        osc_directly_reads_same_hdf5=Path(osc['reference_actions']['path']).resolve()==hdf5.resolve() and osc['reference_actions']['sha256']==ipopt['reference_actions']['sha256'],
        original_osc=osc['controller']=='OSC_POSE' and osc['ipopt_solver_calls']==0,
        original_joint_control=ipopt['controller']=='JointPositionController' and ipopt.get('joint_control_mode','native')=='native',
        same_nominal_kp=osc['native_kp']==[150.0]*6 and ipopt['native_controller_kp']==[150.0]*7,
        ipopt_every_action=all(r['optimizer']['solver_called'] for r in right),
        no_tolerance=ipopt['tolerance']['mode']=='off',
        native_physics=all(r['physics_steps']==25 for r in left) and all(r['physics_steps']==(0 if r['aborted'] else 25) for r in right),
        source_lengths=int(probes[0]['nb_read_frames'])==len(left)==count and int(probes[1]['nb_read_frames'])==len(right),
        source_video_format=all(p['width']==256 and p['height']==256 and p['r_frame_rate']=='20/1' for p in probes),
    )
    if not all(checks.values()):
        raise RuntimeError(f'{rel}: pairing failed {[k for k,v in checks.items() if not v]}')
    dest=output/rel;dest.mkdir(parents=True,exist_ok=False)
    subtitle=dest/'labels.ass'
    from experiments.libero_tasktol_compat.paired_eef_metrics import aligned_tracking_metrics

    metrics=aligned_tracking_metrics(left,right,anchor_position,anchor_rotation,fps=FPS,tail=TAIL)
    (dest/'eef_frame_metrics.json').write_text(json.dumps(dict(
        component_order=['dx_mm','dy_mm','dz_mm','dRx_deg','dRy_deg','dRz_deg'],
        position_definition='Per side: actual minus its OWN issued reference, world-axis signed components, mm',
        rotation_definition='Per side: Log(R_actual @ R_own_reference.T), world-axis rotation-vector components, degrees',
        sign='actual minus reference on each side independently',
        reference_sources=dict(left='native_goal_position_m/native_goal_rotation',right='nominal_target after successful strict IPOPT solve'),
        timing='Video frame i is pre-action, so use already issued reference i-1. Frame zero uses initial anchor; appended final PNGs use last issued reference.',
        after_abort='Right actual pose AND reference hold at last successfully executed command; rejected target is never used.',frames=metrics,
    ),indent=2)+'\n')
    captions(subtitle,task,osc,ipopt,left,right,count,metrics)
    video=dest/'osc_vs_ipopt.mp4'
    # Append actual post-action final PNGs. A rejected IPOPT suffix is held,
    # explicitly captioned; no frames are sped up or motion-interpolated.
    extra_right=count-len(right)+TAIL
    filter_graph=(
        '[0:v]fps=20,setpts=PTS-STARTPTS[a];'
        f'[2:v]trim=end_frame={TAIL},setpts=PTS-STARTPTS[b];'
        '[a][b]concat=n=2:v=1:a=0[l];'
        '[1:v]fps=20,setpts=PTS-STARTPTS[c];'
        f'[3:v]trim=end_frame={extra_right},setpts=PTS-STARTPTS[d];'
        '[c][d]concat=n=2:v=1:a=0[r];'
        '[l][r]hstack=inputs=2:shortest=1,scale=1024:512:flags=lanczos,'
        f"pad=1024:{HEIGHT}:0:112:black,ass=filename='{subtitle.as_posix()}'[out]"
    )
    command=['ffmpeg','-v','error','-nostdin','-threads','1','-filter_complex_threads','1',
        '-i',str(source_left),'-i',str(source_right),
        '-loop','1','-framerate','20','-i',str(osc_dir/'final.png'),
        '-loop','1','-framerate','20','-i',str(ipopt_dir/'final.png'),
        '-filter_complex',filter_graph,'-map','[out]','-frames:v',str(count+TAIL),
        '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
        '-threads','1','-movflags','+faststart',str(video)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    result=probe(video)
    decoded=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert result['width']==1024 and result['height']==HEIGHT and result['r_frame_rate']=='20/1'
    assert int(result['nb_read_frames'])==count+TAIL and decoded.returncode==0 and not decoded.stderr
    record=dict(task=str(rel),task_name=task['task_name'],demo=task['demo_id'],seed=0,
        left='Original native OSC, accumulated relabelled reference',right='IPOPT + original native JOINT_POSITION kp150',
        left_final_success=osc['task_success'],right_final_success=ipopt['task_success'],
        right_ever_success=any(r['success'] and not r['aborted'] for r in right),
        source_reference_frames=count,ipopt_recorded_frames=len(right),
        ipopt_rejected_reference_index=next((i for i,r in enumerate(right) if r['aborted']),None),
        video=str(video),video_frames=count+TAIL,duration_s=(count+TAIL)/FPS,
        image_timing='Original frames are pre-action; added last 2s show actual post-action final PNGs; aborted right suffix is a labelled hold',
        source_left=str(source_left),source_right=str(source_right),
        source_video_sha256=[sha(source_left),sha(source_right)],reference_hdf5=str(hdf5),reference_sha256=sha(hdf5),
        eef_frame_metrics=str(dest/'eef_frame_metrics.json'),eef_metric_frames=len(metrics),
        metric_definition='Each side independently: signed 6D actual-minus-own-issued-reference error; xyz in mm and world rotation-vector xyz in degrees',
        checks={k:bool(v) for k,v in checks.items()},output_validated=True)
    (dest/'pairing.json').write_text(json.dumps(record,indent=2)+'\n')
    for name,seconds in [('start',0),('middle',count/FPS/2),('terminal',count/FPS+1)]:
        subprocess.run(['ffmpeg','-v','error','-threads','1','-ss',str(seconds),'-i',str(video),
            '-frames:v','1',str(dest/f'{name}.png')],check=True,capture_output=True,text=True)
    return record


def concatenate(records, output, name):
    listing=output/f'{name}.concat.txt'
    listing.write_text(''.join(f"file '{r['video']}'\n" for r in records))
    metadata=output/f'{name}.ffmetadata'
    lines=[';FFMETADATA1','title=Original OSC vs IPOPT | identical relabelled actions | seed 0']
    offset=0
    for row in records:
        duration=round(row['duration_s']*1000)
        lines.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={offset}',f'END={offset+duration}',
                      f"title={row['task']} {row['demo']} | OSC vs IPOPT"])
        offset+=duration
    metadata.write_text('\n'.join(lines)+'\n')
    video=output/f'{name}.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','concat','-safe','0','-i',str(listing),
        '-i',str(metadata),'-map','0:v:0','-map_metadata','1','-map_chapters','1','-c','copy',
        '-movflags','+faststart',str(video)],check=True,capture_output=True,text=True)
    p=probe(video)
    assert int(p['nb_read_frames'])==sum(r['video_frames'] for r in records)
    decoded=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    assert decoded.returncode==0 and not decoded.stderr
    return str(video)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--wait-for-source',action='store_true',help='Pipeline composition with the original OSC recording batch')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    tasks=json.loads((ROOT/'data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json').read_text())['tasks']
    records=[]
    def render_task(task):
        report=BASE/'osc_reference_hdf5_for_pair40_v1'/task['suite']/f"{task['task_id']:02d}"/'report.json'
        if args.wait_for_source:
            deadline=time.monotonic()+1200
            while not report.exists():
                if time.monotonic()>deadline:
                    raise TimeoutError(f'OSC recording not ready: {report}')
                time.sleep(1)
        return make(task,args.output)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(render_task,t) for t in tasks]):
            record=future.result();records.append(record)
            print(f"{len(records)}/40 {record['task']} verified",flush=True)
    order={f"{t['suite']}/{t['task_id']:02d}":i for i,t in enumerate(tasks)}
    records.sort(key=lambda r:order[r['task']])
    all_video=concatenate(records,args.output,'all40_osc_vs_ipopt')
    focus_names=['libero_spatial/00','libero_spatial/06','libero_goal/05','libero_10/00','libero_goal/02','libero_10/03']
    focus=concatenate([next(r for r in records if r['task']==n) for n in focus_names],args.output,'focus6_osc_vs_ipopt')
    manifest=dict(scope='OSC freshly recorded by reading saved HDF5 actions; IPOPT reuses validated HDF5-driven recordings; composition adds no simulation steps; no HTML report',
        tasks=len(records),checks_passed=sum(sum(r['checks'].values()) for r in records),checks_total=sum(len(r['checks']) for r in records),
        outputs_validated=True,all40_video=all_video,focus6_video=focus,focus6_selection='One normal case and five contrasting failure/success cases; all 40 included separately and in all40 video',
        script_sha256=sha(Path(__file__)),metrics_module_sha256=sha(Path(__file__).with_name('paired_eef_metrics.py')),results=records)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    md=['# 原版 OSC 与 IPOPT 并排视频','',
        '左：原版 OSC；右：IPOPT + 原生关节控制 kp150。保持同一重构参考 action、seed=0、场景和起始状态。',
        '视频保持20Hz原速；最后2秒为实际终态。求解中止的右侧明确标注并保持终态，不继续仿真。','',
        '每侧分别显示实际末端相对自身已下发参考的六维有符号误差：Δx/Δy/Δz（mm）、ΔRx/ΔRy/ΔRz（世界轴旋转向量分量，°）。动作前画面对应上一条已下发参考。',
        '首帧为初始参考锚点；IPOPT中止后保持最后成功下发的参考，不使用求解失败的候选目标。逐帧数据保存在各任务 eef_frame_metrics.json。','',
        f'[六案例合集]({focus}) · [全部40任务合集]({all_video})','',
        '| 任务 | Demo | OSC末帧 | IPOPT末帧 | 并排视频 |','| --- | --- | --- | --- | --- |']
    for row in records:
        md.append(f"| {row['task']} | {row['demo']} | {'成功' if row['left_final_success'] else '失败'} | {'成功' if row['right_final_success'] else '失败'} | [MP4]({row['video']}) |")
    (args.output/'VIDEOS.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    main()
