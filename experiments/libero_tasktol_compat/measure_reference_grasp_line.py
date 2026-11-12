"""Reference states only: object-center to physical jaw-line decomposition.

No rollout states, controller, optimizer or physics steps are used. Original
dataset states are the primary reference. Reconstructed nominal pose targets
are measured separately against those same reference object states.
"""
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_tasktol_compat.annotate_original_closing_vertical import remap_assets

RUN = ROOT / 'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'
OUT = ROOT / 'analysis/20261101_alphabet_soup_grasp_audit/reference_line_geometry'


def decompose(center, direction, point):
    center, direction, point = (np.asarray(x, dtype=float) for x in (center, direction, point))
    direction = direction / np.linalg.norm(direction)
    along_signed = float((point - center) @ direction)
    foot = center + along_signed * direction
    perpendicular = float(np.linalg.norm(point - foot))
    parallel = abs(along_signed)
    np.testing.assert_allclose((point-foot) @ direction, 0., atol=1e-12, rtol=0)
    np.testing.assert_allclose(perpendicular**2 + parallel**2,
                               np.sum((point-center)**2), atol=1e-12, rtol=1e-12)
    return dict(perpendicular_mm=perpendicular*1000, along_mm=parallel*1000,
                along_signed_mm=along_signed*1000, foot_m=foot.tolist(),
                center_distance_mm=float(np.linalg.norm(point-center)*1000))


def summary(rows):
    result = dict(frames=len(rows))
    for name in ['perpendicular_mm','along_mm','horizontal_perpendicular_mm','horizontal_along_mm']:
        a = np.asarray([r[name] for r in rows])
        result[name] = dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                           median=float(np.median(a)), p95=float(np.percentile(a,95)))
    return result


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((RUN/'plan.json').read_text())
    entries = [e for e in plan['entries'] if e['task']['task_name']=='pick_up_the_alphabet_soup_and_place_it_in_the_basket']
    results=[]; all_frames=[]; keyframes=[]; hashes={}
    for e in entries:
        source = Path(e['job']['source_hdf5']); refpath=Path(e['job']['reference_path']); demo=e['job']['demo']
        with h5py.File(source) as h:
            group=h['data'][demo]; states=group['states'][()]; actions=group['actions'][()]
            xml=group.attrs['model_file']
        with h5py.File(refpath) as h:
            ref={k:v[()]for k,v in h.items()}
        n=len(ref['positions']); np.testing.assert_array_equal(ref['source_state_coordinate'],np.arange(n))
        assert len(states)==n and len(ref['actions'])==n-1
        np.testing.assert_array_equal(ref['actions'][:,6],actions[:n-1,6])
        tree=ET.fromstring(xml)
        remap_assets(tree,ROOT/'third_party/libero/libero/libero/assets',Path(importlib.util.find_spec('robosuite').origin).parent)
        model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));data=mujoco.MjData(model)
        body=model.body('alphabet_soup_1_main').id;hand=model.body('robot0_right_hand').id
        site=model.site('gripper0_grip_site').id
        pads=[model.geom(f'gripper0_finger{i}_pad_collision').id for i in (1,2)]
        # Explicit geometric center of the can collision bounding box. Keep
        # body-origin measurements alongside it because the two differ by mm.
        vertices=[]
        for g in np.flatnonzero(model.geom_bodyid==body):
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:continue
            corners=np.array(list(itertools.product((-1.,1.),repeat=3)))*model.geom_size[g]
            r=np.empty(9);mujoco.mju_quat2Mat(r,model.geom_quat[g])
            vertices.extend(corners@r.reshape(3,3).T+model.geom_pos[g])
        vertices=np.array(vertices);bounds=np.stack([vertices.min(axis=0),vertices.max(axis=0)])
        offset=bounds.mean(axis=0)
        measurements={name:[]for name in ['reference_state_pad_midpoint','reference_state_tcp',
            'reconstructed_nominal_tcp','reconstructed_nominal_pad_midpoint']}
        full=[]
        for i,state in enumerate(states):
            data.qpos[:]=state[1:1+model.nq]
            mujoco.mj_kinematics(model,data)
            p=data.site_xpos[site].copy();r=data.xmat[hand].reshape(3,3).copy()
            np.testing.assert_allclose(p,ref['source_positions_m'][i],atol=5e-7,rtol=0)
            np.testing.assert_allclose(r,ref['source_rotations'][i],atol=5e-7,rtol=0)
            origin=data.xpos[body].copy();obj=origin+data.xmat[body].reshape(3,3)@offset
            padcenters=data.geom_xpos[pads].copy();center=padcenters.mean(axis=0)
            axis=padcenters[0]-padcenters[1];axis/=np.linalg.norm(axis)
            localaxis=r.T@axis;localoffset=r.T@(center-p)
            pn,rn=ref['positions'][i],ref['rotations'][i]
            options=dict(reference_state_pad_midpoint=(center,axis),reference_state_tcp=(p,axis),
                reconstructed_nominal_tcp=(pn,rn@localaxis),
                reconstructed_nominal_pad_midpoint=(pn+rn@localoffset,rn@localaxis))
            variants={}
            for name,(cp,u) in options.items():
                values=decompose(cp,u,obj);xy=decompose(cp[:2],u[:2],obj[:2]);bodyvalues=decompose(cp,u,origin)
                row=dict(demo=demo,frame=i,time_s=i*.05,variant=name,
                    command=float(ref['actions'][i,6])if i<n-1 else None,
                    perpendicular_mm=values['perpendicular_mm'],along_mm=values['along_mm'],
                    horizontal_perpendicular_mm=xy['perpendicular_mm'],horizontal_along_mm=xy['along_mm'],
                    along_signed_mm=values['along_signed_mm'],center_distance_mm=values['center_distance_mm'],
                    vertical_center_offset_mm=float((obj[2]-cp[2])*1000),
                    body_origin_perpendicular_mm=bodyvalues['perpendicular_mm'],body_origin_along_mm=bodyvalues['along_mm'])
                measurements[name].append(row);all_frames.append(row)
                variants[name]=dict(gripper_center_m=np.asarray(cp).tolist(),closing_direction_world=np.asarray(u).tolist(),
                    foot_m=values['foot_m'],horizontal_foot_m=xy['foot_m'],**row)
            full.append(dict(frame=i,time_s=i*.05,cylinder_center_m=obj.tolist(),object_body_origin_m=origin.tolist(),measurements=variants))
        starts=[i for i,a in enumerate(ref['actions']) if a[6]>0 and (i==0 or ref['actions'][i-1,6]<=0)]
        attempts=[]
        for number,c in enumerate(starts):
            release=next((i for i in range(c+1,n-1)if ref['actions'][i,6]<=0),n-1)
            event=dict(attempt=number+1,close_frame=c,close_time_s=c*.05,release_frame=release,measurements={})
            for name,rows in measurements.items():
                event['measurements'][name]=dict(close_state=rows[c],
                    approach_last_0p5s=summary(rows[max(c-10,0):c]),
                    closing_first_0p3s=summary(rows[c:min(c+6,release)]))
                keyframes.append(dict(attempt=number+1,**rows[c]))
            attempts.append(event)
        write_csv(OUT/f'{demo}_frames.csv',[r for r in all_frames if r['demo']==demo])
        hashes.setdefault(str(source),hashlib.sha256(source.read_bytes()).hexdigest()) if str(source)not in hashes else None
        results.append(dict(key=e['key'],demo=demo,frames=n,source_hdf5=str(source),reference_hdf5=str(refpath),
            source_sha256=hashes[str(source)],reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
            original_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            cylinder_center_body_offset_m=offset.tolist(),collision_bbox_body_m=bounds.tolist(),
            full_trajectory={name:summary(rows)for name,rows in measurements.items()},attempts=attempts))
        (OUT/f'{demo}_geometry.json').write_text(json.dumps(full,ensure_ascii=False,indent=2)+'\n')
        a=attempts[0]['measurements']['reference_state_pad_midpoint'];s=a['close_state']
        print(demo,'close',s['frame'],'dperp',round(s['perpendicular_mm'],2),'along',round(s['along_mm'],2),
            'XY',round(s['horizontal_perpendicular_mm'],2),round(s['horizontal_along_mm'],2),flush=True)
    write_csv(OUT/'all_frames.csv',all_frames);write_csv(OUT/'close_states.csv',keyframes)
    result=dict(schema='reference_gripper_line_geometry.1',episodes=results,
        definitions=dict(primary='Original dataset reference states, physical finger-pad midpoint C, unit closing line u through C; no tolerance rollout actual states.',
            cylinder_center='Center of local collision-box bounding bounds, transformed by reference object pose; body-origin alternatives saved.',
            equation='H=C+u*u.T*(O-C); d_perp=||O-H||; d_along=||H-C||=|u.T*(O-C)|',
            dimension='3D primary; XY horizontal projection computed independently, normalized projected closing axis.',
            reconstructed_nominal='Independent positions/rotations from state_reference.hdf5, paired with reference object poses. Separate from source reference-state measurements.',
            time='20Hz reference state i precedes action i; first close index from unchanged original gripper command; no interpolation or time warping.',
            phase_windows='Last10 preclose states=0.5s; first6 closing states=0.3s. All grasp attempts separately included.'),
        validation=dict(ten_episodes=len(results)==10,no_rollout_states_used=True,no_physics_steps=True,
            source_state_coordinates_identity=True,source_fk_pose_matches_reference=True,
            pythagorean_identity_and_orthogonal_foot_checked_every_frame=True))
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Tabulate close states, with the second original attempt explicitly kept.
    lines=['# 参考state：圆柱中心到夹爪闭合直线的距离','',
        '仅分析10条原始示范参考state，不使用容差rollout实际state。C为两指指垫几何中心的中点，u为两指连线的单位方向，O为罐体碰撞几何局部包围盒中心。',
        '', '`H=C+u[uᵀ(O−C)]`；`d⊥=‖O−H‖`；`d∥=‖H−C‖=|uᵀ(O−C)|`。单位均为mm。水平口径先将C/O/u投影XY，再重新归一化u，避免接近高度差混入偏心。',
        '', '主表：每次闭爪命令发出前的参考state。demo29保留两次原始抓取。', '',
        '|demo|尝试|闭爪帧|3D圆心到线 d⊥|垂足到夹爪中心 d∥|XY圆心到线|XY沿线距离|接近末0.5s d⊥均值|接近末0.5s d∥均值|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ep in results:
        for a in ep['attempts']:
            m=a['measurements']['reference_state_pad_midpoint'];s=m['close_state'];w=m['approach_last_0p5s']
            lines.append(f"|{ep['demo']}|{a['attempt']}|{s['frame']}|{s['perpendicular_mm']:.2f}|{s['along_mm']:.2f}|{s['horizontal_perpendicular_mm']:.2f}|{s['horizontal_along_mm']:.2f}|{w['perpendicular_mm']['mean']:.2f}|{w['along_mm']['mean']:.2f}|")
    lines+=['','## 重构后的独立名义TCP轨迹（分开口径）','',
        '下表使用重构命令的参考位置/姿态及同帧原始参考物体state；中心是TCP，方向由参考手部闭合轴映射到该名义姿态。不能与上表原始state的物理指垫中点混为一谈。','',
        '|demo|尝试|3D d⊥|d∥|XY d⊥|XY d∥|','|---|---:|---:|---:|---:|---:|']
    for ep in results:
        for a in ep['attempts']:
            s=a['measurements']['reconstructed_nominal_tcp']['close_state']
            lines.append(f"|{ep['demo']}|{a['attempt']}|{s['perpendicular_mm']:.2f}|{s['along_mm']:.2f}|{s['horizontal_perpendicular_mm']:.2f}|{s['horizontal_along_mm']:.2f}|")
    lines+=['','## 中心定义与留存','',
        f"该罐体碰撞几何中心相对body原点的局部偏置为{np.array(results[0]['cylinder_center_body_offset_m'])*1000} mm；因此另外保存body原点口径。几何包围盒中心是明确可复核的近似圆柱中心，不是额外拟合出的精确曲面轴。",'',
        'summary.json：各示例完整轨迹均值/中位数/P95/最小/最大、各抓取尝试和分窗统计、输入SHA256；all_frames.csv：全部参考帧四种中心/姿态口径；close_states.csv：各次闭爪帧；demo_*_geometry.json：点、方向与垂足坐标。',
        '', '每帧校验垂足正交及d⊥²+d∥²=‖O−C‖²；原state FK位置/旋转与state_reference中的source分支核对。完整轨迹包括接近/搬运/放置，判断抓取偏心应优先查看闭爪及接近末段。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
