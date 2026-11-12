"""Read saved states only: paired grasp geometry/contact audit, without rollout.

Uses each original episode XML with asset paths remapped in memory. Contacts
are geometric reconstructions at saved20Hz states, not measured force traces.
"""
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_tasktol_compat.annotate_original_closing_vertical import remap_assets

RUN = ROOT / 'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'
OUT = ROOT / 'analysis/20261101_alphabet_soup_grasp_audit'


def rot(v):
    return Rotation.from_rotvec(v).as_matrix()


def degrees(r):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(r).as_rotvec())))


def sample(model, data, state, row, previous_nominal):
    data.time = state[0]
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:]
    mujoco.mj_forward(model, data)
    site = model.site('gripper0_grip_site').id
    body = model.body('alphabet_soup_1_main').id
    p = data.site_xpos[site].copy()
    site_r = data.site_xmat[site].reshape(3, 3).copy()
    r = data.xmat[model.body('robot0_right_hand').id].reshape(3, 3).copy()
    np.testing.assert_allclose(p, row['pre_action']['position_m'], atol=1e-10, rtol=0)
    # Client uses grip_site POSITION plus right_hand ROTATION. The site has a
    # fixed 90deg yaw offset; using its rotation silently swaps physical axes.
    # Archived XML rounds one fixed gripper quaternion; sub-microradian FK
    # mismatch is expected, while positions must match to numerical precision.
    np.testing.assert_allclose(r, rot(row['pre_action']['rotation_vector_rad']), atol=5e-7, rtol=0)
    obj = data.xpos[body].copy()
    obj_r = data.xmat[body].reshape(3, 3).copy()
    pads = [model.geom(f'gripper0_finger{j}_pad_collision').id for j in (1, 2)]
    centers = np.array([data.geom_xpos[g] for g in pads])
    axis = centers[0] - centers[1]
    axis /= np.linalg.norm(axis)
    contacts = []
    robot_scene_contacts = []
    for c in data.contact[:data.ncon]:
        a, b = int(c.geom1), int(c.geom2)
        na, nb = model.geom(a).name, model.geom(b).name
        robot_a = na.startswith(('robot0_', 'gripper0_'))
        robot_b = nb.startswith(('robot0_', 'gripper0_'))
        if robot_a != robot_b and c.dist <= 1e-7:
            robot_scene_contacts.append(dict(robot_geom=na if robot_a else nb,
                scene_geom=nb if robot_a else na, distance_mm=float(c.dist * 1000), position_m=c.pos.tolist()))
        if model.geom_bodyid[b] == body:
            a, b = b, a
        if model.geom_bodyid[a] == body and c.dist <= 1e-7:
            contacts.append(dict(other_geom=model.geom(b).name, distance_mm=float(c.dist * 1000),
                                 position_m=c.pos.tolist()))
    names = [c['other_geom'] for c in contacts]
    finger_contact = [any(f'gripper0_finger{j}_' in name for name in names) for j in (1, 2)]
    vertices = []
    for g in np.flatnonzero(model.geom_bodyid == body):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        corners = np.array(list(itertools.product((-1., 1.), repeat=3))) * model.geom_size[g]
        vertices.extend(corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    # Projection width of ALL soup collision boxes along the closing line.
    projections = np.array(vertices) @ axis
    local = r.T @ (obj - p)
    return dict(frame=int(row['index']), time_s=float(row['index'] * .05), phase=row['phase'],
        effective_mask=row['effective_mask'], command=float(row['action'][6]),
        ee_position_m=p.tolist(), ee_rotation=r.tolist(), grip_site_rotation=site_r.tolist(), object_position_m=obj.tolist(),
        object_rotation=obj_r.tolist(), object_in_ee_m=local.tolist(),
        nominal_position_m=previous_nominal[0].tolist(), nominal_rotation=previous_nominal[1].tolist(),
        ee_nominal_position_error_mm=float(np.linalg.norm(p - previous_nominal[0]) * 1000),
        ee_nominal_rotation_error_deg=degrees(r @ previous_nominal[1].T),
        closing_vertical_angle_deg=float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0, 1)))),
        finger_aperture_mm=float(row['phase_aperture_m'] * 1000),
        pad_center_gap_mm=float(np.linalg.norm(centers[0] - centers[1]) * 1000),
        pad_midpoint_m=centers.mean(axis=0).tolist(),
        object_projection_width_mm=float(np.ptp(projections) * 1000),
        finger_contact=finger_contact, bilateral_contact=all(finger_contact), contacts=contacts,
        robot_scene_contacts=robot_scene_contacts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((RUN / 'plan.json').read_text())
    entries = [e for e in plan['entries'] if e['task']['task_name'] == 'pick_up_the_alphabet_soup_and_place_it_in_the_basket']
    results = []
    for e in entries:
        with h5py.File(e['job']['source_hdf5']) as h:
            xml = h['data'][e['job']['demo']].attrs['model_file']
        tree = ET.fromstring(xml)
        remap_assets(tree, ROOT / 'third_party/libero/libero/libero/assets',
                     Path(importlib.util.find_spec('robosuite').origin).parent)
        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
        data = mujoco.MjData(model)
        sides = {}
        for side, folder in [('baseline', Path(e['baseline_folder'])), ('tolerance', RUN / 'episodes' / e['key'])]:
            rows = json.loads((folder / 'trajectory.json').read_text())
            states = np.load(folder / 'states.npz')['pre_action_states']
            with h5py.File(e['job']['reference_path']) as h:
                previous = (h['positions'][0], h['rotations'][0])
            samples = []
            for state, row in zip(states, rows, strict=True):
                samples.append(sample(model, data, state, row, previous))
                previous = (np.array(row['nominal_target']['position_m']), rot(row['nominal_target']['rotation_vector_rad']))
            close = next(i for i, r in enumerate(rows) if r['action'][6] > 0)
            release = next((i for i in range(close + 1, len(rows)) if rows[i]['action'][6] <= 0), len(rows))
            first_contact = next((s['frame'] for s in samples if any('gripper0' in c['other_geom'] for c in s['contacts'])), None)
            first_bilateral = next((s['frame'] for s in samples if s['bilateral_contact']), None)
            obj = np.array([s['object_position_m'] for s in samples])
            moved = np.linalg.norm(obj - obj[0], axis=1) * 1000
            first_move = next((i for i, v in enumerate(moved) if v > 2), None)
            report = json.loads((folder / 'report.json').read_text())
            attempts = []
            for begin in [i for i, r in enumerate(rows) if r['action'][6] > 0 and (i == 0 or rows[i-1]['action'][6] <= 0)]:
                end = next((i for i in range(begin + 1, len(rows)) if rows[i]['action'][6] <= 0), len(rows))
                attempts.append(dict(close_frame=begin, release_frame=end,
                    max_lift_mm=float((obj[begin:end, 2].max() - obj[0, 2]) * 1000),
                    bilateral_contact_frames=sum(s['bilateral_contact'] for s in samples[begin:end]),
                    bilateral3_frame=next((i for i in range(begin, end - 2) if all(samples[z]['bilateral_contact'] for z in range(i, i + 3))), None)))
            sides[side] = dict(success=report['success'], rejected=report['execution_rejected'],
                attempts=attempts,
                close_frame=close, release_frame=release, first_robot_contact_frame=first_contact,
                first_bilateral_contact_frame=first_bilateral, first_object_move_gt2mm_frame=first_move,
                preclose_object_movement_mm=float(moved[:close + 1].max()),
                max_lift_while_closed_mm=float((obj[close:release, 2].max() - obj[0, 2]) * 1000),
                bilateral_contact_closed_frames=sum(s['bilateral_contact'] for s in samples[close:release]),
                closed_frames=release - close,
                input_sha256={str(folder / name):hashlib.sha256((folder / name).read_bytes()).hexdigest()
                              for name in ['trajectory.json', 'states.npz', 'report.json']},
                samples=samples)
        b, t = sides['baseline'], sides['tolerance']
        assert b['close_frame'] == t['close_frame']
        c = b['close_frame']
        bs, ts = b['samples'][c], t['samples'][c]
        pair = dict(ee_separation_mm=float(np.linalg.norm(np.array(bs['ee_position_m']) - ts['ee_position_m']) * 1000),
            object_separation_mm=float(np.linalg.norm(np.array(bs['object_position_m']) - ts['object_position_m']) * 1000),
            ee_rotation_separation_deg=degrees(np.array(ts['ee_rotation']) @ np.array(bs['ee_rotation']).T))
        results.append(dict(key=e['key'], source_hdf5=e['job']['source_hdf5'],
            original_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(), close_comparison=pair, **sides))
        print(e['key'], 'success', t['success'], 'close', c, 'B/T lift',
              round(b['max_lift_while_closed_mm'], 1), round(t['max_lift_while_closed_mm'], 1),
              'ee/obj/rot', {k:round(v, 2) for k, v in pair.items()},
              'first contacts', b['first_robot_contact_frame'], t['first_robot_contact_frame'],
              'preclose move', round(t['preclose_object_movement_mm'], 2),
              'both', b['bilateral_contact_closed_frames'], t['bilateral_contact_closed_frames'], flush=True)
    result = dict(schema='alphabet_soup_paired_grasp_audit.1',
        method='Saved20Hz pre-action states; original per-demo XML; MuJoCo FK and geometric contacts only, no physics stepping, no controller/solver changes. Contact forces not reconstructed.',
        episodes=results)
    (OUT / 'diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    flat = []
    for ep in results:
        for side in ['baseline', 'tolerance']:
            s = ep[side]; frame = s['samples'][s['close_frame']]
            flat.append(dict(key=ep['key'], variant=side, **{k:v for k,v in s.items() if k not in ['samples', 'input_sha256']},
                close_ee_nominal_position_error_mm=frame['ee_nominal_position_error_mm'],
                close_ee_nominal_rotation_error_deg=frame['ee_nominal_rotation_error_deg'],
                close_closing_vertical_angle_deg=frame['closing_vertical_angle_deg'],
                close_object_projection_width_mm=frame['object_projection_width_mm']))
    with (OUT / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)


if __name__ == '__main__':
    main()
