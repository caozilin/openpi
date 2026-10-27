"""Read original LIBERO states and annotate the physical closing-axis line angle.

Only forward kinematics is evaluated; no controller, optimizer or physics step
is run. Source HDF5 files and shared MuJoCo code are opened read-only.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
THRESHOLDS = (1, 5, 10, 15, 30, 45)
HAND = 'robot0_right_hand'
FINGERS = ('gripper0_finger_joint1', 'gripper0_finger_joint2')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def closing_vertical_angle_deg(axes):
    """Unoriented line angle in [0, 90], matching acute_vertical_angle_deg."""
    axes = np.asarray(axes, dtype=float)
    if axes.shape[-1] != 3 or not np.all(np.isfinite(axes)):
        raise ValueError('Expected finite closing-axis three-vectors')
    if np.any(np.linalg.norm(axes, axis=-1) < 1e-12):
        raise ValueError('Closing axis must be nonzero')
    return np.degrees(np.arctan2(np.linalg.norm(axes[..., :2], axis=-1), np.abs(axes[..., 2])))


def remap_assets(tree, assets_root, robosuite_root):
    changes = []
    for node in tree.iter():
        old = node.get('file')
        if old is None:
            continue
        if '/robosuite/' in old:
            new = robosuite_root / old.rsplit('/robosuite/', 1)[1]
        elif '/assets/' in old:
            new = assets_root / old.split('/assets/', 1)[1]
        else:
            raise ValueError(f'Unknown archived asset path: {old}')
        if not new.is_file():
            raise FileNotFoundError(new)
        node.set('file', str(new))
        changes.append(dict(original=old, local=str(new)))
    return changes


def robot_key(tree):
    """Reuse FK only if robot frames, defaults, and full qpos ordering agree.

    Episode XML varies object placements. Those placements cannot affect this
    fixed-base robot's FK. The complete robot subtree must be identical.
    """
    world = tree.find('worldbody')
    bases = [b for b in world.findall('body') if b.get('name') == 'robot0_base']
    if len(bases) != 1:
        raise ValueError('Expected one world-rooted robot0_base')
    layout = [(n.tag, n.get('name'), n.get('type', 'hinge'))
              for n in world.iter() if n.tag in ('joint', 'freejoint')]
    payload = dict(base=ET.tostring(bases[0], encoding='unicode'),
                   compiler=[ET.tostring(n, encoding='unicode') for n in tree.findall('compiler')],
                   defaults=[ET.tostring(n, encoding='unicode') for n in tree.findall('default')],
                   world_attributes=world.attrib, joint_order=layout)
    return sha(json.dumps(payload, sort_keys=True).encode())


def summarize(angles):
    values = np.asarray(angles)
    return dict(frames=len(values), min_deg=float(values.min()), mean_deg=float(values.mean()),
                max_deg=float(values.max()),
                **{f'p{p:02d}_deg': float(np.percentile(values, p)) for p in (1, 5, 50, 95, 99)},
                **{f'below_{t}_deg_frames': int(np.count_nonzero(values < t)) for t in THRESHOLDS})


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotate(source, output, assets_root, robosuite_root):
    output.mkdir(parents=True, exist_ok=False)
    definition_sources = [Path(__file__), ROOT.parent/'franka_mujoco/benchmark_passive_frame_drift.py',
                          ROOT.parent/'franka_mujoco/tasktol/tolerance/frame.py',
                          ROOT.parent/'franka_mujoco/tasktol/simulation/libero.py']
    snapshots = output/'source_snapshot'
    snapshots.mkdir()
    source_hashes = {}
    for index, path in enumerate(definition_sources):
        source_hashes[str(path)] = sha(path.read_bytes())
        shutil.copyfile(path, snapshots/f'{index}_{path.name}')
    files = [(suite, p) for suite in SUITES for p in sorted((source/suite).glob('*.hdf5'))]
    for suite in SUITES:
        if sum(name == suite for name, _ in files) != 10:
            raise ValueError(f'Expected all 10 original tasks in {suite}')
    episode_rows, task_rows, all_angles, inventory = [], [], [], []
    maximum_axis_disagreement = 0.
    destination = output/'closing_vertical_angles.hdf5'
    with h5py.File(destination, 'w') as dest:
        dest.attrs.update(schema='libero_original_state_closing_vertical_v1',
                          definition='atan2(norm(axis_world[:2]), abs(axis_world[2])) in degrees',
                          range_deg='[0,90]; horizontal=90, vertical=0; axis sign ignored',
                          pose_source='original states qpos + each archived model robot kinematics',
                          direction_source='physical first finger slide axis; checked parallel to hand local Y and second finger',
                          source_root=str(source), physics_steps=0, optimizer_calls=0,
                          source_sha256=json.dumps(source_hashes))
        for suite, path in files:
            before_stat = path.stat()
            task_name = path.stem.removesuffix('_demo')
            task_group = dest.require_group(suite).create_group(task_name)
            task_group.attrs['source_hdf5'] = str(path)
            task_angles = []
            previous_key, model, data = None, None, None
            model_records = []
            with h5py.File(path, 'r') as original:
                source_data = original['data']
                demos = sorted(source_data, key=lambda x: int(x.removeprefix('demo_')))
                assert len(demos) == int(source_data.attrs['num_demos'])
                config = json.loads(source_data.attrs['env_args'])
                control_hz = config['env_kwargs']['control_freq']
                assert control_hz == 20
                for demo in demos:
                    group = source_data[demo]
                    states = group['states'][()]
                    xml = str(group.attrs['model_file'])
                    tree = ET.fromstring(xml)
                    key = robot_key(tree)
                    if key != previous_key:
                        changes = remap_assets(tree, assets_root, robosuite_root)
                        model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
                        data = mujoco.MjData(model)
                        hand_id = model.body(HAND).id
                        joint_ids = [model.joint(name).id for name in FINGERS]
                        assert all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in joint_ids)
                        model_records.append(dict(first_demo=demo, robot_fk_key=key, nq=model.nq, nv=model.nv,
                                                  archived_xml_sha256=sha(xml.encode()), asset_path_mappings=changes))
                        previous_key = key
                    assert states.ndim == 2 and states.shape[1] == 1 + model.nq + model.nv
                    assert np.all(np.isfinite(states))
                    np.testing.assert_allclose(np.diff(states[:, 0]), 1/control_hz, rtol=0, atol=1e-9)
                    axes = np.empty((len(states), 3))
                    hand_axes = np.empty_like(axes)
                    second_axes = np.empty_like(axes)
                    for index, state in enumerate(states):
                        data.time = state[0]
                        data.qpos[:] = state[1:1 + model.nq]
                        mujoco.mj_kinematics(model, data)
                        hand_axes[index] = data.xmat[hand_id].reshape(3, 3)[:, 1]
                        for values, joint in zip((axes, second_axes), joint_ids, strict=True):
                            values[index] = data.xmat[model.jnt_bodyid[joint]].reshape(3, 3) @ model.jnt_axis[joint]
                    disagreement = max(float(np.max(1-np.abs(np.sum(axes*hand_axes, axis=1)))),
                                       float(np.max(1-np.abs(np.sum(axes*second_axes, axis=1)))))
                    maximum_axis_disagreement = max(maximum_axis_disagreement, disagreement)
                    assert disagreement < 1e-12, 'Closing axis differs from corrected tool Y'
                    angles = closing_vertical_angle_deg(axes)
                    np.testing.assert_allclose(angles, closing_vertical_angle_deg(hand_axes), rtol=0, atol=1e-10)
                    horizontal_norm = np.linalg.norm(axes[:, :2], axis=1)
                    sample = task_group.create_group(demo)
                    for name, value in dict(frame_index=np.arange(len(states)), state_time_s=states[:, 0],
                                            elapsed_time_s=states[:, 0]-states[0, 0],
                                            closing_axis_world=axes, angle_deg=angles,
                                            horizontal_projection_norm=horizontal_norm).items():
                        sample.create_dataset(name, data=value, compression='gzip', shuffle=True)
                    minimum = int(np.argmin(angles))
                    row = dict(suite=suite, task=task_name, demo=demo, **summarize(angles),
                               initial_deg=float(angles[0]), final_deg=float(angles[-1]),
                               min_frame=minimum, min_elapsed_s=float(states[minimum, 0]-states[0, 0]),
                               heading_fallback_frames=int(np.count_nonzero(horizontal_norm < 1e-8)),
                               libero_heading_guard_frames=int(np.count_nonzero(horizontal_norm < 1e-6)))
                    sample.attrs.update(**row, source_states_sha256=sha(states.tobytes()),
                                        source_model_xml_sha256=sha(xml.encode()), robot_fk_key=key)
                    episode_rows.append(row)
                    task_angles.append(angles)
                values = np.concatenate(task_angles)
                task_rows.append(dict(suite=suite, task=task_name, episodes=len(demos), **summarize(values),
                                      initial_min_deg=min(r['initial_deg'] for r in episode_rows[-len(demos):]),
                                      initial_max_deg=max(r['initial_deg'] for r in episode_rows[-len(demos):])))
                task_group.attrs['models'] = json.dumps(model_records)
            after_stat = path.stat()
            assert (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
            inventory.append(dict(suite=suite, source_hdf5=str(path), size_bytes=before_stat.st_size,
                                  mtime_ns=before_stat.st_mtime_ns, episodes=len(demos), frames=len(values),
                                  compiled_models=len(model_records)))
            all_angles.append(values)
            print(json.dumps(dict(tasks_complete=len(task_rows), tasks_total=len(files), suite=suite,
                                  task=task_name, episodes=len(demos), frames=len(values), min_deg=float(values.min()))), flush=True)
    write_csv(output/'episodes.csv', episode_rows)
    write_csv(output/'tasks.csv', task_rows)
    values = np.concatenate(all_angles)
    summary = dict(complete=True, task_sets=len(SUITES), tasks=len(files), episodes=len(episode_rows),
                   **summarize(values), initial_min_deg=min(r['initial_deg'] for r in episode_rows),
                   initial_max_deg=max(r['initial_deg'] for r in episode_rows),
                   max_closing_axis_alignment_error=maximum_axis_disagreement,
                   heading_fallback_frames=sum(r['heading_fallback_frames'] for r in episode_rows),
                   libero_heading_guard_frames=sum(r['libero_heading_guard_frames'] for r in episode_rows),
                   source_sha256=source_hashes, inventory=inventory,
                   smallest_angle_episodes=sorted(episode_rows, key=lambda r: r['min_deg'])[:20])
    summary['suites'] = {}
    for suite in SUITES:
        subset = np.concatenate([v for (s, _), v in zip(files, all_angles, strict=True) if s == suite])
        summary['suites'][suite] = dict(tasks=sum(r['suite'] == suite for r in task_rows),
                                       episodes=sum(r['suite'] == suite for r in episode_rows), **summarize(subset))
    summary['angles_hdf5_sha256'] = sha(destination.read_bytes())
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'inventory', 'smallest_angle_episodes', 'suites')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'third_party/libero/libero/datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=ROOT/'third_party/libero/libero/libero/assets')
    args = parser.parse_args()
    spec = importlib.util.find_spec('robosuite')
    annotate(args.source.resolve(), args.output.resolve(), args.assets.resolve(), Path(spec.origin).parent)
