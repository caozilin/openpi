"""Validate all saved angles using an independent, rotation-only XML FK chain."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def hand_chain(node, ancestors=()):
    path = ancestors + ((node,) if node.tag == 'body' else ())
    if node.get('name') == 'robot0_right_hand':
        return path
    for child in node:
        result = hand_chain(child, path)
        if result is not None:
            return result
    return None


def independent_closing_axis(xml, states):
    """Compose archived body quaternions and measured hinge rotations only."""
    tree = ET.fromstring(xml)
    assert tree.find('compiler').get('angle') == 'radian'
    assert not any(list(node.iter('joint')) for node in tree.findall('default'))
    world = tree.find('worldbody')
    qpos, address = {}, 0
    for joint in world.iter():
        if joint.tag not in ('joint', 'freejoint'):
            continue
        kind = 'free' if joint.tag == 'freejoint' else joint.get('type', 'hinge')
        qpos[joint.get('name')] = address
        address += {'free': 7, 'ball': 4, 'hinge': 1, 'slide': 1}[kind]
    result = np.broadcast_to(np.eye(3), (len(states), 3, 3)).copy()
    chain = hand_chain(world)
    assert chain is not None
    for body in chain:
        assert not any(name in body.attrib for name in ('euler', 'axisangle', 'xyaxes', 'zaxis'))
        wxyz = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
        result = result @ Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        assert not body.findall('freejoint')
        for joint in body.findall('joint'):
            assert joint.get('type', 'hinge') == 'hinge'
            axis = np.fromstring(joint.get('axis', '0 0 1'), sep=' ')
            axis /= np.linalg.norm(axis)
            angle = states[:, 1 + qpos[joint.get('name')]] - float(joint.get('ref', '0'))
            result = result @ Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
    return result[:, :, 1]


def validate(output):
    summary = json.loads((output/'summary.json').read_text())
    assert summary['complete']
    destination = output/'closing_vertical_angles.hdf5'
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == summary['angles_hdf5_sha256']
    with (output/'episodes.csv').open() as stream:
        csv_rows = {(r['suite'], r['task'], r['demo']): r for r in csv.DictReader(stream)}
    frames, episodes, tasks, max_axis_error, max_angle_error = 0, 0, 0, 0., 0.
    with h5py.File(destination, 'r') as annotations:
        for file in summary['inventory']:
            path = Path(file['source_hdf5'])
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (file['size_bytes'], file['mtime_ns'])
            task = path.stem.removesuffix('_demo')
            with h5py.File(path, 'r') as original:
                target = annotations[file['suite']][task]
                assert set(original['data']) == set(target)
                for demo, group in original['data'].items():
                    states = group['states'][()]
                    xml = str(group.attrs['model_file'])
                    record = target[demo]
                    assert hashlib.sha256(states.tobytes()).hexdigest() == record.attrs['source_states_sha256']
                    assert hashlib.sha256(xml.encode()).hexdigest() == record.attrs['source_model_xml_sha256']
                    np.testing.assert_array_equal(record['state_time_s'][()], states[:, 0])
                    np.testing.assert_array_equal(record['frame_index'][()], np.arange(len(states)))
                    expected_axis = independent_closing_axis(xml, states)
                    actual_axis = record['closing_axis_world'][()]
                    axis_error = np.minimum(np.linalg.norm(expected_axis-actual_axis, axis=1),
                                            np.linalg.norm(expected_axis+actual_axis, axis=1))
                    assert axis_error.max() < 1e-12
                    # Use acos rather than the annotator's atan2 expression.
                    expected = np.degrees(np.arccos(np.clip(np.abs(expected_axis[:, 2]), 0., 1.)))
                    actual = record['angle_deg'][()]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
                    row = csv_rows[(file['suite'], task, demo)]
                    assert int(row['frames']) == len(states)
                    assert float(row['min_deg']) == float(actual.min())
                    assert float(row['initial_deg']) == float(actual[0])
                    for threshold in (1, 5, 10, 15, 30, 45):
                        assert int(row[f'below_{threshold}_deg_frames']) == np.count_nonzero(actual < threshold)
                    max_axis_error = max(max_axis_error, float(axis_error.max()))
                    max_angle_error = max(max_angle_error, float(np.max(np.abs(actual-expected))))
                    frames += len(states)
                    episodes += 1
            tasks += 1
    assert (tasks, episodes, frames) == (summary['tasks'], summary['episodes'], summary['frames'])
    assert episodes == len(csv_rows)
    result = dict(validated=True, tasks=tasks, episodes=episodes, frames=frames,
                  all_source_state_and_xml_hashes_match=True, independent_xml_rotation_fk=True,
                  max_axis_error=max_axis_error, max_angle_error_deg=max_angle_error,
                  output_hdf5_sha256=summary['angles_hdf5_sha256'],
                  validation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'validation_source.py').write_bytes(Path(__file__).read_bytes())
    (output/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args().output.resolve())
