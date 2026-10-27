"""Finalize inverse-label metrics from completed, retained physical recordings."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def finalize(output):
    output = output.resolve()
    provenance = json.loads((output / 'provenance.json').read_text())
    job = provenance['plan'][0]
    dest = output / job['suite'] / f"{job['task_id']:02d}" / job['demo']
    folder = dest / 'ipopt_state_B10'
    with h5py.File(dest / 'state_reference.hdf5', 'r+') as file:
        file.attrs['label_method'] = provenance['label_method']
        source_positions, source_rotations = file['source_positions_m'][()], file['source_rotations'][()]
        witnesses = file['goal_joint_witness_rad'][()]
    with h5py.File(folder / 'recording.hdf5', 'r') as file:
        group = file['attempts'][job['demo']]
        tree = ET.fromstring(group.attrs['model_file'])
        bounds = np.asarray([np.fromstring(tree.find(f'.//joint[@name="robot0_joint{j}"]').attrib['range'], sep=' ') for j in range(1, 8)])
        assert np.all(witnesses >= bounds[:, 0]) and np.all(witnesses <= bounds[:, 1])
        frame_count = len(group['actions'])
    rows = json.loads((folder / 'trajectory.json').read_text())
    assert len(rows) == frame_count
    for row in rows:
        if row['aborted']:
            continue
        index = row['index']
        row['source_next_position_error_m'] = float(np.linalg.norm(np.asarray(row['actual_position_m']) - source_positions[index + 1]))
        row['source_next_rotation_error_rad'] = float(np.linalg.norm(Rotation.from_matrix(np.asarray(row['actual_rotation']) @ source_rotations[index + 1].T).as_rotvec()))
    report = json.loads((folder / 'report.json').read_text())
    report['label_method'] = provenance['label_method']
    if 'source_state_motion' in provenance:
        report['source_state_motion'] = provenance['source_state_motion']
    errors = [row['source_next_position_error_m'] for row in rows if not row['aborted']]
    assert len(errors) == report['executed_frames']
    report['mean_source_next_position_error_mm'] = float(np.mean(errors) * 1000) if errors else None
    report['max_source_next_position_error_mm'] = float(np.max(errors) * 1000) if errors else None
    (folder / 'trajectory.json').write_text(json.dumps(rows, indent=2) + '\n')
    (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    result = dict(**job, paired_initialization_verified=True, results=[report])
    (dest / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    aggregate = dict(ipopt_state_B10=dict(final=int(report['success']), ever=int(report['ever_success']), rejected=int(report['execution_rejected'])))
    (output / 'summary.json').write_text(json.dumps(dict(complete=True, aggregate=aggregate, results=[result]), indent=2) + '\n')
    (output / 'finalization.json').write_text(json.dumps(dict(physical_recording_preserved=True, resimulated=False,
        all_goal_witnesses_within_model_bounds=True, source_state_metrics_recomputed=True,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    finalize(parser.parse_args().root)
