"""Independently verify persisted pilot artifacts; never execute a trial."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/media/czl/sata/franka_my_code/franka_mujoco')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    checked = {}
    episodes = []
    trial_count = 0
    for run_name in ('final_default16', 'final_smoke2'):
        run = ROOT / run_name
        summary = read(run / 'annotation_summary.json')
        assert summary['processing_complete'] and summary['adapter_errors'] == 0
        for path in sorted(run.rglob('prepared_input.json')):
            d = read(path)
            source = d['source']
            output = path.parent
            status = read(output / 'annotation_status.json')
            assert status['complete'] and not status['preflight']
            provenance = d['provenance']
            for mapping in (provenance['input_sha256'], provenance['source_code_sha256'], d['metadata']['asset_sha256']):
                for name, expected in mapping.items():
                    if name == 'source_xml_sha256':
                        continue
                    if name not in checked:
                        checked[name] = sha(name)
                    assert checked[name] == expected, name
            assert provenance['control_hz'] == 20
            metadata = d['metadata']
            assert metadata['physics_steps'] == 0
            assert not metadata['task_scene_instantiated'] and not metadata['task_objects_loaded']
            xml = ET.fromstring(metadata['robot_xml'])
            assert {b.get('name') for b in xml.find('worldbody').findall('body')} == {'robot0_base', 'work_surface'}
            assert all(j.get('type') != 'free' for j in xml.iter('joint'))
            # Verify frozen source actions reconstruct consecutive nominal goals.
            frames = d['frames']
            assert len(frames) == source['frames'] == status['frames']
            assert [f['source_frame_index'] for f in frames] == list(range(len(frames)))
            for i, frame in enumerate(frames):
                assert frame['time_s'] == i / 20
                assert np.isfinite(frame['action']).all()
                if i:
                    previous = frames[i - 1]['nominal_target_pose']
                    goal = frame['nominal_target_pose']
                    assert np.allclose(np.asarray(previous['position_m']) + frame['action'][:3], goal['position_m'], atol=1e-8, rtol=0)
                    predicted = Rotation.from_rotvec(frame['action'][3:6]).as_matrix() @ Rotation.from_rotvec(previous['rotation_vector_base_rad']).as_matrix()
                    actual = Rotation.from_rotvec(goal['rotation_vector_base_rad']).as_matrix()
                    assert np.linalg.norm(Rotation.from_matrix(predicted @ actual.T).as_rotvec()) < 1e-6
            selections = []
            segments = []
            for index, subtask in enumerate(d['subtasks']):
                a = read(output / f'subtask_{index:02d}/annotation.json')
                observations = a['observations']
                expected_queries = a['design']['unique_initial_count'] + status['extra']
                assert len(observations) == expected_queries
                assert a['additional_queries'] == status['extra']
                feasible = [o for o in observations if o['success']]
                best = min(feasible, key=lambda o: (o['velocity_rad_s'], o['index'])) if feasible else None
                assert a['best'] == best
                reasons = Counter()
                minima = []
                for observation in observations:
                    raw_path = Path(observation['raw_trial_path'])
                    assert sha(raw_path) == observation['raw_trial_sha256']
                    raw = read(raw_path)
                    result = raw['result']
                    evaluation = result['evaluation']
                    assert evaluation['physics_steps'] == 0 and evaluation['control_rate_hz'] == 20
                    assert not evaluation['task_interaction'] and not evaluation['objects_considered']
                    assert evaluation['clearance_domain'] == 'solved_joint_positions_20hz_fk; no physics/interpolation'
                    assert evaluation['table_contact_policy'] == 'record_and_continue_fk_only'
                    assert raw['point'] == observation['actual_pre_caps_deg'] + observation['actual_post_caps_deg']
                    assert all(0 <= x <= 30 and math.isfinite(x) for x in raw['point'])
                    masks = np.repeat(subtask['APP_mask'] + subtask['EXEC_mask'], 2)
                    assert all(x == 0 for x, bit in zip(raw['point'], masks) if bit == 0)
                    if observation['success']:
                        assert observation['trajectory_complete'] and observation['solver_pass']
                        assert observation['minimum_clearance_m'] >= .005
                    reasons[observation['reason']] += 1
                    minima.append(evaluation['robot_table_clearance']['minimum'])
                    trial_count += 1
                selections.append(best)
                segments.append(dict(index=index, queries=len(observations), qualified=len(feasible), reasons=dict(reasons),
                    best_velocity_rad_s=best['velocity_rad_s'] if best else None,
                    best_clearance_m=best['minimum_clearance_m'] if best else None,
                    clearance_bottlenecks=minima))
            found = all(s is not None for s in selections)
            label_path = output / 'tolerance_labels.json'
            assert found == label_path.exists() == (status['status'] == 'labeled')
            if found:
                labels = read(label_path)
                assert labels['label_policy'] == 'speed_best_feasible_search_caps_v1'
                assert len(labels['frames']) == len(frames)
                for source_frame, labeled in zip(frames, labels['frames']):
                    assert labeled['source_frame_index'] == source_frame['source_frame_index']
                    assert labeled['phase'] == source_frame['phase']
                    chosen = selections[source_frame['subtask_index']]
                    caps = chosen['actual_pre_caps_deg'] if labeled['phase'] == 'pregrasp' else chosen['actual_post_caps_deg'] if labeled['phase'] == 'postgrasp' else [0.] * 6
                    assert np.allclose(labeled['rotation_tolerance_rad'], np.deg2rad(caps), atol=1e-15, rtol=0)
            episodes.append(dict(source=source, status=status['status'], extra=status['extra'], segments=segments,
                prepared_input_sha256=sha(path), status_sha256=sha(output / 'annotation_status.json')))
    # Check active code imports, excluding historical snapshots and old data.
    retired = ('annotate_tolerance.py', 'annotate_custom_layouts.py', 'tasktol/counterfactual/search.py')
    assert not any((PROJECT / name).exists() for name in retired)
    active = [*PROJECT.glob('*.py'), *(PROJECT / 'tasktol').rglob('*.py'), *(PROJECT / 'tests').rglob('*.py')]
    for path in active:
        source = path.read_text()
        assert 'from annotate_custom_layouts import' not in source, path
        assert 'from annotate_tolerance import' not in source, path
        assert 'from tasktol.counterfactual.search import' not in source, path
    result = dict(complete=True, scope='five small episodes only; no full dataset annotation',
        episodes=episodes, labeled=sum(e['status'] == 'labeled' for e in episodes),
        no_feasible_label=sum(e['status'] == 'no_feasible_label' for e in episodes),
        verified_raw_trials=trial_count, verified_immutable_files=len(checked),
        physics_steps=0, task_scene_instantiated=False, task_objects_loaded=False,
        retired_active_files=list(retired), active_python_files_checked=len(active))
    (ROOT / 'final_independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print({k: v for k, v in result.items() if k != 'episodes'})


if __name__ == '__main__':
    main()
