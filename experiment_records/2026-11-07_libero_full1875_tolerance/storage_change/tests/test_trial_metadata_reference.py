import copy
import json
import shutil

import pytest

from tasktol.data.trial_io import compact_trial, read_json, REFERENCE_KEY, STATIC_KEYS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def fixture(tmp_path, suite='libero_spatial', task=0, demo='demo_0'):
    folder = tmp_path / suite / f'task_{task:02d}' / demo
    metadata = dict(robot_xml='<robot/>', assets=['mesh.obj'], asset_sha256={'mesh.obj': 'abc'})
    source = dict(suite=suite, task_id=task, demo=demo, episode_index=task)
    prepared = dict(source=source, metadata=metadata)
    save(folder / 'prepared_input.json', prepared)
    payload = dict(identity_sha256='identity', point=[0, 30], row={'success': True},
                   result=dict(domain_randomization=metadata, trajectory_kinematics={'speed': 2},
                               evaluation={'solver_statuses': ['accepted'], 'clearance': [.1, .2]}))
    return folder, prepared, payload


@pytest.mark.parametrize('segment', ['subtask_00', 'zero_baselines/subtask_00'])
def test_trial_roundtrip_preserves_all_results_without_inline_metadata(tmp_path, segment):
    folder, prepared, payload = fixture(tmp_path)
    path = folder / segment / 'evaluations/digest/trial.json'
    original = copy.deepcopy(payload)
    packed = compact_trial(payload, path)
    assert payload == original
    assert not any(k in packed['result']['domain_randomization'] for k in STATIC_KEYS)
    assert packed['result'][REFERENCE_KEY]['source'] == prepared['source']
    save(path, packed)
    assert read_json(path) == original
    assert json.loads(path.read_text()) == packed


def test_same_seed_number_in_different_tasks_has_distinct_binding(tmp_path):
    bindings = []
    for task in (0, 1):
        folder, prepared, payload = fixture(tmp_path, task=task)
        path = folder / 'subtask_00/evaluations/digest/trial.json'
        bindings.append(compact_trial(payload, path)['result'][REFERENCE_KEY]['source'])
    assert bindings[0] != bindings[1]


def test_changed_source_fails_instead_of_silently_reconstructing(tmp_path):
    folder, prepared, payload = fixture(tmp_path)
    path = folder / 'subtask_00/evaluations/digest/trial.json'
    save(path, compact_trial(payload, path))
    prepared['metadata']['robot_xml'] = '<changed/>'
    save(folder / 'prepared_input.json', prepared)
    with pytest.raises(ValueError, match='checksum'):
        read_json(path)


def test_changed_candidate_metadata_is_never_discarded(tmp_path):
    folder, prepared, payload = fixture(tmp_path)
    payload['result']['domain_randomization']['robot_xml'] = '<different/>'
    with pytest.raises(ValueError, match='differs'):
        compact_trial(payload, folder / 'subtask_00/evaluations/digest/trial.json')


def test_reference_cannot_switch_source_identity(tmp_path):
    folder, prepared, payload = fixture(tmp_path)
    path = folder / 'subtask_00/evaluations/digest/trial.json'
    packed = compact_trial(payload, path)
    packed['result'][REFERENCE_KEY]['source']['demo'] = 'demo_1'
    save(path, packed)
    with pytest.raises(ValueError, match='identity'):
        read_json(path)


def test_reference_is_portable_with_the_source_demo_tree(tmp_path):
    folder, prepared, payload = fixture(tmp_path / 'original')
    path = folder / 'subtask_00/evaluations/digest/trial.json'
    save(path, compact_trial(payload, path))
    shutil.copytree(tmp_path / 'original', tmp_path / 'moved')
    assert read_json(tmp_path / 'moved' / path.relative_to(tmp_path / 'original')) == payload


def test_non_libero_inline_trials_and_other_json_remain_compatible(tmp_path):
    path = tmp_path / 'other/trial.json'
    payload = dict(result=dict(domain_randomization={'seed': 42}))
    assert compact_trial(payload, path) is payload
    save(path, payload)
    assert read_json(path) == payload
    save(tmp_path / 'other/list.json', [1, 2])
    assert read_json(tmp_path / 'other/list.json') == [1, 2]


def test_reference_cannot_point_outside_demo_tree(tmp_path):
    folder, prepared, payload = fixture(tmp_path)
    path = folder / 'subtask_00/evaluations/digest/trial.json'
    packed = compact_trial(payload, path)
    other, _, _ = fixture(tmp_path, task=1)
    import os
    packed['result'][REFERENCE_KEY]['path'] = os.path.relpath(other / 'prepared_input.json', path.parent)
    save(path, packed)
    with pytest.raises(ValueError, match='outside'):
        read_json(path)


def test_candidate_specific_extra_metadata_is_kept_inline(tmp_path):
    folder, prepared, payload = fixture(tmp_path)
    payload['result']['domain_randomization'].update(execution_mode='kinematic', task_interaction=False)
    path = folder / 'subtask_00/evaluations/digest/trial.json'
    packed = compact_trial(payload, path)
    assert packed['result']['domain_randomization'] == dict(execution_mode='kinematic', task_interaction=False)
    save(path, packed)
    assert read_json(path) == payload
