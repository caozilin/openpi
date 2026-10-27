"""Correct legacy first-frame robot_states in the package, retaining raw shards.

Actions, simulator states, reference anchors and all execution inputs are
unchanged. The corrected group exposes the recorder's actual native
pre-command observation instead of its upstream-compatible legacy first row.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def align(dataset):
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    changes = []
    for task in manifest['task_files']:
        with h5py.File(task['path'], 'r+') as file:
            data = file['data']
            for demo in list(data):
                existing = data[demo]
                if existing.attrs.get('policy_state_contract'):
                    changes.append(dict(case=task['case'], demo=demo,
                                        legacy_first_row_max_abs_delta=float(np.max(np.abs(existing['legacy_robot_states'][0] - existing['robot_states'][0]))),
                                        actions_sha256=hashlib.sha256(existing['actions'][()].tobytes()).hexdigest(),
                                        actual_states_and_execution_inputs_unchanged=True))
                    continue
                link = data.get(demo, getlink=True)
                assert isinstance(link, h5py.ExternalLink)
                with h5py.File(link.filename, 'r') as source:
                    original = source[link.path]
                    legacy = original['robot_states'][()]
                    aligned = original['tasktol/pre_action_robot_states'][()]
                    np.testing.assert_array_equal(legacy[1:], aligned[1:])
                    action_sha = hashlib.sha256(original['actions'][()].tobytes()).hexdigest()
                    del data[demo]
                    training = data.create_group(demo)
                    for key, value in original.attrs.items():
                        training.attrs[key] = value
                    for key in original:
                        if key != 'robot_states':
                            training[key] = h5py.ExternalLink(link.filename, f'{link.path}/{key}')
                    training['robot_states'] = h5py.ExternalLink(link.filename, f'{link.path}/tasktol/pre_action_robot_states')
                    training['legacy_robot_states'] = h5py.ExternalLink(link.filename, f'{link.path}/robot_states')
                    policy_state = np.concatenate([original['obs/ee_states'][()], original['obs/gripper_states'][()]], axis=1).astype(np.float32)
                    assert policy_state.shape == (len(aligned), 8) and np.all(np.isfinite(policy_state))
                    training.create_dataset('policy_state', data=policy_state)
                    training.attrs['policy_state_contract'] = '8D native cached pre-command observation: xyz metres, axis-angle radians, two finger qpos metres'
                    changes.append(dict(case=task['case'], demo=demo, legacy_first_row_max_abs_delta=float(np.max(np.abs(legacy[0] - aligned[0]))),
                                        actions_sha256=action_sha, actual_states_and_execution_inputs_unchanged=True))
                # Close the explicit read-only source handle before resolving
                # an external link from the r+ package handle (HDF5 flags).
                assert hashlib.sha256(training['actions'][()].tobytes()).hexdigest() == action_sha
            contract = json.loads(file.attrs['action_contract'])
            contract.update(training_state='policy_state: xyz, axis-angle, two finger qpos; native pre-command observations',
                            robot_states='aligned pre-command gripper qpos, xyz, xyzw quaternion; legacy first-frame array retained separately',
                            observation_cache_lag_s=.002)
            file.attrs['action_contract'] = json.dumps(contract)
            data.attrs['action_contract'] = json.dumps(contract)
    manifest['action_contract'].update(training_state='policy_state: xyz, axis-angle, two finger qpos; native pre-command observations',
                                     robot_states='aligned pre-command observations; legacy first-frame array retained separately',
                                     observation_cache_lag_s=.002)
    manifest['training_state_alignment_fix'] = 'training_state_alignment.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    evidence = dict(corrected_episodes=len(changes), recorder_shards_modified=False,
                    original_dataset_modified=False, physics_actions_and_reference_inputs_modified=False,
                    changes=changes)
    (dataset / 'training_state_alignment.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({key: value for key, value in evidence.items() if key != 'changes'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    align(parser.parse_args().dataset.resolve())
