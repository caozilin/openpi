"""Offline geometric audit of rejected targets from the paired recording pilot."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from audit_original_osc_samples import (
    ReachBound, ControlEnv, IpoptIK, Rotation, get_libero_path, libero_handles,
    mujoco, np, plain, find_handles,
)


def handles_for(model, spec):
    h = find_handles(model, spec)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    offset = data.site_xmat[h.ee_site_id].reshape(3, 3).T @ data.xmat[model.body('robot0_right_hand').id].reshape(3, 3)
    return replace(h, ee_rotation_offset=offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    comparison = json.loads((root / 'comparison.json').read_text())
    first = comparison['episode_results'][0]
    summary = json.loads((root / first['suite'] / f"{first['task_id']:02d}" / first['demo'] / 'summary.json').read_text())
    env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / first['suite'] / (summary['task_name'] + '.bddl')),
                     controller='OSC_POSE', use_camera_obs=False, has_offscreen_renderer=False, control_freq=20)
    import h5py
    results = []
    try:
        env.reset()
        spec = libero_handles(env).spec
        for episode in comparison['episode_results']:
            result = next(r for r in episode['results'] if r['mode'] == 'ipopt_original_remap_B10')
            if result['rejected_source_index'] is None:
                continue
            dest = root / episode['suite'] / f"{episode['task_id']:02d}" / episode['demo']
            folder = dest / 'ipopt_original_remap_B10'
            rows = json.loads((folder / 'trajectory.json').read_text())
            row = rows[-1]
            assert row['aborted'] and row['physics_steps'] == 0
            with h5py.File(folder / 'recording.hdf5', 'r') as file:
                current_xml = file['attempts'][episode['demo']].attrs['model_file']
            original = mujoco.MjModel.from_xml_path(str(dest / 'original_model_local_paths.xml'))
            current = mujoco.MjModel.from_xml_string(current_xml)
            oh, ch = handles_for(original, spec), handles_for(current, spec)
            old_fk = IpoptIK(original, oh).kinematics.evaluate(np.zeros(7), include_manipulability=False, include_link_points=False)
            local_fk = IpoptIK(current, ch).kinematics.evaluate(np.zeros(7), include_manipulability=False, include_link_points=False)
            fixed = old_fk.rotation.T @ local_fk.rotation
            target_r = Rotation.from_rotvec(row['nominal_target']['rotation_vector_rad']).as_matrix() @ fixed.T
            geometry = ReachBound(original, oh).bound(np.array(row['nominal_target']['position_m']), target_r)
            first_success = next((r['raw_source_index'] for r in rows if not r['aborted'] and r['success']), None)
            item = dict(suite=episode['suite'], task_id=episode['task_id'], demo=episode['demo'],
                        first_success_source_index=first_success, rejected_source_index=row['raw_source_index'],
                        optimizer_status=row['optimizer']['status'], geometric_certificate=geometry)
            results.append(item)
        output = dict(rejected=len(results), certified_unreachable=sum(r['geometric_certificate']['certified_unreachable_even_with_acceptance_tolerance'] for r in results),
                      physics_steps=0, results=results)
        (root / 'rejection_geometry.json').write_text(json.dumps(plain(output), indent=2) + '\n')
        print(json.dumps(plain(output), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
