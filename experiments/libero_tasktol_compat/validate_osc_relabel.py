"""Read back every exported episode and independently verify file alignment."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from tasktol.geometry import rotation_vector
from experiments.libero_tasktol_compat.osc_reference import integrate_reference_actions


def validate(dataset, accumulated):
    summary=json.loads((dataset/'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('Dataset export is incomplete')
    files=[];episodes=frames=0
    for task in summary['results']:
        path=Path(task['output'])
        checks=[];position_max=rotation_max=0.0
        with h5py.File(path,'r') as f:
            if f.attrs['schema']!='libero_osc_independent_reference_actions_v1':
                raise ValueError(f'Unknown schema: {path}')
            for demo in f['data'].values():
                actions=demo['actions'][()]
                source=demo['source_demo']
                source_actions=source['actions'][()]
                states=source['tasktol/pre_action_states'][()]
                positions,rotations=integrate_reference_actions(actions,demo['anchor_position_m'][()],demo['anchor_tool_rotation'][()])
                p_error=float(np.max(np.linalg.norm(positions-demo['goal_positions_m'][()],axis=1)))
                r_error=float(max(np.linalg.norm(rotation_vector(a@b.T)) for a,b in zip(rotations,demo['goal_tool_rotations'][()],strict=True)))
                position_max=max(position_max,p_error);rotation_max=max(rotation_max,r_error)
                checks.append(dict(
                    shapes_match=actions.shape==source_actions.shape and len(states)==len(actions),
                    source_input_hash=hashlib.sha256(states.tobytes()+source_actions.tobytes()).hexdigest()==demo.attrs['input_state_action_sha256'],
                    gripper_unchanged=np.array_equal(actions[:,6],source_actions[:,6]),
                    image_count_aligned=source['obs/agentview_rgb'].shape[0]==len(actions),
                    reference_reconstructed=p_error<1e-10 and r_error<1e-9,
                ))
                episodes+=1;frames+=len(actions)
            rel=Path(task['suite'])/f"{task['task_id']:02d}"
            labels_path=accumulated/rel/'relabelled_reference.npz'
            with np.load(labels_path) as labels:
                # Locate the selected episode by its native replay report.
                report=json.loads((labels_path.parent/'report.json').read_text())
                demo=f['data'][report['demo']]
                label_difference=float(np.max(np.abs(demo['actions'][()]-labels['reference_delta_actions_m_rad_gripper'])))
                goal_difference=float(np.max(np.abs(demo['goal_positions_m'][()]-labels['source_goal_positions_m'])))
                pose_difference=float(np.max(np.abs(demo['goal_tool_rotations'][()]-labels['source_goal_tool_rotations'])))
            selected_matches=label_difference<1e-12 and goal_difference<1e-12 and pose_difference<1e-12
        files.append(dict(file=str(path),episodes=len(checks),episode_checks_passed=sum(sum(c.values()) for c in checks),
                          episode_checks_total=sum(len(c) for c in checks),selected_native_labels_match=selected_matches,
                          selected_label_max_difference=label_difference,
                          reconstruction_position_max_m=position_max,reconstruction_rotation_max_rad=rotation_max))
    passed=all(x['episode_checks_passed']==x['episode_checks_total'] and x['selected_native_labels_match'] for x in files)
    return dict(files=len(files),episodes=episodes,frames=frames,passed=passed,
                episode_checks_passed=sum(x['episode_checks_passed'] for x in files),
                episode_checks_total=sum(x['episode_checks_total'] for x in files),
                selected_native_label_matches=sum(x['selected_native_labels_match'] for x in files),
                reconstruction_position_max_m=max(x['reconstruction_position_max_m'] for x in files),
                reconstruction_rotation_max_rad=max(x['reconstruction_rotation_max_rad'] for x in files),
                results=files)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--accumulated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=validate(args.dataset,args.accumulated)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
