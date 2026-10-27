"""Compare native torques at a genuinely shared pre-physics state and goal."""

import argparse
import json
from pathlib import Path

import numpy as np

from tasktol.geometry import rotation_matrix,rotation_vector


def compare(ipopt,osc):
    tasks=[]
    for path in sorted(ipopt.glob('*/*/control_trace.npz')):
        rel=path.parent.relative_to(ipopt)
        with np.load(path) as a,np.load(osc/rel/'control_trace.npz') as b:
            j={k:a[k][0] for k in a.files};o={k:b[k][0] for k in b.files}
        report=json.loads((path.parent/'report.json').read_text())
        row=json.loads((path.parent/'trajectory.json').read_text())['frames'][0]
        m=j['mass'];jac=j['jac'];bias=j['bias']
        state_errors={k:float(np.max(np.abs(j[k]-o[k]))) for k in ['q','v','mass','jac','bias']}
        target_p_error=float(np.linalg.norm(np.asarray(row['nominal_target']['position_m'])-o['goal_p']))
        target_site_r=rotation_matrix(np.asarray(row['nominal_target']['rotation_vector_rad'])) @ np.asarray(report['ee_rotation_offset']).T
        target_r_error=float(np.linalg.norm(rotation_vector(target_site_r @ o['goal_r'].T)))
        mobility=jac@np.linalg.solve(m,jac.T)
        full=np.linalg.pinv(mobility)
        split=np.zeros((6,6));split[:3,:3]=np.linalg.pinv(mobility[:3,:3]);split[3:,3:]=np.linalg.pinv(mobility[3:,3:])
        ori_error=.5*sum(np.cross(o['ee_r'][:,i],o['goal_r'][:,i]) for i in range(3))
        error=np.r_[o['goal_p']-o['ee_p'],ori_error]
        joint_p=m@(j['kp']*(j['q_goal']-j['q']))
        joint_d=m@(-j['kd']*j['v'])
        osc_p=jac.T@split@(o['kp']*error)
        osc_d=jac.T@split@(-o['kd']*o['ee_v'])
        osc_null=o['tau']-bias-osc_p-osc_d
        request=o['kp']*error-o['kd']*o['ee_v']
        joint_task=jac.T@full@jac@np.linalg.solve(m,j['tau']-bias)
        joint_null=j['tau']-bias-joint_task
        inertia_change=jac.T@(full-split)@request
        task_error_change=joint_task-jac.T@full@request
        null_change=joint_null-osc_null
        delta=j['tau']-o['tau']
        residual=float(np.max(np.abs(delta-inertia_change-task_error_change-null_change)))
        null_task_residual=float(np.max(np.abs(jac@np.linalg.solve(m,osc_null))))
        jp_reconstruction=float(np.max(np.abs(j['tau']-joint_p-joint_d-bias)))
        checks=dict(state_identical=max(state_errors.values())<1e-12,
                    goal_identical_within_native_rotation_conversion=target_p_error<1e-10 and target_r_error<1e-6,
                    no_torque_clipping=np.array_equal(j['tau'],j['tau_clipped']) and np.array_equal(o['tau'],o['tau_clipped']),
                    no_robot_external_contact=j['robot_external_contacts']==0 and o['robot_external_contacts']==0,
                    joint_law_reconstructs=jp_reconstruction<1e-9,
                    osc_remainder_is_nullspace=null_task_residual<1e-9,
                    torque_difference_decomposes=residual<1e-9)
        tasks.append(dict(task=str(rel),sample='first native controller call, before physics',units='Nm',
                          shared_state_max_differences=state_errors,goal_position_difference_m=target_p_error,
                          goal_rotation_difference_rad=target_r_error,
                          osc_torque_nm=o['tau'].tolist(),joint_torque_nm=j['tau'].tolist(),
                          joint_minus_osc_nm=delta.tolist(),difference_norm_nm=float(np.linalg.norm(delta)),
                          gravity_bias_nm=bias.tolist(),joint_proportional_nm=joint_p.tolist(),
                          osc_proportional_nm=osc_p.tolist(),joint_damping_nm=joint_d.tolist(),
                          osc_damping_nm=osc_d.tolist(),osc_nullspace_nm=osc_null.tolist(),
                          decomposition=dict(inertia_mapping_nm=inertia_change.tolist(),
                                             task_error_mapping_nm=task_error_change.tolist(),
                                             nullspace_tracking_nm=null_change.tolist(),
                                             max_reconstruction_residual_nm=residual),checks={k:bool(v) for k,v in checks.items()}))
    return dict(scope='Same state and same target at the first physics tick in each of three trajectories; later cross-run samples are not compared as if their states were identical',
                sign='joint minus OSC',architecture_changed=False,checks_passed=sum(sum(t['checks'].values()) for t in tasks),
                checks_total=sum(len(t['checks']) for t in tasks),tasks=tasks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ipopt',type=Path,required=True);parser.add_argument('--osc',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compare(args.ipopt,args.osc)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('cases',len(result['tasks']),'checks',result['checks_passed'],'/',result['checks_total'])
    for t in result['tasks']:
        print(t['task'],'delta Nm',np.round(t['joint_minus_osc_nm'],3).tolist(),'decomposition residual Nm',t['decomposition']['max_reconstruction_residual_nm'])
    if result['checks_passed']!=result['checks_total']:
        raise SystemExit(1)


if __name__=='__main__':main()
