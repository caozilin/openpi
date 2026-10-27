"""Explain measured native joint tracking without replacing the control chain."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(ipopt,osc):
    tasks=[]
    for p in sorted(ipopt.glob('*/*/control_trace.npz')):
        rel=p.parent.relative_to(ipopt)
        with np.load(p) as a,np.load(osc/rel/'control_trace.npz') as b:
            joint={k:a[k] for k in a.files};cart={k:b[k] for k in b.files}
        traces=json.loads((p.parent/'trajectory.json').read_text())['frames']
        initial_q=joint['q'][0];initial_v=joint['v'][0];target=joint['q_goal'][0]
        omega=np.sqrt(joint['kp'][0]);dt=.05
        error0=initial_q-target
        ideal=target+(error0+(initial_v+omega*error0)*dt)*np.exp(-omega*dt)
        actual=np.asarray(traces[0]['post_action']['joint_position_rad'])
        finite_q=np.asarray([r['joint_target_rad'] for r in traces if not r['aborted']])
        actual_q=np.asarray([r['post_action']['joint_position_rad'] for r in traces if not r['aborted']])
        lag_errors=[dict(lag_frames=k,rms_rad=float(np.sqrt(np.mean((finite_q[:len(finite_q)-k or None]-actual_q[k:])**2)))) for k in range(9)]
        best=min(lag_errors,key=lambda x:x['rms_rad'])
        contact=joint['robot_external_contacts']>0
        clipped=np.any(np.abs(joint['tau']-joint['tau_clipped'])>1e-8,axis=1)
        first_q_difference=float(np.max(np.abs(joint['q'][0]-cart['q'][0])))
        first_v_difference=float(np.max(np.abs(joint['v'][0]-cart['v'][0])))
        m=joint['mass'][0];jac=joint['jac'][0]
        acc_joint=jac@np.linalg.solve(m,joint['tau'][0]-joint['bias'][0])
        acc_osc=jac@np.linalg.solve(m,cart['tau'][0]-cart['bias'][0])
        # Use the native OSC cross-column orientation error, not a log error.
        current=cart['ee_r'][0];desired=cart['goal_r'][0]
        ori_error=.5*sum(np.cross(current[:,i],desired[:,i]) for i in range(3))
        pose_error=np.r_[cart['goal_p'][0]-cart['ee_p'][0],ori_error]
        requested=cart['kp'][0]*pose_error-cart['kd'][0]*cart['ee_v'][0]
        mobility=jac@np.linalg.solve(m,jac.T)
        block_lambda=np.zeros((6,6));block_lambda[:3,:3]=np.linalg.pinv(mobility[:3,:3]);block_lambda[3:,3:]=np.linalg.pinv(mobility[3:,3:])
        cross_effect=(mobility@block_lambda-np.eye(6))@requested
        nonlinear_joint_effect=requested-acc_joint
        unexplained=acc_osc-acc_joint-cross_effect-nonlinear_joint_effect
        trace_count=len(joint['frame']);counts=np.unique(joint['frame'],return_counts=True)[1]
        tasks.append(dict(
            task=str(rel),native_physics_samples=trace_count,calls_per_control_frame=np.unique(counts).tolist(),
            native_period_s=float(np.median(np.diff(joint['time_s']))),
            joint_error_rms_rad=float(np.sqrt(np.mean((finite_q-actual_q)**2))),
            joint_error_max_rad=float(np.max(np.abs(finite_q-actual_q))),
            best_lag_frames=best['lag_frames'],lag_scan=lag_errors,
            torque_clipped_samples=int(clipped.sum()),contact_samples=int(contact.sum()),
            clipped_free_samples=int(np.sum(clipped&~contact)),free_samples=int(np.sum(~contact)),
            initial_state_joint_difference_rad=first_q_difference,initial_state_velocity_difference_rad_s=first_v_difference,
            initial_mass_matrix_max_difference=float(np.max(np.abs(joint['mass'][0]-cart['mass'][0]))),
            initial_jacobian_max_difference=float(np.max(np.abs(joint['jac'][0]-cart['jac'][0]))),
            initial_bias_torque_max_difference_nm=float(np.max(np.abs(joint['bias'][0]-cart['bias'][0]))),
            first_frame=dict(
                robot_contact_samples=int(contact[:25].sum()),torque_clipped_samples=int(clipped[:25].sum()),
                actual_joint_motion_rad=(actual-initial_q).tolist(),reference_joint_step_rad=(target-initial_q).tolist(),
                ideal_critical_damping_joint_motion_rad=(ideal-initial_q).tolist(),
                ideal_vs_actual_max_joint_difference_rad=float(np.max(np.abs(ideal-actual))),
                zero_initial_velocity_ideal_step_completed_fraction=float(1-(1+np.sqrt(150)*dt)*np.exp(-np.sqrt(150)*dt)),
                translational_acceleration_joint_m_s2=acc_joint[:3].tolist(),translational_acceleration_osc_m_s2=acc_osc[:3].tolist(),
                rotational_acceleration_joint_rad_s2=acc_joint[3:].tolist(),rotational_acceleration_osc_rad_s2=acc_osc[3:].tolist(),
                osc_block_inertia_cross_term_translation_m_s2=cross_effect[:3].tolist(),
                osc_block_inertia_cross_term_rotation_rad_s2=cross_effect[3:].tolist(),
                finite_ik_joint_error_mapping_term_translation_m_s2=nonlinear_joint_effect[:3].tolist(),
                finite_ik_joint_error_mapping_term_rotation_rad_s2=nonlinear_joint_effect[3:].tolist(),
                decomposition_unexplained_translation_m_s2=unexplained[:3].tolist(),
                decomposition_unexplained_rotation_rad_s2=unexplained[3:].tolist(),
                dynamics_scope='Instantaneous raw-torque acceleration contribution J M^-1 (tau-bias) at the identical initial state; omits shared Jdot*qdot and external contact forces',
            ),
        ))
    return dict(scope='Read-only native controller instrumentation on representative trajectories; joint-reference architecture unchanged',tasks=tasks)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ipopt',type=Path,required=True);parser.add_argument('--osc',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.ipopt,args.osc)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
