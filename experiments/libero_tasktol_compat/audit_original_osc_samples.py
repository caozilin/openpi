"""Diagnose original LIBERO samples separately from OpenVLA regenerated data.

Only asset paths in the archived model XML are remapped. No dynamics are run.
Native OSC.set_goal reconstructs targets from archived states/actions/config.
A shoulder/elbow/wrist triangle bound can certify some targets unreachable
independently of any numerical IK branch or optimizer.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from audit_failed_ik_geometry import (
    BASE, SOURCE, ControlEnv, IpoptIK, Rotation, get_libero_path, mujoco,
    np, plain, rotation_vector,
)
import h5py
from PIL import Image
import robosuite
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.binding_utils import MjSim
from scipy.optimize import least_squares
from tasktol.simulation.libero import libero_handles
from tasktol.simulation.robot import find_handles

OUTPUT = BASE / 'original_dataset_target_audit_v2'


def remap_xml(xml):
    tree = ET.fromstring(xml)
    changes = []
    for node in tree.iter():
        path = node.get('file')
        if path is None:
            continue
        if '/robosuite/' in path:
            new = Path(robosuite.__file__).parent / path.rsplit('/robosuite/', 1)[1]
        elif '/assets/' in path:
            new = Path(get_libero_path('assets')) / path.split('/assets/', 1)[1]
        else:
            raise ValueError(f'Unrecognized archived asset path: {path}')
        assert new.is_file(), str(new)
        node.set('file', str(new))
        changes.append(dict(original=path, local=str(new)))
    return ET.tostring(tree, encoding='unicode'), changes


class ReachBound:
    """Necessary global reach bound for this verified Panda chain.

    S=joint2, E=joint4, W=joint6, H=joint7, P=control site.
    |SW| <= |SE|+|EW|. Fixing tool orientation fixes R7; for unrestricted
    joint7, W lies on a circle centered near H. Minimize |SW| over that
    ENTIRE circle. If this minimum exceeds |SE|+|EW|, no IK branch can work.
    Joint7 limits and all other joint limits only shrink this relaxed set.
    """

    def __init__(self, model, handles):
        self.model, self.handles = model, handles
        self.data = mujoco.MjData(model)
        self.ids = [model.body(f'robot0_link{i}').id for i in [2, 4, 6, 7]]
        self.hand_id = model.body('robot0_right_hand').id
        self.evaluate_state(np.zeros(7))
        s, e, w, h = self.data.xpos[self.ids].copy()
        r7 = self.data.xmat[self.ids[-1]].reshape(3, 3).copy()
        tool = self.data.site_xmat[handles.ee_site_id].reshape(3, 3) @ handles.ee_rotation_offset
        self.shoulder = s
        self.upper_length = float(np.linalg.norm(e - s))
        self.fore_length = float(np.linalg.norm(w - e))
        self.max_reach = self.upper_length + self.fore_length
        self.tool_in_r7 = r7.T @ tool
        self.site_in_r7 = r7.T @ (self.data.site_xpos[handles.ee_site_id] - h)
        self.wrist_offset_in_r7_at_zero = r7.T @ (h - w)
        self.radius = float(np.linalg.norm(self.wrist_offset_in_r7_at_zero[:2]))
        self.height = float(self.wrist_offset_in_r7_at_zero[2])
        self.validation = self.validate()

    def evaluate_state(self, q):
        self.data.qpos[self.handles.qpos_ids] = q
        mujoco.mj_kinematics(self.model, self.data)

    def bound(self, position, tool_rotation):
        r7 = tool_rotation @ self.tool_in_r7.T
        h = position - r7 @ self.site_in_r7
        center = h - r7[:, 2] * self.height
        delta = center - self.shoulder
        axial = float(delta @ r7[:, 2])
        radial = float(np.linalg.norm(delta - axial * r7[:, 2]))
        minimum = float(np.hypot(radial - self.radius, axial))
        gap = minimum - self.max_reach
        # Any pose satisfying componentwise production tolerances lies within
        # these Euclidean translation/rotation balls. Circle motion under
        # orientation changes is bounded by the site and wrist lever arms.
        allowance = np.sqrt(3) * 1e-5 + (np.linalg.norm(self.site_in_r7) + np.linalg.norm(self.wrist_offset_in_r7_at_zero)) * 2 * np.sin(np.sqrt(3) * 1e-4 / 2)
        return dict(minimum_shoulder_wrist_distance_m=minimum, maximum_two_link_reach_m=self.max_reach,
                    excess_mm=gap * 1000, acceptance_pose_allowance_mm=allowance * 1000,
                    excess_after_acceptance_allowance_mm=(gap - allowance) * 1000,
                    certified_unreachable_even_with_acceptance_tolerance=bool(gap > allowance + 1e-10))

    def validate(self):
        rng = np.random.default_rng(20261021)
        errors = []
        for q in rng.uniform(self.handles.joint_lower, self.handles.joint_upper, (100, 7)):
            self.evaluate_state(q)
            s, e, w, h = self.data.xpos[self.ids].copy()
            r7 = self.data.xmat[self.ids[-1]].reshape(3, 3)
            rz = Rotation.from_rotvec([0, 0, -q[6]]).as_matrix()
            predicted_w = h - r7 @ rz @ self.wrist_offset_in_r7_at_zero
            errors.extend([np.linalg.norm(s - self.shoulder),
                           abs(np.linalg.norm(e - s) - self.upper_length),
                           abs(np.linalg.norm(w - e) - self.fore_length),
                           np.linalg.norm(w - predicted_w),
                           np.linalg.norm(self.data.site_xpos[self.handles.ee_site_id] - h - r7 @ self.site_in_r7)])
            tool = self.data.site_xmat[self.handles.ee_site_id].reshape(3, 3) @ self.handles.ee_rotation_offset
            errors.append(np.max(np.abs(tool - r7 @ self.tool_in_r7)))
            assert not self.bound(self.data.site_xpos[self.handles.ee_site_id], tool)['certified_unreachable_even_with_acceptance_tolerance']
        maximum = float(max(errors))
        assert maximum < 1e-12, maximum
        return dict(random_configurations=100, max_chain_identity_error=maximum,
                    reachable_fk_targets_never_rejected=True)


def audit(rel):
    dest = OUTPUT / rel
    dest.mkdir(parents=True, exist_ok=False)
    replay = json.loads((SOURCE / rel / 'report.json').read_text())
    failure = json.loads((SOURCE / rel / 'trajectory.json').read_text())['frames'][-1]
    sample = int(failure['source_index'])
    demo = replay['task']['demo']
    with h5py.File(replay['source_original_hdf5'], 'r') as file:
        group = file['data'][demo]
        states = group['states'][()]
        actions = group['actions'][()]
        obs_p = group['obs/ee_pos'][()]
        obs_r = group['obs/ee_ori'][()]
        obs_q = group['obs/joint_states'][()]
        config = json.loads(file['data'].attrs['env_args'])['env_kwargs']['controller_configs']
        xml = str(group.attrs['model_file'])
        dones = group['dones'][()]
        rewards = group['rewards'][()]
        for index in range(max(0, sample - 2), min(len(actions), sample + 3)):
            pixels = group['obs/agentview_rgb'][index]
            Image.fromarray(pixels).save(dest / f'raw_obs_{index:04d}_stored.png')
            Image.fromarray(pixels[::-1, ::-1].copy()).save(dest / f'raw_obs_{index:04d}_display.png')
    with h5py.File(replay['source_hdf5'], 'r') as file:
        processed = file['data'][demo]
        indices = processed['tasktol/source_step_indices'][()]
        proc_state = processed['tasktol/pre_action_states'][failure['index']]
        proc_action = processed['actions'][failure['index']]
    assert indices[failure['index']] == sample
    assert np.array_equal(proc_action, actions[sample])
    assert np.array_equal(actions[sample], failure['raw_source_osc_action'])
    env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / replay['task']['suite'] / (replay['task']['name'] + '.bddl')),
                     controller='OSC_POSE', use_camera_obs=False, has_offscreen_renderer=False, control_freq=20)
    try:
        env.seed(0)
        env.reset()
        current_handles = libero_handles(env)
        local_xml, mappings = remap_xml(xml)
        (dest / 'original_model.xml').write_text(xml)
        (dest / 'original_model_local_paths.xml').write_text(local_xml)
        model = mujoco.MjModel.from_xml_string(local_xml)
        handles = find_handles(model, current_handles.spec)
        sim = MjSim(model)
        assert states.shape[1] == 1 + model.nq + model.nv
        sim.data.qpos[:] = states[0, 1:1 + model.nq]
        sim.forward()
        site_r = sim.data.site_xmat[handles.ee_site_id].reshape(3, 3)
        hand_r = sim.data.body_xmat[model.body('robot0_right_hand').id].reshape(3, 3)
        handles = replace(handles, ee_rotation_offset=site_r.T @ hand_r)
        controller = OperationalSpaceController(
            sim=sim, eef_name=handles.spec.ee_site,
            joint_indexes=dict(joints=handles.joint_ids, qpos=handles.qpos_ids, qvel=handles.dof_ids),
            actuator_range=(model.actuator_ctrlrange[handles.actuator_ids, 0], model.actuator_ctrlrange[handles.actuator_ids, 1]),
            **config,
        )
        kin = IpoptIK(model, handles).kinematics
        current_kin = IpoptIK(env.sim.model._model, current_handles).kinematics
        model_cross_checks = []
        hand_frame_transforms = []
        for q in [states[sample, 1 + handles.qpos_ids], proc_state[1 + handles.qpos_ids]]:
            raw_fk = kin.evaluate(q, include_manipulability=False, include_link_points=False)
            local_fk = current_kin.evaluate(q, include_manipulability=False, include_link_points=False)
            hand_frame_transforms.append(raw_fk.rotation.T @ local_fk.rotation)
            model_cross_checks.append(dict(position_difference_m=float(np.linalg.norm(raw_fk.position - local_fk.position)),
                                           hand_rotation_difference_rad=float(np.linalg.norm(rotation_vector(raw_fk.rotation.T @ local_fk.rotation))),
                                           control_site_rotation_difference_rad=float(np.linalg.norm(rotation_vector(
                                               (raw_fk.rotation @ handles.ee_rotation_offset.T).T @ (local_fk.rotation @ current_handles.ee_rotation_offset.T))))))
        assert max(c['position_difference_m'] for c in model_cross_checks) < 1e-12
        fixed_hand_frame_change = hand_frame_transforms[0]
        assert np.max(np.abs(hand_frame_transforms[0] - hand_frame_transforms[1])) < 1e-12
        assert np.array_equal(handles.joint_lower, current_handles.joint_lower)
        assert np.array_equal(handles.joint_upper, current_handles.joint_upper)
        bound = ReachBound(model, handles)
        # Check q/observation timing and model FK against original observations.
        align_q = float(np.max(np.abs(states[1:, 1 + handles.qpos_ids] - obs_q[:-1])))
        assert align_q < 1e-12
        position_alignment = []
        orientation_alignment = []
        previous_physics_position_alignment = []
        previous_physics_orientation_alignment = []
        goals = []
        orientation_known = False
        for index, (state, action) in enumerate(zip(states, actions, strict=True)):
            sim.data.qpos[:] = state[1:1 + model.nq]
            sim.data.qvel[:] = state[1 + model.nq:]
            sim.forward()
            controller.new_update = True
            controller.set_goal(action[:6])
            if np.any(controller.scale_action(action[:6])[3:] != 0):
                orientation_known = True
            p = controller.ee_pos.copy()
            r = controller.ee_ori_mat @ handles.ee_rotation_offset
            goal_p = controller.goal_pos.copy()
            goal_r = controller.goal_ori @ handles.ee_rotation_offset
            if index > 0:
                position_alignment.append(float(np.linalg.norm(p - obs_p[index - 1])))
                # Original obs/eef_quat is HAND-body orientation, whereas
                # OSC ee_ori_mat is the control SITE orientation.
                orientation_alignment.append(float(np.linalg.norm(rotation_vector(r.T @ Rotation.from_rotvec(obs_r[index - 1]).as_matrix()))))
                previous_q = state[1 + handles.qpos_ids] - model.opt.timestep * state[1 + model.nq + handles.dof_ids]
                previous_fk = kin.evaluate(previous_q, include_manipulability=False, include_link_points=False)
                previous_physics_position_alignment.append(float(np.linalg.norm(previous_fk.position - obs_p[index - 1])))
                previous_physics_orientation_alignment.append(float(np.linalg.norm(rotation_vector(previous_fk.rotation.T @ Rotation.from_rotvec(obs_r[index - 1]).as_matrix()))))
            fk = kin.evaluate(state[1 + handles.qpos_ids], include_manipulability=False, include_link_points=False)
            assert np.linalg.norm(fk.position - p) < 1e-12
            sv = np.linalg.svd(fk.jacobian, compute_uv=False)
            margins = np.minimum(state[1 + handles.qpos_ids] - handles.joint_lower, handles.joint_upper - state[1 + handles.qpos_ids])
            goals.append(dict(raw_index=index, raw_time_s=float(state[0]), action=action,
                              actual_position_m=p, actual_tool_rotation=r, goal_position_m=goal_p, goal_tool_rotation=goal_r,
                              orientation_goal_history_known=orientation_known,
                              geometric_sigma_min=float(sv[-1]), min_limit_margin_deg=float(np.rad2deg(margins.min())),
                              native_goal_bound=bound.bound(goal_p, goal_r) if orientation_known else None,
                              actual_pose_bound=bound.bound(p, r)))
        # Original EEF observations can retain the last physics step's cached
        # kinematics, while joint observations match the final qpos exactly.
        # Native set_goal calls forward(), so archived qpos is authoritative.
        assert max(position_alignment) < 0.002, max(position_alignment)
        assert max(previous_physics_position_alignment) < 1e-12
        assert max(previous_physics_orientation_alignment) < 1e-12
        selected = goals[sample]
        assert selected['orientation_goal_history_known']
        original_target_p = np.asarray(selected['goal_position_m'])
        original_target_r = np.asarray(selected['goal_tool_rotation'])
        # Compare the exact archived-state goal with the regenerated-state goal.
        proc_fk = kin.evaluate(proc_state[1 + handles.qpos_ids], include_manipulability=False, include_link_points=False)
        proc_target_p = np.array(failure['nominal_target']['position_m'])
        proc_target_r = Rotation.from_rotvec(failure['nominal_target']['rotation_vector_rad']).as_matrix()
        proc_target_raw_r = proc_target_r @ fixed_hand_frame_change.T
        # Numerical witness search is secondary to the global necessary bound.
        raw_q = states[sample, 1 + handles.qpos_ids]

        def error(q):
            state = kin.evaluate(q, include_manipulability=False, include_link_points=False)
            return np.r_[(state.position - original_target_p) / 0.3, rotation_vector(original_target_r.T @ state.rotation)]

        def jac(q):
            state = kin.evaluate(q, include_manipulability=False, include_link_points=False)
            value = kin.pose_error_jacobian(state, original_target_r)
            value[:3] /= 0.3
            return value

        rng = np.random.default_rng(20261021)
        seeds = [raw_q, proc_state[1 + handles.qpos_ids]]
        seeds.extend(np.clip(raw_q + rng.normal(0, 0.4, 7), handles.joint_lower + 1e-7, handles.joint_upper - 1e-7) for _ in range(6))
        seeds.extend(rng.uniform(handles.joint_lower + 1e-7, handles.joint_upper - 1e-7) for _ in range(6))
        probes = []
        for i, seed in enumerate(seeds):
            solve = least_squares(error, seed, jac=jac, bounds=(handles.joint_lower, handles.joint_upper),
                                  max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            residual = error(solve.x)
            feasible = np.max(np.abs(residual[:3] * 0.3)) <= 1e-5 and np.max(np.abs(residual[3:])) <= 1e-4
            probes.append(dict(seed_index=i, seed_q=seed, q=solve.x, feasible=bool(feasible), nfev=solve.nfev,
                               position_error_max_m=float(np.max(np.abs(residual[:3])) * 0.3),
                               position_error_norm_mm=float(np.linalg.norm(residual[:3]) * 300),
                               rotation_error_max_rad=float(np.max(np.abs(residual[3:]))),
                               rotation_error_norm_deg=float(np.rad2deg(np.linalg.norm(residual[3:])))))
        output = dict(task=replay['task'], raw_hdf5=replay['source_original_hdf5'], processed_hdf5=replay['source_hdf5'],
                      raw_index=sample, processed_index=failure['index'], native_controller_config=config,
                      archived_model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(), asset_path_mappings=mappings,
                      raw_sample_sha256=hashlib.sha256(states[sample].tobytes() + actions[sample].tobytes()).hexdigest(),
                      original_demo_final_done=int(dones[-1]), original_demo_final_reward=float(rewards[-1]),
                      alignment=dict(raw_obs_is_post_action=True, raw_states_are_pre_action=True,
                                     joint_alignment_max_rad=align_q, position_alignment_max_m=max(position_alignment),
                                     orientation_alignment_max_rad=max(orientation_alignment),
                                     previous_physics_position_alignment_max_m=max(previous_physics_position_alignment),
                                     previous_physics_orientation_alignment_max_rad=max(previous_physics_orientation_alignment),
                                     matched_action_exact=True, original_geometry_used=True),
                      original_vs_current_robot_model_checks=model_cross_checks,
                      original_to_current_fixed_hand_rotation=fixed_hand_frame_change,
                      bound_geometry=dict(upper_length_m=bound.upper_length, fore_length_m=bound.fore_length,
                                          wrist_circle_radius_m=bound.radius, max_reach_m=bound.max_reach,
                                          site_in_link7=bound.site_in_r7, validation=bound.validation),
                      original_sample=selected,
                      original_after_action=dict(position_m=obs_p[sample], orientation_rotvec_hand=obs_r[sample],
                                                 target_position_error_mm=float(np.linalg.norm(obs_p[sample] - original_target_p) * 1000),
                                                 target_rotation_error_deg=float(np.rad2deg(np.linalg.norm(rotation_vector(original_target_r.T @ Rotation.from_rotvec(obs_r[sample]).as_matrix()))))),
                      regenerated_vs_original=dict(actual_position_difference_mm=float(np.linalg.norm(proc_fk.position - selected['actual_position_m']) * 1000),
                                                   actual_joint_max_difference_rad=float(np.max(np.abs(raw_q - proc_state[1 + handles.qpos_ids]))),
                                                   target_position_difference_mm=float(np.linalg.norm(proc_target_p - original_target_p) * 1000),
                                                   target_rotation_difference_deg=float(np.rad2deg(np.linalg.norm(rotation_vector(proc_target_raw_r.T @ original_target_r)))),
                                                   target_rotation_comparison='common link7 frame; fixed archived/current hand rotation converted',
                                                   regenerated_target_on_original_model_bound=bound.bound(proc_target_p, proc_target_raw_r)),
                      original_target_numerical_probes=probes, original_target_feasible_witnesses=sum(p['feasible'] for p in probes),
                      original_episode_goals=goals, physics_steps=0,
                      image_timing='raw_obs[j] is AFTER action[j]; raw_obs[j-1] corresponds to before action[j], checked by q and FK',
                      image_display_transform='180 degree rotation from stored OpenGL image; raw pixels also saved unchanged')
        (dest / 'audit.json').write_text(json.dumps(plain(output), indent=2) + '\n')
        print(json.dumps(plain({k: output[k] for k in ['task', 'raw_index', 'alignment', 'bound_geometry', 'original_sample', 'regenerated_vs_original', 'original_target_feasible_witnesses']})), flush=True)
        return output
    finally:
        env.close()


def main():
    OUTPUT.mkdir(exist_ok=False)
    results = [audit(rel) for rel in ['libero_spatial/04', 'libero_goal/02', 'libero_10/08']]
    (OUTPUT / 'summary.json').write_text(json.dumps(plain(results), indent=2) + '\n')


if __name__ == '__main__':
    main()
