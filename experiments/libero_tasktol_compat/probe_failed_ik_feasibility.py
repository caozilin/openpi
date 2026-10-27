"""Independent bounded pose-only IK probes, never used for robot execution."""

import json
from pathlib import Path

from audit_failed_ik_geometry import (
    OUTPUT, SOURCE, ControlEnv, IpoptIK, Rotation, get_libero_path, libero_handles,
    np, plain, rotation_vector,
)
from scipy.optimize import least_squares


def main():
    results = []
    for rel in ['libero_spatial/04', 'libero_goal/02', 'libero_10/08']:
        audit = json.loads((OUTPUT / rel / 'audit.json').read_text())
        rows = json.loads((SOURCE / rel / 'trajectory.json').read_text())['frames']
        target_p = np.array(audit['target']['position_m'])
        target_r = Rotation.from_rotvec(audit['target']['rotation_vector_rad']).as_matrix()
        task = audit['task']
        env = ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task['suite'] / (task['name'] + '.bddl')),
                         controller='OSC_POSE', use_camera_obs=False, has_offscreen_renderer=False, control_freq=20)
        try:
            env.seed(0)
            env.reset()
            handles = libero_handles(env)
            kin = IpoptIK(env.sim.model._model, handles).kinematics
            lower, upper = handles.joint_lower, handles.joint_upper

            def residual(q):
                state = kin.evaluate(q, include_manipulability=False, include_link_points=False)
                return np.r_[(state.position - target_p) / 0.3, rotation_vector(target_r.T @ state.rotation)]

            def jacobian(q):
                state = kin.evaluate(q, include_manipulability=False, include_link_points=False)
                jac = kin.pose_error_jacobian(state, target_r)
                jac[:3] /= 0.3
                return jac

            rng = np.random.default_rng(20261021)
            previous = np.array(audit['previous_accepted_reference']['q_rad'])
            actual = np.array(audit['actual']['q_rad'])
            seeds = [previous, actual]
            seeds.extend(np.clip(previous + rng.normal(0, 0.4, 7), lower + 1e-7, upper - 1e-7) for _ in range(6))
            seeds.extend(rng.uniform(lower + 1e-7, upper - 1e-7) for _ in range(6))
            probes = []
            for index, seed in enumerate(seeds):
                solve = least_squares(residual, seed, jac=jacobian, bounds=(lower, upper),
                                      max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                error = residual(solve.x)
                feasible = np.max(np.abs(error[:3] * 0.3)) <= 1e-5 and np.max(np.abs(error[3:])) <= 1e-4
                probes.append(dict(seed_index=index, seed_q=seed, q=solve.x, feasible=bool(feasible),
                                   nfev=solve.nfev, status=int(solve.status), cost=float(solve.cost),
                                   position_error_norm_mm=float(np.linalg.norm(error[:3]) * 300),
                                   position_error_max_m=float(np.max(np.abs(error[:3])) * 0.3),
                                   rotation_error_norm_deg=float(np.rad2deg(np.linalg.norm(error[3:]))),
                                   rotation_error_max_rad=float(np.max(np.abs(error[3:])))))
            # Confirm independent solver and error/Jacobian conventions on the
            # last ACCEPTED target, using its saved exact joint solution.
            control_q = np.array(rows[-2]['joint_target_rad'])
            control_state = kin.evaluate(control_q, include_manipulability=False, include_link_points=False)
            prior = rows[-2]['nominal_target']
            prior_p = np.array(prior['position_m'])
            prior_r = Rotation.from_rotvec(prior['rotation_vector_rad']).as_matrix()
            assert np.max(np.abs(control_state.position - prior_p)) < 1e-5
            assert np.max(np.abs(rotation_vector(prior_r.T @ control_state.rotation))) < 1e-4
            # Central difference checks the independent probe's analytic Jacobian.
            eps = 1e-6
            numeric = np.column_stack([(residual(previous + np.eye(7)[j] * eps) - residual(previous - np.eye(7)[j] * eps)) / (2 * eps) for j in range(7)])
            jac_error = float(np.max(np.abs(numeric - jacobian(previous))))
            assert jac_error < 1e-7
            item = dict(task=rel, trials=len(probes), feasible_count=sum(p['feasible'] for p in probes),
                        method='scipy bounded least_squares, pose only, analytic Jacobian, position divided by 0.3m',
                        constraints='identical joint bounds and strict pose acceptance thresholds; no conditioning/motion objective',
                        scope='failure to find a feasible point is not a global infeasibility proof',
                        random_seed=20261021, prior_accepted_fk_verified=True, jacobian_max_fd_error=jac_error,
                        probes=probes)
            (OUTPUT / rel / 'independent_feasibility.json').write_text(json.dumps(plain(item), indent=2) + '\n')
            results.append(item)
            print(json.dumps(plain({k: item[k] for k in ['task', 'trials', 'feasible_count', 'jacobian_max_fd_error']})), flush=True)
        finally:
            env.close()
    (OUTPUT / 'independent_feasibility_summary.json').write_text(json.dumps(plain(results), indent=2) + '\n')


if __name__ == '__main__':
    main()
