"""Offline fixed-state IK diagnosis, no rollout and no policy inference."""
import os,sys,json,pathlib,dataclasses,random,time
ROOT=pathlib.Path('/media/czl/sata/franka_my_code/openpi');F=ROOT/'data/libero_tasktol_joint_servo/20261025_success10_v1/frozen_shared_runtime'
os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=[str(F),str(ROOT),str(ROOT/'third_party/libero')]
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, minimize
from libero.libero.envs.env_wrapper import ControlEnv
from tasktol.simulation.libero import libero_handles
from tasktol.control.ipopt import IpoptIK,IpoptSettings
from tasktol.control.pose import TargetPose
from tasktol.control.task_space.objective import ObjectiveSettings
RUN=ROOT/'data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_40tasks_10seeds_extension_20261104'
p=RUN/'episodes/libero_spatial/task_04/seed_7';rows=json.loads((p/'trajectory.json').read_text());states=np.load(p/'trajectory.npz')['pre_action_states'];task=next(x for x in json.loads((RUN/'official_tasks.json').read_text()) if x['suite']=='libero_spatial' and x['task_id']==4)
np.random.seed(7);random.seed(7)
env=ControlEnv(bddl_file_name=str(ROOT/'third_party/libero/libero/libero/bddl_files'/task['problem_folder']/task['bddl_file']),controller='JOINT_POSITION',use_camera_obs=False,has_offscreen_renderer=False,control_freq=20)
env.seed(7);env.reset();h=libero_handles(env);model=env.sim.model._model;data=env.sim.data._data
out=dict(scope='Fixed recorded states/targets; no physics stepping or model policy calls; diagnostic budgets do not alter benchmark',cases=[])
try:
 for idx in [76,77,96]:
  env.set_state(states[idx]);env.sim.forward();row=rows[idx];pos=np.array(row.get('rejected_nominal_target',row['nominal_target'])['position_m']);rot=Rotation.from_rotvec(row.get('rejected_nominal_target',row['nominal_target'])['rotation_vector_rad']).as_matrix();target=TargetPose(pos,rot);last=np.array(rows[idx-1]['joint_target_rad']);measured=data.qpos[h.qpos_ids].copy()
  case=dict(frame_zero_based=idx,recorded_optimizer=row['optimizer'],target_position=pos.tolist(),measured_q=measured.tolist(),last_accepted_q=last.tolist(),ipopt=[])
  for budget in [20,200]:
   ik=IpoptIK(model,h,dataclasses.replace(IpoptSettings(),max_iterations=budget,max_cpu_time_s=2.0),ObjectiveSettings.delta_q_squared_conditioning_safeguard());ik._xopt=last.copy();ik._previous_state=np.array(rows[max(0,idx-2)]['joint_target_rad']);ik._previous_state2=np.array(rows[max(0,idx-3)]['joint_target_rad']);q,d=ik.solve(data,target,np.ones(6,dtype=bool));fk=ik.kinematics.evaluate(q)
   case['ipopt'].append(dict(budget=budget,feasible=d.feasible,status=d.status,iterations=d.iterations,elapsed_ms=d.elapsed_ms,returned_q=q.tolist(),position_error_m=float(np.max(np.abs(fk.position-pos))),rotation_error_rad=float(np.max(np.abs(Rotation.from_matrix(rot.T@fk.rotation).as_rotvec())))))
  fk=ik.kinematics.evaluate(last);sing=np.linalg.svd(fk.jacobian,compute_uv=False);case.update(joint_lower=h.joint_lower.tolist(),joint_upper=h.joint_upper.tolist(),distance_to_joint_bound_rad=np.minimum(last-h.joint_lower,h.joint_upper-last).tolist(),jacobian_singular_values=sing.tolist(),target_delta_from_last_fk_m=(pos-fk.position).tolist())
  def residual(q):
   k=ik.kinematics.evaluate(q,include_manipulability=False,include_link_points=False)
   return np.r_[k.position-pos,Rotation.from_matrix(rot.T@k.rotation).as_rotvec()]
  def jac(q):
   k=ik.kinematics.evaluate(q,include_manipulability=False,include_link_points=False)
   return ik.kinematics.pose_error_jacobian(k,rot)
  sols=[]
  seeds=[last,measured]+[np.clip(last+np.random.default_rng(k).normal(0,.35,7),h.joint_lower+1e-8,h.joint_upper-1e-8) for k in range(6)]
  for seed in seeds:
   sol=least_squares(residual,np.clip(seed,h.joint_lower+1e-8,h.joint_upper-1e-8),jac=jac,bounds=(h.joint_lower,h.joint_upper),max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
   err=residual(sol.x);sols.append(dict(position_error_m=float(np.max(np.abs(err[:3]))),rotation_error_rad=float(np.max(np.abs(err[3:]))),q=sol.x.tolist(),nfev=sol.nfev,status=int(sol.status)))
  case['bounded_least_squares_multistart']=sols
  from tasktol.geometry import rotation_tolerance_coordinates
  from tasktol.tolerance.frame import box_tolerance_frame
  def position_residual(q):return ik.kinematics.evaluate(q,include_manipulability=False,include_link_points=False).position-pos
  def position_jac(q):return ik.kinematics.evaluate(q,include_manipulability=False,include_link_points=False).jacobian[:3]
  relaxed=minimize(lambda q: .5*float(np.sum((q-last)**2)),last,jac=lambda q:q-last,method='SLSQP',bounds=list(zip(h.joint_lower,h.joint_upper)),constraints=[dict(type='eq',fun=lambda q:position_residual(q)*100,jac=lambda q:position_jac(q)*100)],options=dict(maxiter=300,ftol=1e-12))
  rk=ik.kinematics.evaluate(relaxed.x);ang=rotation_tolerance_coordinates(rk.rotation,rot,box_tolerance_frame(rot))
  case['same_position_rotation_relaxed_witness']=dict(q=relaxed.x.tolist(),position_error_m=float(np.max(np.abs(rk.position-pos))),rotation_difference_deg=np.rad2deg(ang).tolist(),within_target_centered_xyz_30deg=bool(np.all(np.abs(ang)<np.pi/6)),scope='Witness around the identical rejected target orientation; not a replay of the full stage-relative tolerance controller')
  out['cases'].append(case);print('checked',idx,case['ipopt'], 'best_ls',min(x['position_error_m']+x['rotation_error_rad'] for x in sols),'relaxed',case['same_position_rotation_relaxed_witness'],flush=True)
finally:env.close()
(ROOT/'analysis/spatial04_seed7_ik_20261104/diagnosis.json').write_text(json.dumps(out,indent=2));print('COMPLETE')
