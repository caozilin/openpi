"""Freeze the authorized400 cases and existing shared runtime; start no processes."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from examples.libero.eval_joint13_predicted_libero_local import assignments, IDENTITY, SUITES, LIMITS, PORTS
from examples.libero.success_continuation import request_fields
from examples.libero.predicted_tolerance_policy import POLICY
REC=Path(__file__).resolve().parent
RUN=ROOT/'data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110'
BASE=ROOT/'data/libero_evaluation/pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104'
SHARED=ROOT.parent/'franka_mujoco'
REMOTE='/mnt/SharedData/czl/runs/libero_eval_joint13_successmix20_predtol_400_20261110'
TRAIN='/mnt/SharedData/czl/runs/pi05_libero_joint13_successmix20_lora_6gpu_bs72_10k_20261109'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 if (RUN/'evaluation_request.json').exists():
  from examples.libero.eval_joint13_predicted_libero_local import validate
  validate(json.loads((RUN/'evaluation_request.json').read_text()));print('Existing frozen request verified; no restart');return
 snapshot=RUN/'frozen_shared_runtime/tasktol'
 shutil.copytree(SHARED/'tasktol',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
 for n in ['official_tasks.json','official_initial_states.npz']:
  shutil.copyfile(BASE/n,RUN/n)
 server=ROOT/'examples/libero/serve_joint13_libero_eval_server.py';shutil.copyfile(server,RUN/server.name)
 paths=[ROOT/'examples/libero'/n for n in ['eval_joint13_predicted_libero_local.py','predicted_tolerance_policy.py','tasktol_actions.py','solver_failure_hold.py','success_continuation.py','audit_joint13_predicted_evaluation.py','mask_viewer/phase_adapter.py','serve_joint13_libero_eval_server.py']]
 paths += [ROOT/'experiments/libero_tasktol_compat/mujoco_joint_servo.py']
 paths += list(snapshot.rglob('*.py'))
 paths += list((SHARED/'tasktol').rglob('*.py'))
 paths += list((SHARED/'assets/mujoco_menagerie/franka_emika_panda').rglob('*.xml'))
 paths += [ROOT/'third_party/libero/libero/libero/bddl_files'/t['problem_folder']/t['bddl_file'] for t in json.loads((RUN/'official_tasks.json').read_text())]
 paths += [Path(t['source_init_states']) for t in json.loads((RUN/'official_tasks.json').read_text())]
 paths += list((ROOT/'packages/openpi-client/src/openpi_client').rglob('*.py'))
 paths=[p.resolve() for p in paths if '__pycache__' not in p.parts]
 req=dict(authorized=True,protocol='libero.joint13.predicted_tolerance.400.v1',client_location='local',server_role='six inference replicas only',
  eval_identity=IDENTITY,server_eval_identity=IDENTITY,expected_episodes=400,tasks=40,seeds=list(range(10)),gpu_ids=[0,1,2,3,4,6],ports=PORTS,
  checkpoint_step=10000,replan_steps=5,action_horizon=10,state_enabled=False,max_steps=dict(zip(SUITES,LIMITS)),gripper_deadband=.5,
  tolerance_enabled=True,tolerance_bounds_policy=POLICY,tolerance_order=['rx-','rx+','ry-','ry+','rz-','rz+'],tolerance_units='rad',tolerance_clip_rad=[0.,3.141592653589793/6],
  phase_policy='existing causal shared finger classifier; ENG/DIS strict and heading degeneracy protection; no authored task mask or future frames',
  runtime=str(RUN/'frozen_shared_runtime'),runtime_version='current shared runtime frozen20261110; distinct from historical20261025 benchmark snapshot',
  solver_failure_policy='hold_last_accepted_reference',max_consecutive_solver_failures=20,solver_failure_timeout_seconds=1.,
  connection_policy='ping disabled; finite600s response timeout',
  remote_inference_run=REMOTE,training_run=TRAIN,training_request_sha256='4c9a7fc7d88dc7c1c07df20c7276a264ed6f9df052cca151318370c8a7268fb0',
  norm_stats_sha256='e26cd8d5baade92e2f081435cb497df85f569439fa1e12e4f93fa7b66d728e42',
  selected_episodes=[dict(suite=s,task_id=t,seed=z) for s in SUITES for t in range(10) for z in range(10)],worker_assignments=assignments(),
  source_sha256={str(p):sha(p) for p in sorted(set(paths))},preserved_artifact_sha256={str(RUN/n):sha(RUN/n) for n in ['official_tasks.json','official_initial_states.npz',server.name]},
  server_entry_sha256=sha(server),server_entry_name=server.name,created_at=datetime.datetime.now().astimezone().isoformat(),**request_fields())
 (RUN/'evaluation_request.json').write_text(json.dumps(req,ensure_ascii=False,indent=2)+'\n')
 (REC/'evaluation_request.json').write_bytes((RUN/'evaluation_request.json').read_bytes())
 from examples.libero.eval_joint13_predicted_libero_local import validate
 validate(req)
 out=dict(complete=True,run=str(RUN),request_sha256=sha(RUN/'evaluation_request.json'),episodes=400,worker_counts={k:len(v) for k,v in assignments().items()},source_files=len(req['source_sha256']),official_inputs_match_baseline=True,local_free_gib=shutil.disk_usage(RUN).free/2**30,estimated_output_gib=.8,processes_started=False)
 (REC/'preparation_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
