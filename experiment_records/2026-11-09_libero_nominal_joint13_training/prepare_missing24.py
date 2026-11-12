"""Prepare a new isolated continuation; never modify the completed1875 run."""
from pathlib import Path
import hashlib,json,shutil,difflib
ROOT=Path('/media/czl/sata/franka_my_code')
OLD=ROOT/'franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107'
OUT=ROOT/'franka_mujoco/analysis_results/libero_missing24_baseline_observation_20261109'
REC=ROOT/'openpi/experiment_records/2026-11-09_libero_nominal_joint13_training'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def change(rel,edits):
 p=OUT/'source_snapshot'/rel;s=p.read_text();orig=s
 for old,new in edits:
  assert s.count(old)==1,(rel,old[:100],s.count(old));s=s.replace(old,new)
 p.write_text(s)
 return ''.join(difflib.unified_diff(orig.splitlines(True),s.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
def main():
 assert not (OUT/'annotation_request.json').exists(), 'Do not overwrite a frozen experiment'
 OUT.mkdir(exist_ok=True);shutil.copytree(OLD/'source_snapshot',OUT/'source_snapshot',dirs_exist_ok=True)
 patch=''
 patch+=change('tasktol/counterfactual/runner.py',[
 ('    kinematic_continue_after_table_contact: bool = False\n','    kinematic_continue_after_table_contact: bool = False\n    # Zero-reference observation only; failures remain failures. Never a candidate mode.\n    kinematic_baseline_observe_after_failure: bool = False\n'),
 ('    release_state = RotationReleaseState(\n        inherit_stage_offset=not zero_tolerance_nominal_reference,','    if spec.kinematic_baseline_observe_after_failure and not zero_tolerance_nominal_reference:\n        raise ValueError("Baseline observation requires strictly zero rotational tolerance")\n    release_state = RotationReleaseState(\n        inherit_stage_offset=not zero_tolerance_nominal_reference,'),
 ('            break\n        q_next = np.asarray(q_next, dtype=float)\n','            if not spec.kinematic_baseline_observe_after_failure:\n                break\n            # Rejected optimizer output must never become a reference.\n            q_next = q.copy()\n        q_next = np.asarray(q_next, dtype=float)\n'),
 ('            failure_details.append({"frame": frame_index, **collision})\n            break\n','            failure_details.append({"frame": frame_index, **collision})\n            if not spec.kinematic_baseline_observe_after_failure:\n                break\n'),
 ('        "backend": "kinematic_collision_checked",\n','        "backend": "kinematic_collision_checked",\n        "baseline_observation_only": spec.kinematic_baseline_observe_after_failure,\n        "baseline_rejection_policy": "hold_last_accepted_joints" if spec.kinematic_baseline_observe_after_failure else None,\n        "baseline_observation_complete": bool(spec.kinematic_baseline_observe_after_failure and executed == len(actions)),\n'),
 ('    if spec.kinematic_continue_after_table_contact and (\n','    if spec.kinematic_baseline_observe_after_failure and (\n        spec.execution_mode != "kinematic" or not spec.kinematic_record_clearance_samples\n        or not spec.kinematic_continue_after_table_contact or spec.kinematic_sampling_stride != 1\n    ):\n        raise ValueError("Baseline observation requires dense FK-only clearance recording")\n    if spec.kinematic_continue_after_table_contact and (\n'),
 ])
 patch+=change('tasktol/tolerance/clearance_policy.py',[
 ('def make_baseline(result, source_frame_indices, *, minimum_m=.005, epsilon_m=DEFAULT_EPSILON_M):','def make_baseline(result, source_frame_indices, *, minimum_m=.005, epsilon_m=DEFAULT_EPSILON_M, require_feasible=True):'),
 ("    if not (evaluation['trajectory_complete'] and evaluation.get('kinematic_feasible_without_table')):\n        raise BaselineUnavailable('Zero-tolerance replay lacks complete admissible kinematics; no baseline fabricated')", "    admissible = evaluation['trajectory_complete'] and evaluation.get('kinematic_feasible_without_table')\n    if require_feasible and not admissible:\n        raise BaselineUnavailable('Zero-tolerance replay lacks complete admissible kinematics; no baseline fabricated')\n    if not require_feasible and not (evaluation.get('baseline_observation_only')\n            and evaluation.get('baseline_observation_complete')\n            and evaluation.get('baseline_rejection_policy') == 'hold_last_accepted_joints'):\n        raise BaselineUnavailable('Require complete explicitly marked baseline observation; no samples fabricated')"),
 ])
 patch+=change('tasktol/data/libero_annotation/dsp.py',[
 ('from pathlib import Path\n','from pathlib import Path\nfrom dataclasses import replace\n'),
 ('    all_frames = document["frames"]\n','    all_frames = document["frames"]\n    baseline_require_feasible = document.get("baseline_require_feasible", True)\n'),
 ('            zero_spec = stage_caps(spec, [0.] * 12, masks)\n','            zero_spec = replace(stage_caps(spec, [0.] * 12, masks),\n                kinematic_baseline_observe_after_failure=not baseline_require_feasible)\n'),
 ('                                       epsilon_m=provenance["clearance_policy"]["epsilon_m"])','                                       epsilon_m=provenance["clearance_policy"]["epsilon_m"],\n                                       require_feasible=baseline_require_feasible)'),
 ('            current.update(clearance_m=0., clearance_policy=policy,\n','            current.update(clearance_m=0., clearance_policy=policy,\n                           reuse_zero_baseline=baseline_require_feasible,\n'),
 ('                                  zero_result=read(current["zero_baseline_trial"])["result"])','                                  zero_result=read(current["zero_baseline_trial"])["result"] if baseline_require_feasible else None)'),
 ])
 patch+=change('label_franka_nominal_dsp.py',[
 ("zero_result=read(baseline_path)['result'] if not any(point) else None)","zero_result=read(baseline_path)['result'] if not any(point) and settings.get('reuse_zero_baseline', True) else None)"),
 ])
 patch+=change('tasktol/data/libero_annotation/audit.py',[
 ("                        assert z['trajectory_complete'] and z['kinematic_feasible_without_table']", "                        if request.get('baseline_require_feasible', True):\n                            assert z['trajectory_complete'] and z['kinematic_feasible_without_table']\n                        else:\n                            assert z['baseline_observation_only'] and z['baseline_observation_complete']\n                            assert z['baseline_rejection_policy'] == 'hold_last_accepted_joints'\n                            assert not spec['kinematic_baseline_observe_after_failure']"),
 ("                        if not any(point):\n                            assert read(trial)['result'] == zero", "                        if not any(point) and request.get('baseline_require_feasible', True):\n                            assert read(trial)['result'] == zero"),
 ])
 write(REC/'missing24_source_changes.json',dict(original_snapshot=str(OLD/'source_snapshot'),new_snapshot=str(OUT/'source_snapshot'),modified_files=['tasktol/counterfactual/runner.py','tasktol/tolerance/clearance_policy.py','tasktol/data/libero_annotation/dsp.py','label_franka_nominal_dsp.py','tasktol/data/libero_annotation/audit.py']))
 (REC/'missing24_source_changes.patch').write_text(patch)
 selected=[r for r in read(OLD/'annotation_independent_final_audit.json')['results'] if r['status']=='baseline_unavailable'];assert len(selected)==24
 keys={(r['suite'],r['task_id'],r['demo']) for r in selected}
 oldsel=read(OLD/'selection.json');sources=[s for s in oldsel['sources'] if (s['suite'],s['task_id'],s['demo']) in keys];assert len(sources)==24
 frozen={str(p):sha(p) for p in (OUT/'source_snapshot').rglob('*') if p.is_file() and p.suffix in ('.py','.lock')}
 preserve={str(OLD/'annotation_independent_final_audit.json'):sha(OLD/'annotation_independent_final_audit.json')}
 for row in read(OLD/'annotation_independent_final_audit.json')['results']:
  folder=OLD/row['suite']/f"task_{row['task_id']:02d}"/row['demo']
  for name in ('annotation_status.json','tolerance_labels.json','zero_baseline.json'):
   p=folder/name
   if p.exists():preserve[str(p)]=sha(p)
 for s in sources:
  folder=OLD/s['suite']/f"task_{s['task_id']:02d}"/s['demo'];p=folder/'prepared_input.json';doc=read(p)
  for name,h in doc['provenance']['source_code_sha256'].items():
   relative=Path(name).relative_to(ROOT/'franka_mujoco');target=OUT/'source_snapshot'/relative
   assert target.exists(),target
   assert sha(OLD/'source_snapshot'/relative)==h,(name,'original snapshot mismatch')
  doc['provenance']['source_code_sha256']=frozen
  doc['provenance']['supplement']=dict(original_prepared_input=str(p),original_prepared_sha256=sha(p),baseline_require_feasible=False,rejection_policy='hold_last_accepted_joints',original_status_sha256=sha(folder/'annotation_status.json'))
  doc['baseline_require_feasible']=False
  write(OUT/s['suite']/f"task_{s['task_id']:02d}"/s['demo']/'prepared_input.json',doc)
 oldreq=read(OLD/'annotation_request.json');req=dict(oldreq,planned_episodes=24,sources=sources,selection='All24 baseline_unavailable sources; no outcome selection within this group',code_sha256=frozen,baseline_require_feasible=False,baseline_rejection_policy='hold_last_accepted_joints',original_request=str(OLD/'annotation_request.json'),original_request_sha256=sha(OLD/'annotation_request.json'))
 write(OUT/'annotation_request.json',req);write(OUT/'selection.json',dict(complete=True,selection=req['selection'],sources=sources));write(OUT/'original_results_protection.json',dict(sha256=preserve))
 print(json.dumps(dict(output=str(OUT),sources=len(sources),protected_files=len(preserve),frozen_files=len(frozen))))
if __name__=='__main__': main()
