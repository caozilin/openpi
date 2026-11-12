from pathlib import Path
import json,hashlib,tempfile,shutil
from tasktol.data.trial_io import compact_trial,read_json,REFERENCE_KEY,STATIC_KEYS
from tasktol.data.libero_annotation.source import read as source_read
import label_franka_nominal_dsp as label
H=Path('/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance')
R=Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107')
prior=json.loads((H/'storage_change_verification.json').read_text());checks=[]
with tempfile.TemporaryDirectory(prefix='libero-trial-reference-verify-') as temporary:
 root=Path(temporary)
 for row in prior['real_trial_roundtrip_checks']:
  old=Path(row['original']);before=hashlib.sha256(old.read_bytes()).hexdigest();assert before==row['sha256']
  raw=json.loads(old.read_bytes());assert read_json(old)==source_read(old)==label.read(old)==raw
  target=root/old.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True)
  demo=next(p for p in old.parents if p.name.startswith('demo_'))
  newdemo=root/demo.relative_to(R);shutil.copy2(demo/'prepared_input.json',newdemo/'prepared_input.json')
  packed=compact_trial(raw,target);assert REFERENCE_KEY in packed['result']
  assert all(k not in packed['result']['domain_randomization'] for k in STATIC_KEYS)
  target.write_text(json.dumps(packed,ensure_ascii=False,indent=2)+'\n')
  assert read_json(target)==source_read(target)==label.read(target)==raw
  assert hashlib.sha256(old.read_bytes()).hexdigest()==before
  checks.append(dict(source=str(old),sha256=before,old_read_equal=True,new_roundtrip_equal=True,static_keys_omitted=True))
value=dict(complete=True,checks=checks,new_trial_files_only_in_temporary_directory=True,existing_trials_unchanged=True,production_reader_integration=True)
(H/'storage_change_postapply_verification.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(f'Passed {len(checks)} existing and referenced trial roundtrips through all three readers')
