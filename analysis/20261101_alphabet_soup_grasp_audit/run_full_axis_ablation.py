"""Complete the ten fixed paired demos; same existing rollout worker."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path('/media/czl/sata/franka_my_code/openpi')
sys.path.insert(0,str(ROOT))
from experiments.libero_tasktol_compat import ablate_alphabet_soup_grasp_axes as ab
OUT=ab.OUT
entries=[e for e in json.loads((ab.PREVIOUS/'plan.json').read_text())['entries']if e['task']['task_name']=='pick_up_the_alphabet_soup_and_place_it_in_the_basket']
jobs=[(e['key'],mask)for e in entries for mask in ['001','010']]+[(f'libero_object/00/{demo}','011')for demo in ['demo_4','demo_30']]
manifest=[]
for n,(name,digest)in enumerate(ab.freeze().items()):
 source=Path(name)
 if source.is_relative_to(ab.prior.runner.SHARED/'tasktol'):source=ab.prior.runner.ACTIVE_SHARED/source.relative_to(ab.prior.runner.SHARED)
 assert ab.prior.runner.sha(source)==digest
 dest=OUT/'source_snapshot'/f'{n:03d}_{source.name}';dest.parent.mkdir(exist_ok=True)
 dest.write_bytes(source.read_bytes());manifest.append(dict(canonical=name,executed=str(source),snapshot=str(dest.relative_to(OUT)),sha256=digest))
ab.prior.runner.dump(OUT/'source_snapshot_manifest.json',manifest)
ab.prior.runner.dump(OUT/'full10_design.json',dict(jobs=jobs,selection='All ten fixed alphabet soup demos; every001 and010 variant run once, and two011 exact reproduction checks.',changes='PRE mask only. Original EXEC111 retained. Bounds30deg, native20Hz/500Hz, original actions/gripper, optimizer, servo and initial states unchanged.',driver_sha256=ab.prior.runner.sha(Path(__file__))))
env=dict(os.environ,MUJOCO_GL='egl',NUMBA_CACHE_DIR=str(OUT/'numba_cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
def run(job):
 key,mask=job;record=OUT/mask/key/'record.json'
 if not record.exists():
  with (OUT/f'{key.split("/")[-1]}_{mask}.log').open('w')as f:subprocess.run([sys.executable,'-u',str(Path(ab.__file__)),'--worker','--key',key,'--mask',mask],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 return json.loads(record.read_text())
results=[]
with ThreadPoolExecutor(max_workers=2)as pool:
 for future in as_completed([pool.submit(run,j)for j in jobs]):
  r=future.result();results.append(r);print(json.dumps(r),flush=True)
  ab.prior.runner.dump(OUT/'full10_summary.json',dict(complete=len(results)==len(jobs),results=results))
