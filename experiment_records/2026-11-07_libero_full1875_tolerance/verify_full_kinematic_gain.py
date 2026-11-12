from pathlib import Path
import json, math, hashlib
H=Path('/media/czl/sata/franka_my_code/openpi/experiment_records/2026-11-07_libero_full1875_tolerance')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(H/'kinematic_gain_summary.json');rows=read(H/'kinematic_gain_pairs.json');assert s['complete'] and len(rows)==1851
assert len({(r['suite'],r['task_id'],r['demo']) for r in rows})==1851
checks={}
for group, subset in [('all_labeled',rows),('excluding_gp_recovered_case',[r for r in rows if (r['suite'],r['task_id'],r['demo'])!=('libero_10',2,'demo_40')])]:
 for key in ('velocity','acceleration','mY','sigma_min'):
  a=math.fsum(r['zero'][key] for r in subset)/len(subset);b=math.fsum(r['best'][key] for r in subset)/len(subset)
  sign=-1 if key in ('velocity','acceleration') else 1
  gain=100*sign*(b-a)/a;actual=s[group]['metrics'][key]
  for x,y in [(a,actual['zero_mean']),(b,actual['best_mean']),(gain,actual['improvement_percent_of_means'])]:assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),(group,key,x,y)
  checks[f'{group}.{key}']=dict(zero_mean=a,best_mean=b,improvement_percent=gain)
assert len(s['missing'])==24 and all(r['status']=='baseline_unavailable' for r in s['missing'])
v=dict(complete=True,episodes=1851,excluded_baseline_unavailable=24,unique_sources=True,independent_aggregation='math.fsum over per-demo pairs; ratio of overall means',checks=checks,summary_sha256=sha(H/'kinematic_gain_summary.json'),pairs_sha256=sha(H/'kinematic_gain_pairs.json'))
(H/'kinematic_gain_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,indent=2))
