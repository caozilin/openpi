"""Independently recount saved episode pairs and verify aggregate statistics."""
from pathlib import Path
import json,hashlib,math,datetime
H=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(H/'best_vs_full_summary.json');rows=read(H/'best_vs_full_episodes.json')
assert len(rows)==1851 and len({(r['suite'],r['task_id'],r['demo']) for r in rows})==1851
checks={}
for key,group in [('aggregate',rows)]+[(suite,[r for r in rows if r['suite']==suite]) for suite in s['suites']]:
 actual=s[key] if key=='aggregate' else s['suites'][key]
 good=[r for r in group if all(b['full']['success'] for b in r['blocks'])]
 badsolver=[r for r in group if any(not b['full']['solver_pass'] for b in r['blocks'])]
 badclear=[r for r in group if r not in good and r not in badsolver]
 assert len(good)+len(badsolver)+len(badclear)==len(group)
 assert (len(good),len(badsolver),len(badclear))==(actual['full_success'],actual['full_failure_solver'],actual['full_failure_clearance_only'])
 assert all(r['best_success'] and all(b['best']['success'] for b in r['blocks']) for r in group)
 totals={name:math.fsum(r[name+'_velocity'] for r in good)/len(good) for name in ('full','best')}
 gain=100*(1-totals['best']/totals['full'])
 for a,b in [(totals['full'],actual['paired_both_accepted']['full_mean_velocity_rad_s']),
             (totals['best'],actual['paired_both_accepted']['best_mean_velocity_rad_s']),
             (gain,actual['paired_both_accepted']['best_reduction_percent'])]:assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 checks[key]=dict(episodes=len(group),full_success=len(good),solver_failure=len(badsolver),clearance_only_failure=len(badclear),paired_speed_gain_percent=gain)
assert len(s['excluded'])==24 and all(r['status']=='baseline_unavailable' for r in s['excluded'])
assert s['verified_block_pairs']==sum(len(r['blocks']) for r in rows)==2275
assert len([r for r in rows if r['strictly_zero_domain']])==83
assert sha(Path(s['audit']))==s['audit_sha256']
assert sha(Path(s['audit']).parent/'annotation_request.json')==s['request_sha256']
out=dict(complete=True,checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 episodes=1851,blocks=2275,excluded_baseline_unavailable=24,checks=checks,
 source_feedback_recomputed_from_raw_kinematic_and_per_frame_clearance=True,
 common_subset_only_for_speed=True,partial_failed_speed_excluded=True,
 summary_sha256=sha(H/'best_vs_full_summary.json'),episode_pairs_sha256=sha(H/'best_vs_full_episodes.json'))
(H/'best_vs_full_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
