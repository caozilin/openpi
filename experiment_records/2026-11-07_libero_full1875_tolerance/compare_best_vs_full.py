"""Read-only comparison of audited selected candidates and authored full caps.

No execution/search imports: validate existing feedback from complete kinematic
state and per-frame clearance, then aggregate paired full episodes only.
"""
from pathlib import Path
from collections import Counter, defaultdict
from statistics import fmean
import datetime, hashlib, json, math

HERE = Path(__file__).resolve().parent
RUN = Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_ema85_full1875_20261107')

def read(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load_trial(row, expected_point, policy, expected_frames):
    raw, digest = read(row['raw_trial_path'])
    assert digest == row['raw_trial_sha256']
    assert raw['point'] == expected_point
    assert row['actual_pre_caps_deg'] + row['actual_post_caps_deg'] == expected_point
    for key,value in raw['row'].items():
        assert row[key] == value, (row['raw_trial_path'],key)
    r=raw['result'];e=r['evaluation'];k=r['trajectory_kinematics']
    assert e['physics_steps']==0 and e['control_rate_hz']==20
    assert e['expected_frames']==expected_frames
    assert not r['termination_reason'].startswith('runner_error:')
    complete=bool(e['trajectory_complete'] and e['kinematic_feasible_without_table'])
    assert complete==row['trajectory_complete']==row['solver_pass']
    if complete:
        assert k['samples']==expected_frames
        assert k['mean_joint_velocity_rad_s']==row['observed_velocity_rad_s']
        distances=e['robot_table_clearance']['samples']
        required=policy['required_clearance_m']
        assert len(required)==len(distances)==expected_frames
        assert [x['frame_index'] for x in distances]==list(range(expected_frames))
        assert required==[min(policy['minimum_m'],x) for x in policy['baseline_clearance_m']]
        assert policy['minimum_m']==.005 and policy['epsilon_m']==.0001
        g=min(x['distance_m']-req for x,req in zip(distances,required,strict=True))
        assert math.isclose(g,row['baseline_relative_g_m'],rel_tol=0,abs_tol=1e-12)
        success=g+policy['epsilon_m']>=0
        assert success==row['clearance_pass']==row['success']
        assert row['velocity_rad_s']==(k['mean_joint_velocity_rad_s'] if success else None)
        assert math.isfinite(k['mean_joint_velocity_rad_s']) and k['mean_joint_velocity_rad_s']>=0
    else:
        assert not row['success'] and row['velocity_rad_s'] is None
    return dict(path=row['raw_trial_path'],sha256=digest,point=expected_point,success=row['success'],
                solver_pass=complete,reason=row['reason'],termination_reason=r['termination_reason'],
                samples=k['samples'],velocity=k['mean_joint_velocity_rad_s'] if complete else None,
                relative_clearance_g_m=row.get('baseline_relative_g_m'))

def speed(blocks,key):
    assert all(x[key]['solver_pass'] for x in blocks)
    weights=[x[key]['samples']-1 for x in blocks]
    assert all(w>=0 for w in weights) and sum(weights)>0
    return math.fsum(x[key]['velocity']*w for x,w in zip(blocks,weights))/sum(weights)

def summarize(rows):
    n=len(rows);full=[r for r in rows if r['full_success']]
    complete=[r for r in rows if r['full_solver_pass']]
    def paired(subset):
        if not subset:return None
        a=fmean(r['full_velocity'] for r in subset);b=fmean(r['best_velocity'] for r in subset)
        return dict(episodes=len(subset),full_mean_velocity_rad_s=a,best_mean_velocity_rad_s=b,
                    best_reduction_percent=100*(a-b)/a,
                    best_faster=sum(r['best_velocity']<r['full_velocity']-1e-10 for r in subset),
                    equal=sum(abs(r['best_velocity']-r['full_velocity'])<=1e-10 for r in subset),
                    best_slower=sum(r['best_velocity']>r['full_velocity']+1e-10 for r in subset))
    return dict(episodes=n,search_success=sum(r['best_success'] for r in rows),
                full_success=len(full),search_success_rate_percent=100*sum(r['best_success'] for r in rows)/n,
                full_success_rate_percent=100*len(full)/n,
                full_failure_solver=sum(not r['full_solver_pass'] for r in rows),
                full_failure_clearance_only=sum(r['full_solver_pass'] and not r['full_success'] for r in rows),
                full_failure_any_clearance_block=sum(any(b['full']['solver_pass'] and not b['full']['success'] for b in r['blocks']) for r in rows),
                paired_both_accepted=paired(full),paired_kinematic_complete_including_clearance_rejections=paired(complete))

def main():
    audit,ah=read(RUN/'annotation_independent_final_audit.json')
    assert audit['complete'] and audit['episodes']==1875 and audit['adapter_errors']==0
    request,rh=read(RUN/'annotation_request.json')
    gains,gh=read(HERE/'kinematic_gain_pairs.json')
    gain_map={(r['suite'],r['task_id'],r['demo']):r for r in gains}
    rows=[];excluded=[];evidence=[];block_pairs=0
    for item in audit['results']:
        folder=RUN/item['suite']/f"task_{item['task_id']:02d}"/item['demo']
        if item['status']!='labeled':
            assert item['status']=='baseline_unavailable'
            excluded.append(dict(suite=item['suite'],task_id=item['task_id'],demo=item['demo'],status=item['status'],
                                 exclusion='Full-max candidate not evaluated: baseline-relative criterion unavailable'))
            continue
        prepared,ph=read(folder/'prepared_input.json');_,lh=read(folder/'tolerance_labels.json')
        assert lh==item['labels_sha256']
        blocks=[]
        for index,subtask in enumerate(prepared['subtasks']):
            ap=folder/f'subtask_{index:02d}'/'annotation.json';a,an_hash=read(ap)
            maximum=[v for phase in ('APP','EXEC') for v in subtask[phase+'_caps_deg'] for _ in range(2)]
            assert len(maximum)==12
            if any(maximum):
                assert maximum==a['design']['maximum_caps_deg']
                assert maximum in a['design']['initial_caps_deg']
            else:assert a['strict_authored_task']
            matches=[r for r in a['observations'] if r['actual_pre_caps_deg']+r['actual_post_caps_deg']==maximum]
            assert len(matches)==1
            m=matches[0];b=a['best'];assert b['success']
            assert b==min([r for r in a['observations'] if r['success']],key=lambda r:(r['velocity_rad_s'],r.get('index',0)))
            n=sum(f['subtask_index']==index for f in prepared['frames'])
            full=load_trial(m,maximum,a['clearance_policy'],n)
            best=load_trial(b,b['actual_pre_caps_deg']+b['actual_post_caps_deg'],a['clearance_policy'],n)
            assert best['success']
            if full['success']:assert best['velocity']<=full['velocity']+1e-12
            blocks.append(dict(index=index,maximum_caps_deg=maximum,full=full,best=best,annotation=str(ap),annotation_sha256=an_hash))
            block_pairs+=1
        full_pass=all(b['full']['success'] for b in blocks);solver=all(b['full']['solver_pass'] for b in blocks)
        row=dict(suite=item['suite'],task_id=item['task_id'],demo=item['demo'],blocks=blocks,
                 best_success=all(b['best']['success'] for b in blocks),full_success=full_pass,full_solver_pass=solver,
                 full_velocity=speed(blocks,'full') if solver else None,best_velocity=speed(blocks,'best'),
                 strictly_zero_domain=all(not any(b['maximum_caps_deg']) for b in blocks),
                 gp_numerical_recovered=(item['suite'],item['task_id'],item['demo'])==('libero_10',2,'demo_40'))
        assert math.isclose(row['best_velocity'],gain_map[(row['suite'],row['task_id'],row['demo'])]['best']['velocity'],rel_tol=1e-12,abs_tol=1e-12)
        rows.append(row)
        evidence.append(dict(suite=row['suite'],task_id=row['task_id'],demo=row['demo'],prepared_input_sha256=ph,labels_sha256=lh))
    assert len(rows)==1851 and len(excluded)==24 and block_pairs==2275
    assert len({(r['suite'],r['task_id'],r['demo']) for r in rows})==1851
    aggregate=summarize(rows)
    assert aggregate['paired_both_accepted']['best_slower']==0
    suites={suite:summarize([r for r in rows if r['suite']==suite]) for suite in ('libero_spatial','libero_object','libero_goal','libero_10')}
    tasks=[]
    for suite,task in sorted({(r['suite'],r['task_id']) for r in rows}):
        subset=[r for r in rows if (r['suite'],r['task_id'])==(suite,task)]
        tasks.append(dict(suite=suite,task_id=task,**summarize(subset)))
    result=dict(complete=True,checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                protocol='read-only audited cache85 existing initial maximum-cap candidates; per-episode conjunction across original independently initialized blocks; no new execution or rollout',
                maximum_definition='Both PRE/APP and POST/EXEC use each authored 5/15/30 degree cap in both signed directions; authored strict zeros stay zero; not unrestricted 111x30deg',
                success_definition='Every original block has complete admissible kinematic trajectory and min_t[c_t-min(5mm,c0_t)] >= -0.1mm; not LIBERO physical task success',
                conditional_population='1851 labeled episodes with usable frozen zero baseline; 24 baseline-unavailable sources excluded from both arms, not counted as maximum failures',
                speed_definition='Pool within each demo independent-block N-1 velocity samples; episode-equal mean on common accepted subset; no partial failed speed',
                selection_scope='Search best selected from same initial+16 candidate set including maximum; in-sample selection, not held-out generalization',
                aggregate=aggregate,suites=suites,tasks=tasks,
                excluding_strict_zero_domains=summarize([r for r in rows if not r['strictly_zero_domain']]),
                excluding_gp_numerical_recovered=summarize([r for r in rows if not r['gp_numerical_recovered']]),
                strict_zero_domain_count=sum(r['strictly_zero_domain'] for r in rows),excluded=excluded,
                verified_episodes=len(rows),verified_block_pairs=block_pairs,
                audit=str(RUN/'annotation_independent_final_audit.json'),audit_sha256=ah,request_sha256=rh,
                source_stats_pairs_sha256=gh,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(HERE/'best_vs_full_episodes.json',rows)
    write(HERE/'best_vs_full_sources.json',dict(audit=result['audit'],audit_sha256=ah,run=str(RUN),request_sha256=rh,episodes=evidence,excluded=excluded))
    write(HERE/'best_vs_full_summary.json',result)
    print(json.dumps(dict(aggregate=aggregate,suites=suites,excluding_zero=result['excluding_strict_zero_domains'],strict_count=result['strict_zero_domain_count']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
