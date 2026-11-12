"""Formal400: newly rerun allseed0..5 plus preserved first-pass seed6..9."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.libero.eval_formal_nominal import RUN, SUITES, read, dump
from examples.libero.summarize_nominal10 import audited_rows, stats


def merge_formal(new240, preserved160):
    """Select by the predeclared seed partition only, never outcome."""
    expected_a = {(s,t,z) for s in SUITES for t in range(10) for z in range(6)}
    expected_b = {(s,t,z) for s in SUITES for t in range(10) for z in range(6,10)}
    key = lambda r: (r['suite'],r['task_id'],r['seed'])
    assert len(new240) == len(expected_a) == 240 and {key(r) for r in new240} == expected_a
    assert len(preserved160) == len(expected_b) == 160 and {key(r) for r in preserved160} == expected_b
    assert all(r['solver_failure_policy'] == 'hold_last_accepted_reference' for r in new240 + preserved160)
    return sorted(new240 + preserved160, key=lambda r:(SUITES.index(r['suite']),r['task_id'],r['seed']))


def main():
    request = read(RUN / 'evaluation_request.json')
    audit = read(RUN / 'evaluation_independent_final_audit.json')
    assert audit['complete'] and audit['episodes'] == 240
    extension = Path(request['preserved_seed6to9_run'])
    for path, expected in request['preserved_artifact_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    a, aa = audited_rows(RUN, 'results', 'formal_new_all_seed0to5_hold20')
    b, ba = audited_rows(extension, 'results', 'preserved_first_pass_seed6to9_hold20')
    rows = merge_formal(a,b)
    # Even old successful trials may not leak into this manifest.
    for row in rows:
        parent = RUN if row['seed'] < 6 else extension
        assert Path(row['source_result']).resolve().is_relative_to(parent.resolve())
    summary = stats(rows)
    assert all(x['episodes'] == 10 for x in summary['per_task'])
    manifest = dict(complete=True, formal_baseline=True, **summary, results=rows,
                    source_selection='Predeclared seed partition: all newly rerun0..5 plus preserved first-pass6..9; outcomes never used for inclusion/replacement',
                    targeted_failed_retests_included=False, newly_executed_episodes=240,
                    preserved_episodes=160, all_400_rerun_in_current_invocation=False,
                    checked_at=datetime.datetime.now().astimezone().isoformat(), audit_chain=[aa,ba])
    dump(RUN / 'combined_formal_400.json',manifest)
    with (RUN / 'combined_formal_400.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tol=ROOT/'data/libero_evaluation/pi05_reconstructed_10000_tol30_confirmedmask_hold20_40tasks_10seeds_20261104'
    tr,ta=audited_rows(tol,'results','confirmed_task_mask_tolerance30_hold20')
    tm={(r['suite'],r['task_id'],r['seed']):r for r in tr}
    assert len(tm)==400
    paired=[];counts=collections.Counter()
    for r in rows:
        t=tm[(r['suite'],r['task_id'],r['seed'])]
        assert r['initial_state_sha256']==t['initial_state_sha256']
        with np.load(Path(r['source_result']).parent/'trajectory.npz',allow_pickle=False) as nz, np.load(Path(t['source_result']).parent/'trajectory.npz',allow_pickle=False) as tz:
            assert np.array_equal(nz['initial_state'],tz['initial_state'])
            initial_prestate_delta=float(np.max(np.abs(nz['pre_action_states'][0]-tz['pre_action_states'][0])))
            assert initial_prestate_delta<=1e-12
            count=min(5,len(nz['actions']),len(tz['actions']))
            delta=np.max(np.abs(nz['actions'][:count]-tz['actions'][:count]),axis=0)
        outcome=('both_success' if r['success'] and t['success'] else 'tolerance_improved' if t['success'] else 'tolerance_regressed' if r['success'] else 'both_failed')
        counts[outcome]+=1
        paired.append(dict(suite=r['suite'],task_id=r['task_id'],seed=r['seed'],nominal_success=r['success'],tolerance_success=t['success'],paired_outcome=outcome,nominal_termination=r['termination'],tolerance_termination=t['termination'],nominal_source=r['source_result'],tolerance_source=t['source_result'],initial_prestate_max_delta=initial_prestate_delta,first_chunk_translation_max_delta_m=float(max(delta[:3])),first_chunk_rotation_max_delta_rad=float(max(delta[3:6])),first_chunk_gripper_max_delta=float(delta[6])))
    with (RUN/'formal_nominal_vs_tolerance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    comparison=dict(complete=True,formal_nominal=summary,tolerance=stats(tr),paired_counts=dict(counts),pairs=paired,tolerance_audit=ta,source_selection=manifest['source_selection'],targeted_failed_retests_included=False)
    dump(RUN/'formal_nominal_vs_tolerance.json',comparison)
    dump(RUN/'formal_400_audit.json',dict(complete=True,unique_task_seed_pairs=400,tasks=40,episodes_per_task=10,seeds=list(range(10)),newly_executed_episodes=240,preserved_first_pass_episodes=160,targeted_failed_retests_included=False,selection_predeclared_by_seed_only=True,all_1600_nominal_artifact_hashes_match_independent_audits=True,all_400_official_initial_states_match_tolerance=True,first_pre_action_state_max_delta=max(r['initial_prestate_max_delta'] for r in paired),source_audits=[aa,ba],formal_manifest_sha256=hashlib.sha256((RUN/'combined_formal_400.json').read_bytes()).hexdigest(),paired_comparison_sha256=hashlib.sha256((RUN/'formal_nominal_vs_tolerance.json').read_bytes()).hexdigest(),checked_at=datetime.datetime.now().astimezone().isoformat()))
    lines=['# 正式无容差测评：40任务×同seed0–9','',
           f"正式无容差：**{summary['successes']}/400（{summary['success_rate']*100:.2f}%）**。本次所有任务的seed0–5共240回合完整新测；seed6–9保留此前完整一次性测评的160回合。按预定seed分区合并，不根据成功/失败选择来源，不纳入之前17条失败定向重测。所有400个task/seed组合唯一，每任务10次。",'',
           '|任务集|新测seed0–5成功/60|保留seed6–9成功/40|正式无容差成功/100|对应mask容差成功/100|','|---|---:|---:|---:|---:|']
    sa,sb,st=stats(a),stats(b),stats(tr)
    for suite in SUITES:lines.append(f"|{suite}|{sa['per_suite'][suite]['successes']}/60|{sb['per_suite'][suite]['successes']}/40|{summary['per_suite'][suite]['successes']}/100|{st['per_suite'][suite]['successes']}/100|")
    lines+=['','固定最终10000步checkpoint，nostate、无容差mask000/boundsNone、默认IPOPT/shared关节伺服、20Hz/500Hz、replan5/horizon10、原步数上限和夹爪deadband0.5。求解失败不更新名义参考，保持上一接受关节/夹爪参考，连续20帧（1秒）失败终止。服务器GPU0/1/2/3/4/6只推理，本地六客户端仿真。','',
            '本次240回合全部客户端exit0、基础设施错误0，逐回合官方初态、全部动作/state/25物理步、失败回滚与20帧计数、原生成功谓词、视频全帧解码和真实服务器身份独立审计通过。保留160的审计链及全部四类artifact SHA重新核对通过。首次策略动作数值差如实记录，不假设不同GPU运行逐位一致。','',
            '## 与对应mask容差组的正式同seed对照','',f"容差组370/400（92.50%）。配对：{json.dumps(dict(counts),ensure_ascii=False)}。两组相同官方初态、相同失败策略；各组独立闭环产生后续动作。该对照替代此前包含17次失败定向重测替换的探索性基线；旧报告和原始数据完整保留。",'',
            '## 逐任务正式结果','','|任务集|task_id（0起始）|无容差成功/10|','|---|---:|---:|']
    for x in summary['per_task']:lines.append(f"|{x['suite']}|{x['task_id']:02d}|{x['successes']}/10|")
    lines+=['','来源清单：combined_formal_400.json/csv；正式容差对照：formal_nominal_vs_tolerance.json/csv；新240独立审计：evaluation_independent_final_audit.json。']
    (RUN/'FORMAL_NOMINAL_400_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(formal400=summary,new240=sa,preserved160=sb,paired_counts=dict(counts)),ensure_ascii=False))


if __name__=='__main__':main()
