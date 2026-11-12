"""Build provenance-preserving 400-episode views from independently audited runs."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'data/libero_evaluation/pi05_reconstructed_10000_nominal_hold20_40tasks_10seeds_extension_20261104'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return row['suite'], row['task_id'], row['seed']


def merge_views(original, retry, extension):
    a, b, c = ({key(r): r for r in group} for group in (original, retry, extension))
    assert [len(a), len(b), len(c)] == [len(original), len(retry), len(extension)]
    assert set(a) == {(s, t, z) for s in SUITES for t in range(10) for z in range(6)}
    assert set(c) == {(s, t, z) for s in SUITES for t in range(10) for z in range(6, 10)}
    assert set(b) == {k for k, r in a.items() if not r['success']}
    original_plus_new = dict(a, **{})
    original_plus_new.update(c)
    latest = dict(original_plus_new)
    latest.update(b)
    assert len(latest) == len(original_plus_new) == 400
    return list(original_plus_new.values()), list(latest.values())


def stats(rows):
    return dict(episodes=len(rows), successes=sum(r['success'] for r in rows), success_rate=sum(r['success'] for r in rows) / len(rows),
                terminations=dict(collections.Counter(r['termination'] for r in rows)),
                per_suite={s: dict(episodes=sum(r['suite'] == s for r in rows), successes=sum(r['suite'] == s and r['success'] for r in rows)) for s in SUITES},
                per_task=[dict(suite=s, task_id=t, episodes=sum(r['suite'] == s and r['task_id'] == t for r in rows), successes=sum(r['suite'] == s and r['task_id'] == t and r['success'] for r in rows)) for s in SUITES for t in range(10)])


def audited_rows(folder, detail_key, source_group):
    audit_path = folder / 'evaluation_independent_final_audit.json'
    audit = read(audit_path)
    assert audit['complete']
    rows = []
    for e in audit[detail_key]:
        episode = folder / 'episodes' / e['suite'] / f"task_{e['task_id']:02d}" / f"seed_{e['seed']}"
        for name, expected in e['artifact_sha256'].items():
            assert sha(episode / name) == expected, str(episode / name)
        result_path = episode / 'result.json'
        result = read(result_path)
        assert result['complete'] and result['success'] == e['success']
        assert result['server_metadata']['checkpoint'].endswith('/Pi05_libero_reconstructed_NoState_LoRA_6gpu_bs72_10k_20261101/10000')
        rows.append(dict(suite=e['suite'], task_id=e['task_id'], seed=e['seed'], success=e['success'],
                         termination=result['termination'], steps=result['steps'], source_group=source_group,
                         source_result=str(result_path), source_audit=str(audit_path),
                         solver_failure_policy=result.get('solver_failure_policy', 'stop_on_first_rejected_solve'),
                         initial_state_sha256=result['initial_state_sha256']))
    return rows, dict(path=str(audit_path), sha256=sha(audit_path), episodes=len(rows), all_artifact_hashes_match=True)


def main():
    request = read(RUN / 'evaluation_request.json')
    original, oa = audited_rows(Path(request['existing_baseline_run']), 'episodes_detail', 'original_seeds0to5')
    retry, ra = audited_rows(Path(request['existing_failure_retry_run']), 'results', 'failed_nominal_retry_hold20')
    extension, ea = audited_rows(RUN, 'results', 'new_seeds6to9_hold20')
    historical, latest = merge_views(original, retry, extension)
    for name, rows in [('combined_original240_plus_new160', historical), ('combined_latest_400', latest)]:
        rows.sort(key=lambda r: (SUITES.index(r['suite']), r['task_id'], r['seed']))
        summary = stats(rows)
        assert all(t['episodes'] == 10 for t in summary['per_task'])
        (RUN / (name + '.json')).write_text(json.dumps(dict(complete=True, **summary, results=rows), ensure_ascii=False, indent=2) + '\n')
        with (RUN / (name + '.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    e, h, l = stats(extension), stats(historical), stats(latest)
    report = dict(complete=True, checked_at=datetime.datetime.now().astimezone().isoformat(), unique_latest_episodes=400,
                  tasks=40, episodes_per_task=10, seeds=list(range(10)), newly_executed_episodes=160,
                  historical_retries_not_counted_as_extra_episodes=17, all_417_artifact_sets_unchanged=True,
                  audit_chain=[oa, ra, ea], new160=e, original240_plus_new160=h, latest_with_17_retry_replacements=l,
                  all_400_rerun_in_current_invocation=False,
                  scope='160 new episodes. Latest400 uses 223 original successful episodes, all17 newer failed-episode retests and160 extensions. Original240+new160 is reported separately.')
    (RUN / 'combined_400_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines=['# 无容差评测补齐至每任务10次', '', f"新增seed6–9共160回合已完成：{e['successes']}/160成功（{e['success_rate']*100:.2f}%）。原seed0–5保留，40任务各10个唯一task/seed组合，共400。", '',
           f"采用17条失败回合最新重测结果的400集合：**{l['successes']}/400（{l['success_rate']*100:.2f}%）**。保留原240结果、仅追加本次160的统计：{h['successes']}/400（{h['success_rate']*100:.2f}%）。两种统计均提供来源清单，不将17条重测计为额外17条回合。", '',
           '本次仅执行160条补测，未重新执行全部400。最新集合包含223条原成功、17条已完成的失败重测及160条新增。', '',
           '## 套件统计', '', '| 套件 | 本次新增成功/40 | 最新来源成功/100 | 原240直接追加来源成功/100 |', '|---|---:|---:|---:|']
    for s in SUITES:lines.append(f"| {s} | {e['per_suite'][s]['successes']}/40 | {l['per_suite'][s]['successes']}/100 | {h['per_suite'][s]['successes']}/100 |")
    lines += ['', '## 固定协议', '', '最终10000checkpoint、nostate、无容差mask000/boundsNone、默认IPOPT/shared关节伺服、20Hz/500Hz、每5步推理/预测10步、原任务步数上限、夹爪deadband0.5。失败动作不更新名义参考，保持上一接受关节参考和夹爪命令继续25物理步；连续20失败帧终止。新seed和官方初态索引均为6–9；6worker各绑定一张服务器GPU，真实seed独立于replica索引，确定性noise按seed/case/chunk生成。', '',
              '## 审计与来源', '', '新增160回合全部客户正常退出，基础设施错误0；逐回合官方初态、动作/状态/物理步、失败参考保持和计数、无容差配置、最终状态原生成功谓词、视频全帧解码及六副本真实模型身份独立通过。旧240、17重测和新160共417条留存记录的所有结果/轨迹/视频SHA与各自独立审计一致。按task/seed去重后400组合完整，每任务10次。', '',
              '逐回合CSV：combined_latest_400.csv；原始240直接追加版本：combined_original240_plus_new160.csv。汇总审计：combined_400_audit.json。服务器退出与隧道清理：inference_shutdown_verification.json、local_cleanup_verification.json。', '',
              '## 最新集合逐任务结果', '', '| 套件 | task_id（0起始） | 成功/10 |', '|---|---:|---:|']
    for x in l['per_task']:lines.append(f"| {x['suite']} | {x['task_id']:02d} | {x['successes']}/10 |")
    (RUN / 'NOMINAL_10SEED_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['audit_chain','new160','original240_plus_new160','latest_with_17_retry_replacements']},ensure_ascii=False))
    print('new160',e['successes'],'latest400',l['successes'],'original_plus_new',h['successes'])


if __name__ == '__main__':
    main()
