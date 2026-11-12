"""Report the complete confirmed-mask400 run against both saved nominal views."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.libero.eval_confirmed_mask_libero_local import DEFAULT as RUN, SUITES, read, dump


def stats(rows):
    return dict(episodes=len(rows), successes=sum(bool(r['success']) for r in rows),
                success_rate=sum(bool(r['success']) for r in rows) / len(rows),
                terminations=dict(collections.Counter(r['termination'] for r in rows)),
                per_suite={s: dict(episodes=sum(r['suite'] == s for r in rows), successes=sum(r['suite'] == s and r['success'] for r in rows)) for s in SUITES},
                per_task=[dict(suite=s, task_id=t, episodes=sum(r['suite'] == s and r['task_id'] == t for r in rows), successes=sum(r['suite'] == s and r['task_id'] == t and r['success'] for r in rows)) for s in SUITES for t in range(10)])


def main():
    audit = read(RUN / 'evaluation_independent_final_audit.json')
    completion = read(RUN / 'evaluation_completion.json')
    assert audit['complete'] and audit['full_evaluation_audited'] and audit['episodes'] == 400
    assert completion['complete'] and completion['exit_codes'] == [0] * 6
    records = []
    for row in audit['results']:
        folder = RUN / 'episodes' / row['suite'] / f"task_{row['task_id']:02d}" / f"seed_{row['seed']}"
        for name, expected in row['artifact_sha256'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
        value = read(folder / 'result.json')
        assert value['success'] == row['success']
        records.append(dict(suite=value['suite'], task_id=value['task_id'], seed=value['seed'],
                            success=value['success'], termination=value['termination'], steps=value['steps'],
                            solver_failed_frames=value['total_solver_failures'], source_result=str(folder / 'result.json')))
    actual = {(r['suite'], r['task_id'], r['seed']): r for r in records}
    assert len(actual) == len(records) == 400
    assert set(actual) == {(s, t, z) for s in SUITES for t in range(10) for z in range(10)}
    paired, comparisons = [], {}
    for name, file in [('latest_nominal', 'paired_baseline_latest400.json'), ('original_nominal', 'paired_baseline_original400.json')]:
        baseline = read(RUN / file)
        assert baseline['complete']
        counts = collections.Counter()
        for nominal in baseline['results']:
            key = nominal['suite'], nominal['task_id'], nominal['seed']
            tolerance = actual[key]
            category = ('both_success' if nominal['success'] and tolerance['success'] else
                        'improved' if tolerance['success'] else
                        'regressed' if nominal['success'] else 'both_failed')
            counts[category] += 1
            paired.append(dict(baseline_view=name, suite=key[0], task_id=key[1], seed=key[2],
                               nominal_success=nominal['success'], tolerance_success=tolerance['success'],
                               paired_outcome=category, nominal_termination=nominal['termination'],
                               tolerance_termination=tolerance['termination'], nominal_source=nominal['source_result'],
                               tolerance_source=tolerance['source_result']))
        comparisons[name] = dict(nominal=stats(baseline['results']), paired_counts=dict(counts))
    records.sort(key=lambda r: (SUITES.index(r['suite']), r['task_id'], r['seed']))
    summary = stats(records)
    report = dict(complete=True, checked_at=datetime.datetime.now().astimezone().isoformat(), **summary,
                  results=records, comparisons=comparisons, infrastructure_errors=0,
                  independent_audit=str(RUN / 'evaluation_independent_final_audit.json'),
                  confirmed_masks_sha256=read(RUN / 'evaluation_request.json')['confirmed_masks_sha256'],
                  new_rollouts_in_this_run=400, unique_task_seed_pairs=400,
                  causal_mask_routing=read(RUN / 'evaluation_request.json')['semantic_routing'])
    dump(RUN / 'confirmed_mask400_report.json', report)
    for filename, rows in [('confirmed_mask400_results.csv', records), ('nominal_tolerance_paired.csv', paired)]:
        with (RUN / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# 对应任务mask ±30°：40任务×同seed0–9', '',
             f"本次完整新执行400回合，成功 **{summary['successes']}/400（{summary['success_rate']*100:.2f}%）**。六个本地客户端均exit0，基础设施错误0。服务器六卡仅推理最终10000步模型。", '',
             '|任务集|对应mask容差成功/100|名义最新来源成功/100|名义原来源成功/100|', '|---|---:|---:|---:|']
    for s in SUITES:
        lines.append(f"|{s}|{summary['per_suite'][s]['successes']}/100|{comparisons['latest_nominal']['nominal']['per_suite'][s]['successes']}/100|{comparisons['original_nominal']['nominal']['per_suite'][s]['successes']}/100|")
    lines += ['', '采用用户2026-10-31确认的每任务/子任务APP、ENG、EXEC、DIS表，启用轴正负各30°；ENG/DIS严格、heading退化保护保留。不是全任务111。默认IPOPT、共享关节伺服、20Hz/500Hz、replan5/predict10、nostate、步数上限及夹爪deadband0.5保持；失败不更新名义参考，保留上一接受关节/夹爪参考，连续20失败帧1秒终止。', '',
              '阶段来自实时手指反馈的既有三稳定帧分类器。子任务身份为因果仿真语义先验：Long03/09使用当前原生放置谓词转到机构子任务，Long07接近阶段按两个指定对象的当前距离选身份，抓取/执行/释放时锁存；Goal03开抽屉至真正抓碗属于APP，ENG/DIS仍严格。不是学习出的VLA语义识别，也未使用配对轨迹未来帧。', '',
              '400回合官方初态和首个动作前物理状态与名义配对记录逐值相同。首次推理动作差异如实保存在独立审计；后续动作由各组闭环独立产生。不能声称是固定动作重放。', '',
              '名义最新来源385/400包含17条原失败的定向重测替换，具有选择重测影响；另提供不替换原240、仅追加seed6–9的372/400来源。两种400基线逐条均保留来源。该比较不应冒充相同失败策略下两组各全新400回合的严格随机对照。', '',
              '## 同seed配对', '']
    for name, value in comparisons.items():
        lines.append(f"- {name}：{json.dumps(value['paired_counts'], ensure_ascii=False)}")
    lines += ['', '## 逐任务', '', '|任务集|task_id（0起始）|成功/10|', '|---|---:|---:|']
    for task in summary['per_task']:
        lines.append(f"|{task['suite']}|{task['task_id']:02d}|{task['successes']}/10|")
    lines += ['', '全部400回合已独立检查每帧动作/state、25物理步、失败参考回滚和计数、实时阶段及确认mask、容差优化边界、官方初态、原生成功谓词和视频全帧解码。源文件SHA与六副本真实身份已核对。', '',
              '文件：confirmed_mask400_report.json、confirmed_mask400_results.csv、nominal_tolerance_paired.csv、evaluation_independent_final_audit.json。']
    (RUN / 'CONFIRMED_MASK_10SEED_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
