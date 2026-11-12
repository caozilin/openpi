"""Re-read the existing same-seed results; do not run any rollout."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
GROUPS = ['old_nominal', 'joint13_nominal', 'joint13_predicted']


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    original_sha = json.loads((OUT / 'sources.json').read_text())
    sources = {}
    tasks = {}
    paired = Counter()
    seen = set()
    for pair in summary['paired_results']:
        key = (pair['suite'], pair['task_id'], pair['seed'])
        assert key not in seen
        seen.add(key)
        actual = {}
        for group in GROUPS:
            path = Path(pair[group]['source_result'])
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == original_sha[str(path)]
            sources[str(path)] = digest
            row = json.loads(data)
            assert row['complete'] and (row['suite'], row['task_id'], row['seed']) == key
            assert row['success'] == pair[group]['success']
            assert row['termination'] == pair[group]['termination']
            actual[group] = row
        assert len({row['initial_state_sha256'] for row in actual.values()}) == 1
        old, new = actual['old_nominal']['success'], actual['joint13_predicted']['success']
        paired['both_success' if old and new else 'both_failed' if not old and not new else 'improved' if new else 'regressed'] += 1
        task = tasks.setdefault(key[:2], dict(suite=key[0], task_id=key[1],
            language=actual['joint13_predicted']['language'], episodes=[]))
        task['episodes'].append(dict(seed=key[2], **{g: dict(success=x['success'], termination=x['termination'],
            source_result=pair[g]['source_result']) for g, x in actual.items()}))
    assert len(seen) == 400 and len(tasks) == 40
    for task in tasks.values():
        episodes = sorted(task['episodes'], key=lambda x: x['seed'])
        assert [x['seed'] for x in episodes] == list(range(10))
        task['episodes'] = episodes
        task['successes'] = {g: sum(x[g]['success'] for x in episodes) for g in GROUPS}
        task['extra_failures'] = task['successes']['old_nominal'] - task['successes']['joint13_predicted']
        task['regressed_seeds'] = [x['seed'] for x in episodes if x['old_nominal']['success'] and not x['joint13_predicted']['success']]
        task['improved_seeds'] = [x['seed'] for x in episodes if not x['old_nominal']['success'] and x['joint13_predicted']['success']]
        task['new_failure_terminations'] = dict(Counter(x['joint13_predicted']['termination'] for x in episodes if not x['joint13_predicted']['success']))
    tasks = sorted(tasks.values(), key=lambda t: (-t['extra_failures'], t['suite'], t['task_id']))
    totals = {g: sum(t['successes'][g] for t in tasks) for g in GROUPS}
    assert totals == dict(old_nominal=380, joint13_nominal=368, joint13_predicted=366)
    report = dict(checked_at=datetime.now().astimezone().isoformat(), completed=True,
        scope='Same 40 tasks and official seeds0..9, 400 per group, all audited; original7d trained model vs joint13 predicted tolerance',
        totals=totals, paired=dict(paired), noticeably_worse_observed_threshold='at least2 additional failures out of10, descriptive not significance test',
        tasks=tasks, at_least_two_extra_failures=[dict(suite=t['suite'],task_id=t['task_id']) for t in tasks if t['extra_failures'] >= 2],
        source_results_sha256=sources,
        limitation='Cross-model differences also include training/runtime changes; same-joint13 nominal column helps separate observed changes, not causal attribution')
    (OUT / 'new_predtol_vs_old_nominal.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(totals=totals, paired=dict(paired)), ensure_ascii=False))
    for task in tasks:
        if task['extra_failures'] > 0:
            print(json.dumps({k: v for k, v in task.items() if k != 'episodes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
