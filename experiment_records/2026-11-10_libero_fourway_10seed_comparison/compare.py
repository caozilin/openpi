"""Read existing official evaluation result JSON; no inference or simulation."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib
import json

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
OUT = Path(__file__).parent
BASE = ROOT / 'data/libero_evaluation'
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
EXPECTED = {(s, t, z) for s in SUITES for t in range(10) for z in range(10)}
sources = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def tally(rows):
    rows = list(rows)
    return dict(completed=len(rows), successes=sum(bool(x['success']) for x in rows),
                terminations=dict(Counter(x['termination'] for x in rows)))


def main():
    pairs = read(ROOT / 'experiment_records/2026-11-04_libero_preset30_rollout400/per_pair.json')
    models = {}
    for label, field in [('old_nominal', 'nominal_source'), ('old_preset30', 'tolerance_source')]:
        models[label] = {}
        for pair in pairs:
            p = pair[field]
            x = read(p)
            key = (x['suite'], x['task_id'], x['seed'])
            assert x['complete'] and key not in models[label]
            models[label][key] = dict(x, source_result=p)
    run_dirs = {
        'joint13_nominal': BASE / 'pi05_joint13_successmix20_10000_nominal_hold20_40tasks_10seeds_20261110',
        'joint13_predicted': BASE / 'pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110',
    }
    audits = {}
    for label, run in run_dirs.items():
        models[label] = {}
        for p in sorted(run.glob('episodes/*/*/*/result.json')):
            x = read(p)
            key = (x['suite'], x['task_id'], x['seed'])
            assert x['complete'] and key not in models[label]
            models[label][key] = dict(x, source_result=str(p))
        p = run / 'evaluation_independent_final_audit.json'
        if p.exists():
            audit = read(p)
            audits[label] = dict(complete=audit['complete'], episodes=audit['episodes'], checked_at=audit['checked_at'])
        else:
            audits[label] = read(run / 'independent_audit_state.json')
    for label, rows in models.items():
        assert set(rows) == EXPECTED
    for key in EXPECTED:
        reference = models['old_nominal'][key]
        assert all(rows[key]['initial_state_sha256'] == reference['initial_state_sha256'] for rows in models.values())
        assert len({rows[key]['server_metadata']['params_values_sha256'] for name, rows in models.items() if name.startswith('joint13')}) == 1
    summaries = {}
    for label, rows in models.items():
        summaries[label] = dict(
            overall=tally(rows.values()),
            per_suite={s: tally(x for k, x in rows.items() if k[0] == s) for s in SUITES},
            per_seed={str(z): tally(x for k, x in rows.items() if k[2] == z) for z in range(10)},
            per_task=[dict(suite=s, task_id=t, **tally(x for k, x in rows.items() if k[:2] == (s, t))) for s in SUITES for t in range(10)],
        )
    contrasts = {}
    for a, b in [('old_nominal', 'old_preset30'), ('joint13_nominal', 'joint13_predicted'),
                 ('old_nominal', 'joint13_nominal'), ('old_preset30', 'joint13_predicted')]:
        c = Counter()
        for key in EXPECTED:
            x, y = models[a][key]['success'], models[b][key]['success']
            c['both_success' if x and y else 'both_failed' if not x and not y else 'improved' if y else 'regressed'] += 1
        contrasts[a + '_to_' + b] = dict(c)
    rows = []
    for s, t, z in sorted(EXPECTED):
        key = (s, t, z)
        rows.append(dict(suite=s, task_id=t, seed=z,
                         **{label: dict(success=records[key]['success'], termination=records[key]['termination'],
                                       source_result=records[key]['source_result']) for label, records in models.items()}))
    report = dict(observed_at=datetime.now().astimezone().isoformat(), episodes_per_group=400,
                  scope='Four existing official400 groups, same40 tasks/seed0..9; no failure-only retests',
                  new_group_audit_status=audits, summaries=summaries, paired_contrasts=contrasts,
                  verification=dict(unique400_each=True, all400_initial_state_hashes_same_across4=True,
                                    same_joint13_checkpoint_digest=True), paired_results=rows)
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items())
    print(json.dumps({k: v for k, v in report.items() if k not in ['paired_results', 'summaries']}, indent=2))
    for label, s in summaries.items():
        print(label, 'TOTAL', s['overall'], 'SEEDS', [s['per_seed'][str(z)]['successes'] for z in range(10)])


if __name__ == '__main__':
    main()
