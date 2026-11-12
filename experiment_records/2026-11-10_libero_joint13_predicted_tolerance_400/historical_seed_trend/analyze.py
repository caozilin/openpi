"""Compare historical official400 seed trends; immutable result JSON only."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib
import json

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
REC = ROOT / 'experiment_records/2026-11-10_libero_joint13_predicted_tolerance_400'
OUT = Path(__file__).parent
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
sources = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def tally(rows):
    rows = list(rows)
    return dict(success=sum(bool(r['success']) for r in rows), completed=len(rows))


def main():
    pairs = read(ROOT / 'experiment_records/2026-11-04_libero_preset30_rollout400/per_pair.json')
    models = {}
    expected = {(s, t, z) for s in SUITES for t in range(10) for z in range(10)}
    for label, field in [('old_nominal', 'nominal_source'), ('old_preset30', 'tolerance_source')]:
        rows = {}
        for pair in pairs:
            row = read(pair[field])
            key = (row['suite'], row['task_id'], row['seed'])
            assert row['complete'] and key not in rows
            rows[key] = row
        assert set(rows) == expected
        models[label] = rows
    assert sum(r['success'] for r in models['old_nominal'].values()) == 380
    assert sum(r['success'] for r in models['old_preset30'].values()) == 370
    diagnostic = read(REC / 'success_rate_rebound_diagnosis.json')
    joint = {}
    for path, digest in diagnostic['result_sha256'].items():
        row = read(path)
        assert sources[path] == digest
        joint[(row['suite'], row['task_id'], row['seed'])] = row
    assert len(joint) == 307
    models['joint13_snapshot307'] = joint
    early_time = read(REC / 'monitor_snapshot_20261110_1726.json')['observed_at']
    early = {k for k, r in joint.items() if r['finished_at'] <= early_time}
    late = set(joint) - early
    assert len(early) == 178 and len(late) == 129
    results = {}
    for label, rows in models.items():
        results[label] = dict(
            overall=tally(rows.values()),
            per_seed={str(z): tally(r for k, r in rows.items() if k[2] == z) for z in range(10)},
            halves={str(lo): tally(r for k, r in rows.items() if lo <= k[2] < lo + 5) for lo in [0, 5]},
            per_suite_halves={s: {str(lo): tally(r for k, r in rows.items() if k[0] == s and lo <= k[2] < lo + 5) for lo in [0, 5]} for s in SUITES},
            aligned_cohorts={name: dict(overall=tally(rows[k] for k in keys), per_suite={s: tally(rows[k] for k in keys if k[0] == s) for s in SUITES}) for name, keys in [('earlier178', early), ('later129', late)]},
        )
    for k in joint:
        for label in ['old_nominal', 'old_preset30']:
            assert joint[k]['initial_state_sha256'] == models[label][k]['initial_state_sha256']
    report = dict(at=datetime.now().astimezone().isoformat(), results=results,
                  checks=dict(historical_unique400_each=True, historical_totals_match_audited_reports=True,
                              joint_snapshot307_immutable=True, all307_official_initial_states_identical=True),
                  scope='Historical official400 (not failure-only retests); joint13 fixed307 snapshot incomplete. Matched cohorts follow current run completion order, not historical wallclock order. Runtime/model differences are not isolated.')
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items())
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
