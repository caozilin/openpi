"""Read committed telemetry only; no inference, simulation, or run mutation."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import csv
import hashlib
import json
import numpy as np

ROOT = Path('/media/czl/sata/franka_my_code/openpi')
RUN = ROOT / 'data/libero_evaluation/pi05_joint13_successmix20_10000_predtol_hold20_40tasks_10seeds_20261110'
OUT = Path(__file__).parent / 'prediction_stability'
AXES = ['rx-', 'rx+', 'ry-', 'ry+', 'rz-', 'rz+']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return {'n': 0}
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)),
                maximum=float(a.max()), fraction_le_1deg=float(np.mean(a <= 1)),
                fraction_le_2deg=float(np.mean(a <= 2)), fraction_gt_5deg=float(np.mean(a > 5)))


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    paths = sorted(RUN.glob('episodes/*/*/*/result.json'))  # immutable snapshot
    sources = {}
    counts = Counter()
    blocks = []
    pairs = []
    groups = []
    raw_values = []
    seen = set()
    for result_path in paths:
        data = result_path.read_bytes()
        result = json.loads(data)
        identity = (result['suite'], result['task_id'], result['seed'])
        assert result['complete'] and identity not in seen
        seen.add(identity)
        sources[str(result_path)] = sha(data)
        trajectory = result_path.with_name('trajectory.json')
        data = trajectory.read_bytes()
        sources[str(trajectory)] = sha(data)
        rows = json.loads(data)
        assert len(rows) == result['steps']
        counts['episodes'] += 1
        counts['successes'] += int(result['success'])
        counts['frames'] += len(rows)
        by_chunk = defaultdict(list)
        segment = -1
        last_key = None
        for row in rows:
            key = (row['subtask_index'], row['phase'])
            if key != last_key:
                segment += 1
            last_key = key
            by_chunk[row['inference_chunk']].append((segment, row))
            raw_values.append(row['predicted_bounds_raw_rad'])
        episode_blocks = []
        for chunk, entries in sorted(by_chunk.items()):
            if len(entries) != 5:
                counts['partial_chunks_excluded'] += 1
                continue
            assert [r['inference_chunk_offset'] for _, r in entries] == list(range(5))
            if len({s for s, _ in entries}) != 1:
                counts['phase_crossing_chunks_excluded'] += 1
                continue
            s, first = entries[0]
            counts['full_same_segment_chunks'] += 1
            for kind, field in [('raw', 'predicted_bounds_raw_rad'), ('clipped', 'bounds_rad')]:
                array = np.rad2deg(np.asarray([r[field] for _, r in entries], dtype=float))
                assert array.shape == (5, 6) and np.isfinite(array).all()
                if kind == 'clipped':
                    expected = np.clip(np.asarray([r['predicted_bounds_raw_rad'] for _, r in entries]), 0, np.pi / 6)
                    assert np.allclose(np.deg2rad(array), expected, atol=1e-14, rtol=0)
                block = dict(suite=identity[0], task_id=identity[1], seed=identity[2],
                             segment=s, subtask=first['subtask_index'], phase=first['phase'],
                             chunk=chunk, kind=kind)
                for i, axis in enumerate(AXES):
                    block[f'{axis}_mean_deg'] = float(array[:, i].mean())
                    block[f'{axis}_std_deg'] = float(array[:, i].std())
                    block[f'{axis}_range_deg'] = float(np.ptp(array[:, i]))
                blocks.append(block)
                episode_blocks.append(block)
        grouped = defaultdict(list)
        for block in episode_blocks:
            grouped[(block['segment'], block['kind'])].append(block)
        for (_, kind), bs in grouped.items():
            means = np.array([[b[f'{a}_mean_deg'] for a in AXES] for b in bs])
            if len(bs) >= 2:
                group = {k: bs[0][k] for k in ['suite', 'task_id', 'seed', 'segment', 'subtask', 'phase', 'kind']}
                group['blocks'] = len(bs)
                for i, a in enumerate(AXES):
                    group[f'{a}_mean_std_deg'] = float(means[:, i].std())
                    group[f'{a}_mean_range_deg'] = float(np.ptp(means[:, i]))
                groups.append(group)
            for i in range(1, len(bs)):
                # Exclude skipped transition/partial chunks from adjacent pairs.
                if bs[i]['chunk'] != bs[i-1]['chunk'] + 1:
                    continue
                pair = {k: bs[i][k] for k in ['suite', 'task_id', 'seed', 'segment', 'subtask', 'phase', 'kind']}
                pair.update(previous_chunk=bs[i-1]['chunk'], chunk=bs[i]['chunk'])
                for j, a in enumerate(AXES):
                    pair[f'{a}_mean_delta_deg'] = float(abs(means[i, j] - means[i-1, j]))
                pairs.append(pair)
    report = dict(observed_at=datetime.now().astimezone().isoformat(), run=str(RUN),
                  scope='Committed snapshot including successes and failures; executed first5 of predicted10 only; pre-EMA bounds',
                  counts=dict(counts), axes=AXES, results={})
    for kind in ['raw', 'clipped']:
        report['results'][kind] = {}
        for phase in ['APP_EXEC', 'APP', 'ENG', 'EXEC', 'DIS', 'ALL']:
            def selected(row):
                return row['kind'] == kind and (phase == 'ALL' or row['phase'] == phase or (phase == 'APP_EXEC' and row['phase'] in ('APP', 'EXEC')))
            bb = [b for b in blocks if selected(b)]
            pp = [p for p in pairs if selected(p)]
            gg = [g for g in groups if selected(g)]
            result = dict(blocks=len(bb), adjacent_pairs=len(pp), continuous_segments=len(gg))
            for metric, source, suffix in [('within_std', bb, 'std_deg'), ('within_range', bb, 'range_deg'),
                                           ('adjacent_block_mean_delta', pp, 'mean_delta_deg'),
                                           ('segment_block_mean_std', gg, 'mean_std_deg'),
                                           ('segment_block_mean_range', gg, 'mean_range_deg')]:
                result[metric] = stats([r[f'{a}_{suffix}'] for r in source for a in AXES])
                result[metric + '_per_axis'] = {a: stats([r[f'{a}_{suffix}'] for r in source]) for a in AXES}
            result['fraction_blocks_all6_range_le_1deg'] = float(np.mean([max(b[f'{a}_range_deg'] for a in AXES) <= 1 for b in bb])) if bb else None
            result['fraction_blocks_all6_range_le_2deg'] = float(np.mean([max(b[f'{a}_range_deg'] for a in AXES) <= 2 for b in bb])) if bb else None
            report['results'][kind][phase] = result
    v = np.rad2deg(np.asarray(raw_values))
    report['raw_outside_0_30_fraction'] = float(np.mean((v < 0) | (v > 30)))
    report['snapshot_input_sha256'] = sha(json.dumps(sources, sort_keys=True).encode())
    (OUT / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    write_csv('blocks.csv', blocks)
    write_csv('adjacent_block_pairs.csv', pairs)
    write_csv('continuous_segments.csv', groups)
    # Verify source immutability after analysis; never touch the live run.
    assert all(sha(Path(p).read_bytes()) == h for p, h in sources.items())
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, ensure_ascii=False, indent=2))
    for kind in ['raw', 'clipped']:
        for phase in ['APP_EXEC', 'APP', 'EXEC', 'ENG', 'DIS']:
            z = report['results'][kind][phase]
            print(kind, phase, json.dumps({k: v for k, v in z.items() if '_per_axis' not in k}, ensure_ascii=False))


if __name__ == '__main__':
    main()
