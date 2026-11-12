"""Compare bs24 training runs with different data-parallel device counts."""
from pathlib import Path
import importlib.util
import json
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'pi05_bs24_vs_bs84_20261029'
spec = importlib.util.spec_from_file_location('earlier_analysis', PREVIOUS / 'analyze.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def summarize_rate(run, num_devices, start=None, end=None):
    points = [p for p in run['progress'] if p['step'] >= 20
              and (start is None or p['unix'] >= start)
              and (end is None or p['unix'] <= end)]
    assert len(points) >= 2
    first, last = points[0], points[-1]
    seconds = last['unix'] - first['unix']
    updates = last['step'] - first['step']
    x = np.array([p['step'] for p in points])
    y = np.array([p['unix'] for p in points])
    slope = float(np.polyfit(x - x[0], y - y[0], 1)[0])
    intervals = [(q['unix'] - p['unix']) / (q['step'] - p['step'])
                 for p, q in zip(points, points[1:]) if q['step'] > p['step']]
    return {'first_step': first['step'], 'last_step': last['step'], 'points': len(points),
            'seconds': seconds, 'seconds_per_step': seconds / updates,
            'regression_seconds_per_step': slope,
            'interval_seconds_per_step_p25_p50_p75': np.percentile(intervals, [25, 50, 75]).tolist(),
            'samples_per_second': updates * 24 / seconds,
            'samples_per_second_per_gpu': updates * 24 / seconds / num_devices}


def parse_resource(path, ids, start, end):
    util, power, load = [], [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not start <= row['unix'] <= end or 'gpus' not in row:
            continue
        load.append(row['load_average'][0])
        readings = {int(s.split(',')[0]): [float(v) for v in s.split(',')[1:]]
                    for s in row['gpus'].strip().splitlines()}
        util.extend(readings[i][1] for i in ids)
        power.extend(readings[i][2] for i in ids)
    return {'median_gpu_utilization_percent': statistics.median(util) if util else None,
            'median_gpu_power_watts': statistics.median(power) if power else None,
            'load_average_1m_min_max': [min(load), max(load)] if load else None}


def main():
    runs = {6: parser.parse_log(PREVIOUS / 'short_bs24_train.log', 24, '2026-10-29')}
    statuses = {}
    for n in (3, 2):
        folder = ROOT / f'server_{n}gpu'
        runs[n] = parser.parse_log(folder / 'train.log', 24, '2026-10-29')
        statuses[n] = json.loads((folder / 'status.json').read_text())
        assert runs[n]['metrics'], f'No training metrics for {n} GPUs'
    overlap_start = max(runs[n]['progress'][0]['unix'] for n in (3, 2)) + 60
    overlap_end = min(runs[n]['progress'][-1]['unix'] for n in (3, 2))
    throughputs = {n: summarize_rate(run, n) for n, run in runs.items()}
    overlap = {n: summarize_rate(runs[n], n, overlap_start, overlap_end) for n in (3, 2)}
    common = sorted(set.intersection(*[set(m['step'] for m in run['metrics']) for run in runs.values()]))
    index = {n: {p['step']: p for p in run['metrics']} for n, run in runs.items()}
    selected = []
    for step in [10, 50, 100, 150, 200, 250, 300, 350]:
        if step in common:
            selected.append({'step': step, 'samples': step * 24,
                             **{f'{n}gpu_loss': index[n][step]['metrics']['loss/action'] for n in (6, 3, 2)}})
    differences = {}
    for a, b in ((3, 2), (6, 3), (6, 2)):
        diffs = [abs(index[a][s]['metrics']['loss/action'] - index[b][s]['metrics']['loss/action']) for s in common]
        differences[f'{a}gpu_vs_{b}gpu'] = {'mean_absolute_loss_difference': statistics.mean(diffs),
                                           'max_absolute_loss_difference': max(diffs), 'points': len(diffs)}
    windows = []
    for lower, upper in [(0, 1680), (1680, 3360), (3360, 5040), (5040, 6720), (6720, 8400)]:
        if upper <= common[-1] * 24:
            windows.append({'sample_interval': [lower, upper],
                            **{f'{n}gpu_loss': parser.weighted_window(runs[n], lower, upper) for n in (6, 3, 2)}})
    runtimes = {}
    for n in (3, 2):
        s = statuses[n]
        runtimes[n] = {'final_step': s['last_step'], 'samples': s['last_step'] * 24,
                       'status': s['state'], 'returncode': s.get('returncode'),
                       'actual_postcompile_seconds': s.get('stop_unix', runs[n]['progress'][-1]['unix']) - runs[n]['progress'][0]['unix'],
                       'process_wall_seconds': s.get('wall_seconds')}
    result = {'global_batch': 24, 'throughput_all_steady': throughputs,
              'throughput_simultaneous_overlap': overlap,
              'simultaneous_overlap_unix': [overlap_start, overlap_end],
              'selected_same_step_losses': selected, 'same_sample_windows': windows,
              'pointwise_loss_differences': differences, 'runtimes': runtimes,
              'resources_overlap': {n: parse_resource(ROOT / f'server_{n}gpu/resources.jsonl',
                                                       [0, 1, 2] if n == 3 else [3, 4], overlap_start, overlap_end)
                                    for n in (3, 2)},
              'limits': ['3GPU and 2GPU ran concurrently; shared CPU, memory, storage, and PCIe effects are included.',
                         '6GPU measurement was earlier, not simultaneous.',
                         'GPU subsets/topology differ; this is not an isolated communication profile.',
                         'Loss means are printed to four decimals and measured on training batches, not a common held-out validation set.']}
    (ROOT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'curves.json').write_text(json.dumps(runs, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
