"""Compare time throughput and early training-loss sample efficiency."""
from pathlib import Path
import datetime as dt
import json
import re
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent


def parse_log(path, batch, date):
    text = path.read_text(errors='replace')
    metrics = []
    progress = []
    last_timestamp = None
    midnight = dt.datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=dt.timezone(dt.timedelta(hours=8)))
    for line in text.splitlines():
        time_match = re.match(r'(\d\d):(\d\d):(\d\d)\.(\d+)', line)
        if time_match:
            hours, minutes, seconds, fraction = time_match.groups()
            stamp = midnight + dt.timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(fraction))
            if last_timestamp and stamp.timestamp() < last_timestamp - 3600:
                midnight += dt.timedelta(days=1)
                stamp += dt.timedelta(days=1)
            last_timestamp = stamp.timestamp()
        metric = re.search(r'Step\s+(\d+): ([^\n]+)', line)
        if metric:
            values = {key: float(value) for key, value in re.findall(r'([\w/]+)=([-+0-9.eE]+)', metric.group(2))}
            step = int(metric.group(1))
            metrics.append({'step': step, 'samples': step * batch, 'metrics': values})
        match = re.search(r'Progress on:\s+([0-9.]+)(k?)it/', line)
        if match and last_timestamp:
            step = float(match.group(1)) * (1000 if match.group(2) else 1)
            progress.append({'step': step, 'unix': last_timestamp})
    return {'batch': batch, 'metrics': metrics, 'progress': progress}


def throughput(run, max_step=None):
    # Exclude first 20 updates and all setup/compilation; use raw progress timestamps.
    points = [row for row in run['progress'] if row['step'] >= 20 and (max_step is None or row['step'] <= max_step)]
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    steps = last['step'] - first['step']
    elapsed = last['unix'] - first['unix']
    x = np.asarray([p['step'] for p in points])
    y = np.asarray([p['unix'] for p in points])
    slope = float(np.polyfit(x - x[0], y - y[0], 1)[0])
    return {'first_step': first['step'], 'last_step': last['step'], 'points': len(points),
            'elapsed_seconds': elapsed, 'seconds_per_step': elapsed / steps,
            'regression_seconds_per_step': slope,
            'samples_per_second': steps * run['batch'] / elapsed,
            'samples_per_second_per_gpu': steps * run['batch'] / elapsed / (7 if run['batch'] == 84 else 6)}


def weighted_window(run, lower, upper):
    # Each printed scalar is a mean of its preceding 10 training steps.
    weighted = 0.
    width = 0.
    for row in run['metrics']:
        end = row['samples']
        start = end - 10 * run['batch']
        overlap = max(0, min(end, upper) - max(start, lower))
        if overlap:
            weighted += overlap * row['metrics']['loss/action']
            width += overlap
    return weighted / width if width else None


def main():
    base = parse_log(HERE / 'baseline_bs84_train.log', 84, '2026-10-28')
    small = parse_log(HERE / 'short_bs24_train.log', 24, '2026-10-29')
    assert small['metrics'], 'No short-run loss records'
    end = small['metrics'][-1]['samples']
    first_progress = base['progress'][0]['unix']
    base_15m = [p for p in base['progress'] if p['unix'] <= first_progress + 900]
    base_15m_end = base_15m[-1]['step']
    rates = {'baseline_first_15_minutes': throughput(base, base_15m_end),
             'baseline_same_early_steps': throughput(base, small['metrics'][-1]['step']),
             'short_6gpu_bs24': throughput(small)}
    ratio = rates['short_6gpu_bs24']['samples_per_second'] / rates['baseline_first_15_minutes']['samples_per_second']
    comparisons = []
    # Exact overlapping sample intervals smooth noise from different batch sizes.
    for lower, upper in [(0, 1680), (1680, 3360), (3360, 5040), (5040, 6720), (6720, 8400),
                         (8400, 10080), (10080, 11760), (11760, 13440), (13440, 15120),
                         (15120, 16800), (16800, 18480), (18480, 20160), (20160, 21840),
                         (21840, 23520)]:
        if upper > end:
            continue
        a, b = weighted_window(base, lower, upper), weighted_window(small, lower, upper)
        comparisons.append({'sample_interval': [lower, upper], 'bs84_loss': a, 'bs24_loss': b,
                            'bs24_over_bs84_loss': b / a, 'bs84_step_end': upper / 84, 'bs24_step_end': upper / 24})
    same_steps = []
    base_by_step = {m['step']: m for m in base['metrics']}
    for step in [50, 100, 150, 200, 300, 350, 400, 500, 600]:
        row = next((m for m in small['metrics'] if m['step'] == step), None)
        if row and step in base_by_step:
            same_steps.append({'step': step, 'bs84_loss': base_by_step[step]['metrics']['loss/action'],
                               'bs24_loss': row['metrics']['loss/action'], 'bs84_samples': step * 84, 'bs24_samples': step * 24})
    status = json.loads((HERE / 'server_artifacts/status.json').read_text())
    elapsed = status['stop_unix'] - small['progress'][0]['unix']
    time_comparisons = []
    for seconds in [300, 600]:
        row = {'postcompile_seconds': seconds}
        for key, run in [('bs84', base), ('bs24', small)]:
            origin = run['progress'][0]['unix']
            positions = np.array([p['step'] for p in run['progress']])
            timestamps = np.array([p['unix'] - origin for p in run['progress']])
            rows = []
            for metric in run['metrics']:
                time = float(np.interp(metric['step'], positions, timestamps))
                if time <= seconds and metric['step'] <= positions[-1]:
                    rows.append(metric)
            last = rows[-1]
            row[key] = {'step': last['step'], 'samples': last['samples'],
                        'last_30_updates_loss_mean': statistics.mean(m['metrics']['loss/action'] for m in rows[-3:])}
        time_comparisons.append(row)
    result = {'throughput': rates, 'throughput_bs24_over_bs84': ratio,
              'same_sample_windows': comparisons, 'same_update_steps': same_steps,
              'same_postcompile_time_approximate': time_comparisons,
              'short_actual_postcompile_seconds': elapsed,
              'short_compile_seconds': small['progress'][0]['unix'] - status['first_progress_unix'],
              'supervisor_timer_note': 'Timer began on the empty initial progress line and included first-step compilation; measured throughput excludes it.',
              'baseline_first_15m_completed_steps': base_15m_end,
              'short_last_metric': small['metrics'][-1],
              'limitations': ['Separate wall-clock runs; CPU load and data cache state differ.',
                              'Training losses, not a common held-out validation set or closed-loop success.',
                              'Same per-update LR schedule; at equal samples bs24 performs 3.5 times as many updates and advances warmup faster.',
                              'Batch-dependent random augmentation/noise and BF16 collective reductions differ.']}
    (HERE / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (HERE / 'parsed_curves.json').write_text(json.dumps({'bs84': base, 'bs24': small}, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
