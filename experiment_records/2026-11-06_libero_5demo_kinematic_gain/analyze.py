"""Read selected, audited candidates and their paired zero baselines only."""
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
MANIFEST = Path('/media/czl/sata/franka_my_code/franka_mujoco/analysis_results/libero_relative_clearance_eps01_zero_nominal_continuation_20261106/combined_200_source_manifest.json')
METRICS = {
    'velocity': 'mean_joint_velocity_rad_s',
    'acceleration': 'mean_joint_acceleration_rad_s2',
    'mY': 'mean_manipulability',
    'sigma_min': 'minimum_sigma_min_velocity_normalized',
}

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def combine(blocks):
    # Each block is initialized separately by the existing annotation protocol.
    # Do not invent cross-block velocity/acceleration differences.
    result = {}
    for name, field in METRICS.items():
        if name == 'sigma_min':
            result[name] = min(b[field] for b in blocks)
        else:
            offset = {'velocity': 1, 'acceleration': 2, 'mY': 0}[name]
            weights = [max(0, b['samples'] - offset) for b in blocks]
            assert sum(weights) > 0
            result[name] = sum(b[field] * w for b, w in zip(blocks, weights)) / sum(weights)
    return result

def summarize(rows):
    output = {'episodes': len(rows), 'tasks': len({(r['suite'], r['task_id']) for r in rows}), 'metrics': {}}
    for name in METRICS:
        baseline = fmean(r['zero'][name] for r in rows)
        best = fmean(r['best'][name] for r in rows)
        sign = -1 if name in ('velocity', 'acceleration') else 1
        output['metrics'][name] = dict(
            zero_mean=baseline, best_mean=best,
            improvement_percent_of_means=100 * sign * (best - baseline) / baseline,
            mean_paired_improvement_percent=fmean(100 * sign * (r['best'][name] - r['zero'][name]) / r['zero'][name] for r in rows),
            improved_episodes=sum(sign * (r['best'][name] - r['zero'][name]) > 1e-10 for r in rows),
            worsened_episodes=sum(sign * (r['best'][name] - r['zero'][name]) < -1e-10 for r in rows),
        )
    return output

def main():
    manifest = read(MANIFEST)
    rows = []
    for item in manifest['episodes']:
        if item['status'] != 'labeled':
            continue
        folder = Path(item['source_run']) / item['suite'] / f"task_{item['task_id']:02d}" / item['demo']
        assert sha(folder / 'tolerance_labels.json') == item['labels_sha256']
        prepared = read(folder / 'prepared_input.json')
        blocks = []
        for index in range(len(prepared['subtasks'])):
            annotation_path = folder / f'subtask_{index:02d}' / 'annotation.json'
            annotation = read(annotation_path)
            selected = annotation['best']
            pair = []
            evidence = []
            for path, expected in [(annotation['zero_baseline_trial'], annotation['zero_baseline_trial_sha256']),
                                   (selected['raw_trial_path'], selected['raw_trial_sha256'])]:
                assert sha(path) == expected
                raw = read(path)
                result = raw['result']
                e = result['evaluation']
                assert e['trajectory_complete'] and e['kinematic_feasible_without_table']
                assert e['control_rate_hz'] == 20 and e['physics_steps'] == 0
                assert result['spec'].get('kinematic_sampling_stride', 1) == 1
                metrics = result['trajectory_kinematics']
                assert all(math.isfinite(metrics[field]) and metrics[field] >= 0 for field in METRICS.values())
                pair.append(metrics)
                evidence.append(dict(path=path, sha256=expected))
            assert pair[0]['samples'] == pair[1]['samples']
            assert pair[0]['samples'] == sum(f['subtask_index'] == index for f in prepared['frames'])
            blocks.append(dict(index=index, zero=pair[0], best=pair[1], evidence=evidence, annotation=str(annotation_path)))
        rows.append(dict(suite=item['suite'], task_id=item['task_id'], demo=item['demo'],
                         reference_protocol=item['reference_protocol'], blocks=blocks,
                         zero=combine([b['zero'] for b in blocks]), best=combine([b['best'] for b in blocks])))
    summary = dict(complete=True, manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST),
        protocol='20Hz ideal kinematic command replay; paired disjoint blocks; valid N-1/N-2 differences pooled, mY weighted by N, sigma_min minimum over blocks; episode-equal mean',
        selected_best_objective='minimum observed mean joint velocity among accepted candidates, not proven global optimum or four-objective optimization',
        all199=summarize(rows), original_protocol198=summarize([r for r in rows if not (r['suite']=='libero_10' and r['task_id']==8 and r['demo']=='demo_4')]),
        excluding_strict_tasks189=summarize([r for r in rows if not (r['suite']=='libero_goal' and r['task_id'] in (0,5))]),
        suites={s:summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
        missing=[{k:i[k] for k in ('suite','task_id','demo','status')} for i in manifest['episodes'] if i['status']!='labeled'],
        verified_label_files=len(rows), verified_trial_pairs=sum(len(r['blocks']) for r in rows))
    (HERE/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (HERE/'paired_metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
