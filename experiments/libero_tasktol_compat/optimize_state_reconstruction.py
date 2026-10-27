"""Uniform offline reference refinement with state-cost acceptance/backtracking."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from closed_loop_state_metrics import evaluate_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', type=Path, nargs='+', required=True)
    parser.add_argument('--cached-first', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for path in args.baselines:
        summary = json.loads(path.read_text())
        assert summary['complete']
        jobs.extend(summary['results'])
    assert len({(j['case'], j['demo']) for j in jobs}) == len(jobs)
    cached = {}
    if args.cached_first:
        first = json.loads(args.cached_first.read_text())
        assert first['complete'] and all(j['returncode'] == 0 for j in first['results'])
        cached = {(j['case'], j['demo']):j for j in first['results']}
    policy = dict(method='IRLC_filtered_reference_update_with_full_physical_state_acceptance',
                  gains=[.5, .25], anticipation_steps=1, time_scale=1, gripper='original unchanged',
                  apply_to='every episode, including already-successful recordings',
                  acceptance='strictly lower fixed physical-state cost; full feasible execution; retain existing task success',
                  initialization='global source intent projection with .01 next-state posture regularization',
                  sources=['https://lup.lub.lu.se/search/files/206302733/IROS24.pdf', 'https://arxiv.org/html/2604.16850v1'],
                  scope='offline data refinement, not VLA training evaluation')
    (output/'policy.json').write_text(json.dumps(policy, indent=2)+'\n')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', NUMBA_CACHE_DIR='/tmp/tasktol_state_relabel_numba',
               MPLCONFIGDIR='/tmp/tasktol_state_relabel_mpl')

    def run(job):
        key = (job['case'], job['demo'])
        parent = Path(job['root']).resolve()
        baseline = evaluate_run(parent)
        trials = []
        for index, gain in enumerate(policy['gains']):
            if index == 0 and key in cached:
                item = cached[key]
                assert Path(item['parent']).resolve() == parent
                root = Path(item['root']).resolve()
            else:
                root = output/f"case{job['case']:02d}_{job['demo']}"/f'iteration_{index+1}'
                root.parent.mkdir(parents=True, exist_ok=True)
                with (root.parent/f'iteration_{index+1}.log').open('w') as log:
                    done = subprocess.run([sys.executable, str(Path(__file__).with_name('refine_reference_iteration.py')),
                        '--parent',str(parent),'--output',str(root),'--gain',str(gain),'--anticipation','1'],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                assert done.returncode == 0, f'Execution process failed: {root}'
            decision = json.loads((root/'acceptance.json').read_text())
            metrics = evaluate_run(root)
            assert metrics['objective'] == decision['objective']
            previous = evaluate_run(parent)
            accepted = (metrics['full_length'] and not metrics['rejected']
                        and (not previous['success'] or metrics['success'])
                        and metrics['objective'] < previous['objective'])
            assert accepted == decision['accepted']
            trials.append(dict(root=str(root), parent=str(parent), gain=gain, **decision))
            if accepted:
                parent = root
        final = evaluate_run(parent)
        return dict(case=job['case'], demo=job['demo'], suite=baseline['suite'], task_id=baseline['task_id'],
                    baseline_root=job['root'], baseline_objective=baseline['objective'], baseline_success=baseline['success'],
                    selected_root=str(parent), objective=final['objective'], success=final['success'], trials=trials)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            row = future.result()
            results.append(row)
            summary = dict(complete=len(results)==len(jobs), policy='policy.json', results=results,
                           baseline_success=sum(r['baseline_success'] for r in results), success=sum(r['success'] for r in results),
                           baseline_mean_cost=sum(r['baseline_objective'] for r in results)/len(results),
                           final_mean_cost=sum(r['objective'] for r in results)/len(results))
            temp = output/'summary.json.tmp'
            temp.write_text(json.dumps(summary,indent=2)+'\n')
            temp.replace(output/'summary.json')
            print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
