"""Small diagnostic PRE-axis ablation through the existing native backend.

No planner/controller/physics implementation: only configure existing worker.
The formal400 results and confirmed annotations are never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_tasktol_compat import rollout_confirmed_tolerance400 as prior

OUT = ROOT / 'analysis/20261101_alphabet_soup_grasp_audit/axis_ablation'
PREVIOUS = ROOT / 'data/libero_tasktol_joint_servo/20261031_mask30_confirmed_v1'


def freeze():
    return dict(prior.freeze(), **{str(Path(__file__)):prior.runner.sha(Path(__file__))})


def worker(key, mask):
    import numpy as np
    prior.runner.freeze_sources = freeze
    entry = next(e for e in json.loads((PREVIOUS / 'plan.json').read_text())['entries'] if e['key'] == key)
    task = copy.deepcopy(entry['task'])
    task['masks']['APP'] = mask
    task['semantic_subtasks'][0]['APP'] = mask
    folder = OUT / mask / key
    sources = freeze()
    plan = dict(case=entry['case'], task=task, jobs=[entry['job']], source_sha256=sources,
                phase_policy=dict(entry['phase_policy'], subtasks=task['semantic_subtasks']))
    prior.runner.dump(folder / 'joint_native/plan.json', plan)
    phase_policy = prior.make_phase_policy(task['semantic_subtasks'], entry['baseline_cycles'], entry['attempt_to_semantic_subtask'])
    report = prior.servo.worker(folder, 0, 'joint_native', reference_transform=lambda r,s:r,
                               mode='mask30', phase_policy=phase_policy)
    dest = folder / 'joint_native/seed_00/mask30'
    base_report = json.loads((Path(entry['baseline_folder']) / 'report.json').read_text())
    for k in ['actions_sha256', 'initial_state_sha256', 'model_sha256', 'solver_settings', 'objective_settings', 'shared_joint_servo']:
        assert report[k] == base_report[k], k
    exact_old = None
    if mask == '011':
        for name in ['states.npz', 'servo_physics.npz']:
            with np.load(dest / name) as a, np.load(PREVIOUS / 'episodes' / key / name) as b:
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k])
        exact_old = True
    prior.runner.dump(folder / 'record.json', dict(key=key, mask=mask, success=report['success'],
        execution_rejected=report['execution_rejected'], folder=str(dest),
        original011_bitwise_identical=exact_old, unchanged_solver_model_actions_servo=True))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--worker', action='store_true'); p.add_argument('--key'); p.add_argument('--mask')
    args = p.parse_args()
    if args.worker:
        worker(args.key, args.mask); return
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(f'libero_object/00/{demo}', mask) for demo in ['demo_4', 'demo_30'] for mask in ['011', '001', '010']]
    prior.runner.dump(OUT / 'design.json', dict(jobs=jobs,
        selection='Two diagnostic failure cases: demo4 early object displacement and demo30 little preclose object displacement. Not a population success-rate estimate.',
        change='PRE mask only:011 versus001(zF only) versus010(yF only); EXEC111, ENG/DIS000 unchanged; all bounds30deg.',
        source_sha256=freeze(), python=sys.executable))
    def run(job):
        key, mask = job
        record = OUT / mask / key / 'record.json'
        if not record.exists():
            log = OUT / f'{key.split("/")[-1]}_{mask}.log'
            env = dict(os.environ, MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       NUMBA_CACHE_DIR=str(OUT / 'numba_cache'))
            with log.open('w') as f:
                subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker', '--key', key, '--mask', mask],
                               cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        result = json.loads(record.read_text()); print(json.dumps(result), flush=True); return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    prior.runner.dump(OUT / 'summary.json', dict(complete=True, results=results))


if __name__ == '__main__':
    main()
