"""Resumable DSP labels on immutable recorded nominal trajectories.

Only orchestration, data conversion and checkpoint I/O live here. All queries
use the public counterfactual backend and the existing DSP search algorithm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('TASKTOL_NOMINAL_CANDIDATE_WORKERS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
from tasktol.data.collector import _atomic_write_json as write
from tasktol.data.trial_io import compact_trial, read_json
from tasktol.data.task_sets import training_tasks
from tasktol.data.tolerance_labels import trajectory_sha256, write_tolerance_labels, load_rotation_tolerance
from tasktol.counterfactual.runner import CounterfactualTrialSpec, prepare_dataset_nominal_trial, run_counterfactual_trial
from tasktol.tolerance.context import declared_stage_rotation_masks, long_task_subtask_stage_rotation_masks
from tasktol.tolerance.labeling import SEARCH_SOLUTION_LABEL_POLICY
from benchmark_joint_pre_post_t1_60 import result_outcome
from benchmark_long_task_reannotation import stage_caps
from benchmark_reduced_joint_init_t1_60 import reduced_initial_design
from benchmark_paper_cbo import stable_seed

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / 'datasets/franka_100_nominal'
DEFAULT_OUTPUT = ROOT / 'analysis_results/franka_50_nominal_dsp_joint32_20261026'
METRICS = ('mean_joint_velocity_rad_s', 'mean_joint_acceleration_rad_s2',
           'mean_manipulability', 'minimum_sigma_min_velocity_normalized')


def read(path):
    return read_json(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def registry():
    return {t.task_id: (s, t) for s, t in training_tasks()}


def split_recorded_subtasks(document, semantic_count):
    """Retain observed phase IDs; clamp terminal return to the last skill.

    The same clamping is used by the public backend's stage_values. Do not
    interpret a terminal open-gripper PRE as another pick/place operation.
    """
    groups = [[] for _ in range(semantic_count)]
    aliases = {}
    previous = -1
    for f in document['trajectory']:
        observed = f['subtask_index']
        if type(observed) is not int or observed < 0:
            raise ValueError('Invalid recorded subtask index')
        i = min(observed, semantic_count - 1)
        if i < previous:
            raise ValueError('Nonchronological recorded subtask index')
        previous = i
        groups[i].append(f)
        aliases[str(observed)] = i
    if not all(groups):
        raise ValueError('Recorded trajectory is missing a semantic subtask')
    # Do not renumber frames or edit the source JSON; the backend consumes
    # these exact original frame dictionaries, actions and finger positions.
    return [tuple(g) for g in groups], aliases


def variant_index(scene, document, preferred=None):
    from tasktol.simulation.scenes import SceneManager
    active = set(document['episode']['domain_randomization']['active_scene_bodies'])
    variants = SceneManager._SEEDED_VARIANTS.get(scene, ())
    matches = [i for i, names in enumerate(variants) if set(names) == active]
    if preferred is not None and preferred in matches:
        return preferred
    if len(matches) > 1:
        # Some registered variants intentionally share the same bodies (pot
        # lid kinds / block ordering). Recover the original seeded index,
        # rather than guessing a different variant from identical body sets.
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((
            int(document['episode']['seed']), SceneManager._VARIANT_STREAM_ID))))
        selected = int(rng.integers(0, len(variants)))
        if selected in matches:
            return selected
    if variants and len(matches) != 1:
        raise ValueError(f'Cannot uniquely identify recorded variant in {scene}: {matches}')
    return matches[0] if matches else None


def base_spec(job, document):
    from tasktol.tasks.catalog import pot_lid_task_variant_index
    scene, task = registry()[job['task']]
    routing = document['episode']['domain_randomization']['seed_routing']
    return CounterfactualTrialSpec(scene=scene.runtime_scene, task=task.runtime_task,
        robot_uid='panda', seed=job['seed'], environment_seed=routing['environment_seed'],
        environment_randomization_probability=routing['environment_randomization_probability'],
        scene_variant_index=variant_index(scene.runtime_scene, document,
            pot_lid_task_variant_index(task.runtime_task)),
        angle_deg=0, maximum_frames=10000, execution_mode='kinematic',
        controller_mode='sqp_release', sqp_loss_profile='delta_q_squared_only',
        sqp_loss_weights=(1., 0., 10.), sqp_max_time_s=0., ipopt_max_iterations=20,
        kinematic_check_table_collisions=True, kinematic_record_robot_table_clearance=True,
        kinematic_continue_after_table_contact=True)


def source_files():
    return sorted(set([*ROOT.glob('*.py'), *(ROOT / 'tasktol').rglob('*.py'),
        *(ROOT / 'assets').rglob('*.xml'), ROOT / 'scene.xml', ROOT / 'pyproject.toml', ROOT / 'uv.lock']))


def verify_code(manifest, output=None):
    expected_hashes = dict(manifest['code_sha256'])
    history = Path(output)/'source_update_history.json' if output else None
    if history and history.exists():
        changes = read(history)
        if changes['original_manifest_sha256'] != sha(Path(output)/'manifest.json'):
            raise ValueError('Source amendment belongs to another run')
        for update in changes['updates']:
            expected_hashes.update(update['code_sha256'])
        if changes.get('constraint_policy_sha256') != sha(Path(output)/'constraint_policy.json'):
            raise ValueError('Authorized task-specific constraint policy changed')
    for name, expected in expected_hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Source changed during experiment: {name}')


def verify_base(job):
    for name, expected in job['files_sha256'].items():
        if sha(Path(job['episode']) / name) != expected:
            raise ValueError(f'Immutable base file changed: {job["episode"]}/{name}')


def prepare_manifest(dataset, output, extra):
    path = output / 'manifest.json'
    settings = dict(extra=extra, label_policy=SEARCH_SOLUTION_LABEL_POLICY,
        profile='dsp', fit_maxiter=100, sobol_points=1024, restarts=8, optimizer_maxiter=100,
        model_fit_max_attempts=5, clearance_m=.005, seed=20261026,
        objective='full-subtask mean joint velocity', domain='authored signed directions, continuous [0,30] degrees',
        nominal='exact stored actions/finger observations; no replanning',
        terminal_return='clamp observed trailing index to last semantic subtask, matching backend')
    if path.exists():
        result = read(path)
        if result['dataset'] != str(dataset) or result['settings'] != settings:
            raise ValueError('Changed dataset or search settings')
        verify_code(result, output)
        return result
    plan = read(dataset / 'collection_plan.json')
    jobs = []
    for task in plan['tasks']:
        directory = dataset / task['relative_path']
        entries = read(directory / 'manifest.json')['episodes']
        if len(entries) != 50:
            raise ValueError(f'Expected exactly 50 successful episodes for {task["task"]}')
        scene, registration = registry()[task['task']]
        descriptors = long_task_subtask_stage_rotation_masks(registration.runtime_task)
        for e in entries:
            episode = directory / e['path']
            trajectory_sha256(episode)  # Validate schema before creating any jobs.
            jobs.append(dict(task=task['task'], seed=e['seed'], episode=str(episode.resolve()),
                episode_index=e['episode_index'], semantic_count=max(1, len(descriptors)),
                relative_case=f'{task["task"]}/{e["path"]}',
                files_sha256={n: sha(episode / n) for n in
                    ('trajectory.json', 'main_rgb.mp4', 'wrist_rgb.mp4')}))
    if len(jobs) != 1250 or len(plan['tasks']) != 25:
        raise ValueError('Require the full 25-task, 1250-episode dataset')
    # Round-robin tasks exposes task-specific issues early and avoids a long
    # sequence of hard cases being postponed until the end of the night.
    jobs.sort(key=lambda j: (j['episode_index'], j['task']))
    result = dict(dataset=str(dataset), settings=settings, jobs=jobs,
        created_utc=datetime.now(timezone.utc).isoformat(),
        collection_plan_sha256=sha(dataset / 'collection_plan.json'),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files()})
    output.mkdir(parents=True, exist_ok=True)
    write(path, result)
    return read(path)


def feedback(result, clearance_m=.005, clearance_strict=False, clearance_policy=None):
    row = result_outcome(result)
    if clearance_policy is not None:
        from tasktol.tolerance.clearance_policy import relative_feedback
        return relative_feedback(row, result, clearance_policy)
    if clearance_m != .005 or clearance_strict:
        if row['trajectory_complete']:
            c = row['minimum_clearance_m']
            qualified = c > clearance_m if clearance_strict else c >= clearance_m
            row.update(success=qualified, clearance_pass=qualified,
                velocity_rad_s=row['observed_velocity_rad_s'] if qualified else None,
                reason='kinematic_success' if qualified else 'robot_table_clearance')
    return row


def case_settings(job, manifest, output):
    settings = dict(manifest['settings'])
    if job['task'] == 'wipe_whiteboard':
        policy_path = Path(output)/'constraint_policy.json'
        policy = read(policy_path)
        expected = {'minimum_clearance_m': 0.0, 'comparison': '>'}
        rule = policy['task_overrides']['wipe_whiteboard']
        if any(rule[k] != v for k, v in expected.items()) or set(policy['task_overrides']) != {'wipe_whiteboard'}:
            raise ValueError('Only the authorized whiteboard 5mm exception is allowed')
        settings.update(clearance_m=0., clearance_strict=True,
                        constraint_policy_sha256=sha(policy_path))
    return settings


def evaluate(folder, spec, prepared, frames, point, identity, clearance_m=.005, clearance_strict=False,
             *, clearance_policy=None, zero_result=None):
    path = folder / 'evaluations' / digest(point) / 'trial.json'
    provenance = dict(identity=identity, point=point, spec=asdict(spec))
    if clearance_policy is not None:
        provenance['clearance_policy'] = clearance_policy
    expected = digest(provenance)
    if path.exists():
        raw = read(path)
        if raw['identity_sha256'] != expected:
            raise ValueError('Cached point has different input provenance')
    else:
        start = time.monotonic()
        if zero_result is not None:
            if any(point) or digest(zero_result['spec']) != digest(asdict(spec)):
                raise ValueError('Only an identical zero-tolerance replay may reuse the baseline')
            result = zero_result
        else:
            result = run_counterfactual_trial(spec, prepared_nominal=prepared,
                dataset_nominal_frames=frames).as_dict()
        row = feedback(result, clearance_m, clearance_strict, clearance_policy)
        row.update(actual_pre_caps_deg=point[:6], actual_post_caps_deg=point[6:])
        raw = dict(identity_sha256=expected, point=point, row=row,
                   result=result, seconds=time.monotonic()-start)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, compact_trial(raw, path))
    return dict(raw['row'], raw_trial_path=str(path), raw_trial_sha256=sha(path)), raw['seconds']


def search_maximum_caps(masks, settings):
    """Keep legacy 30-degree defaults, or use explicit per-direction authored bounds."""
    allowed = np.repeat(np.asarray(masks, float), 2, axis=1).ravel()
    maximum = np.asarray(settings.get('maximum_caps_deg', (allowed * 30.).tolist()), float)
    if (allowed.shape != (12,) or not np.isin(allowed, [0., 1.]).all()
            or maximum.shape != (12,) or not np.isfinite(maximum).all()
            or np.any(maximum < 0) or np.any(maximum > 30)
            or np.any((maximum > 0) != (allowed > 0))):
        raise ValueError('Authored caps must be finite 0..30 degrees and match declared strict axes')
    return maximum.tolist()


def search_subtask(folder, spec, frames, descriptor, settings, identity, *, preflight=False, prepared_nominal=None):
    from literature_botorch_search import LiteratureJointSearch
    folder.mkdir(parents=True, exist_ok=True)
    prepared = prepared_nominal if prepared_nominal is not None else prepare_dataset_nominal_trial(spec, frames)
    if prepared_nominal is None:
        expected = identity['source_environment']
        if (set(prepared.metadata['active_scene_bodies']) != set(expected['active_scene_bodies'])
                or prepared.metadata['table_offset_m'] != expected['table_offset_m']):
            raise ValueError('Prepared scene does not match recorded bodies/table randomization')
    elif prepared.stored_context is None:
        raise ValueError('Imported trajectory search requires a validated stored scene context')
    masks = descriptor or declared_stage_rotation_masks(prepared.model, spec.scene, spec.task,
        has_grasp=any(f['action'][6] > 0 for f in frames))
    maximum = search_maximum_caps(masks, settings)
    if not any(maximum):
        raise ValueError('No authored rotational search dimensions; cannot fabricate 32 distinct points')
    design = reduced_initial_design(maximum[:6], maximum[6:])
    identity = digest(dict(parent=identity, frames=frames, spec=asdict(spec), masks=masks,
                           settings=settings, design=design))
    policy = settings.get('clearance_policy')
    def query(point):
        extra_options = {}
        if policy is not None:
            baseline_path = Path(settings['zero_baseline_trial'])
            if sha(baseline_path) != settings['zero_baseline_trial_sha256']:
                raise ValueError('Frozen zero baseline trial changed')
            extra_options = dict(clearance_policy=policy,
                zero_result=read(baseline_path)['result'] if not any(point) else None)
        return evaluate(folder, stage_caps(spec, point, masks), prepared, frames, point, identity,
            settings['clearance_m'], settings.get('clearance_strict', False), **extra_options)
    if preflight:
        probes = [query(point)[0] for point in design['initial_caps_deg'][:2]]
        report = dict(identity_sha256=identity, design=design, probes=probes,
                      nominal_frames=len(frames), domain_randomization=prepared.metadata)
        write(folder / 'preflight.json', report)
        return report
    initial, initial_times = [], []
    for i, point in enumerate(design['initial_caps_deg']):
        row, seconds = query(point)
        initial.append(dict(row, index=i, query_index=i))
        initial_times.append(dict(index=i, kind='initial', evaluation_seconds=seconds))
    checkpoint = folder / 'checkpoint.json'
    saved = read(checkpoint) if checkpoint.exists() else None
    if saved and saved['identity_sha256'] != identity:
        raise ValueError('Search checkpoint input identity changed')
    search = LiteratureJointSearch(maximum, settings['search_seed'], initial,
        state=saved['state'] if saved else None, profile='dsp', fit_maxiter=settings['fit_maxiter'],
        sobol_points=settings['sobol_points'], restarts=settings['restarts'],
        optimizer_maxiter=settings['optimizer_maxiter'], clearance_m=settings['clearance_m'],
        clearance_strict=settings.get('clearance_strict', False),
        **({'constraint_metric': 'clearance_margin_m'} if policy is not None else {}))
    timings = saved['timings'] if saved else initial_times
    while len(search.state['observations']) < len(initial)+settings['extra']:
        point = search.ask()
        row, seconds = query(point)
        search.tell(point, row)
        timings.append(dict(index=row.get('index', len(search.state['observations'])-1),
            kind='adaptive', evaluation_seconds=seconds, **search.last_timing))
        write(checkpoint, dict(identity_sha256=identity, state=search.state, timings=timings))
        write(folder / 'progress.json', dict(initial=len(initial),
            extra_done=len(search.state['observations'])-len(initial), target=settings['extra'],
            updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
    rows = search.state['observations']
    feasible = [r for r in rows if r['success']]
    selected = min(feasible, key=lambda r: (r['velocity_rad_s'], r['index'])) if feasible else None
    for r in rows:
        if sha(r['raw_trial_path']) != r['raw_trial_sha256']:
            raise ValueError('Changed saved raw evaluation')
    report = dict(identity_sha256=identity, design=design, observations=rows, timings=timings,
        complete=True, additional_queries=len(rows)-len(initial), best=selected,
        clearance_m=settings['clearance_m'], clearance_strict=settings.get('clearance_strict', False),
        metrics={k: read(selected['raw_trial_path'])['result']['trajectory_kinematics'][k]
                 for k in METRICS} if selected else None)
    if policy is not None:
        report['clearance_policy'] = policy
        report['zero_baseline_trial'] = settings['zero_baseline_trial']
        report['zero_baseline_trial_sha256'] = settings['zero_baseline_trial_sha256']
    write(folder / 'annotation.json', report)
    return report


def run_case(job, manifest, output, preflight=False):
    import torch
    import mujoco
    import botorch_joint_search as engine
    from threadpoolctl import threadpool_limits
    from recover_dsp_random_fit import retry_factory
    torch.set_num_threads(1)
    verify_code(manifest, output)
    verify_base(job)
    root = Path(output) / 'cases' / job['relative_case']
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    trace = []
    document = read(Path(job['episode']) / 'trajectory.json')
    segments, aliases = split_recorded_subtasks(document, job['semantic_count'])
    spec = base_spec(job, document)
    descriptors = long_task_subtask_stage_rotation_masks(spec.task)
    annotations = []
    with threadpool_limits(limits=1), \
         patch('mujoco.mj_step', side_effect=AssertionError('Physics forbidden in annotation')), \
         patch.object(engine, 'fit_gpytorch_mll', retry_factory(engine.fit_gpytorch_mll, trace)):
        for i, frames in enumerate(segments):
            settings = dict(case_settings(job, manifest, output), search_seed=stable_seed(
                manifest['settings']['seed'], job['task'], job['seed'], i))
            identity = dict(job=job, semantic_index=i, aliases=aliases,
                            source_environment=document['episode']['domain_randomization'])
            report = search_subtask(root / f'subtask_{i:02d}', spec, frames,
                descriptors[i][1] if descriptors else None, settings, identity, preflight=preflight)
            annotations.append(report)
            write(root / 'status.json', dict(semantic_subtasks=job['semantic_count'],
                subtasks_done=len(annotations), task=job['task'], seed=job['seed'], pid=os.getpid()))
            write(root / 'fit_attempts.json', dict(attempts=trace))
    verify_base(job)
    if preflight:
        return dict(task=job['task'], seed=job['seed'], preflight=True, seconds=time.monotonic()-start)
    found = all(a['best'] for a in annotations)
    if found:
        labels = []
        for observed, i in sorted(aliases.items(), key=lambda item: int(item[0])):
            row = annotations[i]['best']
            labels.append(dict(subtask_index=int(observed),
                pregrasp=np.radians(row['actual_pre_caps_deg']).tolist(),
                postgrasp=np.radians(row['actual_post_caps_deg']).tolist()))
        source = dict(method='BoTorch DSP three-model CBO; full continuous PRE+POST',
            label_policy=SEARCH_SOLUTION_LABEL_POLICY, search_manifest_sha256=sha(Path(output)/'manifest.json'),
            trajectory_sha256=job['files_sha256']['trajectory.json'], observed_to_semantic_subtask=aliases,
            selected_trials=[dict(path=a['best']['raw_trial_path'], sha256=a['best']['raw_trial_sha256'])
                             for a in annotations], additional_queries_per_subtask=manifest['settings']['extra'])
        if job['task'] == 'wipe_whiteboard':
            source['clearance_policy'] = dict(minimum_m=0., comparison='>',
                constraint_policy_sha256=sha(Path(output)/'constraint_policy.json'))
        label_path = Path(job['episode']) / 'tolerance_labels.json'
        if label_path.exists():
            old = read(label_path)
            load_rotation_tolerance(Path(job['episode']))
            if old['subtasks'] != labels or old['source'] != source:
                raise ValueError('Refusing to overwrite different existing labels')
        else:
            write_tolerance_labels(Path(job['episode']), labels,
                expected_trajectory_sha256=job['files_sha256']['trajectory.json'], source=source)
    report = dict(task=job['task'], seed=job['seed'], complete=True, labeled=found,
        seconds=time.monotonic()-start, semantic_subtasks=len(annotations), aliases=aliases,
        selected=[a['best'] for a in annotations], label_policy=SEARCH_SOLUTION_LABEL_POLICY,
        manifest_sha256=sha(Path(output)/'manifest.json'))
    write(root / 'report.json', report)
    (root / 'error.json').unlink(missing_ok=True)
    return report


def progress(output, manifest, workers, wall_seconds):
    reports, errors, subs = [], [], 0
    for job in manifest['jobs']:
        root = output / 'cases' / job['relative_case']
        if (root/'report.json').exists():
            report = read(root/'report.json')
            reports.append(report)
            subs += report['semantic_subtasks']
        elif (root/'error.json').exists():
            errors.append(read(root/'error.json'))
        elif (root/'status.json').exists():
            subs += read(root/'status.json')['subtasks_done']
    payload = dict(complete=len(reports) == len(manifest['jobs']) and all(r['labeled'] for r in reports),
        total_episodes=len(manifest['jobs']), searched_episodes=len(reports),
        labeled_episodes=sum(r['labeled'] for r in reports), completed_subtask_searches=subs,
        total_subtask_searches=sum(j['semantic_count'] for j in manifest['jobs']),
        no_feasible_label=[dict(task=r['task'], seed=r['seed']) for r in reports if not r['labeled']],
        errors=errors, workers=workers, coordinator_pid=os.getpid(), wall_seconds=wall_seconds,
        updated_utc=datetime.now(timezone.utc).isoformat())
    write(output/'progress.json', payload)
    return payload


def audit_all(output, manifest):
    """Independently check complete budgets, feedback, selection and labels."""
    verify_code(manifest, output)
    episodes, subtask_count, query_count = 0, 0, 0
    for job in manifest['jobs']:
        verify_base(job)
        root = output/'cases'/job['relative_case']
        report = read(root/'report.json')
        if (not report['complete'] or not report['labeled']
                or report['manifest_sha256'] != sha(output/'manifest.json')):
            raise ValueError(f'Incomplete case: {job["relative_case"]}')
        original = read(Path(job['episode'])/'trajectory.json')
        _, aliases = split_recorded_subtasks(original, job['semantic_count'])
        label = read(Path(job['episode'])/'tolerance_labels.json')
        load_rotation_tolerance(Path(job['episode']))
        if label['source']['observed_to_semantic_subtask'] != aliases:
            raise ValueError('Changed label-to-semantic-subtask mapping')
        selected = []
        for i in range(job['semantic_count']):
            folder = root/f'subtask_{i:02d}'
            annotation, checkpoint = read(folder/'annotation.json'), read(folder/'checkpoint.json')
            rows = annotation['observations']
            ninit = annotation['design']['unique_initial_count']
            if (annotation['additional_queries'] != manifest['settings']['extra']
                    or len(rows) != ninit+manifest['settings']['extra']
                    or checkpoint['state']['observations'] != rows
                    or len(checkpoint['state']['diagnostics']) != manifest['settings']['extra']):
                raise ValueError('Incorrect query budget or checkpoint history')
            if [r['actual_pre_caps_deg']+r['actual_post_caps_deg'] for r in rows[:ninit]] != annotation['design']['initial_caps_deg']:
                raise ValueError('Structured initialization was changed')
            for index, row in enumerate(rows):
                if row['index'] != index or sha(row['raw_trial_path']) != row['raw_trial_sha256']:
                    raise ValueError('Changed raw history or query order')
                raw = read(row['raw_trial_path'])
                policy = case_settings(job, manifest, output)
                expected = feedback(raw['result'], policy['clearance_m'], policy.get('clearance_strict', False))
                if any(row[key] != value for key, value in expected.items()):
                    raise ValueError('Observation differs from full-trajectory feedback')
                if raw['point'] != row['actual_pre_caps_deg']+row['actual_post_caps_deg']:
                    raise ValueError('Search point differs from executed caps')
                maximum = np.asarray(annotation['design']['maximum_caps_deg'])
                if np.any(np.asarray(raw['point']) < 0) or np.any(np.asarray(raw['point']) > maximum):
                    raise ValueError('Point violates authored domain')
            for step, diagnostic in enumerate(checkpoint['state']['diagnostics']):
                n = ninit+step
                complete = [r['index'] for r in rows[:n] if r['trajectory_complete']]
                if (diagnostic['after_queries'] != n
                        or diagnostic['objective_indices'] != complete
                        or diagnostic['clearance_indices'] != complete
                        or diagnostic['solver_indices'] != list(range(n))):
                    raise ValueError('GP used future data or wrong failure information')
            feasible = [r for r in rows if r['success']]
            best = min(feasible, key=lambda r: (r['velocity_rad_s'], r['index'])) if feasible else None
            if best is None or annotation['best'] != best or report['selected'][i] != best:
                raise ValueError('Final selection is not the speed-best feasible observation')
            selected.append(best)
            subtask_count += 1
            query_count += len(rows)
        for item in label['subtasks']:
            best = selected[aliases[str(item['subtask_index'])]]
            for phase, key in (('pregrasp', 'actual_pre_caps_deg'), ('postgrasp', 'actual_post_caps_deg')):
                if item[phase] != np.radians(best[key]).tolist():
                    raise ValueError('Label clipped to actual release or differs from selected caps')
        episodes += 1
    result = dict(complete=True, immutable_base_files=episodes*3, verified_labels=episodes,
        verified_subtask_searches=subtask_count, verified_queries=query_count,
        manifest_sha256=sha(output/'manifest.json'),
        checks=['base SHA256', 'full query budgets and initial design', 'raw feedback SHA256',
                'no future training data', '5mm acceptance except authorized whiteboard contact-only rule', 'minimum feasible velocity',
                'selected search caps, not release peaks', 'row-aligned labels'])
    write(output/'completion_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--extra', type=int, default=32)
    parser.add_argument('--run-first', type=int)
    parser.add_argument('--preflight', action='store_true', help='Zero/full initial boxes for one episode of every task; no labels')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.extra < 1:
        parser.error('Require positive workers and additional budget')
    dataset, output = args.dataset.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for lock_path in (dataset/'.collection.lock', output/'.annotation.lock'):
            handle = stack.enter_context(lock_path.open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare_manifest(dataset, output, args.extra)
        start = time.monotonic()
        if args.audit_only:
            audit_all(output, manifest)
            print(json.dumps(progress(output, manifest, args.workers, 0)), flush=True)
            return
        jobs = manifest['jobs']
        if args.preflight:
            jobs = [j for j in jobs if j['episode_index'] == 0]
        elif args.run_first:
            jobs = jobs[:args.run_first]
        # Completed results are still independently hash-checked by audit-only.
        if not args.preflight:
            jobs = [j for j in jobs if not (output/'cases'/j['relative_case']/'report.json').exists()]
        print(json.dumps(dict(event='start', jobs=len(jobs), workers=args.workers,
            output=str(output), extra=args.extra, preflight=args.preflight)), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_case, j, manifest, str(output), args.preflight): j for j in jobs}
            while futures:
                done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    try:
                        print(json.dumps(dict(event='case_complete', **future.result())), flush=True)
                    except Exception as error:
                        failure = dict(task=job['task'], seed=job['seed'], error=str(error), traceback=traceback.format_exc())
                        path = output/'cases'/job['relative_case']/'error.json'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        write(path, failure)
                        print(json.dumps(dict(event='case_error', **failure)), flush=True)
                if not args.preflight:
                    status = progress(output, manifest, args.workers, time.monotonic()-start)
                    print(json.dumps(dict(event='progress', **status)), flush=True)
        if not args.preflight:
            status = progress(output, manifest, args.workers, time.monotonic()-start)
            if status['complete']:
                audit_all(output, manifest)


if __name__ == '__main__':
    main()
