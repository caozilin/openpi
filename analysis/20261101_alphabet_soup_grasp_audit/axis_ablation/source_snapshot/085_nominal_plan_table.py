"""Shared persistent nominal candidate selection, with no execution loop.

Tasks supply candidate definitions and an on-demand reference builder. Every
observation is obtained from run_counterfactual_trial. Only the discrete
selection and validation summaries are cached, never executable trajectories.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import copy
from dataclasses import replace
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import tempfile
import threading
import time

from ..tasks.planning.types import PlanningInfeasibleError

SCHEMA_VERSION = 8
SCREENING_POLICY = dict(strict_angle_deg=0, failure_policy="minimum_reference_rotation")
SELECTION_METRIC = "log_joint_travel_plus_rotation"
# Calibrated on the frozen first100 DR-on seeds. Half the smallest anchor
# break-even coefficient retains >=50% of every protected log-travel gain.
# This is a fixed policy at runtime, not a seed-dependent coefficient.
ROTATION_SCORE_POLICY = dict(weight=0.10674311180193222, power=1.5,
                             scale_deg=180., travel_floor_rad=1e-12)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "analysis_results/nominal_plan_table/v8"
_SCREENING_POOLS = {}
_SCREENING_POOL_LOCK = threading.Lock()
_SCREENING_POOL_FINALIZER = None


class NominalChoiceNotCached(LookupError):
    """A cache-only request must schedule screening outside the UI thread."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_digest():
    h = hashlib.sha256()
    for path in sorted((REPO_ROOT/"tasktol").rglob("*.py")):
        h.update(str(path.relative_to(REPO_ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def frame_records(frames):
    return [dict(position_m=f.position.tolist(), rotation_matrix=f.rotation.tolist(),
                 gripper_closed=bool(f.gripper_closed), phase=f.phase) for f in frames]


def candidate_rotation_summary(frames, initial_rotation):
    """Geodesic Home->grasp plus grasp->release rotation, in degrees.

    Each screening entry covers one live pick/place subtask. Long tasks use
    separate entries and rank each subtask from its own boundary rotation.
    """
    from ..tasks.subtasks import identify_pick_place_subtasks

    spans = identify_pick_place_subtasks(frames)
    if len(spans) != 1:
        raise ValueError("Nominal rotation ranking requires exactly one pick/place cycle")
    span = spans[0]
    grasp = frames[span.pregrasp_index].rotation
    place = frames[span.postgrasp_index].rotation
    return reference_rotation_summary(initial_rotation, grasp, place)


def reference_rotation_summary(initial_rotation, grasp, place):
    """The same geometric angle for frames and targets rejected before IK."""
    import numpy as np
    from ..geometry import rotation_vector

    first = float(np.degrees(np.linalg.norm(rotation_vector(initial_rotation.T @ grasp))))
    second = float(np.degrees(np.linalg.norm(rotation_vector(grasp.T @ place))))
    return dict(grasp_rotation_deg=first, placement_rotation_deg=second,
                total_rotation_deg=first + second)


def nominal_selection_score(observation, policy=None):
    """Minimize log(J / 1 rad) + weight * (total rotation / scale)**power.

    Log-travel makes the exchange depend on relative kinematic improvement,
    not a scene's absolute path length. Failed/incomplete solves never rank.
    """
    if not observation['feasible']:
        return math.inf
    policy = ROTATION_SCORE_POLICY if policy is None else policy
    weight, power = policy['weight'], policy['power']
    scale, floor = policy['scale_deg'], policy['travel_floor_rad']
    if not all(math.isfinite(v) for v in (weight, power, scale, floor)) or (
        weight < 0 or power < 1 or scale <= 0 or floor <= 0
    ):
        raise ValueError('Invalid nominal rotation scoring policy')
    travel, rotation = observation['total_joint_travel_rad'], observation['total_rotation_deg']
    if not math.isfinite(travel) or travel < 0 or not math.isfinite(rotation) or rotation < 0:
        raise ValueError('Invalid nominal travel/rotation observation')
    return math.log(max(travel, floor)) + weight * (rotation / scale) ** power


def rank_nominal_observations(observations, policy=None):
    """Fixed candidate order breaks exact score ties deterministically."""
    return min((i for i, o in enumerate(observations) if o['feasible']),
               key=lambda i: (nominal_selection_score(observations[i], policy), i), default=None)


def rank_reference_rotations(observations):
    """Choose a reference, without claiming failed FK/IK is executable."""
    return min((i for i, o in enumerate(observations)
                if o.get('total_rotation_deg') is not None
                and math.isfinite(o['total_rotation_deg']) and o['total_rotation_deg'] >= 0),
               key=lambda i: (observations[i]['total_rotation_deg'], i), default=None)


def selection_identity(*, scene, task, robot_uid, episode_seed, candidates,
                       domain_randomization_enabled):
    """Discrete cache routing ONLY, independent of model/data/solver state."""
    identity = dict(schema_version=SCHEMA_VERSION, scene=scene, task=task,
                robot_uid=robot_uid, episode_seed=int(episode_seed),
                domain_randomization_enabled=bool(domain_randomization_enabled),
                selection_metric=SELECTION_METRIC,
                screening_policy=dict(SCREENING_POLICY),
                rotation_score_policy=dict(ROTATION_SCORE_POLICY),
                candidate_definitions=json.loads(json.dumps(candidates, allow_nan=False)))
    if scene == "Whiteboard":
        from ..tasks.planning.wipe_whiteboard import (
            WIPE_COMPRESSION_DEPTH_M, WIPE_COMPRESSION_JITTER_M,
        )
        # Only the affected task's old candidate rankings are invalidated.
        identity["whiteboard_contact_policy"] = dict(revision=3,
            nominal_compression_m=WIPE_COMPRESSION_DEPTH_M,
            compression_jitter_m=WIPE_COMPRESSION_JITTER_M,
            compensate_grasp_height=True, hold_contact_height_during_wipe=True)
    return identity


def selection_cache_path(identity, *, root=None):
    root = Path(os.environ.get("TASKTOL_NOMINAL_PLAN_TABLE_ROOT", str(DEFAULT_ROOT))) if root is None else Path(root)
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in identity["scene"])
    return root/slug/identity["robot_uid"]/f"seed_{identity['episode_seed']:05d}"/f"{digest(identity)}.json"


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k:v for k,v in value.items() if k != "content_sha256"}
    payload = dict(clean, content_sha256=digest(clean))
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name+".", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


@contextmanager
def entry_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_entry(path, identity, candidates):
    entry = json.loads(path.read_text())
    if entry.get("content_sha256") != digest({k:v for k,v in entry.items() if k != "content_sha256"}):
        raise ValueError("Nominal table contents changed/corrupt")
    if entry["identity"] != identity or entry["entry_key"] != digest(identity) or entry["candidates"] != candidates:
        raise ValueError("Nominal table identity changed/corrupt; refusing cached results")
    return entry


def _selected_result(entry, path, *, cache_hit, started):
    selected = None if entry['selected_index'] is None else entry['observations'][entry['selected_index']]
    metadata = dict(path=str(path.resolve()), entry_key=entry["entry_key"], cache_hit=cache_hit,
        selected_candidate_id=entry["selected_candidate_id"],
        selection_metric=SELECTION_METRIC,
        selected_total_joint_travel_rad=None if selected is None else selected['total_joint_travel_rad'],
        rotation_score_policy=dict(entry['identity']['rotation_score_policy']),
        selected_rotation_deg=None if selected is None else selected['total_rotation_deg'],
        selected_score=None if selected is None else selected['selection_score'],
        selection_reason=entry.get("selection_reason", "feasible_joint_travel_plus_rotation"),
        selected_nominal_feasible=bool(selected and selected['feasible']),
        feasible_candidates=sum(o["feasible"] for o in entry["observations"]),
        candidate_count=len(entry["candidates"]),
        screening_workers=entry.get("screening_workers", 1),
        validation_seconds_this_call=time.monotonic()-started)
    if entry["selected_index"] is None:
        error = PlanningInfeasibleError("No candidate passes strict nominal FK screening")
        error.context["nominal_plan_table"] = metadata
        raise error
    return entry["selected_index"], metadata


def _initialize_screening_worker():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _screening_pool(workers):
    # Reuse imports/solver runtimes across cold seeds.
    # Warm cache reads never request a pool. UI owns the whole process group.
    global _SCREENING_POOL_FINALIZER
    with _SCREENING_POOL_LOCK:
        if workers not in _SCREENING_POOLS:
            _SCREENING_POOLS[workers] = ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_screening_worker)
        if _SCREENING_POOL_FINALIZER is None or not _SCREENING_POOL_FINALIZER.still_active():
            # Spawned collection workers run multiprocessing finalizers BEFORE
            # threading shutdown. Waiting for children there would deadlock:
            # the executor's threading exit hook has not sent its sentinels yet.
            # Close our pools explicitly before multiprocessing joins children.
            _SCREENING_POOL_FINALIZER = Finalize(
                None, close_screening_workers, exitpriority=20,
            )
        return _SCREENING_POOLS[workers]


def close_screening_workers():
    """Release this process's candidate workers (no scene/execution state)."""
    with _SCREENING_POOL_LOCK:
        pools = list(_SCREENING_POOLS.values())
        _SCREENING_POOLS.clear()
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


def _validate_candidate(spec, prepared, definition, initial_rotation, tolerance_deg):
    """One isolated call to the public backend, returning only a cache summary."""
    from ..counterfactual.runner import run_counterfactual_trial

    result = run_counterfactual_trial(spec, prepared_nominal=prepared)
    if result.termination_reason.startswith("runner_error:"):
        raise RuntimeError(f"Nominal candidate backend error: {result.evaluation}")
    e = result.evaluation
    velocity = result.trajectory_kinematics.get("mean_joint_velocity_rad_s")
    travel = result.trajectory_kinematics.get("total_joint_travel_rad")
    complete = bool(e.get("trajectory_complete"))
    feasible = bool(result.task_success and complete and e.get("optimizer_feasible")
                    and velocity is not None and math.isfinite(velocity)
                    and travel is not None and math.isfinite(travel) and travel >= 0)
    minimum = e["robot_table_clearance"]["distribution"]["minimum_m"]
    observation = dict(candidate_id=definition["candidate_id"], feasible=feasible,
        trajectory_complete=complete, velocity_rad_s=velocity if feasible else None,
        total_joint_travel_rad=travel if feasible else None, minimum_clearance_m=minimum,
        annotation_5mm_pass=bool(feasible and minimum is not None and minimum >= .005),
        optimizer_feasible=bool(e.get("optimizer_feasible")),
        termination_reason=result.termination_reason,
        hard_failure_reasons=list(e.get("hard_failure_reasons", [])),
        physics_steps=e.get("physics_steps"),
        validation_domain=e.get("validation_domain", "full_rate_nominal_fk"),
        sampled_frame_indices=e.get("sampled_frame_indices", []),
        solved_samples=result.trajectory_kinematics.get("samples", 0),
        reference_sha256=digest(frame_records(prepared.frames)),
        validation_tolerance_deg=tolerance_deg,
        **candidate_rotation_summary(prepared.frames, initial_rotation))
    observation["selection_score"] = nominal_selection_score(observation) if feasible else None
    return observation


def _planner_failure(definition, error, tolerance_deg, initial_rotation):
    observation = dict(candidate_id=definition["candidate_id"], feasible=False,
        trajectory_complete=False, optimizer_feasible=False,
        velocity_rad_s=None, total_joint_travel_rad=None, minimum_clearance_m=None,
        annotation_5mm_pass=False, termination_reason="planner_infeasible",
        hard_failure_reasons=["planner_infeasible"], planner_message=str(error),
        physics_steps=0, sampled_frame_indices=[], solved_samples=0,
        reference_sha256=None, grasp_rotation_deg=None, placement_rotation_deg=None,
        total_rotation_deg=None, selection_score=None, validation_tolerance_deg=tolerance_deg)
    rotations = error.context.get("nominal_reference_rotations")
    if rotations is not None:
        observation.update(reference_rotation_summary(initial_rotation, *rotations))
    return observation


def _screen_pass(entry, path, model, data, handles, *, scene, task, episode_seed,
                 candidates, build_candidate, settings, workers, built):
    import mujoco
    from ..counterfactual.runner import CounterfactualTrialSpec, PreparedNominalTrial
    from .robot import ee_pose

    initial_rotation = ee_pose(data, handles)[1]
    tolerance_deg = 0
    pending = entry.setdefault("pending_observations", {})
    remaining = [i for i in range(len(entry["observations"]), len(candidates)) if str(i) not in pending]

    def store_observation(index, observation):
        pending[str(index)] = observation
        while str(len(entry["observations"])) in pending:
            entry["observations"].append(pending.pop(str(len(entry["observations"]))))
        write_record(path, entry)

    # Mutable task builders stay serial. Only immutable scene/reference
    # snapshots cross process boundaries; IPOPT instances never share threads.
    pool = None
    futures = {}
    try:
        if workers > 1 and len(remaining) > 1:
            pool = _screening_pool(workers)
            frozen_model = copy.copy(model)
            frozen_data = mujoco.MjData(frozen_model)
            mujoco.mj_copyData(frozen_data, frozen_model, data)
        else:
            frozen_model, frozen_data = model, data
        for index in remaining:
            definition = candidates[index]
            if index not in built:
                try:
                    built[index] = build_candidate(index)
                except PlanningInfeasibleError as error:
                    built[index] = error
            candidate = built[index]
            if isinstance(candidate, PlanningInfeasibleError):
                store_observation(index, _planner_failure(definition, candidate, tolerance_deg, initial_rotation))
                continue
            frames = tuple(replace(frame, position=frame.position.copy(), rotation=frame.rotation.copy())
                           for frame in candidate["frames"])
            if not frames:
                raise ValueError("Nominal candidate has no reference frames")
            spec = CounterfactualTrialSpec(scene=scene, task=task, robot_uid=handles.spec.uid,
                seed=episode_seed, maximum_frames=max(600, len(frames)), **settings)
            prepared = PreparedNominalTrial(frozen_model, frozen_data, handles.spec.uid, frames,
                dict(semantic_goal_perturbations=candidate["perturbations"],
                     nominal_candidate_id=definition["candidate_id"], nominal_selection_entry_key=entry["entry_key"]))
            args = (spec, prepared, definition, initial_rotation, tolerance_deg)
            if pool is None:
                store_observation(index, _validate_candidate(*args))
            else:
                futures[pool.submit(_validate_candidate, *args)] = index
        for future in as_completed(futures):
            store_observation(futures[future], future.result())
    except BrokenProcessPool:
        with _SCREENING_POOL_LOCK:
            for key, candidate_pool in list(_SCREENING_POOLS.items()):
                if candidate_pool is pool:
                    _SCREENING_POOLS.pop(key)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if pool is not None:
            for future in futures:
                future.cancel()


def _legacy_entry(identity, candidates):
    """Reuse strict observations; never import the relaxed v6 policy."""
    for version in (7, 5):
        legacy = dict(identity, schema_version=version)
        if version == 5:
            legacy.pop("screening_policy")
        else:
            legacy["screening_policy"] = dict(strict_angle_deg=0, failure_policy="report_infeasible")
        root = Path(os.environ.get("TASKTOL_NOMINAL_PLAN_TABLE_ROOT", str(DEFAULT_ROOT.parent / f"v{version}")))
        path = selection_cache_path(legacy, root=root)
        if path.exists():
            entry = _read_entry(path, legacy, candidates)
            # Old CPU rejections contain no geometric orientation targets.
            # Rebuild these once under the new policy rather than invent angles.
            if (not entry["complete"] or entry["selected_index"] is None) and any(
                o.get("total_rotation_deg") is None for o in entry["observations"]
            ):
                continue
            return entry, path
    return None, None


def select_nominal_candidate(model, data, handles, *, scene, task, episode_seed,
                             candidates, build_candidate, planning_conditions=None,
                             options=None, domain_randomization_enabled=False,
                             cache_only=False):
    """Return the winning index; the caller generates its LIVE reference.

    Candidate IDs and order are supplied by the task. No simulation, planner,
    controller or tolerance implementation is duplicated here.
    """
    settings = dict(sqp_loss_profile="delta_q_squared_only", sqp_loss_weights=(1.,0.,0.),
        sqp_conditioning_eta_low=.05, sqp_conditioning_eta_high=.10,
        sqp_conditioning_sigma_epsilon=1e-12, sqp_conditioning_fd_relative_step=1e-5,
        ipopt_max_iterations=20, sqp_max_time_s=0., execution_mode="kinematic",
        kinematic_record_robot_table_clearance=True,
        kinematic_continue_after_table_contact=False, controller_mode="sqp_specific",
        angle_deg=0, nominal_plan_selection_enabled=False,
        kinematic_sampling_stride=2)
    options = options or {}
    allowed = {"sqp_loss_profile", "sqp_loss_weights", "ipopt_max_iterations",
               "sqp_conditioning_eta_low", "sqp_conditioning_eta_high",
               "sqp_conditioning_sigma_epsilon", "sqp_conditioning_fd_relative_step",
               "kinematic_sampling_stride", "candidate_workers"}
    if set(options)-allowed:
        raise ValueError("Unsupported nominal validation options")
    # Screening fixes PRE/POST release to zero. Validate
    # with one canonical velocity-only objective, independent of the caller's
    # optimizer profile, EMA weight or suspended conditioning configuration.
    # Accept those existing caller fields but never forward them or key the
    # table on them. The ceiling affects COLD solve feasibility; later caller
    # budget changes keep the previously selected discrete index.
    if "ipopt_max_iterations" in options:
        settings["ipopt_max_iterations"] = options["ipopt_max_iterations"]
    if "kinematic_sampling_stride" in options:
        settings["kinematic_sampling_stride"] = options["kinematic_sampling_stride"]
    settings = json.loads(json.dumps(settings))
    workers = options.get("candidate_workers", int(os.environ.get("TASKTOL_NOMINAL_CANDIDATE_WORKERS", "4")))
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("candidate_workers must be an integer in [1, 32]")
    workers = min(workers, len(candidates), max(1, (os.cpu_count() or 2) - 1))
    if not candidates or len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("Require distinct, stable candidate IDs")
    if any(set(c)-{"candidate_id", "metadata"} for c in candidates):
        raise ValueError("Candidate definitions cannot contain trajectories or perturbations")
    serialized = json.loads(json.dumps(candidates, allow_nan=False))
    identity = selection_identity(scene=scene, task=task, robot_uid=handles.spec.uid,
        episode_seed=episode_seed, candidates=serialized,
        domain_randomization_enabled=domain_randomization_enabled)
    key = digest(identity)
    path = selection_cache_path(identity)
    started = time.monotonic()
    # Writes atomically replace the JSON. A complete record is immutable in
    # normal screening, so readers need no lock and never wait behind a solve.
    if path.exists():
        entry = _read_entry(path, identity, serialized)
        if entry["complete"]:
            return _selected_result(entry, path, cache_hit=True, started=started)
    legacy, legacy_path = _legacy_entry(identity, serialized) if not path.exists() else (None, None)
    if legacy is not None and legacy["complete"] and legacy["selected_index"] is not None:
        return _selected_result(legacy, legacy_path, cache_hit=True, started=started)
    if cache_only:
        raise NominalChoiceNotCached(str(path))
    with entry_lock(path):
        if path.exists():
            entry = _read_entry(path, identity, serialized)
            cache_hit = bool(entry["complete"])
        else:
            entry = dict(identity=identity, entry_key=key, candidates=serialized, observations=[], complete=False,
                cache_payload="discrete_selection_and_validation_summaries_only",
                selection_metric=SELECTION_METRIC,
                rotation_score_policy=dict(ROTATION_SCORE_POLICY),
                validation_settings=legacy["validation_settings"] if legacy else settings,
                validation_phase="strict",
                screening_workers=workers, planning_conditions=planning_conditions or {},
                backend_code_sha256=code_digest(),
                runtime_versions={name:importlib.metadata.version(name) for name in ("mujoco", "numpy", "scipy")},
                uv_lock_sha256=hashlib.sha256((REPO_ROOT/"uv.lock").read_bytes()).hexdigest())
            if legacy is not None:
                entry["observations"] = copy.deepcopy(legacy["observations"])
                entry["legacy_provenance"] = dict(path=str(legacy_path.resolve()),
                    content_sha256=legacy["content_sha256"], strict_checkpoints_reused=True)
            cache_hit = False
        # A resumed screening keeps the originally recorded solver budget.
        # Once selected, callers' later budgets do not invalidate the index.
        settings = entry["validation_settings"]
        built = {}
        if not entry["complete"]:
            _screen_pass(entry, path, model, data, handles, scene=scene, task=task,
                episode_seed=episode_seed, candidates=candidates, build_candidate=build_candidate,
                settings=settings, workers=workers, built=built)
            best = rank_nominal_observations(entry['observations'])
            reason = "feasible_joint_travel_plus_rotation"
            if best is None:
                best = rank_reference_rotations(entry['observations'])
                reason = "minimum_reference_rotation" if best is not None else "no_nominal_reference"
            entry.update(complete=True, selected_index=best,
                         selection_reason=reason,
                         selected_nominal_feasible=bool(best is not None and entry['observations'][best]['feasible']),
                         selected_candidate_id=None if best is None else entry['observations'][best]['candidate_id'])
            write_record(path, entry)
    return _selected_result(entry, path, cache_hit=cache_hit, started=started)
