"""Background scheduling for the shared planner; no rollout/physics loop."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import copy
import multiprocessing
import os
import signal

import mujoco

from ..simulation.nominal_plan_table import NominalChoiceNotCached
from ..tasks.planning.types import PreparedTaskPlan


class PreparationPending(Exception):
    """A nonblocking poll has no completed plan yet."""


def _initialize_worker():
    # Candidate validators inherit this owned group. Cancelling a UI request
    # must stop the complete preparation tree, including in-flight validators.
    if hasattr(os, "setsid"):
        os.setsid()
    os.environ.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    if hasattr(os, "nice"):
        os.nice(5)


def _prepare(model, data, robot_uid, scene, task, seed, iterations, table_root,
             domain_randomization_enabled, *, cache_only=False,
             subtask_index=0, initial_position=None, initial_rotation=None, perturbations=None):
    from ..simulation.robot import ee_pose, find_handles
    from ..simulation.robots import get_robot_spec
    from ..tasks.planning.presets import PresetTaskExecutor
    from ..tolerance.context import resolve_task_tolerance_context

    if not cache_only:
        if table_root is not None:
            os.environ["TASKTOL_NOMINAL_PLAN_TABLE_ROOT"] = table_root
        else:
            os.environ.pop("TASKTOL_NOMINAL_PLAN_TABLE_ROOT", None)
    handles = find_handles(model, get_robot_spec(robot_uid))
    mujoco.mj_forward(model, data)
    position, rotation = ee_pose(data, handles)
    executor = PresetTaskExecutor(model, handles)
    if subtask_index:
        executor.start_at_subtask(task, scene, data, initial_position, initial_rotation,
            subtask_index=subtask_index, episode_seed=seed, perturbations=perturbations or {},
            domain_randomization_enabled=domain_randomization_enabled,
            nominal_cache_only=cache_only, nominal_validation_options={"ipopt_max_iterations": iterations})
        return PreparedTaskPlan(scene, task, robot_uid, data.qpos.copy(),
            tuple(executor.frames), dict(executor.perturbations.samples),
            dict(executor.last_planning_stats), subtask_index), None
    executor.start(task, scene, data, position, rotation, episode_seed=seed,
                   domain_randomization_enabled=domain_randomization_enabled,
                   nominal_cache_only=cache_only,
                   nominal_validation_options={"ipopt_max_iterations": iterations})
    prepared = PreparedTaskPlan(scene, task, robot_uid, data.qpos.copy(),
        tuple(executor.frames), dict(executor.perturbations.samples),
        dict(executor.last_planning_stats))
    context = resolve_task_tolerance_context(model, data, handles, scene, task,
        position, rotation, episode_seed=seed, planned_frames=prepared.frames)
    return prepared, context


class BackgroundNominalPreparation:
    """Cached choice locally; only cold screening uses an isolated worker."""

    def __init__(self):
        self.pool = None
        self.future = None
        self.key = None

    def request(self, key, model, data, *, robot_uid, scene, task, seed, iterations,
                domain_randomization_enabled=False, subtask_index=0,
                initial_position=None, initial_rotation=None, perturbations=None):
        self.cancel()
        self.key = key
        extra = (dict(subtask_index=subtask_index, initial_position=initial_position.copy(),
                      initial_rotation=initial_rotation.copy(), perturbations=dict(perturbations or {}))
                 if subtask_index else {})
        try:
            # The SAME planner regenerates live targets. Cache-only selection
            # cannot solve candidates or wait on a screening lock; a miss
            # schedules the expensive branch below instead.
            result = _prepare(model, data, robot_uid, scene, task, seed, iterations,
                os.environ.get("TASKTOL_NOMINAL_PLAN_TABLE_ROOT"),
                domain_randomization_enabled, cache_only=True, **extra)
        except NominalChoiceNotCached:
            pass
        except Exception as error:
            # Keep the existing asynchronous error contract: callers display
            # failures from poll(), including corrupt/all-failed cached tables.
            self.future = Future()
            self.future.set_exception(error)
            return
        else:
            self.future = Future()
            self.future.set_result(result)
            return
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=1,
                mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_worker)
        # The queue serializes in its own thread; give it immutable clones.
        frozen_model = copy.copy(model)
        frozen_data = mujoco.MjData(frozen_model)
        mujoco.mj_copyData(frozen_data, frozen_model, data)
        self.future = self.pool.submit(_prepare, frozen_model, frozen_data,
            robot_uid, scene, task, seed, iterations,
            os.environ.get("TASKTOL_NOMINAL_PLAN_TABLE_ROOT"), domain_randomization_enabled, **extra)

    def poll(self):
        if self.future is None or not self.future.done():
            raise PreparationPending()
        return self.future.result()

    def cancel(self):
        if self.future is not None and not self.future.done() and not self.future.cancel():
            self._stop_pool()
        self.future = None
        self.key = None

    def close(self):
        self.cancel()
        self._stop_pool()

    def _stop_pool(self):
        if self.pool is not None:
            # Python 3.12 has no public ProcessPool termination API. Snapshot
            # only this owned pool's process handles before shutdown; terminate
            # them so interpreter exit/scene switching cannot wait on a solve.
            # Atomic table writes and OS file locks keep checkpoints resumable.
            processes = tuple(getattr(self.pool, "_processes", {}).values())
            for process in processes:
                if process.is_alive():
                    pid = getattr(process, "pid", None)
                    try:
                        if pid is not None and hasattr(os, "killpg") and os.getpgid(pid) == pid:
                            os.killpg(pid, signal.SIGTERM)
                        else:
                            process.terminate()
                    except ProcessLookupError:
                        pass
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
