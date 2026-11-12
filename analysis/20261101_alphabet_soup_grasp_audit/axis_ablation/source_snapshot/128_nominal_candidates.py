"""Discrete symmetry choices for curated training tasks (no execution code)."""
from functools import lru_cache

import mujoco
import numpy as np


ELIGIBLE_TASK_IDS = frozenset({
    "geometry_region_cylinder_lying", "geometry_plate_cylinder_lying",
    "geometry_plate_cylinder_lying_stable", "cylinder_to_narrow_box",
    "geometry_plate_box_upright", "geometry_plate_box_upright_stable",
    "geometry_region_box_upright", "geometry_plate_box_lying",
    "geometry_plate_box_lying_stable", "geometry_region_box_lying",
    "box_to_narrow_box", "adjust_cylindrical_bottle", "adjust_rectangular_bottle",
    "wipe_whiteboard", "close_handle_pot_lid", "apple_then_banana_to_tray",
    "stack_yellow_red_purple", "round_handle_pot_pear_sequence",
})


@lru_cache(maxsize=1)
def _task_ids():
    from ...data.task_sets import training_tasks
    return {(s.runtime_scene, t.runtime_task): t.task_id for s, t in training_tasks()}


def supports_nominal_selection(scene, task=None):
    return any(s == scene and (task is None or t == task) and uid in ELIGIBLE_TASK_IDS
               for (s, t), uid in _task_ids().items())


def _box_grasp_count(model, data, prefix, *, lying):
    bodies = [i for i in range(model.nbody)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(prefix)
              and np.any(model.geom_contype[np.flatnonzero(model.geom_bodyid == i)] != 0)]
    if len(bodies) != 1:
        raise RuntimeError(f"Expected one active {prefix} box, found {bodies}")
    body = bodies[0]
    geom = int(model.body_geomadr[body])
    size = model.geom_size[geom]
    axes = [i for i in range(3) if abs(data.xmat[body].reshape(3, 3)[2, i]) < .25] if lying else [0, 1]
    if len(axes) != 2:
        raise RuntimeError("Box must expose two horizontal pinch axes")
    narrow = min(size[i] for i in axes)
    return 2 * sum(size[i] <= narrow + 1e-6 for i in axes)


def nominal_candidate_definitions(model, data, scene, task, subtask_index=0):
    uid = _task_ids().get((scene, task))
    if uid not in ELIGIBLE_TASK_IDS:
        return []
    grasp_count, place_count = 2, 1
    multi = uid in {"apple_then_banana_to_tray", "stack_yellow_red_purple", "round_handle_pot_pear_sequence"}
    if uid == "apple_then_banana_to_tray":
        if subtask_index != 1:
            return []
        place_count = 2
    elif uid == "round_handle_pot_pear_sequence":
        if subtask_index != 1:
            return []
    elif uid == "stack_yellow_red_purple":
        if subtask_index not in (0, 1):
            return []
        grasp_count, place_count = 4, 4
    elif subtask_index != 0:
        return []
    elif uid == "box_to_narrow_box" or uid.startswith(("geometry_plate_box_", "geometry_region_box_")):
        grasp_count = _box_grasp_count(model, data,
            "narrow_box_block_" if uid == "box_to_narrow_box" else "paper_object_box_",
            lying="upright" not in uid)
        if "region" in uid or "narrow" in uid:
            place_count = 4 if "upright" in uid else 2
    elif uid in {"geometry_region_cylinder_lying", "cylinder_to_narrow_box",
                 "adjust_cylindrical_bottle", "adjust_rectangular_bottle", "wipe_whiteboard"}:
        place_count = 2
    prefix = f"subtask{subtask_index}_" if multi else ""
    return [dict(candidate_id=f"{prefix}grasp{g}_place{p}",
                 metadata=dict(grasp_index=g, placement_index=p,
                               **({"subtask_index": subtask_index} if multi else {})))
            for g in range(grasp_count) for p in range(place_count)]
