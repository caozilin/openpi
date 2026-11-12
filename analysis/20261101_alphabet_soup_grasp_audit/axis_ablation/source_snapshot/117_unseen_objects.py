"""Held-out object evaluators built from the established task semantics."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .fruit_receptacles import FruitReceptacleEvaluator
from .paper_placement import PaperPlacementEvaluator
from .pick_place import BlockToPlateEvaluator


_SPORTS_BALL_NAMES = (
    "ycb_baseball",
    "ycb_tennis_ball",
    "ycb_golf_ball",
)


def _active_root(model: mujoco.MjModel, names: tuple[str, ...]) -> str:
    active: list[str] = []
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        subtree = []
        for candidate in range(model.nbody):
            current = candidate
            while current > body_id:
                current = int(model.body_parentid[current])
            if current == body_id:
                subtree.append(candidate)
        geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, subtree))
        if np.any(model.geom_contype[geom_ids] != 0):
            active.append(name)
    if len(active) != 1:
        raise RuntimeError(f"Expected one active held-out object, found {active}")
    return active[0]


def unseen_fruit_evaluator(
    model: mujoco.MjModel,
    data: mujoco.MjData | None,
    *,
    fruit: str,
    receptacle: str,
) -> FruitReceptacleEvaluator:
    other_fruit = "fruit_lemon" if fruit == "apple" else "fruit_apple"
    other_receptacle = (
        "ycb_bowl_white" if receptacle == "plate" else "ycb_plate_white"
    )
    return FruitReceptacleEvaluator(
        model,
        data,
        fruit=fruit,
        receptacle=receptacle,
        fruit_body_name=f"fruit_{fruit}",
        receptacle_body_name=f"ycb_{receptacle}_white",
        distractor_body_names=(other_fruit, other_receptacle),
        plate_radius_limit_m=0.105,
        bowl_radius_limit_m=0.070,
    )


def unseen_sports_ball_evaluator(
    model: mujoco.MjModel,
    data: mujoco.MjData | None,
) -> BlockToPlateEvaluator:
    ball = _active_root(model, _SPORTS_BALL_NAMES)
    return BlockToPlateEvaluator(
        model,
        data,
        target_body_name=ball,
        target_geom_name=f"{ball}_geom",
        distractor_body_names=(),
        topple_body_names=(),
        require_target_upright=False,
        require_stability=False,
        plate_body_name="ycb_plate_white",
        plate_geom_name="ycb_plate_white_top",
    )


def unseen_upright_box_plate_evaluator(
    model: mujoco.MjModel,
    data: mujoco.MjData | None,
) -> BlockToPlateEvaluator:
    return BlockToPlateEvaluator(
        model,
        data,
        target_body_name="ycb_sugar_box",
        target_geom_name="ycb_sugar_box_geom",
        distractor_body_names=(),
        topple_body_names=(),
        plate_body_name="ycb_plate_white",
        plate_geom_name="ycb_plate_white_top",
    )


class UnseenUprightBoxRegionEvaluator(PaperPlacementEvaluator):
    """Reuse white-region footprint/height evaluation for the YCB sugar box."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model,
            data,
            object_body_name="ycb_sugar_box",
            object_geom_name="ycb_sugar_box_geom",
            decal_body_name="unseen_box_white_region",
            decal_geom_name="unseen_box_white_region_geom",
            object_kind="box",
            upright=True,
            **kwargs,
        )


__all__ = (
    "UnseenUprightBoxRegionEvaluator",
    "unseen_fruit_evaluator",
    "unseen_sports_ball_evaluator",
    "unseen_upright_box_plate_evaluator",
)
