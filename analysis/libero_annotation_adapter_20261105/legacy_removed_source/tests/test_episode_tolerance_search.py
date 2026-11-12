from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from annotate_tolerance import dataset_targets
from tasktol.counterfactual.search import (
    CounterfactualSearchConfig,
    EpisodeAnnotationTarget,
    ToleranceAnnotationError,
    search_task_tolerance,
)
from tasktol.tolerance.annotations import static_stage_rotation_masks
from tasktol.tolerance.context import long_task_subtask_stage_rotation_masks


def _declared_masks(spec):
    long_masks = long_task_subtask_stage_rotation_masks(spec.task)
    if long_masks:
        return [[list(pre), list(post)] for _, (pre, post) in long_masks]
    masks = static_stage_rotation_masks(spec.scene, spec.task)
    assert masks is not None
    return [[list(masks[0]), list(masks[1])]]


def _result(spec, success: bool, *, negative=0.0, positive=0.0):
    extrema = [{
        "subtask_index": 0,
        "pregrasp": {
            "negative": [0.0, negative, 0.0],
            "positive": [0.0, positive, 0.0],
        },
        "postgrasp": {
            "negative": [0.0, negative, 0.0],
            "positive": [0.0, positive, 0.0],
        },
    }]
    return SimpleNamespace(
        spec=spec,
        success=success,
        termination_reason=("kinematic_success" if success else "optimizer_infeasible"),
        hard_failure_reasons=(() if success else ("optimizer_infeasible",)),
        evaluation={
            "backend": "kinematic_collision_checked",
            "hard_failure_details": [],
            "sqp_subtask_stage_release_extrema_deg": extrema,
            "executed_frames": 10,
            "expected_frames": 10,
            "declared_subtask_stage_rotation_masks": _declared_masks(spec),
        },
    )


def _bounds(spec):
    return (
        spec.sqp_subtask_stage_negative_limits_deg[0],
        spec.sqp_subtask_stage_positive_limits_deg[0],
    )


def test_peak_guided_search_rechecks_largest_signed_peak(tmp_path: Path) -> None:
    target = EpisodeAnnotationTarget(4, tmp_path / "seed_04")
    config = CounterfactualSearchConfig(
        scene="geometry_plate_cylinder_lying",
        task="geometry_plate_cylinder_lying",
        targets=(target,),
    )

    def fake_run(spec, **_kwargs):
        negative, positive = _bounds(spec)
        pre_neg, post_neg = negative
        pre_pos, post_pos = positive
        if not any((*pre_neg, *post_neg, *pre_pos, *post_pos)):
            return _result(spec, True)
        if any(pre_neg) or any(pre_pos):
            return _result(spec, True)
        zero_axes = [
            axis for axis in range(3)
            if _declared_masks(spec)[0][1][axis]
            and post_neg[axis] == 0 and post_pos[axis] == 0
        ]
        if zero_axes:
            return _result(spec, zero_axes == [1], negative=28.0, positive=12.0)
        if post_neg[1] == 30 and post_pos[1] == 30:
            return _result(spec, False, negative=28.0, positive=12.0)
        if post_neg[1] == 25 and post_pos[1] == 30:
            return _result(spec, False, negative=7.0, positive=12.0)
        assert post_neg[1] == 25 and post_pos[1] == 10
        return _result(spec, True, negative=7.0, positive=9.0)

    with patch(
        "tasktol.counterfactual.search.run_counterfactual_trial",
        side_effect=fake_run,
    ):
        (path,) = search_task_tolerance(config)

    payload = json.loads(path.read_text(encoding="utf-8"))
    post = payload["subtasks"][0]["postgrasp"]
    assert post["negative"] == [30, 25, 30]
    assert post["positive"] == [30, 10, 30]
    assert payload["subtasks"][0]["pregrasp"] == {
        "negative": [30, 30, 0],
        "positive": [30, 30, 0],
    }
    reductions = payload["stage_results"][0]["history"][-2:]
    assert [item["selected_side"] for item in reductions] == [
        "negative", "positive",
    ]
    assert [item["selected_bound_deg"] for item in reductions] == [25, 10]


def test_ambiguous_corrective_axes_raise_and_write_failure(tmp_path: Path) -> None:
    target = EpisodeAnnotationTarget(6, tmp_path / "seed_06")
    config = CounterfactualSearchConfig(
        scene="geometry_plate_cylinder_lying",
        task="geometry_plate_cylinder_lying",
        targets=(target,),
    )
    calls = 0

    def fake_run(spec, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _result(spec, True)
        if calls == 2:
            return _result(spec, False, negative=20.0, positive=10.0)
        return _result(spec, True, negative=20.0, positive=10.0)

    with (
        patch(
            "tasktol.counterfactual.search.run_counterfactual_trial",
            side_effect=fake_run,
        ),
        pytest.raises(
            ToleranceAnnotationError,
            match="exactly one corrective axis",
        ),
    ):
        search_task_tolerance(config)

    failure = json.loads((
        target.output_dir / "tolerance_annotation_failure.json"
    ).read_text(encoding="utf-8"))
    assert failure["context"]["successful_zeroed_axes"] == ["x", "y", "z"]


def test_multi_subtasks_are_annotated_last_to_first(tmp_path: Path) -> None:
    target = EpisodeAnnotationTarget(8, tmp_path / "seed_08")
    config = CounterfactualSearchConfig(
        scene="banana_apple_to_tray",
        task="apple_then_banana_to_tray",
        targets=(target,),
    )
    observed_bounds = []

    def fake_run(spec, **_kwargs):
        observed_bounds.append((
            spec.sqp_subtask_stage_negative_limits_deg,
            spec.sqp_subtask_stage_positive_limits_deg,
        ))
        count = len(spec.sqp_subtask_stage_rotation_masks)
        result = _result(spec, True)
        result.evaluation["sqp_subtask_stage_release_extrema_deg"] = [
            {
                "subtask_index": index,
                "pregrasp": {
                    "negative": [0.0, 0.0, 0.0],
                    "positive": [0.0, 0.0, 0.0],
                },
                "postgrasp": {
                    "negative": [0.0, 0.0, 0.0],
                    "positive": [0.0, 0.0, 0.0],
                },
            }
            for index in range(count)
        ]
        return result

    with patch(
        "tasktol.counterfactual.search.run_counterfactual_trial",
        side_effect=fake_run,
    ):
        (path,) = search_task_tolerance(config)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [
        (item["subtask_index"], item["stage"])
        for item in payload["stage_results"]
    ] == [
        (1, "postgrasp"),
        (1, "pregrasp"),
        (0, "postgrasp"),
        (0, "pregrasp"),
    ]
    # When earlier subtask 0 starts, both stages of later subtask 1 are fixed
    # at their already-finalized values.
    earlier_post_negative = observed_bounds[3][0]
    assert earlier_post_negative[1] == (
        (30.0, 30.0, 0.0),
        (30.0, 30.0, 30.0),
    )


def test_dataset_targets_write_inside_each_episode(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    episode = task_dir / "episode_0004_seed_4"
    episode.mkdir(parents=True)
    (episode / "trajectory.json").write_text(
        json.dumps({
            "episode": {
                "domain_randomization": {
                    "seed_routing": {"environment_seed": -1},
                },
            },
            "trajectory": [{"index": 0}],
        }),
        encoding="utf-8",
    )
    (task_dir / "manifest.json").write_text(
        json.dumps({
            "episodes": [{
                "episode_index": 4,
                "seed": 4,
                "path": episode.name,
            }],
        }),
        encoding="utf-8",
    )

    (target,) = dataset_targets(task_dir, selected_seeds={4})

    assert target.seed == 4
    assert target.episode_index == 4
    assert target.output_dir == episode
    assert target.trajectory_path == episode / "trajectory.json"
