import json

import pytest

from examples.libero.rerun_failed_nominal import failed_selection, summarize


def commit(root, suite, task, seed, success):
    p = root / "episodes" / suite / f"task_{task:02d}" / f"seed_{seed}" / "result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(suite=suite, task_id=task, seed=seed, success=success, complete=True, termination="task_success" if success else "ik_infeasible", steps=10, eval_identity="retry")))
    return p


def test_selection_preserves_failed_task_seed_pairs_and_excludes_successes(tmp_path):
    for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        for t in range(10):
            for z in range(6):
                commit(tmp_path, s, t, z, (s, t, z) not in {("libero_goal", 2, 3), ("libero_10", 8, 0)})
    selected = failed_selection(tmp_path)
    assert {(v["suite"], v["task_id"], v["seed"]) for v in selected} == {("libero_goal", 2, 3), ("libero_10", 8, 0)}
    commit(tmp_path, "libero_10", 10, 0, False)
    with pytest.raises(ValueError, match="240 complete unique"):
        failed_selection(tmp_path)


def test_retry_completion_is_selection_based_and_rejects_unrequested_rollouts(tmp_path):
    request = dict(selected_episodes=[dict(suite="libero_spatial", task_id=4, seed=1)], eval_identity="retry")
    assert not summarize(tmp_path, request)["complete"]
    commit(tmp_path, "libero_spatial", 4, 1, True)
    assert summarize(tmp_path, request)["complete"]
    commit(tmp_path, "libero_spatial", 4, 2, True)
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        summarize(tmp_path, request)
