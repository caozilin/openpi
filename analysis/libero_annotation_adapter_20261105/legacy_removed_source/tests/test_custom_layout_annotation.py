from pathlib import Path
from unittest.mock import patch
import json
import pytest

from annotate_custom_layouts import annotate_case, discover
from tasktol.simulation.layouts import CUSTOM_LAYOUT_ROOT
from tasktol.counterfactual.search import (
    CounterfactualSearchConfig, EpisodeAnnotationTarget, search_task_tolerance,
)
from tests.test_episode_tolerance_search import _result


def test_discovery_only_registered_training_layouts():
    cases = discover(CUSTOM_LAYOUT_ROOT)
    assert cases
    assert len({(c["task_id"], c["source_path"]) for c in cases}) == len(cases)
    assert all(Path(c["source_path"]).is_file() for c in cases)


def test_search_accepts_shared_backend_callback_and_records_context(tmp_path):
    config = CounterfactualSearchConfig(
        scene="geometry_region_cylinder_lying",
        task="geometry_region_cylinder_lying", tolerance_step_deg=2,
        targets=(EpisodeAnnotationTarget(0, tmp_path / "annotation"),),
    )
    context = {"custom_layout_path": "/tmp/layout.json", "clearance": 0.005}
    with patch("tasktol.counterfactual.search.run_counterfactual_trial") as legacy:
        paths = search_task_tolerance(
            config, trial_runner=lambda spec, **kw: _result(spec, True),
            trial_context=context,
        )
    legacy.assert_not_called()
    payload = json.loads(paths[0].read_text())
    assert payload["trial_context"] == context
    assert payload["method"] == "per_episode_peak_guided_2deg_search"
    assert payload["search_history"]


@pytest.mark.parametrize("rejected,status,count", [(0, "complete", 3), (1, "failed", 1)])
def test_custom_layout_batch_routes_to_formal_backend(tmp_path, rejected, status, count):
    case = discover(CUSTOM_LAYOUT_ROOT)[0]
    runtime_paths = set()
    def fake_condition(*args, **kwargs):
        assert args[7] not in runtime_paths
        runtime_paths.add(args[7])
        assert kwargs["robot_table_clearance_failure_m"] == 0.005
        assert kwargs["tolerance_frame_mode"] == "realtime_nominal"
        assert kwargs["objective_settings_override"].ema_release_loss_weight == 0
        assert kwargs["ipopt_settings"].max_cpu_time_s == 1e6
        return {
            "task_success": True, "optimizer_rejected_frames": rejected,
            "termination_reason": "success", "frames": 10,
            "declared_stage_rotation_masks": [[1, 1, 1], [1, 1, 1]],
            "trajectory_metric_domains": {},
            "release_statistics": {stage: {"peak_directional_deg": [0]*6}
                                   for stage in ("pregrasp", "postgrasp")},
        }
    with patch("annotate_custom_layouts._run_condition", side_effect=fake_condition):
        result = annotate_case(case, str(tmp_path), 2)
    assert result["status"] == status, result
    assert result["trial_count"] == count
    if rejected:
        failure = json.loads(Path(result["search_failure"]).read_text())
        assert failure["search_history"][0]["termination_reason"] == "optimizer_failure"
