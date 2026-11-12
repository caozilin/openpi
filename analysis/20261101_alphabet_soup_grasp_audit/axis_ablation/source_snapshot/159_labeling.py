"""Selected search-box labels, with executed peaks retained as diagnostics.

This module performs no replay and never changes the selected PRE/POST pair.
The explicit legacy peak extractor remains available for historical audits.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


RELEASE_PEAK_LABEL_POLICY = "speed_best_complete_rollout_signed_release_peaks_v1"
SEARCH_SOLUTION_LABEL_POLICY = "speed_best_feasible_search_caps_v1"
STAGES = ("pregrasp", "postgrasp")


def _peaks(values):
    result = [float(v) for v in values]
    if len(result) != 6 or any(not math.isfinite(v) or v < 0 for v in result):
        raise ValueError("Labels require six finite nonnegative release peaks")
    return result


def release_peak_label(raw):
    """Legacy policy: extract per-stage peaks for historical experiment audits.

    PRE and POST are both measured in the final selected rollout, not spliced
    from the earlier PRE=0 POST search. Long tasks retain separate subtask rows.
    Missing statistics are errors, not evidence of zero release.
    """
    rollout = raw["rollout"]
    if not raw["success"] or not raw["trajectory_complete"]:
        raise ValueError("Only qualified complete trajectories may supply labels")
    if (not math.isfinite(raw["velocity_rad_s"])
            or raw["velocity_rad_s"] < 0):
        raise ValueError("A label requires a finite full-trajectory velocity")
    index = rollout.get("semantic_subtask_index", 0)
    statistics = rollout.get("subtask_release_statistics")
    subtasks = []
    if statistics is not None:
        for i, item in enumerate(statistics):
            stages = {}
            for stage in STAGES:
                stages[stage] = _peaks([
                    item[stage][side][axis]
                    for axis in range(3) for side in ("negative", "positive")
                ])
            subtasks.append(dict(subtask_index=i, **stages))
        if not 0 <= index < len(subtasks):
            raise ValueError("Selected semantic subtask has no release statistics")
    else:
        stages = {}
        for stage in STAGES:
            stat = rollout["release_statistics"][stage]
            if stat["samples"] <= 0:
                raise ValueError("A completed stage must have release samples")
            stages[stage] = _peaks(stat["peak_directional_deg"])
        subtasks.append(dict(subtask_index=index, **stages))
    selected = next(s for s in subtasks if s["subtask_index"] == index)
    return dict(
        label_policy=RELEASE_PEAK_LABEL_POLICY,
        pre_caps_deg=list(selected["pregrasp"]),
        post_caps_deg=list(selected["postgrasp"]),
        search_pre_caps_deg=list(raw["pre_caps_deg"]),
        search_post_caps_deg=list(raw["post_caps_deg"]),
        velocity_rad_s=raw["velocity_rad_s"],
        semantic_subtask_index=index, subtask_release_labels_deg=subtasks,
    )


def search_solution_label(raw):
    """Label the selected feasible physical box, without clipping or replay.

    Only the annotated subtask's search caps are known here. Extrema for other
    subtasks are diagnostics, never substitute labels for their own searches.
    """
    if not raw["success"] or not raw["trajectory_complete"]:
        raise ValueError("Only qualified complete trajectories may supply labels")
    velocity = raw["velocity_rad_s"]
    if not math.isfinite(velocity) or velocity < 0:
        raise ValueError("A label requires a finite full-trajectory velocity")
    caps = {}
    for stage in ("pre", "post"):
        values = [float(v) for v in raw[f"{stage}_caps_deg"]]
        if len(values) != 6 or any(not math.isfinite(v) or not 0 <= v <= 30 for v in values):
            raise ValueError("Search labels require six finite caps in [0, 30] degrees")
        caps[stage] = values
    rollout = raw.get("rollout", {})
    index = rollout.get("semantic_subtask_index", 0)
    if type(index) is not int or index < 0:
        raise ValueError("Selected semantic subtask must be a nonnegative integer")
    label = dict(
        label_policy=SEARCH_SOLUTION_LABEL_POLICY,
        pre_caps_deg=caps["pre"], post_caps_deg=caps["post"],
        search_pre_caps_deg=list(caps["pre"]), search_post_caps_deg=list(caps["post"]),
        velocity_rad_s=velocity, semantic_subtask_index=index,
        subtask_tolerance_labels_deg=[dict(subtask_index=index,
            pregrasp=list(caps["pre"]), postgrasp=list(caps["post"]))],
    )
    if (rollout.get("subtask_release_statistics") is not None
            or "release_statistics" in rollout):
        peaks = release_peak_label(raw)
        label.update(
            actual_pre_release_peaks_deg=peaks["pre_caps_deg"],
            actual_post_release_peaks_deg=peaks["post_caps_deg"],
            subtask_release_peaks_deg=peaks["subtask_release_labels_deg"],
        )
    return label


def label_from_observation(row):
    """Load the selected observation with provenance checks, without replay."""
    path = Path(row["raw_trial_path"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["raw_trial_sha256"]:
        raise ValueError("Selected label trial changed")
    raw = json.loads(content)
    for stage in ("pre", "post"):
        key = f"{stage}_caps_deg"
        # Some GP checkpoints call POST coordinates pre_caps_deg. Prefer their
        # explicitly recorded physical coordinates, never the legacy GP field.
        expected = row.get(f"actual_{key}", row.get(key))
        if expected is not None and expected != raw[key]:
            raise ValueError("Label trial differs from the selected physical box")
    if raw["velocity_rad_s"] != row["velocity_rad_s"]:
        raise ValueError("Label trial differs from the selected velocity")
    return dict(search_solution_label(raw), raw_trial_path=str(path.resolve()),
                raw_trial_sha256=row["raw_trial_sha256"])
