from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tasktol.counterfactual.runner import (
    CounterfactualTrialSpec,
    PreparedNominalTrial,
    _setup_trial,
    _TrajectoryRecorder,
    run_counterfactual_trial,
)
from tasktol.counterfactual.stored import StoredKinematicContext, StoredStageReference, StoredSubtaskReference
from tasktol.data.libero_annotation.source import discover, sha, write
from tasktol.data.libero_annotation.segmentation import authored_masks
from tasktol.simulation.robot import ee_pose
from tasktol.tasks.planning.types import TaskFrame


def test_shared_backend_uses_stored_scene_at_20hz_without_planner_or_physics():
    spec = CounterfactualTrialSpec(
        scene="Geometry Plate - Upright Box",
        task="Blue block -> plate",
        execution_mode="kinematic",
        angle_deg=0,
        maximum_frames=4,
    )
    model, data, handles, _, _, _ = _setup_trial(spec, kinematic_only=True)
    position, rotation = ee_pose(data, handles)
    stage = StoredStageReference(rotation, np.eye(3), (0, 1, 0))
    context = StoredKinematicContext(
        handles, 0.05, ("work_surface",), position, rotation, (StoredSubtaskReference("source object", stage, stage),)
    )
    q = data.qpos.copy()
    time = data.time
    frames = tuple(
        dict(
            action=[0] * 6 + [-1],
            phase="pregrasp",
            subtask_index=0,
            nominal_target_pose=dict(position_m=position.tolist(), rotation_vector_base_rad=[0, 0, 0]),
            joint_state=dict(position_rad=data.qpos[handles.qpos_ids].tolist()),
            gripper_state=dict(
                finger_positions_m=[0.04, 0.04], raw_joint_positions_m=data.qpos[handles.finger_qpos_ids].tolist()
            ),
        )
        for _ in range(4)
    )
    plan = tuple(TaskFrame(position, rotation, False, "pregrasp") for _ in frames)
    prepared = PreparedNominalTrial(model, data, "panda", plan, {}, context)
    with (
        patch("tasktol.counterfactual.runner.mujoco.mj_step", side_effect=AssertionError("physics advanced")),
        patch(
            "tasktol.counterfactual.runner.PresetTaskExecutor",
            side_effect=AssertionError("replacement planner constructed"),
        ),
    ):
        result = run_counterfactual_trial(
            replace(spec, scene="source-scene", task="source-task"),
            prepared_nominal=prepared,
            dataset_nominal_frames=frames,
        )
    assert result.success, result.evaluation
    assert result.evaluation["control_rate_hz"] == 20 and result.evaluation["physics_steps"] == 0
    assert result.evaluation["executed_frames"] == 4 and result.evaluation["trajectory_complete"]
    np.testing.assert_array_equal(data.qpos, q)
    assert data.time == time


def test_recorded_control_period_is_used_for_kinematic_metrics():
    from tasktol.control.pose import TargetPose

    result = []
    for dt in (0.1, 0.05):
        recorder = _TrajectoryRecorder(control_dt_s=dt)
        for q in (np.zeros(7), np.ones(7) * 0.01):
            recorder.sample(TargetPose(np.zeros(3), np.eye(3)), np.zeros(3), np.eye(3), q, 0.0)
        result.append(recorder.result().as_dict())
    assert result[1]["mean_joint_velocity_rad_s"] == pytest.approx(result[0]["mean_joint_velocity_rad_s"] * 2)


def test_nominal_source_manifest_linking_is_read_only_and_rejects_changed_manifest(tmp_path):
    raw = tmp_path / "nominal"
    raw.mkdir()
    write(raw / "manifest.json", dict(complete=True))
    index = tmp_path / "index.json"
    recording = raw / "episode/recording.hdf5"
    write(
        index,
        dict(
            fps=20,
            manifest_sha256=sha(raw / "manifest.json"),
            sources=[
                dict(
                    episode_index=0,
                    suite="libero_spatial",
                    task_id=0,
                    demo="demo_0",
                    frames=3,
                    recording=str(recording.relative_to(tmp_path)),
                    recording_sha256="unchanged",
                )
            ],
        ),
    )
    sources = discover(raw, index)
    assert len(sources) == 1 and sources[0].folder == recording.parent
    write(raw / "manifest.json", dict(complete=False))
    with pytest.raises(ValueError, match="manifest"):
        discover(raw, index)


def test_manual_angle_table_only_defines_authored_axis_eligibility_for_shared_search():
    table = dict(
        tasks=[dict(suite="libero_spatial", task_id=4, subtasks=[dict(label="bowl", APP=[0, 5, 0], EXEC=[15, 15, 5])])]
    )
    assert authored_masks(table, "libero_spatial", 4) == [dict(label="bowl", APP=(0, 1, 0), EXEC=(1, 1, 1))]


def test_robot_table_adapter_never_constructs_a_task_object_model(tmp_path):
    import xml.etree.ElementTree as ET
    from tasktol.data.libero_annotation.scene import robot_and_table_xml

    xml = """<mujoco><asset/><worldbody>
    <body name="robot0_base" pos="-.66 0 .912"><body name="robot0_link0"><geom type="sphere" size=".01"/></body></body>
    <body name="table" pos="0 0 .875"><geom type="box" size=".5 .6 .025"/></body>
    <body name="task_object"><joint name="object_joint" type="free"/><geom type="box" size=".02 .02 .02"/></body>
    </worldbody><actuator/></mujoco>"""
    state = np.r_[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    with patch("mujoco.MjModel.from_xml_string", side_effect=AssertionError("A full task scene was constructed")):
        reduced, _, tabletop = robot_and_table_xml(dict(xml=xml, states=state[None]), tmp_path, tmp_path)
    root = ET.fromstring(reduced)
    assert {b.get("name") for b in root.find("worldbody").iter("body")} == {
        "robot0_base",
        "robot0_link0",
        "work_surface",
    }
    assert all(j.get("type") != "free" for j in root.iter("joint"))
    assert tabletop["height_m"] == pytest.approx(0.9)
    assert tabletop["xy_lower_m"] == [-0.5, -0.6] and tabletop["xy_upper_m"] == [0.5, 0.6]
