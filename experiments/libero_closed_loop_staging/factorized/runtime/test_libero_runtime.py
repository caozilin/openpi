from dataclasses import replace

import mujoco
import numpy as np
import pytest

from tasktol.control.task_space.kinematics import RobotKinematics
from tasktol.geometry import rotation_matrix
from tasktol.simulation.loader import load_scene_model
from tasktol.simulation.robot import find_handles, set_home, ee_pose


def test_fixed_tool_rotation_preserves_world_angular_jacobian_and_position():
    model = load_scene_model("Geometry Region - Lying Cylinder", "panda")
    data = mujoco.MjData(model)
    handles = find_handles(model)
    set_home(model, data, handles)
    q = data.qpos[handles.qpos_ids].copy()
    offset = rotation_matrix(np.array([0.0, 0.0, np.pi / 2]))
    calibrated = replace(handles, ee_rotation_offset=offset)
    original = RobotKinematics(model, handles).evaluate(q)
    actual = RobotKinematics(model, calibrated).evaluate(q)
    np.testing.assert_allclose(actual.position, original.position, atol=1e-12)
    np.testing.assert_allclose(actual.rotation, original.rotation @ offset, atol=1e-12)
    np.testing.assert_allclose(
        actual.rotation @ actual.jacobian[3:], original.rotation @ original.jacobian[3:], atol=1e-12
    )
    pos, rot = ee_pose(data, calibrated)
    np.testing.assert_allclose(pos, actual.position, atol=1e-12)
    np.testing.assert_allclose(rot, actual.rotation, atol=1e-12)


@pytest.mark.skipif(mujoco.__version__ != "3.2.3", reason="Native LIBERO integration uses the pinned 3.2.3 environment")
@pytest.mark.parametrize("joint_control_mode", ["native", "osc_inertia_matched", "osc_task_joint_null"])
def test_native_libero_ipopt_holds_pose_and_gates_rotational_bounds(tmp_path, monkeypatch, joint_control_mode):
    import json
    from pathlib import Path
    import h5py

    root = Path("/media/czl/sata/franka_my_code/openpi")
    monkeypatch.syspath_prepend(str(root / "third_party/libero"))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    from tasktol.simulation.libero import LiberoTaskTolSession
    from tasktol.control.task_space.types import TaskKind

    selected = json.loads(
        (root / "data/libero40_mask_annotation/20261019_mainview_v1/selection_manifest.json").read_text()
    )["tasks"][0]
    with h5py.File(selected["processed_hdf5"], "r") as f:
        initial = f["data"][selected["demo_id"]]["tasktol/pre_action_states"][0]
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / selected["suite"] / (selected["task_name"] + ".bddl")),
        controller="OSC_POSE" if joint_control_mode == "osc_task_joint_null" else "JOINT_POSITION",
        camera_names=["agentview"],
        camera_heights=32,
        camera_widths=32,
        control_freq=20,
    )
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(initial)
        calls = []
        original_step = env.step

        def observed_step(action):
            calls.append(np.asarray(action))
            return original_step(action)

        monkeypatch.setattr(env, "step", observed_step)
        session = LiberoTaskTolSession(env, joint_control_mode=joint_control_mode)
        q = session.data.qpos[session.handles.qpos_ids].copy()
        fk = session.ik.kinematics.evaluate(q)
        hand = session.model.body(env.robots[0].robot_model.eef_name).id
        np.testing.assert_allclose(fk.rotation, session.data.xmat[hand].reshape(3, 3), atol=1e-12)
        start_time = float(session.data.time)
        _, eng = session.step(np.r_[np.zeros(6), -1.0], phase="ENG", semantic_mask="111", bounds_rad=np.full(6, 0.03))
        assert eng["optimizer"]["solver_called"] and eng["optimizer"]["feasible"]
        assert eng["effective_mask"] == "000" and eng["physics_steps"] == 25
        assert session.data.time == pytest.approx(start_time + 0.05)
        assert eng["tracking_position_error_m"] < 0.001
        assert eng["native_joint_goal_error_rad"] < 1e-10
        _, app = session.step(
            np.r_[np.zeros(6), -1.0], phase="APP", semantic_mask="001", bounds_rad=np.tile([0.0, 0.03], 3)
        )
        assert app["effective_mask"] == "001" and app["optimizer"]["feasible"]
        assert max(app["predicted_tolerance_violation_rad"]) < 1e-7
        _, strict = session.step(np.r_[np.zeros(6), -1.0], phase="APP", semantic_mask="111", bounds_rad=None)
        assert strict["effective_mask"] == "000" and strict["optimizer"]["feasible"]
        assert len(calls) == 3
        assert env.env.timestep == 3
        assert strict["native_joint_goal_error_rad"] < 1e-10
        assert session.controller.kp[0] == (150.0 if joint_control_mode == "osc_task_joint_null" else 50.0)
        if joint_control_mode == "osc_task_joint_null":
            assert session.controller.null_kp == 10.0
            assert session.controller.factor_statistics["calls"] == 75
            np.testing.assert_array_equal(session.controller.joint_target, strict["joint_target_rad"])
        if joint_control_mode == "osc_inertia_matched":
            stats = session.controller.inertia_diagnostics
            assert stats["calls"] == 75
            assert stats["min_effective_eigenvalue"] > 0
            assert stats["max_symmetry_error"] < 1e-9

        assert all(t.kind == TaskKind.SPECIFIC for t in session.ik._resolve_axis_tasks(np.ones(6, dtype=bool)))
    finally:
        env.close()
