# ruff: noqa: SLF001
import ast
import hashlib
import json
import os
from pathlib import Path
import threading
import types

import h5py
import numpy as np
import pytest

from examples.libero import regenerate_libero_dataset as regenerate

REFERENCE = Path(__file__).parent / "upstream/regenerate_libero_dataset.py.reference"


def quat_to_axis_angle(quat):
    return quat[:3].copy()


class FakeEnv:
    def __init__(self, resolution=8, stop=None):
        self.resolution = resolution
        self.stop = stop
        self.resets = 0
        self.closed = False
        self.state = np.zeros(5)
        self.sim = types.SimpleNamespace(get_state=lambda: self.state.copy())

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.resets += 1
        self.state = np.zeros(5)
        self.texture = int(np.random.randint(0, 100))

    def set_init_state(self, state):
        self.state = state.copy()
        return self.observation()

    def observation(self):
        image = np.arange(self.resolution**2 * 3, dtype=np.uint8).reshape(self.resolution, self.resolution, 3)
        image = image + np.uint8(self.texture)
        return {
            "robot0_gripper_qpos": np.asarray([0.02, 0.03]),
            "robot0_eef_pos": self.state[1:4].copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_joint_pos": np.arange(7) + self.state[0],
            "agentview_image": image,
            "robot0_eye_in_hand_image": image + np.uint8(1),
        }

    def step(self, action):
        self.state[0] += 1
        self.state[1] += action[0]
        if self.stop is not None and self.resets == 2 and self.state[0] >= 11:
            self.stop.set()
        # Can become true early, then false on a final negative motion.
        done = bool(self.state[-1]) and self.state[1] > 0
        return self.observation(), float(done), done, {}

    def close(self):
        self.closed = True


def make_env(job, stop=None):
    env = FakeEnv(resolution=job["resolution"], stop=stop)
    env.seed(0)
    return env


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "raw/fake_task_demo.hdf5"
    source.parent.mkdir()
    actions = np.zeros((8, 7))
    actions[:, -1] = [-1, -1, -1, 1, 1, 1, -1, -1]
    actions[1, 0] = 1e-4  # Strictly '<', not '<='.
    actions[5, 0] = 0.1
    with h5py.File(source, "w") as file:
        data = file.create_group("data")
        data.attrs["env_args"] = '{"control_freq": 20}'
        for i in range(4):
            group = data.create_group(f"demo_{i}")
            state = np.zeros((len(actions), 5))
            state[:, -1] = int(i != 1)
            episode_actions = actions.copy()
            if i == 2:
                episode_actions[-1, 0] = -1  # Ends unsuccessfully despite earlier success.
            group.create_dataset("actions", data=episode_actions)
            group.create_dataset("states", data=state)
            group.create_dataset("robot_states", data=np.full((len(actions), 9), 123.0))
            group.attrs["model_file"] = "<mujoco/>"
            group.attrs["num_samples"] = len(actions)
    bddl = tmp_path / "fake_task.bddl"
    bddl.write_text("fake task")
    return {
        "suite": "libero_spatial",
        "task_id": 0,
        "name": "fake_task",
        "language": "fake task",
        "bddl": str(bddl),
        "source": str(source),
        "output": str(tmp_path / "processed/libero_spatial_no_noops/fake_task_demo.hdf5"),
        "max_demos": None,
        "resume": False,
        "resolution": 8,
        "compression": "gzip",
        "gzip_level": 1,
        "render_backend": "egl",
    }


def reference_namespace(**overrides):
    tree = ast.parse(REFERENCE.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("is_noop", "main")]
    namespace = {
        "np": np,
        "h5py": h5py,
        "os": os,
        "json": json,
        "tqdm": types.SimpleNamespace(tqdm=lambda x: x),
        "T": types.SimpleNamespace(quat2axisangle=quat_to_axis_angle),
        "IMAGE_RESOLUTION": 8,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(REFERENCE), "exec"), namespace)
    return namespace


def assert_same_datasets(left, right):
    for name, obj in left.items():
        if name == "tasktol":
            continue
        assert name in right
        if isinstance(obj, h5py.Dataset):
            np.testing.assert_array_equal(obj[()], right[name][()])
        else:
            assert_same_datasets(obj, right[name])


def test_reference_is_pinned():
    metadata = json.loads((REFERENCE.parent / "SOURCE.json").read_text())
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == metadata["sha256"]
    assert metadata["commit"] == regenerate.UPSTREAM_COMMIT


def test_motion_executor_uses_same_recorder_filter_and_settling(job):
    seen = []
    env = make_env(job)

    def settled(native, obs, state):
        assert native is env
        assert state[0] == regenerate.SETTLE_STEPS
        seen.append('settled')

    def execute(native, index, action, obs):
        assert seen[0] == 'settled'
        seen.append(index)
        return native.step(action.tolist())

    with h5py.File(job['source'], 'r') as source:
        result = regenerate.replay_demo(env, source['data/demo_0'], quat_to_axis_angle=quat_to_axis_angle,
                                        on_settled=settled, action_executor=execute)
        assert seen == ['settled', 1, 3, 5, 6]
        assert result['success']
        np.testing.assert_array_equal(result['arrays']['actions'], source['data/demo_0/actions'][()][[1, 3, 5, 6]])
        assert env.state[0] == regenerate.SETTLE_STEPS + 4


def test_rejected_motion_is_saved_without_executing_or_claiming_success(job):
    env = make_env(job)

    def reject(native, index, action, obs):
        raise regenerate.ReplayActionRejected('no feasible reference')

    with h5py.File(job['source'], 'r') as source:
        result = regenerate.replay_demo(env, source['data/demo_0'], quat_to_axis_angle=quat_to_axis_angle,
                                        action_executor=reject, retain_failed=True)
    assert env.state[0] == regenerate.SETTLE_STEPS
    assert not result['success']
    assert result['execution_rejected']
    assert result['rejected_source_index'] == 1
    assert result['retained_steps'] == 1
    np.testing.assert_array_equal(result['source_step_indices'], [1])
    np.testing.assert_array_equal(result['arrays']['pre_action_states'][0], result['final_state'])


def test_state_trajectory_replay_preserves_waits_and_exact_initial_state(job):
    env = make_env(job)
    seen = []
    with h5py.File(job['source'], 'r') as source:
        raw = source['data/demo_0']
        result = regenerate.replay_demo(
            env, raw, quat_to_axis_angle=quat_to_axis_angle,
            skip_noops=False, settle_steps=0, max_source_steps=len(raw['states']) - 1,
            action_executor=lambda native, index, action, obs: (
                seen.append(index) or native.step(action.tolist())
            ),
        )
        np.testing.assert_array_equal(result['settled_state'], raw['states'][0])
    assert seen == list(range(7))
    assert result['discarded_noops'] == 0
    assert result['retained_steps'] == 7
    assert env.state[0] == 7


def test_retimed_actions_record_every_tick_with_actual_states(job):
    env = make_env(job)
    with h5py.File(job['source'], 'r') as source:
        raw = source['data/demo_0']
        commands = np.repeat(raw['actions'][:-1], 2, axis=0)
        result = regenerate.replay_demo(
            env, raw, quat_to_axis_angle=quat_to_axis_angle,
            replay_actions=commands, skip_noops=False, settle_steps=0,
        )
    assert result['retained_steps'] == 14
    assert result['discarded_noops'] == 0
    np.testing.assert_array_equal(result['arrays']['actions'], commands)
    np.testing.assert_array_equal(result['arrays']['pre_action_states'][:, 0], np.arange(14))
    assert result['final_state'][0] == 14


@pytest.mark.parametrize("motion", [0.0, 0.999e-4, 1e-4, 1.001e-4, 0.1])
@pytest.mark.parametrize("previous_gripper", [None, -1, 1])
def test_noop_matches_upstream(motion, previous_gripper):
    action = np.asarray([motion, 0, 0, 0, 0, 0, 1])
    previous = None if previous_gripper is None else np.asarray([0, 0, 0, 0, 0, 0, previous_gripper])
    expected = reference_namespace()["is_noop"](action, previous)
    assert regenerate.is_noop(action, previous) == expected


def test_full_replay_matches_upstream_and_adds_aligned_states(job, tmp_path, monkeypatch):
    task = types.SimpleNamespace(name=job["name"], language=job["language"])
    suite = types.SimpleNamespace(n_tasks=1, get_task=lambda _: task)
    reference_envs = []

    def get_env(*_, **__):
        env = make_env(job)
        reference_envs.append(env)
        return env, job["language"]

    namespace = reference_namespace(
        benchmark=types.SimpleNamespace(get_benchmark_dict=lambda: {job["suite"]: lambda: suite}),
        get_libero_env=get_env,
        get_libero_dummy_action=lambda _: regenerate.DUMMY_ACTION,
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments/robot/libero").mkdir(parents=True)
    expected_dir = tmp_path / "official"
    namespace["main"](
        types.SimpleNamespace(
            libero_task_suite=job["suite"],
            libero_raw_data_dir=str(Path(job["source"]).parent),
            libero_target_dir=str(expected_dir),
        )
    )
    metadata = regenerate.process_task(job, env_factory=make_env, quat_to_axis_angle=quat_to_axis_angle)
    assert metadata["successful_demos"] == 2
    assert metadata["failed_demos"] == 2
    assert metadata["episodes"]["demo_0"]["source_step_indices"] == [1, 3, 5, 6]
    assert not metadata["episodes"]["demo_2"]["success"]
    with h5py.File(job["output"], "r") as actual, h5py.File(expected_dir / Path(job["source"]).name, "r") as expected:
        assert set(actual["data"]) == {"demo_0", "demo_3"} == set(expected["data"])
        assert_same_datasets(expected["data"], actual["data"])
        demo = actual["data/demo_0"]
        np.testing.assert_array_equal(demo["states"][0], np.asarray([0, 0, 0, 0, 1]))
        assert demo["tasktol/pre_action_states"][0, 0] == 10
        np.testing.assert_array_equal(demo["tasktol/pre_action_states"][0], demo["tasktol/settled_state"][()])
        np.testing.assert_array_equal(demo["tasktol/pre_action_states"][1:], demo["states"][1:])
        np.testing.assert_array_equal(demo["tasktol/source_step_indices"][()], [1, 3, 5, 6])
        assert demo.attrs["model_file"] == "<mujoco/>"
        assert demo["obs/agentview_rgb"].compression == "gzip"


def test_resume_restores_rng_and_rolls_back_uncommitted_episode(job, tmp_path):
    uninterrupted = {**job, "output": str(tmp_path / "uninterrupted/data.hdf5")}
    regenerate.process_task(uninterrupted, env_factory=make_env, quat_to_axis_angle=quat_to_axis_angle)
    stop = threading.Event()
    stopped = regenerate.process_task(
        job, env_factory=lambda value: make_env(value, stop), stop=stop, quat_to_axis_angle=quat_to_axis_angle
    )
    assert stopped["status"] == "interrupted"
    assert stopped["processed_demos"] == 1
    final = Path(job["output"])
    assert not final.exists()
    with h5py.File(final.with_suffix(".hdf5.inprogress"), "a") as partial:
        partial["data"].create_group("__pending_demo_1")
        partial["regeneration/episodes"].create_group("__pending_demo_1")
        partial["data"].create_group("demo_1")  # Crash after data rename, before ledger rename.
    result = regenerate.process_task(
        {**job, "resume": True}, env_factory=make_env, quat_to_axis_angle=quat_to_axis_angle
    )
    assert result["complete"]
    assert result["processed_demos"] == 4
    with h5py.File(final, "r") as resumed, h5py.File(uninterrupted["output"], "r") as expected:
        assert_same_datasets(expected["data"], resumed["data"])
        assert set(resumed["regeneration/episodes"]) == {"demo_0", "demo_1", "demo_2", "demo_3"}
    before = final.read_bytes()
    skipped = regenerate.process_task(
        {**job, "resume": True}, env_factory=lambda _: pytest.fail("replayed completed task")
    )
    assert skipped["status"] == "skipped"
    assert final.read_bytes() == before


def test_resume_rejects_changed_inputs_and_incomplete_completed_output(job):
    regenerate.process_task(job, env_factory=make_env, quat_to_axis_angle=quat_to_axis_angle)
    with pytest.raises(FileExistsError, match="--resume"):
        regenerate.process_task(job)
    with h5py.File(job["source"], "a") as source:
        source["data/demo_0/actions"][1, 0] = 0.2
    with pytest.raises(ValueError, match="does not match"):
        regenerate.process_task({**job, "resume": True})
    with h5py.File(job["source"], "a") as source:
        source["data/demo_0/actions"][1, 0] = 1e-4
    with h5py.File(job["output"], "a") as output:
        del output["data/demo_0"]
    with pytest.raises(ValueError, match="missing successful episode"):
        regenerate.process_task({**job, "resume": True})


@pytest.mark.parametrize(
    "changed",
    [
        {"compression": "none"},
        {"max_demos": 2},
        {"libero_environment_sha256": "edited_environment"},
        {"blas_threads": 2},
    ],
)
def test_resume_rejects_changed_processing_scope_or_environment(job, changed):
    regenerate.process_task(job, env_factory=make_env, quat_to_axis_angle=quat_to_axis_angle)
    with pytest.raises(ValueError, match="does not match"):
        regenerate.process_task({**job, **changed, "resume": True})


def test_resume_repairs_counters_after_last_episode_commit(job):
    regenerate.process_task(job, env_factory=make_env, quat_to_axis_angle=quat_to_axis_angle)
    final = Path(job["output"])
    partial = final.with_suffix(".hdf5.inprogress")
    final.rename(partial)
    with h5py.File(partial, "a") as output:
        output.attrs["complete"] = False
        output["data"].attrs["num_demos"] = -1
        output["data"].attrs["total"] = -1
    result = regenerate.process_task(
        {**job, "resume": True}, env_factory=lambda _: pytest.fail("replayed fully committed checkpoint")
    )
    assert result["complete"]
    with h5py.File(final, "r") as output:
        assert output["data"].attrs["num_demos"] == len(output["data"]) == 2
        assert output["data"].attrs["total"] == sum(group["actions"].shape[0] for group in output["data"].values())
