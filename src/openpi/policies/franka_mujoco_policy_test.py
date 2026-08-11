import numpy as np

from openpi.models import model as _model
from openpi.policies import franka_mujoco_policy


def test_franka_mujoco_inputs() -> None:
    transform = franka_mujoco_policy.FrankaMujocoInputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "observation/image": np.zeros((3, 20, 30), dtype=np.float32),
            "observation/wrist_image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/state": np.arange(7, dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "prompt": "pick up the block",
        }
    )

    assert result["image"]["base_0_rgb"].shape == (20, 30, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (20, 30, 3)
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert result["state"].shape == (7,)
    assert result["actions"].shape == (10, 7)
    assert result["prompt"] == "pick up the block"


def test_franka_mujoco_outputs_drop_padding() -> None:
    actions = np.arange(10 * 32, dtype=np.float32).reshape(10, 32)
    result = franka_mujoco_policy.FrankaMujocoOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :7])


def test_franka_mujoco_joint_tolerance_inputs_and_outputs() -> None:
    transform = franka_mujoco_policy.FrankaMujocoInputs(
        model_type=_model.ModelType.PI05,
        joint_task_tolerance=True,
    )
    actions = np.arange(2 * 7, dtype=np.float32).reshape(2, 7)
    target = np.full((2, 6), 1.0, dtype=np.float32)
    tolerance = np.full((2, 6), 3.0, dtype=np.float32)
    result = transform(
        {
            "observation/image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/state": np.arange(7, dtype=np.float32),
            "actions": actions,
            "stage_target_pose": target,
            "rotation_tolerance": tolerance,
        }
    )

    assert result["actions"].shape == (2, 19)
    np.testing.assert_array_equal(result["actions"], np.concatenate((actions, target, tolerance), axis=-1))

    padded = np.pad(result["actions"], ((0, 0), (0, 13)))
    outputs = franka_mujoco_policy.FrankaMujocoOutputs(joint_task_tolerance=True)({"actions": padded})
    np.testing.assert_array_equal(outputs["actions"], actions)
    np.testing.assert_array_equal(outputs["stage_target_pose"], target)
    np.testing.assert_array_equal(outputs["rotation_tolerance"], tolerance)
