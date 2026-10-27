import numpy as np
import pytest

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
    assert result["action_loss_weight"] == 1.0


def test_tolerance_trajectory_gets_four_x_action_loss_weight() -> None:
    transform = franka_mujoco_policy.FrankaMujocoInputs(
        model_type=_model.ModelType.PI05,
        tolerance_trajectory_action_loss_weight=4.0,
    )
    result = transform(
        {
            "observation/image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/state": np.arange(7, dtype=np.float32),
            "trajectory_is_tolerance": np.asarray([1], dtype=np.int64),
        }
    )

    assert result["action_loss_weight"] == 4.0


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
    tolerance = np.arange(2 * 6, dtype=np.float32).reshape(2, 6) + 100
    result = transform(
        {
            "observation/image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((20, 30, 3), dtype=np.uint8),
            "observation/state": np.arange(7, dtype=np.float32),
            "actions": actions,
            "rotation_tolerance": tolerance,
        }
    )

    assert result["actions"].shape == (2, 13)
    np.testing.assert_array_equal(result["actions"], np.concatenate((actions, tolerance), axis=-1))

    outputs = franka_mujoco_policy.FrankaMujocoOutputs(joint_task_tolerance=True)({"actions": result["actions"]})
    assert set(outputs) == {"actions", "rotation_tolerance"}
    np.testing.assert_array_equal(outputs["actions"], actions)
    np.testing.assert_array_equal(outputs["rotation_tolerance"], tolerance)


@pytest.mark.parametrize("dimension", [7, 12, 15, 32])
def test_joint_outputs_reject_incompatible_dimensions(dimension: int) -> None:
    with pytest.raises(ValueError, match="exactly 13"):
        franka_mujoco_policy.FrankaMujocoOutputs(joint_task_tolerance=True)(
            {"actions": np.zeros((2, dimension), dtype=np.float32)}
        )
