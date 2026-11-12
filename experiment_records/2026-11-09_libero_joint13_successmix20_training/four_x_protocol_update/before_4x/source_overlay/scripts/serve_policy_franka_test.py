"""Franka mix20 serving must use the training dataset's checkpoint assets."""
from unittest import mock

import pytest
import tyro

from openpi.policies import policy_config
from openpi.training import config
from scripts import serve_policy


@pytest.mark.parametrize("name", ["pi05_franka_mujoco_joint32", "pi05_tasktol_mix20_4xloss_7gpu_9k_joint32"])
def test_checkpoint_asset_override_preserves_training_transforms(name):
    args = serve_policy.Args(policy=serve_policy.Checkpoint(
        config=name, dir="/checkpoints/9000", asset_id="caozilin/my_mix20",
    ))
    with mock.patch.object(policy_config, "create_trained_policy") as create:
        serve_policy.create_policy(args)
    configured, directory = create.call_args.args
    assert directory == "/checkpoints/9000"
    assert configured.data.assets.asset_id == "caozilin/my_mix20"
    assert configured.data.joint_task_tolerance
    assert configured.model.action_dim == 32
    assert configured.model.action_horizon == 16
    if "mix20" in name:
        assert configured.data.tolerance_trajectory_action_loss_weight == 4.
    assert config.get_config(name).data.assets.asset_id is None


def test_missing_mix20_asset_fails_before_loading_weights():
    args = serve_policy.Args(policy=serve_policy.Checkpoint(
        config="pi05_tasktol_mix20_4xloss_7gpu_9k_joint32", dir="/checkpoints/9000",
    ))
    with (
        mock.patch.object(policy_config, "create_trained_policy") as create,
        pytest.raises(ValueError, match="--policy.asset-id"),
    ):
        serve_policy.create_policy(args)
    create.assert_not_called()


def test_default_joint32_serving_is_unchanged():
    args = serve_policy.Args(policy=serve_policy.Checkpoint(
        config="pi05_franka_mujoco_joint32", dir="/checkpoints/9000",
    ))
    with mock.patch.object(policy_config, "create_trained_policy") as create:
        serve_policy.create_policy(args)
    assert create.call_args.args[0] is config.get_config("pi05_franka_mujoco_joint32")


def test_checkpoint_cli_accepts_asset_id():
    args = tyro.cli(serve_policy.Args, args=[
        "--port", "8001", "policy:checkpoint", "--policy.config",
        "pi05_tasktol_mix20_4xloss_7gpu_9k_joint32", "--policy.dir", "/checkpoints/9000",
        "--policy.asset-id", "caozilin/my_mix20",
    ])
    assert args.port == 8001
    assert args.policy.asset_id == "caozilin/my_mix20"
