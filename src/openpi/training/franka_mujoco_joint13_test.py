import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import config
from openpi.training import weight_loaders


def test_joint13_configs_use_exact_descriptor_layout() -> None:
    baseline = config.get_config("pi05_franka_mujoco")
    assert baseline.model.action_dim == 32

    for name in ("pi05_franka_mujoco_joint13", "pi05_franka_mujoco_joint13_state"):
        joint = config.get_config(name)
        assert joint.model.action_dim == 13
        assert joint.model.action_loss_groups == (
            ("action", 0, 7, 1.0),
            ("rotation_tolerance", 7, 13, 0.5),
        )
        data = joint.data.create(joint.assets_dirs, joint.model)
        assert data.action_sequence_keys == ("actions", "rotation_tolerance")
        assert isinstance(joint.weight_loader, weight_loaders.CheckpointWeightLoader)
        assert joint.weight_loader.skip_mismatched_shapes


def test_joint13_checkpoint_merge_keeps_new_action_head_shapes() -> None:
    loaded = {
        "action_head": {
            "kernel": np.ones((4, 32), dtype=np.float64),
            "bias": np.ones((4,), dtype=np.float64),
        },
    }
    reference = {
        "action_head": {
            "kernel": np.zeros((4, 13), dtype=np.float32),
            "bias": np.zeros((4,), dtype=np.float32),
        },
    }

    merged = weight_loaders._merge_params(  # noqa: SLF001
        loaded,
        reference,
        missing_regex=".*lora.*",
        use_reference_for_mismatched=True,
    )

    np.testing.assert_array_equal(merged["action_head"]["kernel"], reference["action_head"]["kernel"])
    np.testing.assert_array_equal(merged["action_head"]["bias"], loaded["action_head"]["bias"])
    assert merged["action_head"]["bias"].dtype == np.float32


@pytest.mark.parametrize("field", ["mean", "std", "q01", "q99"])
@pytest.mark.parametrize("dimension", [7, 15])
def test_joint13_rejects_incompatible_normalization(monkeypatch, tmp_path, field, dimension) -> None:
    fields = {name: np.ones(13) for name in ("mean", "std", "q01", "q99")}
    fields[field] = np.ones(dimension)
    monkeypatch.setattr(
        config.LeRobotFrankaMujocoDataConfig,
        "create_base_config",
        lambda *_: config.DataConfig(norm_stats={"actions": normalize.NormStats(**fields)}),
    )
    joint = config.get_config("pi05_franka_mujoco_joint13")
    with pytest.raises(ValueError, match="Recompute normalization statistics"):
        joint.data.create(tmp_path, joint.model)


def test_action_only_config_sequences_no_unlabeled_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(config.LeRobotFrankaMujocoDataConfig, "create_base_config", lambda *_: config.DataConfig())
    baseline = config.get_config("pi05_franka_mujoco")
    data = baseline.data.create(tmp_path, baseline.model)
    assert data.action_sequence_keys == ("actions",)
    assert data.source_schema == "tasktol_mujoco/8.0"
