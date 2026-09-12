import numpy as np

from openpi.training import config
from openpi.training import weight_loaders


def test_joint15_configs_use_exact_descriptor_layout() -> None:
    baseline = config.get_config("pi05_franka_mujoco")
    assert baseline.model.action_dim == 32

    for name in ("pi05_franka_mujoco_joint15", "pi05_franka_mujoco_joint15_state"):
        joint = config.get_config(name)
        assert joint.model.action_dim == 15
        assert joint.model.action_loss_groups == (
            ("action", 0, 7, 1.0),
            ("tolerance_frame_code", 7, 9, 0.5),
            ("rotation_tolerance", 9, 15, 0.5),
        )
        assert isinstance(joint.weight_loader, weight_loaders.CheckpointWeightLoader)
        assert joint.weight_loader.skip_mismatched_shapes


def test_joint15_checkpoint_merge_keeps_new_action_head_shapes() -> None:
    loaded = {
        "action_head": {
            "kernel": np.ones((4, 32), dtype=np.float64),
            "bias": np.ones((4,), dtype=np.float64),
        },
    }
    reference = {
        "action_head": {
            "kernel": np.zeros((4, 15), dtype=np.float32),
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
