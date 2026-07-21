# Franka MuJoCo dataset conversion

This example converts TaskTol-VLA schema 4.0 episodes to the LeRobot layout used
by the `pi05_franka_mujoco` training config. The initial integration intentionally
excludes phase, stage-target, tolerance, privileged success, collision, and
optimizer fields.

The seven-dimensional state is:

```text
[ee_x, ee_y, ee_z, ee_Rx, ee_Ry, ee_Rz, symmetric_gripper_position]
```

The gripper state is half the recorded aperture, equivalently the mean of the two
finger slide positions. Actions retain the raw schema 4.0 convention:

```text
[dx_base, dy_base, dz_base, dRx_tool, dRy_tool, dRz_tool, gripper_command]
```

Convert locally:

```bash
uv run examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
  --raw-dir /path/to/franka_mujoco/datasets \
  --repo-id caozilin/franka_mujoco
```

Use `--overwrite` to replace an existing local conversion. After logging in to
Hugging Face, `--push-to-hub` publishes a private dataset by default; add
`--no-private` only when a public dataset is intended.

Compute normalization statistics and train:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_mujoco

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_mujoco --exp-name=franka_mujoco_lora
```

`fsdp_devices=1` disables parameter sharding. `CUDA_VISIBLE_DEVICES=0` is what
restricts JAX to one GPU when the host exposes multiple GPUs.
