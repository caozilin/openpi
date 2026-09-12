# Multi-robot MuJoCo dataset conversion

This example converts TaskTol-VLA schema 7.0 / dataset release 4.2 Panda,
xArm7, and UR5e episodes to the LeRobot layout used by the
`pi05_franka_mujoco` training config. The raw hierarchy is
`datasets/<robot_uid>/<task>/...`. It only accepts the current four-stage
format; legacy schema 4.0 integer phases are not supported.

Each converted frame preserves its canonical phase as a one-element `int64`
feature:

```text
0 = pregrasp, 1 = grasp, 2 = postgrasp, 3 = release
```

The current `pi05_franka_mujoco` transforms intentionally consume only images,
state, action, and instruction. The preserved `phase` is therefore neither a
model input nor a prediction target. The converter additionally resolves each
frame's `stage_annotation_id` and stores these future tolerance-head targets:

```text
tolerance_frame_code  [cos(phi), sin(phi)]
rotation_tolerance    [Rx-, Rx+, Ry-, Ry+, Rz-, Rz+]
```

`stage_target_pose` remains raw-dataset provenance only. The converter copies
the annotation's two-dimensional gravity-aligned heading code directly; it is
the VLA supervision target. Runtime normalizes the code and reconstructs the
full SO(3) tolerance frame using world gravity. The data loader sequences both
stored targets over the same 16-step horizon as `actions`; the action-only
config discards them, while the joint15 config packs them into its target.

The `pi05_franka_mujoco_joint15` config uses a native 15-dimensional pi0.5
action interface. It concatenates the seven
physical actions, one two-dimensional frame code, and six bilateral tolerance
magnitudes. Its grouped flow-matching loss weights physical actions by
1.0 and each TaskTol group by 0.5. The original `pi05_franka_mujoco` config remains action-only.

The seven-dimensional state is:

```text
[ee_x, ee_y, ee_z, ee_Rx, ee_Ry, ee_Rz, symmetric_gripper_position]
```

The converter stores `robot_id` as Panda=0, xArm7=1, UR5e=2. Raw joint
positions and velocities are validated as fixed seven-dimensional vectors but
are not copied into LeRobot because the current model does not consume them.
UR5e has six physical joints, so both raw joint vectors must contain a zero in
their seventh dimension. The model state above remains unchanged.

The gripper state is half the recorded aperture, equivalently the mean of the two
finger slide positions. Actions retain the raw schema 7.0 convention:

```text
[dx_base, dy_base, dz_base, dRx_tool, dRy_tool, dRz_tool, gripper_command]
```

Both camera streams must already have been processed with resize-with-pad and
stored as 224x224 RGB video. The converter validates this resolution and writes
the decoded frames unchanged; it does not resize or pad images itself. The
resulting LeRobot dataset therefore stores only 224x224 RGB frames.

Task-level bilateral tolerances are initially read from
`task_metadata.json/annotation.rotation_tolerance_profiles_rad`. When a task
also contains `tolerance_annotation.json`, its optimized
`rotation_tolerance_bounds_rad` becomes the single task-level profile shared by
every episode. Per-episode values in `episode_refinements` are deliberately
ignored, even when present. If neither task metadata nor an optimized
annotation supplies tolerances, the converter prints a warning and uses the
original defaults: Pre-grasp allows +/-30 degrees around tolerance-frame Y,
Post-grasp allows +/-30 degrees around tolerance-frame Z, and Grasp/Release are
strict.

Install the locked OpenPI environment and run conversion on Linux. Conversion,
normalization-statistics generation, and training deliberately share the root
project environment so that LeRobot, Hugging Face Datasets, and PyArrow use the
same schema versions throughout the pipeline:

```bash
uv sync

uv run python examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
  --raw-dir /path/to/franka_mujoco/datasets/v1 \
  --repo-id caozilin/franka_mujoco \
  --workers 12 \
  --image-writer-processes 0 \
  --image-writer-threads 2 \
  --progress-interval-seconds 5
```

The converter displays one global frame-based progress bar with throughput and
ETA. Hugging Face's per-episode map/parquet bars are disabled. Increase
`--progress-interval-seconds` for less frequent terminal updates.

Use `--overwrite` to replace an existing conversion.

In the same Linux environment, compute normalization statistics and train:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_mujoco

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_mujoco --exp-name=franka_mujoco_lora
```

For joint 15-dimensional training, normalization statistics must be recomputed
because its packed action statistics have shape 15 rather than 7:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_mujoco_joint15

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_mujoco_joint15 --exp-name=franka_mujoco_joint15_lora
```

Joint training logs the backward-compatible `loss` metric plus the following
W&B metrics:

```text
loss/total
loss/action
loss/tolerance_frame_code
loss/rotation_tolerance
loss_weighted/action
loss_weighted/tolerance_frame_code
loss_weighted/rotation_tolerance
```

The three `loss/*` components are unweighted MSE values. The corresponding
`loss_weighted/*` values include their configured lambda, and `loss/total` is
their weighted sum.

`fsdp_devices=1` disables parameter sharding. `CUDA_VISIBLE_DEVICES=0` is what
restricts JAX to one GPU when the host exposes multiple GPUs.
