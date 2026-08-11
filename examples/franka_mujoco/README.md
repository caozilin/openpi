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
stage_target_pose   [b1x, b1y, b1z, b2x, b2y, b2z]
rotation_tolerance  [Rx-, Rx+, Ry-, Ry+, Rz-, Rz+]
```

Despite its retained dataset key, `stage_target_pose` contains only the
target orientation and no XYZ position. The converter maps its raw
three-dimensional axis-angle vector to a rotation matrix
and stores its first two columns in order. A decoder can apply Gram-Schmidt to
these two vectors and recover a proper SO(3) matrix. The converter copies the
annotation selected by each frame without applying phase-specific rules.
`tolerance_frame` is deliberately not stored: runtime derives it uniquely from
the target rotation with the dataset's deterministic `box_tolerance_frame`
rule. The data loader sequences both stored targets over the same 16-step horizon as
`actions`; the conventional pi0.5 transforms currently discard them until the
tolerance head is introduced.

The `pi05_franka_mujoco_joint19` config uses the existing 32-dimensional pi0.5
action interface without adding model modules. It concatenates the seven
physical actions, one six-dimensional target rotation, and six bilateral tolerance
magnitudes into the first 19 channels. Its grouped flow-matching loss weights physical actions by
1.0 and each TaskTol group by 0.5; padded channels 19 through 31 are excluded
from the loss. The original `pi05_franka_mujoco` config remains action-only.

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
Post-grasp allows +/-45 degrees around tolerance-frame Z, and Grasp/Release are
strict.

Convert locally on Windows with the converter's dedicated uv environment. The
script metadata and adjacent lock file isolate this command from the root
OpenPI environment, so it does not install JAX, Flax, CUDA packages, or OpenPI:

```powershell
uv run --script examples\franka_mujoco\convert_franka_mujoco_data_to_lerobot.py `
  --raw-dir G:\Code\franka_mujoco\datasets\v1 `
  --repo-id caozilin/franka_mujoco `
  --image-writer-processes 0
```

Use `--overwrite` to replace an existing local conversion. After logging in to
Hugging Face, `--push-to-hub` publishes a private dataset by default; add
`--no-private` only when a public dataset is intended.

Do not run normalization-statistics generation in this Windows-only converter
environment. On the Linux GPU cloud server, use the complete OpenPI environment
to compute normalization statistics and train:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_mujoco

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_mujoco --exp-name=franka_mujoco_lora
```

For joint 19-dimensional training, normalization statistics must be recomputed
because its packed action statistics have shape 19 rather than 7:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_mujoco_joint19

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_mujoco_joint19 --exp-name=franka_mujoco_joint19_lora
```

Joint training logs the backward-compatible `loss` metric plus the following
W&B metrics:

```text
loss/total
loss/action
loss/stage_target_pose
loss/rotation_tolerance
loss_weighted/action
loss_weighted/stage_target_pose
loss_weighted/rotation_tolerance
```

The three `loss/*` components are unweighted MSE values. The corresponding
`loss_weighted/*` values include their configured lambda, and `loss/total` is
their weighted sum.

`fsdp_devices=1` disables parameter sharding. `CUDA_VISIBLE_DEVICES=0` is what
restricts JAX to one GPU when the host exposes multiple GPUs.
