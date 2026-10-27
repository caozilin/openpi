# MuJoCo schema 8.0 data for OpenPI

The converter accepts TaskTol **schema 8.0 / dataset release 5.0** from Panda,
xArm7, and UR5e. Each task must contain `task_metadata.json`, `manifest.json`,
and successful episode directories with `trajectory.json`, `main_rgb.mp4`, and
`wrist_rgb.mp4`. Metadata and both videos must specify 10 Hz and 224×224 RGB.
The source videos already use `resize_with_pad`; conversion preserves their
rows and decoded pixels without resampling or applying a second resize.

Source schema 8.0 uses an observation-to-action convention: image and state
row `i` precede action `i`. Filtered source rows have compact timestamps
`i / 10` and `index == video_frame_index == i`. The converter checks trajectory,
manifest, and video frame counts. It does not run the source frame filter again.

The seven-dimensional state is:

```text
[ee_x, ee_y, ee_z, ee_Rx, ee_Ry, ee_Rz, mean_finger_position]
```

Absolute orientation uses a continuous base-frame rotation vector. Actions are:

```text
[dx_base, dy_base, dz_base, dRx_base, dRy_base, dRz_base, gripper_command]
```

Rotation increments are in radians and integrate by left multiplication:
`R_next = Exp(dR) @ R_current`. Translation is in metres. Gripper commands are
`-1` for open and `+1` for close. Conversion preserves these values directly.
Raw joint vectors have seven entries; UR5e pads its sixth physical joint with a
zero seventh entry. They are validated but are not model inputs.

The export retains `robot_id` (Panda=0, xArm7=1, UR5e=2), `subtask_index`,
`trajectory_is_tolerance`, and `phase` (Pre-grasp=0, Grasp=1, Post-grasp=2,
Release=3). These fields are not prediction targets. The trajectory-type flag
can weight physical action loss using `--data.tolerance-trajectory-action-loss-weight`.
The converter maps manifest episode `execution_variant="nominal"` to 0 and
`execution_variant="tolerance"` to 1. Base collection without an execution variant
is nominal. Invalid or conflicting execution variants are errors.

## Deferred 7+6 supervision

Joint conversion requires each episode's completed `tolerance_labels.json`.
Labels identify the exact base `trajectory.json` by SHA-256 and frame count,
record their source method, and provide six nonnegative radian magnitudes for
each observed `(subtask_index, pregrasp/postgrasp)` pair:

```text
[Rx negative, Rx positive, Ry negative, Ry positive, Rz negative, Rz positive]
```

The label contract uses `fixed_axis_xyz_rpy`, composition
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`, and reference `realtime_nominal`. The converter
validates metadata, dimensions, finite values, nonnegativity, and complete
subtask/phase coverage. It assigns Pre/Post labels to their original rows and
sets Grasp/Release rows to zero. Each subtask keeps its own labels.

Missing or stale labels are errors in joint mode. Source data and labels remain
unchanged. The converter reads no task-level tolerance profiles or search CSVs.

```bash
uv sync

uv run python examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
  --raw-dir /path/to/schema8/datasets \
  --repo-id caozilin/franka_mujoco \
  --workers 12 \
  --image-writer-processes 0 \
  --image-writer-threads 2

uv run scripts/compute_franka_mujoco_norm_stats.py \
  --dataset-root /path/to/lerobot/caozilin/franka_mujoco

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_franka_mujoco_joint13 --exp-name=franka_mujoco_joint13_lora
```

`pi05_franka_mujoco_joint13` and `pi05_franka_mujoco_joint13_state` pack seven
physical actions followed by six tolerances into a native 13-dimensional model
interface. Both fields are sequenced over the same 16-step action horizon.
The loss is `1.0 × action MSE + 0.5 × rotation_tolerance MSE`; tolerance-trajectory
weighting changes only the physical action term. Metrics include `loss/total`,
`loss/action`, `loss/rotation_tolerance`, and the corresponding `loss_weighted/*`.

Inference returns two aligned arrays, `actions[H,7]` and
`rotation_tolerance[H,6]`. MuJoCo constructs the gravity-aligned tolerance frame
from its action-integrated nominal EE orientation every control cycle.

## Seven-action training before labeling

Use a separate action-only export when labels are not yet available. This mode
reads only the base trajectory and synchronized videos and creates no tolerance
column. The action-only training config requests only the action sequence.

```bash
uv run python examples/franka_mujoco/convert_franka_mujoco_data_to_lerobot.py \
  --raw-dir /path/to/schema8/datasets \
  --repo-id caozilin/franka_mujoco_actions \
  --no-joint-task-tolerance

uv run scripts/compute_franka_mujoco_norm_stats.py \
  --dataset-root /path/to/lerobot/caozilin/franka_mujoco_actions \
  --no-joint-task-tolerance \
  --pi05-output-dir assets/pi05_franka_mujoco/caozilin/franka_mujoco_actions

uv run scripts/train.py pi05_franka_mujoco \
  --data.repo-id=caozilin/franka_mujoco_actions \
  --exp-name=franka_mujoco_actions_lora
```

Action-only normalization writes only the seven-dimensional statistics. Joint
normalization writes both 13-dimensional joint statistics and their seven-action
slice. Both Parquet and completed Arrow caches are supported without decoding
images. Pass `--source arrow --arrow-cache-dir /path/to/cache` for Arrow input.

## Resume and rebuild

Exports record the source schema and label contract in `meta/info.json`.
Training and statistics generation require this metadata. `--resume` accepts
only the same conversion mode and unchanged trajectories and labels; the
conversion journal records their hashes, including each manifest entry and its
execution variant. Exports made before execution-variant classification was
corrected must be rebuilt with a fresh Arrow cache. It rolls back an incomplete episode
and continues with new episodes without rewriting committed Parquets.

After attaching or changing labels, create a new joint export or use
`--overwrite`, regenerate its Arrow cache and normalization statistics, and
start a new experiment. The protected append utility also accepts only current
exports and requires explicit existing/append counts plus the original `raw_dir`.

`fsdp_devices=1` disables parameter sharding; `CUDA_VISIBLE_DEVICES=0` restricts
JAX to one GPU. Increase `--progress-interval-seconds` for less frequent progress
updates.
