# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

## Regenerate original LIBERO demonstrations

`regenerate_libero_dataset.py` replays the original HDF5 demonstrations using the preprocessing rules from
[OpenVLA](https://github.com/openvla/openvla/blob/0ebc5e333ee7a916813bf50db240f1af7ba1d295/experiments/robot/libero/regenerate_libero_dataset.py).
It uses the local LIBERO simulation environment and does not load a VLA model. The unmodified upstream source,
its SHA256, and its MIT license are retained in `upstream/` for equivalence tests.

Install the additional HDF5 dependency into the **Python 3.8 simulation environment**:

```bash
uv pip install --python examples/libero/.venv/bin/python -r examples/libero/requirements-regenerate.txt
```

The default input is `third_party/libero/libero/datasets`. Select a stable output disk with room for the rendered
images. Outputs are separate from the originals. Lossless gzip level 1 is enabled by default; `--compression lzf`
and `--compression none` are also supported. Storage compression does not change the arrays used for training.

Start with two tasks and three demonstrations per task. Replace the example output path with a location on your disk:

```bash
examples/libero/.venv/bin/python examples/libero/regenerate_libero_dataset.py \
  --output-dir /path/to/stable-disk/libero-smoke \
  --suites libero_spatial --task-ids 0 1 \
  --max-demos-per-task 3 --workers 2
```

Use `--dry-run` to print the task plan without constructing simulation environments. Once the smoke test passes,
process all four suites with four workers:

```bash
examples/libero/.venv/bin/python examples/libero/regenerate_libero_dataset.py \
  --output-dir /path/to/stable-disk/libero-openvla \
  --workers 4 --resume
```

Workers use `spawn`, independent offscreen rendering contexts, and one BLAS thread each.
Two terminal progress bars show processed demonstrations (including unsuccessful replays) and finished tasks,
with throughput and estimated remaining time. `saved` counts successful demonstrations; resumed episodes are
included as their checkpoints are validated. Use `--no-progress` to disable the bars for log-only runs.

Each task owns its HDF5, lock, and metadata file. The `summary.json` describes the current invocation; per-task `.metainfo.json` files and
the HDF5 episode ledger are the persistent processing records. A failed task makes the command exit nonzero while
other tasks can finish. Only the source demonstrations selected by the current limits are considered complete.

### Replay and alignment rules

- Create the environment at 256×256 and call `env.seed(0)`.
- Restore the original `states[0]`, then execute ten `[0, 0, 0, 0, 0, 0, -1]` settling actions.
- Apply OpenVLA's `1e-4` no-op predicate, including its gripper and first-action special cases. Filtered actions are
  not executed in the simulator.
- Record observations before each retained action, replay the complete retained sequence, and save the episode
  only if its final `done` is true. Original `demo_N` names are preserved; successful episode indices may have gaps.
- Store raw, unrotated camera images. Apply the 180° rotation once in the eventual HDF5→LeRobot exporter, matching
  the official RLDS conversion and this project's LIBERO evaluation preprocessing.

Simulation retains LIBERO's default 20 Hz control rate. Source indices refer to the original sequence; replay
time advances only for the ten settling actions and retained actions. Preserve this simulator time when building
tolerance trajectories; a training export's declared image FPS should not silently change the physical time step.

The original OpenVLA script stores the **reset state** in legacy `states[0]`, while its first observation is from
after settling. This entry point preserves that compatibility behavior. The added `data/demo_N/tasktol` group
contains the original initial state, post-settling state, actual pre-action simulator/robot states, final state,
and indices of retained actions in the original demonstration. Use `tasktol/pre_action_states` to align states
with observations and actions. The `source_metadata` group and original demo attributes retain source environment
metadata and model XML.

Flattened simulator states do not serialize every controller or solver internal variable. To reconstruct a
mid-trajectory execution context, replay the same settling steps and retained action prefix from the original
initial state. The added states provide alignment and verification rather than a promise of arbitrary controller
state restoration.

### Resume and reproducibility

Ctrl+C or SIGTERM requests a graceful stop. An unfinished episode is replayed on the next invocation; committed
episodes remain in a `.hdf5.inprogress` file. Repeat the same command with `--resume`. Completed task files are
checked and skipped. NumPy and Python RNG checkpoints preserve the reset sequence when continuing a task.
An HDF5 that was damaged by a power loss or SIGKILL requires separate inspection; the script will not silently
overwrite it. Final `.hdf5` files are published only after the selected task's episodes have been committed and
flushed.

Resume checks include replay-input hashes, source demo IDs, preprocessing settings, runtime package versions,
this script's hash, local LIBERO Python source hashes, and asset size/mtime inventory. Changing these settings,
editing LIBERO, or expanding a three-demo smoke run into a full run requires a new output directory. These checks
protect the local replay configuration; outputs are not claimed to be byte-identical to a published RLDS release.

The processed HDF5 is ready for tolerance generation and annotation. The current OpenPI training loader still
requires a LeRobot export; this entry point does not perform that conversion or generate tolerance labels.

Run the CPU-only equivalence and checkpoint tests with the main project environment:

```bash
.venv/bin/python -m pytest -q examples/libero/regenerate_libero_dataset_test.py
```
