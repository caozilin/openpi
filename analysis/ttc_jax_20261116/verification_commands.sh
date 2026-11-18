#!/usr/bin/env bash
# Run from the openpi repository root. Local test IPC/loopback must be permitted.
# These commands use synthetic inputs and never start production training.
set -euo pipefail
export JAX_PLATFORMS=cpu
export XDG_CACHE_HOME=/tmp/openpi_ttc_cache
export SWANLAB_SAVE_DIR=/tmp/openpi_ttc_swanlab
export SWANLAB_LOG_DIR=/tmp/openpi_ttc_swanlog
.venv/bin/pytest -q src/openpi/models/rtc_test.py src/openpi/models/pi0_test.py \
  src/openpi/policies/policy_ttc_test.py src/openpi/serving/ttc_test.py \
  src/openpi/serving/tasktol_sampling_test.py scripts/train_ttc_test.py scripts/train_test.py \
  scripts/serve_policy_franka_test.py \
  packages/openpi-client/src/openpi_client/ttc_action_chunk_broker_test.py \
  packages/openpi-client/src/openpi_client/msgpack_numpy_test.py --disable-warnings
XLA_FLAGS=--xla_force_host_platform_device_count=2 .venv/bin/pytest -q \
  src/openpi/models/rtc_test.py -k sharded --disable-warnings
.venv/bin/python analysis/ttc_jax_20261116/verify_legacy.py
.venv/bin/python analysis/ttc_jax_20261116/verify_lerobot.py
