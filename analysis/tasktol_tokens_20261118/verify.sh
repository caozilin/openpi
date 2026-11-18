#!/usr/bin/env bash
# From the repository root. No downloads, robot connections, or production runs.
set -euo pipefail
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_force_host_platform_device_count=2
export XDG_CACHE_HOME=/tmp/openpi_ttc_cache
.venv/bin/python -m pytest -q --disable-warnings --tb=short \
  src/openpi/models/tasktol_test.py src/openpi/models/rtc_test.py src/openpi/models/pi0_test.py \
  src/openpi/policies/policy_ttc_test.py src/openpi/training/franka_mujoco_joint32_test.py \
  src/openpi/training/libero_joint_test.py packages/openpi-client/src/openpi_client/ttc_action_chunk_broker_test.py
