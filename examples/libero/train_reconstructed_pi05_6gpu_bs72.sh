#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."

config_name=pi05_libero_reconstructed_nostate_6gpu_bs72
repo_id=caozilin/libero_reconstructed_full_20261101
python_bin="${OPENPI_PYTHON:-$PWD/.venv/bin/python}"

# Resolve both the local conversion output and the uploaded server layout.
if [[ -z "${HF_LEROBOT_HOME:-}" ]]; then
    if [[ -d /mnt/SharedData/czl/lerobot ]]; then
        export HF_LEROBOT_HOME=/mnt/SharedData/czl/lerobot
    else
        export HF_LEROBOT_HOME="$PWD/data/lerobot"
    fi
fi
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_LEROBOT_HOME/.cache/hf_datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
export ARROW_JEMALLOC_CONF="${ARROW_JEMALLOC_CONF:-background_thread:false}"

# Conversion and its full audit must finish before normalization or training.
"$python_bin" - "$HF_LEROBOT_HOME/$repo_id" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
audit_path = root / "meta/reconstruction_audit.json"
if not audit_path.is_file():
    raise SystemExit(f"Dataset audit is not ready: {audit_path}")
audit = json.loads(audit_path.read_text())
info = json.loads((root / "meta/info.json").read_text())
expected = {"episodes": 1875, "frames": 312374, "tasks": 40, "fps": 20}
for key, value in expected.items():
    info_key = key if key == "fps" else f"total_{key}"
    if audit.get(key) != value or info.get(info_key) != value:
        raise SystemExit(f"Unexpected dataset {key}; expected {value}")
for key in ("complete", "all_camera_pixels_equal", "all_state_action_equal", "reader_smoke_passed"):
    if audit.get(key) is not True:
        raise SystemExit(f"Dataset audit did not pass: {key}")
PY

mkdir -p "$HF_DATASETS_CACHE"
if [[ "${1:-}" == --compute-norm-stats ]]; then
    shift
    export JAX_PLATFORMS=cpu
    exec "$python_bin" scripts/compute_norm_stats.py --config-name "$config_name" "$@"
fi

IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#selected_gpus[@]}" -ne 6 ]]; then
    echo "Select exactly six GPUs in CUDA_VISIBLE_DEVICES (global batch 72, per GPU 12)." >&2
    exit 1
fi

exec "$python_bin" scripts/train.py "$config_name" \
    --exp-name Pi05_libero_reconstructed_nostate_6gpu_bs72_10k "$@"
