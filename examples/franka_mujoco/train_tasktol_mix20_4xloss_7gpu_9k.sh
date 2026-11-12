#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

exec uv run --locked scripts/train.py pi05_tasktol_mix20_4xloss_7gpu_9k_joint32 "$@"
