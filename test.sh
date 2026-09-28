#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export KLX_USE_AUTOTUNE="${KLX_USE_AUTOTUNE:-0}"
export XMLIR_MATMUL_FAST_MODE=1

python3 test/kernel_level.py
python3 test/op_level.py
python3 test/gap_checks.py
python3 test/framework_level.py
