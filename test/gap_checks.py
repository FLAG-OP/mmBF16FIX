#!/usr/bin/env python3
"""Boundary checks around the BF16 MM fast path."""
import importlib.util
import os
from pathlib import Path

os.environ["XMLIR_MATMUL_FAST_MODE"] = "1"

import torch
import flag_gems

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "kernel/triton_level.py"
sys_path = Path(flag_gems.__file__).parent / "runtime/backend"
import sys
sys.path.insert(0, str(sys_path))
spec = importlib.util.spec_from_file_location("mm_gap_under_test", SOURCE)
op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(op)
results = []


def record(name, ok):
    results.append(ok)
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")


# Values beyond normal FP16 range must remain valid BF16 values.
k = 64
a = torch.zeros(1, k, dtype=torch.bfloat16, device=flag_gems.device)
b = torch.ones(k, 1, dtype=torch.bfloat16, device=flag_gems.device)
for value in (65520.0, 1.0e5, 1.0e10, 1.0e20):
    a.fill_(value)
    expected = torch.mm(a, b)
    actual = op.mm(a, b)
    torch.cuda.synchronize()
    record(
        f"bf16 range {value:g}",
        torch.allclose(actual.float(), expected.float(), rtol=2e-2, atol=2e-2),
    )

# Both inputs can be non-contiguous at the same time.  This complements the
# one-side-transposed cases in kernel_level.py.
M, N, K = 33, 47, 63
a = torch.randn(K, M, dtype=torch.bfloat16, device=flag_gems.device).t()
b = torch.randn(N, K, dtype=torch.bfloat16, device=flag_gems.device).t()
expected = torch.mm(a, b)
actual = op.mm(a, b)
torch.cuda.synchronize()
record(
    "both inputs transposed",
    torch.allclose(
        actual.float(), expected.float(), rtol=2e-2, atol=2e-2
    ),
)

raise SystemExit(not all(results))
