#!/usr/bin/env python3
"""A model-style chained matmul check through public torch.mm dispatch."""
import os

os.environ["XMLIR_MATMUL_FAST_MODE"] = "1"

import torch
import flag_gems

M, K, N, R = 64, 128, 96, 80
x = 0.1 * torch.randn(M, K, dtype=torch.bfloat16, device=flag_gems.device)
w1 = 0.1 * torch.randn(K, N, dtype=torch.bfloat16, device=flag_gems.device)
w2 = 0.1 * torch.randn(N, R, dtype=torch.bfloat16, device=flag_gems.device)

# Include a transposed second weight, a common framework storage pattern.
w3_base = 0.1 * torch.randn(N, R, dtype=torch.bfloat16, device=flag_gems.device)
w3 = w3_base.t()

expected = torch.mm(torch.mm(torch.mm(x, w1), w2), w3)
with flag_gems.use_gems():
    actual = torch.mm(torch.mm(torch.mm(x, w1), w2), w3)
torch.cuda.synchronize()

ok = torch.allclose(
    actual.float(), expected.float(), rtol=2e-2, atol=2e-2, equal_nan=True
)
print(f"[{'PASS' if ok else 'FAIL'}] chained BF16 matmul framework scenario")
raise SystemExit(not ok)
