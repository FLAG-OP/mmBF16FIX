#!/usr/bin/env python3
"""Public torch.mm dispatch tests for the deployed KunlunXIN MM operator."""
import os

# The deployment defaults to fast mode; make this test explicit.
os.environ["XMLIR_MATMUL_FAST_MODE"] = "1"

import torch
import flag_gems

PASS = 0
FAIL = 0


def check(name, a, b):
    global PASS, FAIL
    expected = torch.mm(a, b)
    with flag_gems.use_gems():
        actual = torch.mm(a, b)
    torch.cuda.synchronize()

    ok = torch.allclose(
        actual.float(), expected.float(), rtol=2e-2, atol=2e-2, equal_nan=True
    )
    if not ok:
        FAIL += 1
        print(f"  [FAIL] {name}")
    else:
        PASS += 1
        print(f"  [PASS] {name}")


shape = (128, 256, 512)
a = torch.randn(shape[0], shape[2], dtype=torch.bfloat16, device=flag_gems.device)
b = torch.randn(shape[2], shape[1], dtype=torch.bfloat16, device=flag_gems.device)
check("contiguous bf16 torch.mm", a, b)

a_t = torch.randn(shape[2], shape[0], dtype=torch.bfloat16, device=flag_gems.device).t()
check("transposed-a bf16 torch.mm", a_t, b)

b_t = torch.randn(shape[1], shape[2], dtype=torch.bfloat16, device=flag_gems.device).t()
check("transposed-b bf16 torch.mm", a, b_t)

out = torch.empty(shape[0], shape[1], dtype=torch.bfloat16, device=flag_gems.device)
expected = torch.mm(a, b)
with flag_gems.use_gems():
    returned = torch.mm(a, b, out=out)
torch.cuda.synchronize()
ok = returned is out and torch.allclose(
    out.float(), expected.float(), rtol=2e-2, atol=2e-2
)
print(f"  [{'PASS' if ok else 'FAIL'}] bf16 torch.mm out= API")
PASS += ok
FAIL += not ok

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(bool(FAIL))
