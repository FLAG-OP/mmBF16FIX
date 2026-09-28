#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
from pathlib import Path

# The kernel-level contract intentionally tests the fast lowering.  Set it
# before FlagGems/MM are imported so an ambient XMLIR_MATMUL_FAST_MODE=0 cannot
# turn this into a baseline-only run.
os.environ["XMLIR_MATMUL_FAST_MODE"] = "1"

import torch
import flag_gems

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent / "kernel/triton_level.py"
SOURCE = Path(os.environ.get("MM_SOURCE", DEFAULT))
sys.path.insert(0, str(Path(flag_gems.__file__).parent / "runtime/backend"))
spec = importlib.util.spec_from_file_location("mm_under_test", SOURCE)
assert spec and spec.loader
mmmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmmod)

SHAPES = [
    (8, 16, 32),
    (33, 47, 63),
    (128, 256, 512),
    (513, 257, 129),
    (1024, 1024, 1024),
]


def make(m, n, k, layout):
    if layout == "contiguous":
        a = torch.randn(m, k, dtype=torch.bfloat16, device=flag_gems.device)
        b = torch.randn(k, n, dtype=torch.bfloat16, device=flag_gems.device)
    elif layout == "transposed_a":
        a = torch.randn(k, m, dtype=torch.bfloat16, device=flag_gems.device).t()
        b = torch.randn(k, n, dtype=torch.bfloat16, device=flag_gems.device)
    elif layout == "transposed_b":
        a = torch.randn(m, k, dtype=torch.bfloat16, device=flag_gems.device)
        b = torch.randn(n, k, dtype=torch.bfloat16, device=flag_gems.device).t()
    else:
        raise ValueError(layout)
    return a, b


def metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    max_abs = diff.abs().max().item() if diff.numel() else 0.0
    denom = expected.abs().clamp_min(1e-6)
    max_rel = ((diff.abs() / denom)).max().item() if diff.numel() else 0.0
    ok = torch.allclose(actual, expected, rtol=2e-2, atol=2e-2, equal_nan=True)
    return ok, max_abs, max_rel


records = []
failures = 0

# XMLIR_MATMUL_FAST_MODE uses a 16-bit SDNN lowering for BF16.  These diagonal
# outer products check that the fast path retains BF16 range, not only values
# that happen to be FP16-representable.
for value in (1.0, 3.14159265, 65504.0, 65520.0, 1.0e5, 1.0e10, 1.0e20, 1.0e-8):
    k = 64
    a = torch.zeros(1, k, dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.zeros(k, 1, dtype=torch.bfloat16, device=flag_gems.device)
    a[0, :] = value
    b[:, 0] = 1.0
    expected = torch.mm(a, b)
    actual = mmmod.mm(a, b)
    torch.cuda.synchronize()
    ok, max_abs, max_rel = metrics(actual, expected)
    record = {
        "layout": "bf16_range",
        "value": value,
        "status": "PASS" if ok else "FAIL",
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
    }
    records.append(record)
    print(record, flush=True)
    failures += not ok

for layout in ("contiguous", "transposed_a", "transposed_b"):
    for m, n, k in SHAPES:
        a, b = make(m, n, k, layout)
        expected = torch.mm(a, b)
        actual = mmmod.mm(a, b)
        torch.cuda.synchronize()
        ok, max_abs, max_rel = metrics(actual, expected)
        # Exercise out= API on representative shapes.
        if m <= 128:
            out = torch.empty((m, n), dtype=expected.dtype, device=expected.device)
            returned = mmmod.mm_out(a, b, out=out)
            torch.cuda.synchronize()
            ok_out, out_abs, out_rel = metrics(returned, expected)
            ok = ok and ok_out and returned is out
            max_abs = max(max_abs, out_abs)
            max_rel = max(max_rel, out_rel)
        record = {
            "layout": layout,
            "m": m,
            "n": n,
            "k": k,
            "status": "PASS" if ok else "FAIL",
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
        }
        records.append(record)
        print(record, flush=True)
        failures += not ok

print(f"TOTAL: {len(records)-failures} passed, {failures} failed")
json_path = Path(os.environ.get("MM_ACCURACY_JSON", "/dev/null"))
if str(json_path) != "/dev/null":
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "source": str(SOURCE),
        "xmlir_matmul_fast_mode": os.environ.get("XMLIR_MATMUL_FAST_MODE", "unset"),
        "records": records,
    }, indent=2))
raise SystemExit(bool(failures))
