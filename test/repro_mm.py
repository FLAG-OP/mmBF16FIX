#!/usr/bin/env python3
"""Minimal legacy-vs-fixed BF16 MM reproduction.

XMLIR reads the matmul mode while the compiler backend is initialized, so each
variant is launched as a separate Python process.  Using one process can reuse
the first JIT compilation and hide the lowering difference.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child() -> int:
    import importlib.util
    import time

    import torch
    import flag_gems

    source = ROOT / ("kernel/legacy_flaggems.py" if os.environ["MM_REPRO_MODE"] == "legacy" else "kernel/triton_level.py")
    sys.path.insert(0, str(Path(flag_gems.__file__).parent / "runtime/backend"))
    spec = importlib.util.spec_from_file_location(f"mm_{os.environ['MM_REPRO_MODE']}", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    M = N = K = 4096
    a = torch.randn(M, K, dtype=torch.bfloat16, device=flag_gems.device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=flag_gems.device)
    expected = torch.mm(a, b)
    actual = module.mm(a, b)
    torch.cuda.synchronize()
    ok = torch.allclose(
        actual.float(), expected.float(), rtol=2e-2, atol=2e-2, equal_nan=True
    )

    start = time.perf_counter()
    for _ in range(10):
        module.mm(a, b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / 10 * 1000
    tflops = 2 * M * N * K / ms / 1e9
    label = os.environ["MM_REPRO_MODE"]
    print(f"{label:12s} {'PASS' if ok else 'FAIL'} {ms:8.3f} ms {tflops:8.2f} TFLOPS")
    return not ok


def main() -> int:
    all_ok = True
    for mode, xmlir_mode in (("legacy", "0"), ("fixed", "1")):
        env = os.environ.copy()
        env.update(
            MM_REPRO_MODE=mode,
            XMLIR_MATMUL_FAST_MODE=xmlir_mode,
            TRITON_CACHE_DIR=tempfile.mkdtemp(prefix=f"mm-{mode}-cache-"),
        )
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env)
        all_ok = result.returncode == 0 and all_ok
    print("PASS: fixed BF16 MM matches native reference" if all_ok else "FAIL")
    return not all_ok


if __name__ == "__main__":
    raise SystemExit(child() if "MM_REPRO_MODE" in os.environ else main())
