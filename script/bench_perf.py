#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import flag_gems

BACKEND_ROOT = Path(flag_gems.__file__).parent / "runtime" / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
ROOT = Path(__file__).resolve().parents[1]
MM_SOURCE = Path(os.environ.get("MM_SOURCE", ROOT / "kernel/triton_level.py"))
spec = importlib.util.spec_from_file_location("mm_op_under_bench", MM_SOURCE)
assert spec is not None and spec.loader is not None
mm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm_module)
gems_mm = mm_module.mm

SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (4096, 8192, 8192),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json", type=Path)
    p.add_argument("--gems", action="store_true")
    p.add_argument("--dispatch", action="store_true", help="Benchmark torch.mm with FlagGems enabled")
    p.add_argument("--shapes", default="all", help="Comma-separated M:N:K list, or all")
    p.add_argument("--dtypes", default="all", help="Comma-separated dtype names, or all")
    args = p.parse_args()
    records = []
    if args.gems and args.dispatch:
        p.error("choose either --gems or --dispatch")
    mode = "gems_direct" if args.gems else "aten_dispatch" if args.dispatch else "native"
    if args.dispatch:
        flag_gems.enable()
    if args.shapes == "all":
        shapes = SHAPES
    else:
        shapes = []
        for item in args.shapes.split(","):
            m, n, k = map(int, item.split(":"))
            shapes.append((m, n, k))
    if args.dtypes == "all":
        dtypes = DTYPES
    else:
        dtype_map = {str(dtype).removeprefix("torch."): dtype for dtype in DTYPES}
        dtypes = [dtype_map[name.strip()] for name in args.dtypes.split(",")]
    print(
        f"mode={mode} device={flag_gems.device} "
        f"source={MM_SOURCE} XMLIR_MATMUL_FAST_MODE="
        f"{os.environ.get('XMLIR_MATMUL_FAST_MODE', '0')}"
    )
    print("| dtype | M,N,K | ms | TFLOPS |")
    print("|---|---|---:|---:|")
    for dtype in dtypes:
        for m, n, k in shapes:
            a = torch.randn(m,k,dtype=dtype,device=flag_gems.device)
            b = torch.randn(k,n,dtype=dtype,device=flag_gems.device)
            if args.gems:
                fn = lambda: gems_mm(a, b)
            else:
                fn = lambda: torch.mm(a, b)
            ms = bench(fn,args.warmup,args.iters)
            tflops = 2*m*n*k/ms/1e9
            print(f"| {str(dtype).removeprefix('torch.')} | {m},{n},{k} | {ms:.3f} | {tflops:.2f} |")
            records.append(dict(mode=mode,dtype=str(dtype).removeprefix('torch.'),m=m,n=n,k=k,ms=tflops if False else ms,tflops=tflops,warmup=args.warmup,iters=args.iters))
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(records,indent=2))

if __name__ == "__main__": raise SystemExit(main())
