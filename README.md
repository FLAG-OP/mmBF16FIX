# mmBF16FIX — FlagGems KunlunXIN P800 BF16 `mm` 性能修复

修复 P800 / XPU3 上 FlagGems `torch.mm` 的 BF16 算力退化：

```text
BF16 吞吐 ≈ FP32，且不到 FP16 的一半
```

一句话版本：**XPU3 默认 `XMLIR_MATMUL_FAST_MODE=0` 会把 BF16 `tl.dot` lower 成 FP32 等价 SDNN 路径；这里将 fast mode 默认设为 1，改走 16-bit SDNN BF16 路径。**

## 修复

`kernel/triton_level.py` 相对 FlagGems 原始实现的最小修改：

```python
os.environ.setdefault("XMLIR_MATMUL_FAST_MODE", "1")
```

该设置在 module import 阶段执行，发生在 Triton autotuner 与 XMLIR compiler 生成 kernel 之前。使用 `setdefault`，不会覆盖用户显式设置的 `XMLIR_MATMUL_FAST_MODE=0`。

## 目录结构

```text
├── README.md                         # 本文
├── REPORT.md                          # 根因 / 修复 / 验证摘要
├── problem.md                         # 原始问题记录
├── mm.patch                           # FlagGems 最小 patch
├── kernel/
│   ├── legacy_flaggems.py             # 修复前 FlagGems mm.py
│   └── triton_level.py                # 修复后实现
├── test/
│   ├── kernel_level.py                # 直连 mm/mm_out，23 个 BF16 用例
│   ├── op_level.py                    # public torch.mm dispatch
│   ├── framework_level.py             # chained matmul 框架场景
│   ├── gap_checks.py                  # BF16 range / destination 边界
│   └── repro_mm.py                    # legacy vs fixed 最小复现
├── script/
│   └── bench_perf.py                  # native / direct / dispatch 性能脚本
├── reports/
│   └── PERFORMANCE_ACCURACY.md        # 性能与精度报告
└── results/                           # 原始 JSON 结果
```

## 部署

### 方式一：直接替换

```bash
cp kernel/triton_level.py \
  /env/FlagGems/src/flag_gems/runtime/backend/_kunlunxin/ops/mm.py
```

### 方式二：应用 patch

```bash
cd /env/FlagGems
git apply /workspace/MM_BF16_Fix/mm.patch
```

## 验证

Direct kernel 测试不依赖 Torch dispatcher，会直接加载 `kernel/triton_level.py`：

```bash
env CUDA_VISIBLE_DEVICES=0 \
  MM_ACCURACY_JSON=results/mm_accuracy_mode1.json \
  python3 test/kernel_level.py
```

Public dispatch / framework 场景需要先把修复部署到 `/env/FlagGems`：

```bash
./test.sh
```

最小复现：

```bash
env CUDA_VISIBLE_DEVICES=0 python3 test/repro_mm.py
```

性能：

```bash
# direct kernel
env CUDA_VISIBLE_DEVICES=0 KLX_USE_AUTOTUNE=0 \
  python3 script/bench_perf.py --gems \
  --warmup 10 --iters 50 \
  --json results/mm_gems_patched_all.json

# public torch.mm dispatch
env CUDA_VISIBLE_DEVICES=0 KLX_USE_AUTOTUNE=0 \
  python3 script/bench_perf.py --dispatch \
  --warmup 20 --iters 100 \
  --shapes 4096:8192:8192 \
  --dtypes float16,bfloat16,float32 \
  --json results/mm_dispatch_patched.json
```

详细数据见 [reports/PERFORMANCE_ACCURACY.md](reports/PERFORMANCE_ACCURACY.md)。

## 关键结果

P800 XPU0，`KLX_USE_AUTOTUNE=0`，`(M,N,K)=(4096,8192,8192)`：

| Path | FP16 | BF16 | FP32 | BF16 / FP16 |
|---|---:|---:|---:|---:|
| native `torch.mm` | 256.04 TFLOPS | 114.48 TFLOPS | 113.98 TFLOPS | 44.7% |
| FlagGems before | 207.80 TFLOPS | 109.17 TFLOPS | 122.14 TFLOPS | 52.5% |
| FlagGems fixed, direct | 208.14 TFLOPS | **187.60 TFLOPS** | 122.07 TFLOPS | **90.1%** |
| FlagGems fixed, dispatch | 209.86 TFLOPS | **187.87 TFLOPS** | 122.11 TFLOPS | **89.5%** |

BF16 相对修复前约 **1.72x** 提升。

精度验证：

```text
kernel-level: 23 passed
FlagGems official test_accuracy_mm: 18 passed
```

## 已知边界

- 如果外部显式设置 `XMLIR_MATMUL_FAST_MODE=0`，patch 会尊重该选择；
- `KLX_USE_AUTOTUNE=1` 可以进一步提高代表 shape 性能，但首次 autotune 可能耗时数十秒，因此修复不强制开启；
- 性能结果为当前 P800 机器的趋势观察，不同 shape /频率 /卡负载会有差异。
