# P800 / FlagGems `torch.mm` BF16 性能问题分析

日期：2026-09-28

设备与环境：

```text
Device: KunlunXIN P800, XPU 0
Torch: 2.9.0+cu129 / torch_xmlir
FlagGems: /env/FlagGems
Kernel: _kunlunxin.ops.mm.mm_kernel
KLX_USE_AUTOTUNE=0
```

## 1. 现象

大 shape `torch.mm` 中：

```text
FP16   ≈ 208~256 TFLOPS
BF16   ≈ 109~114 TFLOPS
FP32   ≈ 114~122 TFLOPS
```

BF16 与 FP32 几乎相同，只有 FP16 的一半左右。

代表 shape：

```text
M=4096, N=8192, K=8192
```

native `torch.mm` 结果：

| dtype | ms | TFLOPS |
|---|---:|---:|
| FP16 | 2.147 | 256.04 |
| BF16 | 4.802 | 114.48 |
| FP32 | 4.823 | 113.98 |

FlagGems 修复前 direct kernel：

| dtype | ms | TFLOPS |
|---|---:|---:|
| FP16 | 2.646 | 207.80 |
| BF16 | 5.036 | 109.17 |
| FP32 | 4.501 | 122.14 |

## 2. 根因

FlagGems `mm_kernel` 的 TTIR 一开始保持 BF16：

```text
tt.dot tensor<...xbf16> * tensor<...xbf16> -> tensor<...xf32>
```

但 `XMLIR_MATMUL_FAST_MODE=0` 时，XPU3 TTXIR lowering 变成：

```text
triton_sdnn.dma in memref<...xbf16> -> memref<...xf32>
triton_sdnn.mma memref<...xf32> × memref<...xf32> -> memref<...xf32>
```

也就是说，BF16 输入在进入 SDNN MMA 前被展开到 FP32 local memory，MMA 也按
FP32 操作执行，因此吞吐自然与 FP32 相同。

Triton autotuner 中还有对应逻辑：

```python
if str(A.dtype) == "torch.bfloat16" and matmul_mode == 0:
    ele_bytes = ele_bytes * 2
```

默认模式按 4 bytes/element 估算 BF16 block 资源，进一步导向保守的 FP32 等价路径。

开启：

```text
XMLIR_MATMUL_FAST_MODE=1
```

后，TTXIR 变为 16-bit SDNN local memory 与 MMA：

```text
triton_sdnn.dma in memref<...xf16> -> memref<...xf16>
triton_sdnn.mma memref<...xf16> × memref<...xf16> -> memref<...xf32>
```

TTIR 和 public tensor dtype 仍是 BF16。这里的 `f16` memref 是 XPU3 内部 16-bit
SDNN lowering 表示，不等于把用户 BF16 数值转换成 FP16。精度范围测试证明 BF16
动态范围被保留。

## 3. 修复

在 `_kunlunxin/ops/mm.py` import 阶段加入：

```python
os.environ.setdefault("XMLIR_MATMUL_FAST_MODE", "1")
```

完整 patch 见 `mm.patch`。

设计点：

1. 在 autotuner/config generator 和 compiler 工作前设置。
2. 只设置默认值，不覆盖用户显式配置。
3. 不强制 `KLX_USE_AUTOTUNE=1`，避免引入首次调优等待。
4. `mm` 与 `mm_out` 共用同一个 `mm_kernel`，都能受益。

## 4. 修复后性能

### 代表 shape：`(4096,8192,8192)`

`KLX_USE_AUTOTUNE=0`：

| Path | FP16 | BF16 | FP32 |
|---|---:|---:|---:|
| native `torch.mm` | 256.04 | 114.48 | 113.98 |
| FlagGems before | 207.80 | 109.17 | 122.14 |
| FlagGems fixed, direct | 208.14 | **187.60** | 122.07 |
| FlagGems fixed, dispatch | 209.86 | **187.87** | 122.11 |

单位：TFLOPS。

BF16：

```text
修复前: 109.17 TFLOPS
修复后: 187.60 TFLOPS
提升:   1.72x

修复前 BF16/FP16: 52.5%
修复后 BF16/FP16: 90.1%
```

public `torch.mm` dispatch 与 direct kernel 差距很小，说明不是 Python wrapper
主导。

### 多 shape direct kernel

| shape | FP16 | BF16 fixed | FP32 |
|---|---:|---:|---:|
| `(512,512,512)` | 7.94 | 6.19 | 7.63 |
| `(1024,1024,1024)` | 59.49 | 49.01 | 63.76 |
| `(4096,4096,4096)` | 204.02 | 173.88 | 116.79 |
| `(4096,8192,8192)` | 208.14 | 187.60 | 122.07 |

小 shape 主要受 launch/compiler wrapper 开销影响；大 shape 进入 SDNN 计算域后
BF16 恢复正常。

## 5. 精度验证

### 自定义用例

`test/kernel_level.py` 覆盖：

- contiguous；
- A transpose；
- B transpose；
- shape `(8,16,32)`、`(33,47,63)`、`(128,256,512)`、`(513,257,129)`、
  `(1024,1024,1024)`；
- `mm` 与 `mm_out`；
- BF16 range outer-product。

比较 native BF16 `torch.mm`：

```python
torch.allclose(actual, expected, rtol=2e-2, atol=2e-2, equal_nan=True)
```

结果：

```text
TOTAL: 23 passed, 0 failed
```

### BF16 range 检查

输入值包括：

```text
1.0
3.14159265
65504.0
65520.0
1.0e5
1.0e10
1.0e20
1.0e-8
```

其中 `65520`、`1e5`、`1e10`、`1e20` 超出 FP16 正常表示范围。fast mode 输出仍与
native BF16 对齐，说明它不是简单的 BF16→FP16 数值转换。

例如：

```text
value=65520:    PASS, max_abs_err=0
value=1e5:      PASS, max_abs_err=0
value=1e10:     PASS, max_abs_err=0
value=1e20:     PASS, max_rel_err=0.00578
```

### FlagGems 官方测试

部署后运行：

```bash
env CUDA_VISIBLE_DEVICES=0 KLX_USE_AUTOTUNE=0 \
  pytest -q /env/FlagGems/tests/test_blas_ops.py::test_accuracy_mm
```

结果：

```text
18 passed, 1 warning
```

## 6. 可选 autotune

当前环境设置了 `KLX_USE_AUTOTUNE=0`，修复保持该选择。

如果在代表 shape 上允许首次调优开销：

```bash
env CUDA_VISIBLE_DEVICES=0 \
  KLX_USE_AUTOTUNE=1 XMLIR_MATMUL_FAST_MODE=1 \
  python3 script/bench_perf.py --gems \
  --shapes 4096:8192:8192 --dtypes bfloat16
```

实测：

```text
BF16: 281.21 TFLOPS
FP16: 333.13 TFLOPS
FP32: 133.42 TFLOPS
```

最佳配置为：

```text
BLOCK_M=480, BLOCK_N=512, BLOCK_K=512, num_warps=5
```

但该调优首次运行耗时明显，且最佳 tile 随 shape 变化，因此不建议作为默认强制行为。

## 7. 后续优化方向

当前修复已经把 BF16 从 FP32 等价路径恢复到 16-bit SDNN fast path，代表 shape 上
BF16/FP16 达到约 90%。这不是正确性问题的延续，而是剩余 10%~15% 的性能空间。
后续优化建议按以下优先级推进。

### P0：BF16 专属 tile 配置与 shape 分桶

代表 shape 的手动 autotune 已经证明默认 config 不是 BF16 的最优点：

```text
默认 config BF16:  187.60 TFLOPS
autotune BF16:     281.21 TFLOPS
autotune FP16:     333.13 TFLOPS
```

下一步不应简单强制开启 runtime autotune，而是离线完成 BF16 专属 sweep，并把结果
固化成 config 表：

1. Sweep 维度：
   - `BLOCK_M / BLOCK_N / BLOCK_K`；
   - `num_warps`；
   - `num_stages` / software pipeline 相关参数；
   - grid 分块与 CTA 复用策略。
2. 按矩阵规模分桶，而不是只记录 exact shape：
   - large-square / large-rectangular compute-bound GEMM；
   - `M` 较小的 decode / inference linear；
   - `N` 或 `K` 很小的 skinny GEMM；
   - 128 对齐、非 128 对齐、尾块比例较高的 shape。
3. 为常见 LLM、embedding / projection、MLP shape 建立 tuned config cache。
4. Cache 查找优先 exact match，再做 bucket fallback；避免把数十秒的首次调优
   暴露给普通用户。

验收指标：在保留 `KLX_USE_AUTOTUNE=0` 交互路径的前提下，让高频 shape 不需要
在线调优也能接近 offline autotune 结果。

### P0：建立 BF16 / FP16 性能基线与回归门槛

补充一个固定 benchmark 矩阵，覆盖：

```text
small:   M=128/256,  N=512/1024,   K=512/1024
medium:  M=512/1024, N=2048/4096,  K=2048/4096
large:   M=2048/4096,N=4096/8192,  K=4096/8192
skinny:  M=1/8/32,   N=1024~8192,  K=1024~8192
```

同时记录：

- contiguous、A transpose、B transpose；
- `mm` 与 `mm_out`；
- public dispatch 与 direct kernel 的差距；
- XPU 频率、占用进程和迭代稳定性。

建议报告两个门槛：

```text
BF16 / FP16 >= 0.75  # large compute-bound GEMM
BF16 / FP32  >= 1.3  # 防止退回 FP32 等价路径
```

阈值可以根据后续更多 shape 数据调整，关键是每次 FlagGems / XMLIR 更新后自动发现
lowering 或 tile config 回归。

### P1：检查剩余 lowering 差异

BF16 与 FP16 标称峰值通常相同，但实测低 5%~15% 并不罕见。为了确认 P800 剩余
差距来自软件调度而不是硬件路径，应对比 fast mode 下 BF16 与 FP16 的：

1. TTIR / TTXIR：
   - SDNN MMA 输入是否保持 16-bit；
   - 是否仍存在 BF16 到 FP32 的额外展开；
   - BF16 是否多出 layout cast / swizzle / local-memory 转换。
2. MMA 指令形态：
   - instruction tile shape；
   - accumulator dtype 与寄存器压力；
   - local memory bank / vector width。
3. Pipeline 行为：
   - DMA 与 MMA overlap；
   - stages 数；
   - CTA occupancy；
   - tail wave 与 grid 切分。

输出应是可比较的 IR/profiling 证据，而不是只比较最终 TFLOPS。

### P1：布局与访存优化

当前 correctness 用例已经覆盖 transpose，但 transpose、非 128 对齐和尾块仍是常见
性能风险。可以继续尝试：

- A/B local-memory layout 与 swizzle 参数；
- C store 的 vectorization 与 alignment；
- transpose 场景下选择不同的 tile 长；
- 减少 `mm_out` 输出路径上的额外检查或转换；
- 对小 M 场景采用更适合 skinny GEMM 的并行方式。

这些优化必须逐项对比 native `torch.mm` 和 FlagGems fixed path，避免只在一个
代表 shape 上过拟合。

### P1：模型级 E2E 验证

Kernel-level 吞吐不能完全代表实际收益。需要选择 BF16 推理与训练场景，分别在
FlagGems 开启 / 关闭时收集：

- end-to-end latency / throughput；
- `torch.profiler` 中 `mm` 的调用次数与总耗时；
- `mm` shape 分布；
- host launch 与 device kernel 占比；
- 修复前、修复后、tuned config 后三组结果。

重点不是证明所有模型都有同等幅度加速，而是确认：

1. 已修复的 large GEMM 场景能转化为端到端收益；
2. 小 shape / skinny GEMM 没有因额外配置查找或 kernel 选择变慢；
3. 精度指标与 native BF16 对齐。

### P2：Autotune 工程化

如果要把 offline sweep 变成可持续能力，建议增加：

- JSON / SQLite tuned config cache；
- dtype / shape bucket 元数据；
- 编译结果与缓存版本号；
- 单 shape benchmark 与 sweep 脚本；
- cache miss 时的保守 fallback；
- 清晰记录调优时长，避免隐藏冷启动。

默认策略仍应尊重部署环境的 `KLX_USE_AUTOTUNE` 设置，不把在线 autotune 作为
正确性修复的一部分。

### P2：与 XMLIR / XPU3 上游协同

本修复改变了默认 lowering mode。后续更适合上游长期维护的方向包括：

- 明确 `XMLIR_MATMUL_FAST_MODE` 在不同硬件代际和 dtype 下的语义；
- 将 BF16 16-bit SDNN fast path 作为 BF16 `tl.dot` 的默认合法 lowering；
- 提供 compiler flag / IR metadata 说明为什么选择 FP32 等价路径；
- 在 XMLIR 或 backend 测试中加入 BF16 lowering 断言；
- 与硬件文档对齐 BF16 / FP16 的实际峰值与限制。

这样可以避免未来 XMLIR 升级后同一问题通过环境变量静默回归。

## 8. 已尝试但不采用的方案

### BF16 输入先转 FP16，再走 FP16 MMA

该方案需要额外分配 / 转换两个输入，并把中间输出转回 BF16。实测代表 shape：

```text
BF16 host convert + FP16 MMA + cast back: ~179 TFLOPS
XMLIR BF16 fast mode:                     ~188 TFLOPS
```

它比 fast mode 更慢，还引入额外内存流量和中间 dtype 语义问题。

直接在 Triton kernel 内做 `bf16 -> fp16`cast 也不稳定：XPU3 编译路径会出现
SRAM 资源不足或 unsupported DMA type。因此没有采用。

### 强制 KLX_USE_AUTOTUNE=1

Autotune 在代表 shape 上可以把 BF16 提升到约 281 TFLOPS，但首次调优可能需要
数十秒，且当前部署显式选择 `KLX_USE_AUTOTUNE=0`。修复尊重该部署配置，只将其
作为可选手动策略。

## 9. 结论

问题不是 FlagGems `mm_kernel` 的 stride 或 shape 处理错误，而是 XPU3 对 BF16
`tl.dot` 的默认 lowering 走了 FP32 等价 SDNN 路径。

默认开启 `XMLIR_MATMUL_FAST_MODE=1` 后：

```text
BF16 从 FP32 等价性能恢复到 16-bit SDNN 性能；
大 shape BF16/FP16 达到约 85%~90%；
精度和 BF16 动态范围保持一致；
FP16/FP32 路径无正确性回归。
```
