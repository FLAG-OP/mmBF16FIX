# KunlunXIN P800 BF16 `mm` 性能退化报告

## 问题

`problem.md` 原始记录：

```text
P800-BF16精度下算力退化严重，torch.mm 下 BF16 吞吐≈FP32，不到FP16的一半
```

代表 shape `(4096,8192,8192)`：

```text
native FP16   256.04 TFLOPS
native BF16   114.48 TFLOPS
native FP32   113.98 TFLOPS
```

BF16 与 FP32 基本相同，说明问题不在普通 Python wrapper 或内存布局，而在 XPU3 编译 lowering。

## 根因

FlagGems `mm_kernel` 的 TTIR 保持 BF16：

```text
tt.dot tensor<...xbf16> * tensor<...xbf16> -> tensor<...xf32>
```

默认 `XMLIR_MATMUL_FAST_MODE=0` 时，TTXIR 变成：

```text
triton_sdnn.dma in bf16 -> f32
triton_sdnn.mma f32 × f32 -> f32
```

BF16 输入进入 SDNN MMA 前被展开成 FP32 local memory，实际执行 FP32 MMA，吞吐自然与 FP32 相同。

开启 fast mode 后：

```text
triton_sdnn.dma in 16-bit -> 16-bit
triton_sdnn.mma 16-bit × 16-bit -> f32 accumulator
```

该路径保留 BF16 range / 语义，不是简单的 BF16→FP16 数值转换。

## 修复

```python
os.environ.setdefault("XMLIR_MATMUL_FAST_MODE", "1")
```

实现见 `kernel/triton_level.py`，最小 diff 见 `mm.patch`。

## 效果

| Path | FP16 | BF16 | FP32 |
|---|---:|---:|---:|
| FlagGems before | 207.80 | 109.17 | 122.14 |
| FlagGems fixed, direct | 208.14 | **187.60** | 122.07 |
| FlagGems fixed, dispatch | 209.86 | **187.87** | 122.11 |

单位：TFLOPS。BF16 从 FP16 的 52.5% 恢复到约 90%。

## 验证

- `test/kernel_level.py`：23 passed；
- `test/op_level.py`：public `torch.mm` / `out=` API 通过；
- `test/framework_level.py`：chained BF16 matmul 场景通过；
- `test/gap_checks.py`：BF16 range 超出 FP16 的值、边界行为通过；
- FlagGems 官方 `test_accuracy_mm`：18 passed。

## 后续优化方向

1. **P0：BF16 专属 tile 配置**：离线 sweep `BLOCK_M/BLOCK_N/BLOCK_K`、warps、
   stages 与 grid 策略，按 large / skinny / non-aligned shape bucket 固化 config
   cache；目标是保留 `KLX_USE_AUTOTUNE=0` 的即时启动行为，同时接近手动 autotune
   的约 281 TFLOPS。
2. **P0：性能回归门槛**：覆盖 small / medium / large / skinny GEMM 与 transpose
   布局，建议 large compute-bound 场景至少检查 `BF16/FP16 >= 0.75`、
   `BF16/FP32 >= 1.3`。
3. **P1：lowering 对比**：检查 fast mode 后 BF16 与 FP16 的 TTXIR、SDNN 指令、
   local-memory layout、pipeline 和 occupancy，定位剩余 10%~15% 差距。
4. **P1：布局与访存**：优化 transpose、尾块、C 写回向量化、小 M skinny GEMM
   与 `mm_out` 路径。
5. **P1：E2E 验证**：用真实 BF16 模型收集端到端吞吐、profiler 中 `mm` 占比和
   shape 分布，确认大 GEMM 收益且小 shape 不回归。
6. **P2：工程化与上游协同**：建设 tuned config cache / benchmark 脚本，并与
   XMLIR / XPU3 上游明确 BF16 默认 lowering 语义和回归测试。

完整数据与 lowered IR 证据见 [reports/PERFORMANCE_ACCURACY.md](reports/PERFORMANCE_ACCURACY.md)。
