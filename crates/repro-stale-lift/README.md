# repro-stale-lift — 用真实 xray render 模块复现陈旧读

复现 brush 训练里 `eval/loss` 渲染偏少 (~4% density, 现象见
`experiments/output/260828-1316_rxa-static-refinefix/root_cause_report.md`) 的
陈旧读 bug。

## 结构（已接入真实模块）

`src/main.rs` 用**真实的** xray render 模块复刻静态重建训练循环：

```
canonical (inner XRaySplats)
  → lift_xray_splats_to_autodiff        (brush-xray-bwd)
  → render_xray (autodiff, Backward)     (brush-xray-bwd)
  → L1 loss vs 随机 GT → backward
  → AdamScaled 复刻 (per-component scaling [1,10] + momentum 跨步持久)
  → .inner() 写回 canonical (inner)
```

依赖：`brush-cube` / `brush-render` / `brush-xray` / `brush-xray-bwd`（不依赖
`brush-train`，避免数据集/deform/loss 重依赖）。

## 检查方式

每 100 步在**另一个 OS 线程**（`std::thread::spawn` + current-thread runtime，必然
不同 `StreamId`）上对比：

1. `render_xray`（autodiff/lift 路径）— **先做**，是 step 后第一个 drain canonical 的
   路径（等价 brush `eval_view` 的 render）；若读陈旧则密度偏少。
2. `render_xray_forward`（raw，无 lift）— 后做，此时 canonical 已落地，读真值。

打印 `ratio = ad_sum / fwd_sum`；若 <1 即复现陈旧读。

## 当前结果（负结果，2026-08-29）

**`ratio = 1.00000`，未复现陈旧读。** 已尝试的组合：

| 变量 | 结果 |
|---|---|
| 真实 `render_xray` / `render_xray_forward` / `lift_xray_splats_to_autodiff` | ratio=1.0 |
| AdamScaled 复刻（scaling [1,10] + momentum 跨步持久） | ratio=1.0 |
| 跨线程（std::thread spawn, 不同 StreamId） | ratio=1.0 |
| autodiff 先 / forward 后（autodiff 是第一个 drain） | ratio=1.0 |
| **`tokio::spawn`（work-stealing 线程迁移）+ `sleep` 强制迁移** | **step_cur 5↔6 迁移, 但 ratio=1.0** |
| 每步 `grad_remove` refine_weight readback（复刻 brush step()） | ratio=1.0 |
| 更大规模（8192 splats, 128×128）+ 欠密度增长场景 | ratio=1.0 |

## tokio 异步问题的结论

- **work-stealing 迁移是真的**：把训练循环放进 `tokio::spawn` 后，task 在 `.await`
  处被迁到不同 worker 线程（`step_cur` 在 5↔6 间切换）。而放在 `#[tokio::main]`
  的 `block_on` 里则永远停主线程（`step_cur` 恒 0）—— 这是本 crate 之前没迁移的原因。
- **但迁移 + 跨 stream shared_view 仍读正确**（ratio=1.0）。所以陈旧读**不是**
  work-stealing 迁移 + shared_view 单独能触发的。
- 根因报告 §6 也记录「`current_thread` 无效」：单线程（无迁移）下陈旧读仍在 ——
  与 shared_view 保值一致。两者共同指向：触发点在**渲染 pipeline 内部**
  （`resolve_tensor_float`/drain + cubecl kernel raw buffer 分配 + BindOp 绑定）。

## 与 brush 尚存的差异（待逼近）

1. **优化器封装**：brush 用 `OptimizerAdaptor<AdamScaled>` + 每步 `to_record()` /
   `load_record()`；本 crate 是手工复刻。
2. **规模**：brush 图像 862×634 / 14280 splats（内存池 buffer 复用压力大）；
   本 crate 64×64 / 2048。
3. **读回模式**：brush 每步读回 `num_visible`/`num_intersections`（`tr_execute`）、
   `collect_grads`/`collect_loss` 诊断读回、refine/densify 的 `gather_stats` 读回。

## 已知（有价值）结论

- 跨线程 `FusionTensor::clone` 确实产生新 `TensorId`（`shared_view`），但内容与
  真值一致（`tag_shared_view` 正确 materialize src 并共享 buffer）。
- work-stealing 迁移是真实存在的（`tokio::spawn` 下 step_cur 5↔6 切换），但单独
  迁移不触发陈旧读。
- 因此陈旧读的触发点仍在渲染 pipeline 内部，需在**真实 fit_static 训练**里加诊断
  探针定位（或增大本 crate 的规模逼近内存池复用压力）。

## 下一步

1. 在 `brush-xray-bwd` 的 `render_xray` 里加诊断探针（读回 `transforms_inner` 的
   `TensorId` + 内容 first3），在真实 fit_static 训练里对比 canonical，确认陈旧读
   是否仍在、是否依赖线程迁移；
2. 或把本 crate 规模拉到 14280 splats / 862×634，逼近 brush 的内存池 buffer 复用
   压力，看 ratio 是否偏离 1.0；
3. 重点追 burn-fusion 的 `submit`（异步 enqueue）/`submit_blocking`（drain）之间的
   `custom_channel` 双缓冲时序，以及 `MultiStream::drain` 是否漏掉未 flush 的
   pending op。
