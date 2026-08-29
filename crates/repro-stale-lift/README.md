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
| 更大规模（8192 splats, 128×128） + 欠密度增长场景 | ratio=1.0 |

关键观察：`step_cur` 恒为 0 —— 训练 future 在 `yield_now` 后**从不迁移线程**
（GPU 读回走 `submit_blocking` 阻塞当前线程，不真正 yield 到 tokio 调度器）。
而 brush 的 `xray_stream` 里 `step()` 与 `eval_view()` 之间读回会真正 suspend，
导致同一 task 在不同 worker 线程间迁移。

## 与 brush 尚存的差异（待逼近）

1. **线程迁移**：brush 的 step 与 eval 在**同一个** async task 里，读回真正 suspend
   时 task 被 steal 到别的 worker 线程；本 crate 用 `std::thread::spawn` 硬造跨线程，
   但跨线程 shared_view 读到的内容一致（见
   `root_cause_report.md` §10 —— shared_view 保值）。
2. **优化器封装**：brush 用 `OptimizerAdaptor<AdamScaled>` + 每步 `to_record()` /
   `load_record()`；本 crate 是手工复刻。
3. **规模/读回模式**：brush 每步读回 `num_visible`/`num_intersections`（`tr_execute`）
   且图像 862×634 / 14280 splats；本 crate 64×64 / 2048。

## 已知（有价值）结论

- 跨线程 `FusionTensor::clone` 确实产生新 `TensorId`（`shared_view`），但内容与
  真值一致（`tag_shared_view` 正确 materialize src 并共享 buffer）。
- 因此陈旧读**不是** shared_view / lift 本身，触发点更可能在渲染 pipeline 内部
  （`resolve_tensor_float`/drain + 自定义 cubecl kernel 的 raw buffer 分配 + BindOp
  绑定），且依赖**同一 task 跨线程迁移**（brush 的 work-stealing tokio）而非显式
  跨线程克隆。

## 下一步

1. 把 step 与 eval 放进**同一个** tokio task，用真实读回 suspend（大张量/慢读回）
   让 task 迁移线程，而不是 `std::thread::spawn` 硬造；
2. 或直接在 `brush-xray-bwd` 的 `render_xray` 里加诊断探针（读回 `transforms_inner`
   的 `TensorId` + 内容），在真实 fit_static 训练里对比 canonical；
3. 重点追 burn-fusion 的 `submit`（异步 enqueue）/`submit_blocking`（drain）之间的
   `custom_channel` 双缓冲时序，以及 `MultiStream::drain` 是否可能漏掉未 flush 的
   pending op。
