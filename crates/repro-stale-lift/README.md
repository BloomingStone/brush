# repro-stale-lift

应用无关的最小复现 crate：在 burn-fusion 后端 (`burn_wgpu::Wgpu`) 上，复刻 brush
训练循环里 `lift_xray_splats_to_autodiff` → `lift_to_autodiff` 的行为，检查"带
pending op 的 fusion 张量经 lift 后是否读到陈旧 buffer"。

不依赖任何 brush crate，只依赖 `burn` / `burn-wgpu` / `burn-fusion`。

## 背景

见 `experiments/output/260828-1316_rxa-static-refinefix/root_cause_report.md`。

训练 eval/loss 的 autodiff 渲染 (`render_xray`) 读到了陈旧的 canonical 张量
(上一 step 的 FusionTensor，不同 TensorId + 不同内容)，导致渲染密度偏少 ~4%，
训练持续加密度 → 雾累积。而 forward 渲染 (`render_xray_forward`) 读到当前密度，
导出后暴露出雾 (~23 dB vs eval ~31 dB)。

## 这个 crate 做什么

1. 复刻 `lift_to_autodiff`（`AutodiffBackend::from_inner` + `wrap_ad_wgpu_float`）。
2. 维护一个 `Param<Tensor>` (inner)，每步用 Adam 式融合 op 更新
   (momentum / variance 跨步持久)，外加一个模拟 render 大张量压力的 `_noise`。
3. 每 500 步对比：
   - 直接读 `p.val().into_data()`（真值）
   - lift 后读 `lift_to_autodiff(p.val()).into_data()`（bug 路径）

## 运行

```bash
cargo build -p repro-stale-lift
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/debug/repro-stale-lift
```

## 当前状态（重要）

**用简单融合 op（累积更新 + 每步 lift + require_grad）目前没有复现陈旧读
(ndiff=0, maxdiff=0)。**

本 crate 里有一个**跨线程 shared_view 测试**（`std::thread::spawn` 里做 `val()` 克隆 +
读），确认了：

- 跨线程克隆**确实**产生新的 `TensorId`（shared_view）：主线程 `stream=0/id=3506`，
  子线程 `stream=9/id=3507`。
- 但 shared_view 读到的**内容与真值一致**（`v[0] == direct[0]`），即 `tag_shared_view`
  正确地 materialize 了 src 并共享了 buffer。

**结论**：`shared_view`（跨 stream 新 id）**不是**陈旧读的根因——它本身保值。之前
报告里"shared_view 解析到未落地陈旧 buffer"的机制描述需要修正：陈旧读的触发点在
**渲染 pipeline 内部**（`Fusion::render_xray` 的 `resolve_tensor_float`/drain + 自定义
cubecl kernel 的 raw buffer + BindOp 绑定的组合），而非 lift/shared_view 本身。

这是一个有价值的负结果，把搜索空间从"lift/shared_view"收窄到"渲染 pipeline"。

## 如何在这个 crate 里继续逼近

1. 直接调用 `Fusion::render_xray`（Backward pass），对比 forward，并打印
   `resolve_tensor_float` 前后的 `TensorId` / buffer 地址；
2. 复刻渲染 pipeline 的 buffer 分配模式（out_img [H,W]、visible [N]、n_contrib [H,W] 等
   raw cubecl buffer 与融合张量的交错分配/释放）；
3. 观察 `resolve_tensor_float` 返回的 CubeTensor 内容 vs `into_data` 的差异。
