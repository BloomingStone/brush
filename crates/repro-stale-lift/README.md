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

**用简单的融合 op（含 Adam 式 momentum/variance 跨步持久 + 每步 lift + require_grad
+ 大张量压力 `_noise`）目前没有复现陈旧读 (ndiff=0, maxdiff=0)。**

这是一个有价值的负结果：说明陈旧读**不是**单纯由"带 pending op 的 fusion 张量 +
lift (from_inner)"触发，而是与 X-ray 渲染 pipeline 的特定结构有关，可能来自以下任一
（需继续排查）：

- `Fusion<MainBackendBase>::render_xray` 内部的 `resolve_tensor_float`（drain）+ BindOp
  绑定回融合流的组合；
- 自定义 cubecl kernel 的 raw buffer 分配（内存池）与融合流的交互；
- `render_xray` (autodiff) 里 `prep_nodes` / `refine_weight_holder` / `prep.finish`
  与 Backward pass 的组合（单独加 [1] zeros 已排除，但组合未排除）。

## 如何在这个 crate 里继续逼近

1. 把 `lift_to_autodiff` 的 `require_grad()` 也加进"检查"路径；
2. 增加真实 render 的融合张量个数/尺寸（out_img ≈ [H,W] f32，visible [N]，n_contrib [H,W] u32）；
3. 直接调用 `Fusion::render_xray`（Backward pass）对比 forward；
4. 打印 `resolve_tensor_float` 前后的 `TensorId` 与 buffer 地址。
