# 260829-1434 deform-rootcause — 训练 eval 与导出不一致的最终根因: fit_static 意外启用 deform

## 目的
定位 `fit_static` 训练 eval (~31 dB) 与导出 forward 渲染 (~23 dB, +0.05 proj 均匀雾)
不一致的根因。此前多轮调查 (2026-08-27 ~ 2026-08-29) 先后归因于 "cubecl 内存池 buffer
覆盖" → "lift 跨 stream shared_view 读到陈旧 canonical" → "渲染 pipeline 内部",
全部被本次调查推翻。

## 调查过程 (commit 追溯)
- 最小复现 crate `repro-stale-lift`: 应用无关复刻 (真实 lift/render + AdamScaled +
  tokio::spawn work-stealing 迁移 + grad_remove readback + 14280 splats/862x634 规模),
  **始终 ratio=1.0 无法复现**。
- 真实 `fit_static` 加逐层探针 (eval_view 读 canonical / lift 前后 / render 内部
  `splats.transforms.val()`), 发现:
  - lift 实际输出 primitive id = **4** (保值), 但 render 内 `val()` 读到 **155**。
  - burn-fusion `FusionTensor::clone` 打点证明 id 4→155 **不是 shared_view**
    (全程 stream=0, cur0=cur1=0, 无 SHARED_VIEW 事件)。
  - `create_empty_handle` 打点证明 id 155 是 `refine_weight_holder = zeros([1])`
    前后创建的**全新张量** → 即 `splats.transforms` 持有的根本不是 lift 的输出。
- 关键发现: `XRayTrainConfig::default()` 的 **`enable_deform = true`**
  (xray_train.rs:225), 而 fit_static **从未显式关闭** (注释声称 "删除形变场部分" 但
  代码没有) → fit_static 实际带 HexPlane deform 网络在跑。
  - eval 渲染 `deform_splats(canonical, deforms)` (形变后 splats, 新 TensorId, 正常),
  - 导出渲染 `render_xray_forward(&canonical)` (未形变 canonical)。

## 运行命令
```bash
# 探针运行 (debug 定位; 已还原所有探针)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/fit_static images/RXA_chest.dcm --points=5000 \
  --iters=3 --eval-every=1 --eval-views=1 --out=/tmp/opencode/fit_static_c1/

# 验证运行 (enable_deform=false 后)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/fit_static images/RXA_chest.dcm --points=5000 \
  --iters=160 --eval-every=40 --eval-views=2 --out=/tmp/opencode/fit_static_nodeform/
```

## 实验结果 (决定性验证)
`cfg.enable_deform = false` 后重跑 160 iter, 对比最后 eval pred (`gt_pred_00160.nrrd`)
与导出时 forward 重渲 (`gt_pred_10000_FWD.nrrd`):

```
maxdiff = 0.000000, meandiff = 0.000000e+00   (逐位一致)
proj mean 均 = 0.743   (原报告: eval 0.73 vs 导出 0.76)
```

## 结果分析
1. **根因**: fit_static 用 `XRayTrainConfig::default()` (enable_deform=true) 但未
   显式 `enable_deform=false` → 训练带 HexPlane deform 网络。
2. **机制**: deform 网络被训练来拟合 GT, 会把 canonical 里多余的密度 splats 移开,
   使**形变后**渲染更接近 GT (0.73, PSNR 31); canonical 本身保留雾, 导出未形变
   渲染暴露雾 (0.76, PSNR 23)。deform 吸收了雾 → 训练指标虚高。
3. **此前所有 "陈旧读 / shared_view 解析陈旧 buffer / 内存池 buffer 覆盖 /
   异步未 flush" 结论均为错误对比 (canonical vs 形变后 splats) 造成的假象**。
4. `repro-stale-lift` 无法复现的原因: 应用层没有 deform, 自然没有 "eval 渲染 ≠
   canonical" 的现象。

## 修复
1. `crates/brush-train/src/xray_train.rs`: 新增 **`create_static_xray_trainer`**,
   内部**强制** `config.enable_deform = false`, 从 API 层面杜绝静态训练带 deform。
2. `crates/brush-process/src/bin/fit_static.rs`: 改用 `create_static_xray_trainer`,
   清理所有误导性注释 ("与 fit_deform 一致, 删除形变场部分" 等)。
3. 删除调试产物 `crates/repro-stale-lift/` (复现目标为假象, 已无用途)。

## 后续待做
- 若静态场景确需 deform (如呼吸运动), 应改用 `fit_deform` 并确保导出时也施加
  相同 phase 的 deform, 而非渲染未形变 canonical。
- 可选: 在 `XRayTrainer::new` 对 `enable_deform=true` 但 `deform=None` 的不一致
  配置加 debug_assert, 提前暴露此类错误。
