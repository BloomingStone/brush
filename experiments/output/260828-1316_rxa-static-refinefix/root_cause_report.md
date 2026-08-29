# 训练 eval 与导出渲染不一致 — 根因定位报告

日期: 2026-08-29
对象: fit_static 静态 X-ray 重建 (brush-fix-voxelizer)
现象源: `experiments/output/260828-1316_rxa-static-refinefix/`

---

## 1. 摘要

训练循环内 `eval_view` 渲染出的 pred (autodiff 路径) 与导出 `.bin`/`.ply` 后经
`gs2volume` / `render_xray_forward` 渲染的结果**不一致**: eval 报告 PSNR ~31~34 dB
(看似很好), 而导出后 forward 渲染只有 ~23 dB, 多出一层 ~+0.05 proj 的**均匀雾**。

**根因**: 训练 loss/eval 使用的 autodiff 渲染路径 (`render_xray`, 内部经
`lift_xray_splats_to_autodiff` → `lift_to_autodiff` → `AutodiffBackend::from_inner`)
**读到了陈旧的 canonical 张量** (上一 step 的 FusionTensor, 不同的 TensorId + 不同的
内容), 而非当前的 canonical。因此训练过程**持续低估模型密度**, 不断加密度补偿, 最终
在导出的模型上积累出雾。

这不是之前结论所述的 "cubecl 内存池 buffer 覆盖", 而是 burn-fusion 层在 lift/克隆
跨 stream 时 `shared_view` 解析到未落地陈旧 buffer 的问题。

---

## 2. 现象 (可复现)

同一 canonical (14280 splat, 逐字节相同), 同一相机, 同一 GT:

| 路径 | mean proj | PSNR vs GT |
|---|---|---|
| 训练 eval (`render_xray`, autodiff/lift) | ~0.73 | ~31–34 dB |
| 导出 forward (`render_xray_forward`) | ~0.76 | ~21–25 dB |
| gs2volume forward (.bin) | ~0.76 | 与导出一致 |
| gs2volume `--xray-path=bwd` (from_raw + autodiff) | ~0.76 | 与 forward 一致 |

关键点: **只有训练循环内 (live canonical) 的 autodiff/lift 渲染偏少**;
fresh process (gs2volume) 里 forward 与 autodiff (from_raw) 完全一致 (2.4e-7)。

---

## 3. 排查方法

在 `fit_static` eval 块与 `render_xray`/`lift` 内部加诊断探针, 在同一时刻、同一
canonical 上对比:

- `resolve_vs_into`: `canonical.transforms.val()` 的 `into_data` vs `resolve_tensor_float`。
- `lift_ids`: lift 的输入/输出 FusionTensor `TensorId`。
- `eval_canonical_id`: `self.canonical.transforms.val()` 的 `TensorId`。
- `render_xray_input`: `render_xray` 内部 `transforms_inner` 的 `TensorId` + 内容 first3。
- `eval_ad_vs_fwd` / `raw_fwd_vs_bwd` / `fusion_bwd_direct`: 同一时刻 forward vs autodiff vs raw。

所有诊断都用 debug 构建 (含 wgpu validation), GPU 选择
`env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)'`。

---

## 4. 决定性证据

同一时刻 (eval step 2000 view 0), stream 均为 0:

```
DIAG resolve_vs_into[pre-render-2000]: id=TensorId{1950417} first_into=[57.79,-147.7,-101.9]
DIAG eval_canonical_id[k=16]:           id=TensorId{1950417}            (self.canonical.transforms.val())
DIAG render_xray_input[k=2017]:         id=TensorId{1950703} first3=[48.05,...]   (render 内 transforms_inner)
DIAG render_xray_input[k=2016]:         id=TensorId{1949720} first3=[50.14,...]   (training step)
DIAG render_xray_input[k=2025]:         id=TensorId{1950417} first3=[57.79,...]   (diag, 循环后再 lift)
```

即:

1. `self.canonical.transforms.val()` 读到**当前** id `1950417`, 内容 `[57.79,...]`。
2. `render_xray` 内部的 `transforms_inner` 却读到**陈旧** id `1950703`/`1949720`,
   内容 `[48.05,...]`/`[50.14,...]` (与当前差 7+mm, 明显不是同一时刻的值)。
3. 同一 lift 函数, 在 eval 循环内 (首次) 读到陈旧, 循环结束后 (diag 再次 lift)
   读到当前 —— **陈旧读是瞬态的, 取决于调用时刻的 fusion 流状态**。

配套确认 (均为逐位/同刻对比):

- `raw Forward == raw Backward` (maxdiff=0): rasterize pipeline 与 pass 无关。
- 直连 `Fusion<MainBackendBase>::render_xray` (Backward, 不经过 autodiff 包装) == forward。
- lift 的 `val()` 与 `into_data` 内容一致 (`lift_resolve` nonzero=0) —— lift **本身保值**,
  问题出在 lift 后 render 的 resolve 读到另一块 buffer。
- canonical 的 `into_data` 在渲染前/后逐位正确 (nonzero=0) —— canonical 没被改。

---

## 5. 根本原因

**lift → render 的 resolve 读到了陈旧 buffer, 而非当前 canonical。**

具体链条:

1. `step()` 末尾 `self.canonical = canonical_updated.valid()` 把 canonical 设为一个
   **带 pending optimizer op 的 fusion 张量** (新 TensorId, 内容尚未落地)。
2. eval/loss 的 `lift_xray_splats_to_autodiff(self.canonical.clone())` 里
   `Param::val()` → `FusionTensor::clone` / `into_ir` 在**跨 stream** 时走 `shared_view`
   (新建 TensorId), 而 `shared_view` 的 `tag_shared_view` 解析到的 buffer 是**未落地的
   旧 buffer** (上一/更早 step 的内容)。
3. `render_xray` 里 `resolve_tensor_float` 拿到这个陈旧 buffer 去渲染 → 密度偏少 ~4%。
4. 训练 loss 看到偏少的密度 → 认为模型欠密度 → 持续加密度 → 雾累积。
5. 导出/gs2volume 的 forward 渲染 (不经过 lift/shared_view) 读到当前 (含雾) 密度,
   于是暴露出雾 (~23 dB)。

这与 `shared_view` 的 `tag_shared_view` 实现有关 (`burn-fusion/src/stream/multi.rs`):
它只在新 id 上注册 src 的 handle 克隆, 但 src 若仍是 pending (未 drain) 或跨 stream
drain 时机不对, 就会注册到陈旧 buffer。

---

## 6. 已排除的假设 (附证据)

| 假设 | 结论 |
|---|---|
| rasterize Forward vs Backward pass 不同 | 排除: raw fwd==bwd 逐位一致 |
| 相机 / GT / splat 数不同 | 排除: 逐位一致 |
| .bin 导出与 canonical 不一致 | 排除: `into_data` 逐字节相同 |
| lift 内容不保值 | 排除: `lift_resolve` nonzero=0 |
| refine 最后一步剪枝 | 排除: refine_until_frac=0.9 后冻结, 导出==eval |
| cubecl 内存池 slice 复用覆盖 | **更正**: 之前误判, 实为 lift 陈旧读 |
| refine_weight_holder 的 [1] zeros op 引入 | 排除: 渲染前加 [1] zeros 无影响 |
| prep_nodes (autodiff 图准备) | 排除: 直连 Fusion Backward(无 prep) 也偏少时是 lift, 详见下 |
| lift 用 consume() vs val() | 排除: 改 val() 无效 |
| `self.canonical.clone()` 引入陈旧 | 排除: 改 `&XRaySplats` 免 clone 无效 |
| 异步写未 flush (step 末尾物化) | 排除: `resolve_tensor_float` 物化无效 |
| tokio 多线程跨 stream | 排除: `current_thread` 无效 |
| cubecl-common `StreamId::current()` 固定 0 | 排除: 无效 (仍偏少) |

---

## 7. 尝试过的修复 (全部无效, 已回退)

1. lift 用 `val()` 代替 `consume()`。
2. lift 签名改 `&XRaySplats` (免 clone)。
3. `step()` 末尾 `resolve_tensor_float` 强制物化 canonical。
4. `#[tokio::main(flavor="current_thread")]`。
5. cubecl-common `StreamId::current()` 固定返回 0 (含 `cargo clean -p cubecl-common`)。

以上都无法阻止 lift→render 的陈旧读, 说明机制比"跨 stream shared_view"更深
(固定 stream 0 后仍偏少), 需在 burn-fusion 的 handle 生命周期 / resolve 路径上
继续追 (见 §9)。

---

## 8. 因果链总结

```
optimizer 更新 → canonical = 带 pending op 的 fusion 张量
     ↓
eval/loss: lift(canonical)  → shared_view 解析到陈旧 buffer
     ↓
render 读到陈旧(偏低)密度 → loss 误判"欠密度"
     ↓
持续加密度 → 模型积累 +0.05 proj 雾
     ↓
导出 forward 渲染读到当前(含雾)密度 → 暴露雾 (~23 dB), 与 eval (~31 dB) 相差 ~10 dB
```

---

## 9. 下一步建议

- **过渡方案 (立即可用)**: 静态 eval 改用 `render_xray_forward` (已确认与导出一致),
  训练指标以导出 forward 重渲为准。
- **根治方向**: 在 burn-fusion 层追 `lift_to_autodiff`/`from_inner` 之后的
  `resolve_tensor_float` 为何拿到旧 buffer —— 重点看 `shared_view`/`tag_shared_view`
  与 pending op 的 drain 时序、以及 handle 生命周期 (TensorId 为何指向旧 buffer)。
- **最小复现**: 见 `crates/repro-stale-lift/` (应用无关, 只依赖 burn/burn-wgpu/burn-fusion)。

---

## 10. 更正 (2026-08-29 晚, 基于 repro-stale-lift)

用最小复现 crate (`crates/repro-stale-lift`) 的跨线程 shared_view 测试确认:

- 跨线程 `FusionTensor::clone` **确实**产生新的 TensorId (shared_view): 主线程
  `stream=0/id=3506`, 子线程 `stream=9/id=3507`。
- 但 shared_view 读到的**内容与真值一致** (`tag_shared_view` 正确 materialize 了 src
  并共享 buffer)。

**因此 §5 的机制描述需修正**: "shared_view 解析到未落地陈旧 buffer" 不成立 ——
shared_view 本身保值。陈旧读的真正触发点在**渲染 pipeline 内部**
(`Fusion::render_xray` 的 `resolve_tensor_float`/drain + 自定义 cubecl kernel 的 raw
buffer 分配 + BindOp 绑定的组合), 而非 lift / shared_view。搜索空间从 lift/shared_view
收窄到渲染 pipeline。
