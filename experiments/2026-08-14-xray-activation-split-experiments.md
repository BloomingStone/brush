# 2026-08-14 — X-ray 密度控制 split + 激活函数 + LPIPS 并行实验

**日期**: 2026-08-14 17:55–18:20
**分支**: `dev/dicom-r2guassian`（基线）+ 3 个 worktree（`exp5-scale-exp` / `exp6-silu` / `exp7-lpips-loss`）
**数据**: `images/RXA_chest.dcm`（187 帧，149 train / 38 held-out，862×634，像素 0.462mm，弧长 201°）
**硬件**: 4× RTX 3090（每路一张，`DiscreteGpu(0..3)`）

---

## 1. 背景与动机

诊断"目视模糊 + 空物质区残留点云"时发现 brush 的 X-ray density control 与参考项目（`GS-dev-contrast-flow`）有三处关键差异：

1. **缺 split 机制** — 只有 clone，大尺度高梯度 splat 永远不会被拆小 → 边界模糊
2. **缺 screen-size / oversized-scale prune** — 空区大点残留
3. **bwd 密度激活不一致** — `project_bwd.rs:298` 用 sigmoid 的导数 `σ(1−σ)` 且缺 `MU_WATER` 因子（RGB 遗留），前向却是 `MU_WATER·softplus(raw)`

后续又评估了两个激活函数假设：
- **scale** `softplus` 有 1mm 下界且 `raw→−∞` 梯度消失 → 精细结构受限；`exp` 无下界且与 bwd 已有的 `dL/dσ·σ` 链式法则一致
- **density** `softplus` 恒正、空气点卡在 cull 阈值之上残留；`SiLU` 允许负密度 → 直接落入 cull 被剪

## 2. 实验配置

公共配置（四路一致）：
```
fit_static RXA_chest.dcm
  --iters=10000 --init-density=0.005 --refine-every=400
  --eval-split-every=5 --eval-views=8 --eval-every=500
  --fixed-grad-thr=1e-6 --split
```

| 分支 | 差异 |
|---|---|
| **base**（主分支 `1c2981b`） | bwd 激活修复 + split + 固定阈值 + LPIPS 指标 |
| **exp5**（`85758d8`） | + scale 激活 `softplus→exp(clamp(raw,−20,ln1000))`（渲染/refine/init 三处） |
| **exp6**（`026f515`） | + density 激活 `softplus→SiLU`（brush-cube `silu`+`inverse_silu`，渲染前后向，refine prune/reset/init） |
| **exp7**（`0afe12e`） | + LPIPS 感知损失 `--lpips-weight=0.05`（VGG 加载于 autodiff device） |

## 3. 结果（held-out 8 视图，iter 10000）

| 实验 | PSNR (dB) | SSIM | LPIPS↓ | 最终点数 | 说明 |
|---|---|---|---|---|---|
| **base** | 30.44 | 0.959 | 0.6065 | 54,562 | 参照 |
| **exp5** (scale=exp) | **32.87** | **0.962** | **0.5992** | 49,315 | **+2.43 dB** |
| **exp6** (density=SiLU) | 30.63 | 0.959 | 0.6094 | 42,671 | +0.19 dB，点数少 1.2 万 |
| exp7 (LPIPS loss) | — | — | — | — | 1.8 s/step，训练被停止 |

### 历史参照（同数据，供趋势对比）

| 实验 | PSNR | 点数 | 说明 |
|---|---|---|---|
| percentile 无 split（`exp_chest`） | 29.39 | 27,760 | 点数单调流失 |
| 固定 1e-6 + split、无 bwd 修复（`fit_static_chest_split`） | 30.47 | 53,108 | split 首次生效 |
| **base**（本次，含 bwd 修复） | 30.44 | 54,562 | 与上接近，修复未破坏质量 |

## 4. 逐项分析

### base（bwd 修复 + split）
- split 集中在 iter 800–2400（拆分 8853→1535 递减），大点拆完后只剩 prune，点数峰值 ~55,800 后缓慢回落
- bwd 修复后密度梯度从 `2e-4~6e-4` 降到 `2e-6~3e-6`（缩小 50–250 倍，数学正确），短验证（2500 步 29.14 dB）确认训练正常，最终与修复前持平（30.44 vs 30.47）

### exp5（scale = exp+clamp）⭐ 最大赢家
- **+2.43 dB**，SSIM 0.962，LPIPS 最低
- 全程领先（@3000 步 30.11 vs base ~29.3；@5500 步 31.87 vs 30.08）
- 验证假设：`softplus` 的 1mm 下界确实限制了精细结构；`exp` 无下界 + 均匀 log-space 梯度，且与 bwd 已有 `dL/draw = dL/dσ·σ` 完全一致（前向/反向天然匹配）
- 点数略少（49,315）→ 更小尺度点更高效地表达了细节

### exp6（density = SiLU）
- **+0.19 dB**，点数少 1.2 万
- SiLU 负密度触发剪枝按预期工作：每 400 步 prune 600–1000（base 同阶段 ≈0），最终点数 42,671
- 验证假设：空气点 raw 被推负 → activated density < 0 < `cull_density_threshold(5e-5)` → 被剪，空区更干净
- 质量略高于 base（30.63 vs 30.44），说明剪掉的确实是"有害"空气点

### exp7（LPIPS 损失）— 停止
- **1.8 s/step**（比 base 慢 ~35×）：VGG 5 层在 burn autodiff + wgpu 上前向+反向开销无法接受，10000 步需约 5 小时
- **结论**：LPIPS 保留为**评估指标**（已集成到 `eval_view`/`fit_static`，灰度复制 3 通道）；作为**训练损失**在当前 burn+wgpu 实现下不可行（除非优化 kernel 或降到低分辨率/低频更新）

## 5. 结论与建议

1. **scale 激活改用 `exp+clamp`**（exp5）——+2.43 dB，强烈建议合回主分支
2. **density 激活改用 `SiLU`**（exp6）——+0.19 dB 且点云更干净（空区剪枝），建议合回并观察 brain 数据表现
3. **LPIPS 训练损失**暂缓（成本过高），保留为指标
4. 后续可探索：exp5+exp6 组合、screen-size prune（参考项目 max_radii2D>20）、密度软重置

## 6. 复现

```bash
# base（主分支）
target/debug/fit_static images/RXA_chest.dcm --iters=10000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split
# exp5/exp6：在对应 worktree 编译后同命令
# 日志：/tmp/exp_chest_{base,exp5,exp6,exp7}.log；输出：target/fit_chest_{base,exp5,exp6,exp7}/
```
