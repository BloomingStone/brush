# 2026-08-17 — 密度软重置 + proj 域损失实验

**日期**: 2026-08-17 09:20–10:00
**分支**: `dev/dicom-r2guassian`（exp5+exp6 合并版 + 新增两个可开关改进）
**数据**: `images/RXA_chest.dcm`（149 train / 38 held-out）
**硬件**: 4× RTX 3090

## 改进实现

1. **密度软重置**（`xray_refine.rs`）：`--density-reset=N`，每 N 步
   `new_density = min(current_density, init_density)`（只降不升，参考项目 `_reset_density`）。
   `XRayRefineConfig` 新增 `init_density` cap，`create_xray_trainer` 自动与训练初始化同步；
   顺带修正残留 `inverse_softplus` → `inverse_silu`。
2. **proj 域损失**（`xray_train.rs`）：`--proj-weight=W`，loss 追加
   `W·mean(|proj_pred − proj_gt|)`，其中 `proj_pred = clamp(proj, 1e-3, 14)`（=−ln(intensity)）、
   `proj_gt = −ln(clamp(gt, 1e-4, 1))`。在衰减积分域比较，避开 Beer-Lambert exp 压缩。

## 配置

公共：`--iters=20000 --init-density=0.005 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 --fixed-grad-thr=1e-6 --split`

| GPU | 实验 | 参数 |
|---|---|---|
| 0 | softreset | `--density-reset=2000` |
| 1 | projloss | `--proj-weight=1.0` |
| 2 | projloss05 | `--proj-weight=0.5` |
| 3 | resetproj | `--density-reset=2000 --proj-weight=1.0` |

## 结果（held-out 8 视图，iter 20000）

| 实验 | PSNR | SSIM | LPIPS↓ | 最终点数 |
|---|---|---|---|---|
| baseline（参照，上一轮） | 33.75 | 0.963 | 0.5940 | 37,342 |
| softreset | 33.26 | 0.963 | 0.5940 | 36,951 |
| **projloss (w=1.0)** | **34.82** | **0.965** | **0.5740** | 87,096 |
| projloss05 (w=0.5) | 34.44 | 0.964 | 0.5838 | 59,562 |
| resetproj（组合） | 34.61 | 0.965 | 0.5752 | 84,019 |

## 分析

1. **proj 域损失是重大改进**（+1.07 dB @ w=1.0，LPIPS 0.594→0.574）：
   验证假设——在 `-ln(intensity)` 衰减积分域比较避开了 exp 压缩，密度场优化更有效。
   权重敏感（0.5→34.44, 1.0→34.82，越高越好），值得再试更高权重。
   点数从 37k 翻倍到 87k（proj 梯度使 split/clone 更活跃），但 PSNR/LPIPS 双升说明点有效。
2. **密度软重置单独无明显收益**（-0.49 dB），组合（resetproj 34.61）也低于纯 projloss（34.82）：
   SiLU + 当前剪枝下"压帽"边际价值低，甚至轻微干扰密度场学习 → 建议保持默认关闭。
3. **建议**：proj 域损失合入主分支作为默认（权重 1.0）；软重置保持参数化（默认关）。

## 复现

```bash
target/debug/fit_static images/RXA_chest.dcm --iters=20000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split [--density-reset=2000] [--proj-weight=1.0]
# 日志: /tmp/exp_chest_{softreset,projloss,projloss05,resetproj}.log
# 输出: target/fit_chest_{softreset,projloss,projloss05,resetproj}/
```
