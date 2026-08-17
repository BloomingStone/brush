# 2026-08-17 — 结合版 + 进一步优化策略实验

**日期**: 2026-08-17 11:47–12:30
**分支**: `dev/dicom-r2guassian`
**数据**: `images/RXA_chest.dcm`（149 train / 38 held-out）
**硬件**: 4× RTX 3090

## 配置

公共：`--iters=10000 --init-density=0.005 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 --fixed-grad-thr=1e-6 --split`
（含默认：FOV 预投影剔除 + bound×3 + proj 损失 w=1.0）

| GPU | 实验 | 变体 |
|---|---|---|
| 0 | combined | `--proj-ssim-weight=1.0 --cosine-lr --percent-dense=0.0002` |
| 1 | full | combined + `--split-scale=0.6` |
| 2 | scrnsz | combined + `--max-screen-size=20` |
| 3 | multiscale | combined + `--multiscale-weight=0.5` |

## 结果（held-out 8 视图，iter 10000）

| 实验 | PSNR | SSIM | LPIPS↓ | 最终点数 |
|---|---|---|---|---|
| **multiscale** | **34.13** | **0.966** | **0.5290** | 376,299 |
| combined | 33.95 | 0.965 | 0.5570 | 173,901 |
| full | 33.52 | 0.964 | 0.5508 | 229,593 |
| scrnsz | 5.99 | 0.744 | 0.8126 | 11（失败） |
| 参照 fovbase | 33.32 | 0.962 | 0.5830 | 84,025 |

## 分析

1. **多尺度损失（multiscale）是新的最强项**：34.13 dB + LPIPS 0.5290（全场最低，感知质量显著提升）。
   多尺度（1/2、1/4 金字塔）一致性强力细化结构；点数 376k（增长最激进）但 PSNR/LPIPS 双优。
   代价：每步 adaptive_avg_pool2d 开销，训练约慢 1.5×。
2. **combined（projssim+coslr+pden）33.95**：三合一优于单项但略低于 multiscale，点数更省（174k）。
3. **full（+spscale 0.6）33.52**：PSNR 略降，但 LPIPS 0.5508 好——split-scale 0.6 主要增益在感知/结构。
4. **scrnsz 失败**：`--max-screen-size=20` 在初始随机点（scale 5-13mm、近处投影半径可达数十 px）下几乎全剪（剩 11 点）。
   **参考项目是在成熟模型（densify 后小点）+ 2000 步后才启用 size_threshold** —— brush 需修复：延迟启用 + 半径换算修正（归一化 ×min(img_w,img_h)）。
5. **与参考项目调度差异（待修复）**：参考 `%2000>=300` 规避 reset 与 densify 重合；brush 400/2000 整数倍 → 完全重合且 reset 在 densify 之后压掉新点（softreset 无收益的解释）。

## 结论

- **多尺度损失建议作为默认/组合项**（感知质量提升最大）
- screen-size prune 需修复后再试
- reset-densify 重合问题待修复（对齐参考项目）

## 复现

```bash
target/debug/fit_static images/RXA_chest.dcm --iters=10000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split --proj-ssim-weight=1.0 --cosine-lr \
  --percent-dense=0.0002 [--multiscale-weight=0.5 | --split-scale=0.6]
# 日志: /tmp/exp_chest_{combined,full,scrnsz,multiscale}.log
```
