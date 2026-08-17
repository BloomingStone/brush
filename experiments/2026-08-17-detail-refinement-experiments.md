# 2026-08-17 — 细节精细度改进 4 路实验（P0/P1）

**日期**: 2026-08-17 10:40–11:10
**分支**: `dev/dicom-r2guassian`（proj 损失默认 + 新增 cosine LR / proj SSIM / percent_dense / split_scale 参数）
**数据**: `images/RXA_chest.dcm`（149 train / 38 held-out）
**硬件**: 4× RTX 3090

## 改进实现

1. **cosine LR**（`xray_train.rs`）：`--cosine-lr`，`ComposedLrScheduler` 包装
   `CosineAnnealingLrScheduler(initial_lr, num_iters).with_min_lr(lr_mean_end)`，替代指数衰减（参考项目 cosine ramp，解决后期位置冻结）。
2. **proj 域 SSIM**（`xray_train.rs`）：`--proj-ssim-weight=S`，在 `-ln(intensity)` 域追加
   `S·(1-SSIM(proj_pred, proj_gt))` 结构感知项（proj 损失默认仍为纯 L1）。
3. **percent_dense 参数化**：`--percent-dense=F`（clone/split 分界 `scale > F·scene_extent`，降低→更多点进 split 拆小）。
4. **split_scale 参数化**：`--split-scale=F`（split 尺度收缩，默认 1/√2，更小→拆得更细）。

## 配置

公共：`--iters=10000 --init-density=0.005 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 --fixed-grad-thr=1e-6 --split`（proj 损失默认 w=1.0）

| GPU | 实验 | 变体 |
|---|---|---|
| 0 | coslr | `--cosine-lr` |
| 1 | projssim | `--proj-ssim-weight=1.0` |
| 2 | pden | `--percent-dense=0.0002` |
| 3 | spscale | `--split-scale=0.6` |

## 结果（held-out 8 视图，iter 10000）

| 实验 | PSNR | SSIM | LPIPS↓ | 最终点数 | 峰值点数 |
|---|---|---|---|---|---|
| **projssim** | **34.03** | **0.965** | **0.5627** | 167,483 | ~180k |
| coslr | 33.33 | 0.962 | 0.5865 | 85,747 | ~92k |
| pden | 33.18 | 0.962 | 0.5871 | 85,709 | ~91k |
| spscale | 32.64 | 0.962 | 0.5841 | 103,770 | ~112k |
| baseline projloss（参照） | ~31.3–32.4（@9k–11k） | — | ~0.614 | ~91k | — |

趋势（PSNR@iter）：
| iter | coslr | projssim | pden | spscale |
|---|---|---|---|---|
| 3000 | 29.69 | 28.84 | 29.63 | 29.26 |
| 5000–5500 | 30.42 | 30.20 | 31.42 | 30.91 |
| 7500–8000 | 32.22 | 30.77 | 32.11 | 31.54 |
| 10000 | 33.33 | 34.03 | 33.18 | 32.64 |

## 分析

1. **proj 域 SSIM 最强（+1.7~2.7 dB）**：结构感知项让密度场更精细，10000 步已达 baseline projloss 20000 步的 34.82 水平（还差 0.79 但只用一半步数）；点数增长最激进（~180k）但 PSNR/LPIPS 双优，值得 20000 步验证。
2. **cosine LR 显著（+1~2 dB）**：中后期保持更高 LR，验证了"指数衰减后期冻结位置"的假设；后期（7500+）追平 pden。
3. **percent_dense 降低有效（+1~1.8 dB）**：更多点进入 split 通道拆小；与 coslr 结果几乎相同。
4. **split_scale 0.6 提升最小**：拆得更细但点数也多，PSNR 反而最低——1/√2 已足够，过细拆分收益递减。
5. **综合**：projssim 最值得推进；coslr/pden 可作为默认候选（成本低、稳定）。

## 复现

```bash
target/debug/fit_static images/RXA_chest.dcm --iters=10000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split [--cosine-lr | --proj-ssim-weight=1.0 | \
  --percent-dense=0.0002 | --split-scale=0.6]
# 日志: /tmp/exp_chest_{coslr,projssim,pden,spscale}.log
# 输出: target/fit_chest_{coslr,projssim,pden,spscale}/
```
