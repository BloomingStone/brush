# 2026-08-14 — exp5+exp6 合并版 20000 步参数 sweep

**日期**: 2026-08-14 18:26–19:21
**分支**: `dev/dicom-r2guassian`（已合并 exp5 scale=exp + exp6 density=SiLU）
**数据**: `images/RXA_chest.dcm`（149 train / 38 held-out）
**硬件**: 4× RTX 3090（每路一张）

## 配置

公共（四路一致）：`--iters=20000 --init-density=0.005 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 --split`

| GPU | 实验 | 变体 | 动机 |
|---|---|---|---|
| 0 | baseline | 默认（lr-opac 0.012 / lr-scale 5e-3 / thr 1e-6） | 参照 |
| 1 | lropac | `--lr-opac=0.05`（4×） | bwd 修复后密度梯度缩小 50–250×，重校准密度学习率 |
| 2 | lrscale | `--lr-scale=1.5e-2`（3×） | scale 改 exp 后重校准 |
| 3 | thr | `--fixed-grad-thr=3e-6` | 更保守的增长策略 |

## 结果（held-out 8 视图，iter 20000）

| 实验 | PSNR | SSIM | LPIPS↓ | 最终点数 | 峰值点数 |
|---|---|---|---|---|---|
| **baseline** | **33.75** | **0.963** | 0.5940 | 37,342 | 42,809 |
| lropac | 33.08 | 0.962 | 0.5983 | 26,450 | 31,992 |
| lrscale | 33.14 | 0.963 | **0.5937** | 40,903 | 42,062 |
| thr | 32.20 | 0.961 | 0.6005 | 22,212 | 30,262 |

PSNR 曲线（关键点）：

| iter | baseline | lropac | lrscale | thr |
|---|---|---|---|---|
| 4000 | 27.44 | 24.89 | 27.79 | 29.09 |
| 9000 | 31.99 | 30.20 | 31.60 | 31.04 |
| 19000 | 33.99 | 33.32 | 32.68 | 32.10 |
| 20000 | 33.75 | 33.08 | 33.14 | 32.20 |

点数序列（每 2000 步）：
- baseline: 30000 → 42809 → 41894 → 40867 → 40157 → 39722 → 39472 → 39355 → 38640 → 37738
- lropac:   30000 → 31992 → 31066 → 30318 → 29856 → 29549 → 29411 → 29136 → 27980 → 26859
- lrscale:  30000 → 37891 → 38947 → 39289 → 39798 → 40364 → 41019 → 41748 → 41759 → 41193
- thr:      30000 → 27550 → 25200 → 24114 → 23426 → 23014 → 22765 → 22570 → 22406 → 22271

## 分析

1. **baseline 最优（33.75）**：20000 步相比 10000 步（32.87）提升 +0.88 dB，点数从 49k 降到 37k（更干净）。建议 20000 步作为标准训练时长。
2. **lropac（33.08）失败模式**：lr_opac 4× 让 SiLU 负密度过快被推到 cull → 点数从 30000 掉到 26k（过度剪枝），前期 psnr 24.89 明显落后。证明默认 lr_opac=0.012 对 bwd 修复后的梯度尺度恰当。
3. **lrscale（33.14）**：点数持续增长到 41k（最多），LPIPS 最低但 PSNR 反而低——scale 学习率过高造成震荡/冗余。默认 lr_scale=5e-3 合理。
4. **thr（32.20）欠拟合**：split 几乎不触发（总量仅 1308），点数单调下降到 22k；早期靠不增点稳定性领先（29.09@4k），后期表达力封顶停滞。证明 1e-6 阈值对当前梯度量级（~1e-6）正确。

## 结论

- exp5+exp6 合并版在 20000 步下稳健（33.75 dB）
- 三个 sweep 方向均未超越默认参数 → 超参已接近局部最优
- 下一步提升应转向**结构性改进**：参考项目的 screen-size prune（max_radii2D>20）、密度软重置（min(density,init)）、proj 域损失

## 复现

```bash
target/debug/fit_static images/RXA_chest.dcm --iters=20000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split [--lr-opac=0.05 | --lr-scale=1.5e-2 | --fixed-grad-thr=3e-6]
# 日志: /tmp/exp_chest_{exp56_20k,lropac,lrscale,thr}.log
# 输出: target/fit_chest_{exp56_20k,lropac,lrscale,thr}/
```
