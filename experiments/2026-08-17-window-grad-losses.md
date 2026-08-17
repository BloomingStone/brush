# 2026-08-17 — softreset 修复 + 多窗宽窗位 / 梯度损失实验

**日期**: 2026-08-17 13:15–13:50
**分支**: `dev/dicom-r2guassian`
**数据**: `images/RXA_chest.dcm`（149 train / 38 held-out）

## 实现

1. **reset 步跳过 densify**（`xray_refine.rs`）：对齐参考项目 `%2000>=300`——`density_reset_interval` 步不 clone/split（防刚 densify 的新点被 reset 压帽）。
2. **lr_mean_end 2e-7 → 2e-6**（默认）：末期不冻结（cosine min / 指数末段）。
3. **多窗宽窗位损失**（`--window-weight`）：proj 域 3 组窗变换 `clamp((x-(wl-ww/2))/ww,0,1)`（软组织 0.6/0.4、骨 1.2/0.6、细节 0.4/0.25），各窗下 L1。
4. **梯度差分损失**（`--grad-weight`）：intensity 域中心差分 `|∇pred−∇gt|`（x+y 方向）。

## 配置

公共：`--iters=10000 --init-density=0.005 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 --fixed-grad-thr=1e-6 --split --proj-ssim-weight=1.0 --cosine-lr --percent-dense=0.0002`（combined 基线）

| GPU | 实验 | 变体 |
|---|---|---|
| 0 | softreset2 | `--density-reset=2000`（重合修复后重测） |
| 1 | window | `--window-weight=0.5` |
| 2 | grad | `--grad-weight=0.5` |
| 3 | wingrad | `--window-weight=0.5 --grad-weight=0.5` |

## 结果（held-out 8 视图，iter 10000）

| 实验 | PSNR | SSIM | LPIPS↓ | 最终点数 |
|---|---|---|---|---|
| **wingrad**（window+grad） | **33.97** | 0.964 | 0.5463 | 282,330 |
| window | 33.80 | 0.964 | 0.5455 | 285,307 |
| grad | 33.63 | 0.965 | 0.5574 | 177,301 |
| softreset2 | 33.50 | 0.965 | 0.5607 | 151,512 |
| 参照 combined | 33.95 | 0.965 | 0.5570 | 173,901 |
| 参照 multiscale（最强） | 34.13 | 0.966 | 0.5290 | 376,299 |

## 分析

1. **softreset 修复重合后仍无收益**（33.50 vs combined 33.95）：softreset 本身价值有限，与重合 bug 无关 → 保持默认关闭。修复本身保留（正确行为）。
2. **window 多窗宽窗位**：PSNR 略降（-0.15）但 **LPIPS 0.5455 明显更好**（结构感知增强），点数 285k。
3. **grad 梯度差分**：早期波动（25-26）后期追到 33.63；单独增益有限。
4. **wingrad 组合**：33.97（略超 combined）+ LPIPS 0.5463——window+grad 可小幅叠加。
5. **多尺度损失（multiscale）仍是最强项**：34.13 / LPIPS 0.5290 遥遥领先，下一步建议 combined+multiscale+wingrad 全组合。

## 结论

- **lr_mean_end 2e-6** 保留为默认（末期不冻结）
- **reset 跳过 densify** 保留（正确行为）；softreset 仍禁用
- **window / grad 损失**可作为辅助项（感知提升，PSNR 持平或略降）；**多尺度损失**是当前最强的细节损失

## 复现

```bash
target/debug/fit_static images/RXA_chest.dcm --iters=10000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split --proj-ssim-weight=1.0 --cosine-lr \
  --percent-dense=0.0002 [--window-weight=0.5 | --grad-weight=0.5 | \
  --density-reset=2000]
# 日志: /tmp/exp_chest_{softreset2,window,grad,wingrad}.log
```
