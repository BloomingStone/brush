# 2026-08-17 — 离群点治理实验（FOV 预投影剔除 + 激进 bound prune + cull/点数）

**日期**: 2026-08-17 11:24–11:50
**分支**: `dev/dicom-r2guassian`
**数据**: `images/RXA_chest.dcm`（149 train / 38 held-out）
**硬件**: 4× RTX 3090

## 背景

实测 `fit_chest_projloss/canonical_final.ply`（86668 点）：
- **13.75% 的点质心距离 r>160mm**（初始球半径才 164mm），p99=483.7mm、max=1387mm —— 训练中漂移出的离群点
- 这些点 opacity 更高（p95=7.02 vs 全体 3.19）——漂移后无梯度、密度停在较高值
- 根因：refine 的 bound prune 阈值 = `scene_extent×100`（16440mm）形同虚设

## 实现

1. **初始 FOV 预投影剔除**（`create_xray_trainer`）：逐相机 `world_to_local`+针孔投影，剔除从未投影到任何图像平面（±10px 边距）的随机点。实测 30000 点 rejected 14585（**32.7%**）。
2. **激进 bound prune**（`xray_refine`）：`max_bound_factor`（默认 3.0×scene_extent，原 ×100 不剪漂移点）。人体固定区域 → 可激进。
3. fit_static：`--bound-factor` / `--cull-density` 参数。

## 配置

公共：`--iters=10000 --init-density=0.005 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 --fixed-grad-thr=1e-6 --split`（proj 损失默认）

| GPU | 实验 | 变体 |
|---|---|---|
| 0 | fovbase | FOV 剔除 + bound×3（新基线） |
| 1 | cull1e3 | + `--cull-density=1e-3` |
| 2 | pts10k | + `--points=10000`（初始点数减少，自然分裂） |
| 3 | cull5e3 | + `--cull-density=5e-3`（更激进） |

## 结果（held-out 8 视图，iter 10000）

| 实验 | PSNR | SSIM | LPIPS↓ | 点数 | 离群点 r>160mm | p99 半径 |
|---|---|---|---|---|---|---|
| projloss 旧基线（参照） | ~31.3–32.4 | — | ~0.614 | ~91k | 13.75% | 483.7 mm |
| **fovbase** | **33.32** | **0.962** | **0.5830** | 84,025 | **8.15%** | **184.5 mm** |
| cull1e3 | 32.95 | 0.962 | 0.5881 | 58,760 | — | — |
| pts10k | 32.90 | 0.961 | 0.5927 | 64,055 | — | — |
| cull5e3 | 30.19 | 0.956 | 0.5948 | 7,302 | 严重欠拟合 | — |

## 分析

1. **fovbase 是双重胜利**：离群点治理（r>160 从 13.75%→8.15%，p99 483→184mm，max 1387→594mm）+ 质量提升（33.32 vs 旧 baseline ~31.3–32.4，+1~2 dB）。FOV 外点不再干扰渲染/点数预算。
2. **cull1e3（32.95）**：略低于 fovbase，点数 58k——cull=1e-3 误剪了一些有效低密度点（肺实质 μ~5e-4）。
3. **pts10k（32.90）**：初始 10000 点也能到接近水平（结构由分裂生长），但略低；30000 起点仍更好。
4. **cull5e3（30.19）失败**：点数崩到 7k，严重欠拟合——cull>1e-3 不可行。
5. **离群点残留**：fovbase 仍有 8.15% 在 r>160（主要 160-185mm 边界区，接近 FOV 边缘 156mm；max 594mm 为个别 refine 后漂移），已无远漂移点。

## 结论

- **FOV 预投影剔除 + 激进 bound prune（默认 ×3）应保留为默认**：既清离群点又提质量
- cull 阈值保持 5e-5（调高误剪）；初始点数保持 30000

## 复现

```bash
target/debug/fit_static images/RXA_chest.dcm --iters=10000 --init-density=0.005 \
  --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=500 \
  --fixed-grad-thr=1e-6 --split [--cull-density=1e-3 | --points=10000]
# 日志: /tmp/exp_chest_{fovbase,cull1e3,pts10k,cull5e3}.log
# 输出: target/fit_chest_{fovbase,cull1e3,pts10k,cull5e3}/ + /tmp/check_ply.py 验证离群点
```
