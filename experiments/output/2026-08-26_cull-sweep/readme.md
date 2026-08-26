# cull-sweep — cull_density 阈值扫描 (baseline)

## 2026-08-26
- commit: 037d7f0 (scene_extent=264 新默认, max_bound_factor=1.0)
- 动机: R>R0 点密度直方图在 0.005~0.01 有明显台阶 (结构边界), 74-78% 的
  >R0 点 <2x水 (低密度/空气)。默认 cull 5e-4 抓不住 → 提高 cull 阈值
  扫描, 看是否剪掉低密度 >R0 点 (空气) 且不伤真实结构。
- 配置: baseline (ball init, FOV 过滤, 5000 点, 20k) + --cull-density 扫描。
- 对照: baseline 默认 cull 5e-4 = 0.2147 (coronary-new-phase/default, 但
  旧 scene_extent=153; 本实验用新 scene_extent=264)。

## 配置 (20k, baseline 命令 + --cull-density)
① cull003  --cull-density=0.003
② cull004  --cull-density=0.004
③ cull005  --cull-density=0.005

## 结果 (2026-08-26, 20k, scene_extent=264)
| 配置 | cull | PSNR | SSIM | LPIPS | splats |
|---|---|---|---|---|---|
| baseline (旧 5e-4, scene 153) | 5e-4 | 44.14 | 0.9910 | 0.2147 | 38k |
| cull003 | 0.003 | 42.27 | 0.989 | 0.3030 | 12.3k |
| cull004 | 0.004 | 42.04 | 0.989 | 0.3094 | 12.1k |
| cull005 | 0.005 | 40.98 | 0.988 | 0.3417 | 9.7k |

## 结论
- **高 cull 阈值严重过剪**: LPIPS 0.30-0.34 (vs 0.21), splats 10-13k (vs 38k)。
  0.003-0.005 剪掉了真实低密度结构 (软组织), 不只空气点。
- >R0 的"空气"点**无法用全局密度阈值干净分离** — 直方图台阶以下包含
  空气 + 需要的软组织 (密度重叠)。
- 方向: 全局 cull 不宜 >1e-3; 若要剪 >R0 空气, 需**半径感知 prune**
  (R>R0 用激进阈值, FOV 内保持 5e-4)。
