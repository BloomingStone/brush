# fdk-residual — FDK 静态先验 + 有符号残差 GS

## 2026-08-27
- commit: 7b51e1c (FdkPrior + 有符号渲染 P1b+P2a+P2b+P3)
- 目的: 用干净 FDK (fdk_final, 20.29dB) 提供静态解剖, 有符号残差 GS
  (raw=±1e-5 起步, proj = splat + fdk_drr) 只修正动态差异 (心脏/冠脉)。
  对比 cyl_r15 (0.2007, 旧 scene=153) → 需重测新默认 (scene=264)。
- 配置 (20k, 新默认 scene_extent=264, cull 5e-4, max_bound_factor=1):
  ① cyl_r15_ref  --init-shape=cylinder --init-radius-scale=1.5 --no-fov-filter (无 FDK)
  ② fdksplat     同 ① + --fdk-volume=experiments/output/fdk-residual/fdk_final/volume.nii.gz
                  --fdk-meta/--fdk-calib 同目录, --fdk-steps=256 (有符号残差)

## 结果 (20k, 新默认 scene=264, 训练时间含导出)
| 配置 | LPIPS | PSNR | SSIM | splats | 训练时间 |
|---|---|---|---|---|---|
| cyl_r15_ref (纯 GS) | **0.2199** | 44.56 | 0.991 | 46.5k | ~26.6 min |
| fdksplat (FDK+残差) | 0.2891 | 40.10 | 0.972 | 21.0k | ~18.2 min |

## 结论 (负面结果)
- **FDK+残差 明显劣于纯 GS** (LPIPS 0.289 vs 0.220)。
- iter 100 时 FDK+残差 ≈0.455 ≈ FDK-only 先验质量; 残差逐步修正到 0.289,
  但始终到不了纯 GS 的 0.220。FDK 先验反而拖累。
- 诊断:
  1. **体积覆盖不匹配**: FDK 体积只覆盖 R≤118.6 / z≤84.7, 场景 scene=264 /
     cyl_r15 半径 177.9。55-57% 残差 splat 在 FDK=0 区域 (R>118.6), 那里必须
     用受限的微小残差从零重建 → 外围质量差。
  2. **残差容量受限**: init ±1e-5 + signed cull 5e-4 → 21k splats (vs 46k);
     残差需同时修 FDK 静态误差 (20.3dB) + 动态, 力不从心。
- **附带发现**: 新默认 scene=264 比旧 153 差 — cyl_r15 新基准 0.2199 vs 旧
  cyl_r15 0.2007 (scene=153)。scene_extent 增大 hurt LPIPS。
