# 2026-08-21 全量 20k: 放宽点云 + ROI 截取 + time 条件化 (heart_pig_stent)

数据: `/media/data4/sj/brush/images/heart_pig_stent_with_phase.dcm`
(722f/30fps/516x516, 577 训练 / 145 held-out, 呼吸 ~0.83Hz, FOV 边缘 ~15px 暗边)

配置 (全部 20k iters, hexplane):
`--iters=20000 --points=30000 --refine-every=400 --eval-split-every=5
--eval-views=8 --eval-every=500 --split --max-splats=600000
--growth-frac=0.25 --cull-density=1e-4 --fixed-grad-thr=1e-6`
time 变体: `--enable-time --time-min-freq=0.2 --time-max-freq=1.5`

## 结果 (20k)

| run | ROI | time | splats | PSNR@20k | SSIM | LPIPS@20k | bestLPIPS | 耗时 |
|---|---|---|---|---|---|---|---|---|
| full20k_roi20_time | 20px | ✓ | 600k | 42.78 | 0.9848 | 0.3403 | 0.3352@19500 | 1h56m |
| full20k_noroi_time | 无 | ✓ | 600k | 41.48 | 0.9837 | **0.3063** | 0.3040@19500 | 1h58m |
| full20k_roi20_notime | 20px | ✗ | 600k | 38.49 | 0.9836 | 0.3546 | 0.3530@19500 | 1h51m |
| full20k_roi40_time | 40px | ✓ | 600k | **43.17** | 0.9849 | 0.3522 | 0.3513@19500 | 1h52m |

## 结论

1. **ROI 截取 (去暗边) 提升 PSNR**: ROI20 +1.30 dB (42.78 vs 41.48); ROI40
   更深裁剪再 +0.39 dB (43.17)。暗边确实影响重建, 截掉更好。
   - **LPIPS 反转** (noroi 0.304 < roi 0.335): 因不同 run 评估区域不同
     (516² vs 476²/436²), 指标可比性存疑, 需目视 gt_pred_*.nrrd 确认。
   - ROI 实现: `LoadDatasetConfig.roi` + `--roi=no|N|x0,y0,w,h` (默认 N=20,
     四边裁剪; no=关闭; 显式矩形), 像素与相机内参 (fov/主点) 同步调整
     (build_camera_roi), 投影保持精确 (单测通过)。

2. **放宽点云 + 20k 是主要提升**: 相比 10k 动态阈值 (16k splats, 38.90),
   600k splats + 20k iters → 42.78 (+3.9 dB), 且 20k 仍在上升 (42.37→42.78)。
   可视细节应显著改善。

3. **time 条件化在 600k splat 下依然 +4.29 dB** (42.78 vs 38.49, 同 ROI)。

4. learned time freqs 仍贴近初值 (0.200…0.766…1.500) — 网络靠幅度选基函数,
   0.766Hz ≈ 呼吸 0.83Hz。

## 建议
- 默认配置建议: `--roi=40,40,436,436 --enable-time --time-min-freq=0.2
  --time-max-freq=1.5` + 放宽点云参数 (上述 COMMON)。
- 视觉核验: `target/exp/full20k_roi40_time/gt_pred_*.nrrd` (3D Slicer/ParaView
  翻阅), 对比 noroi_time 看暗边与细节。
