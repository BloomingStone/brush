# coronary-old-phase

## 2026-08-25 对照: 旧相位 (未同步, 9周期/111bpm) 同配置
- commit: 4b8fa39
- 数据: images/pig-data-coronary.dcm (相位未同步)
- 目的: 与校正相位 (coronary-new-phase) 对照, 判断相位校正的影响
- 命令: ./target/release/fit_deform images/pig-data-coronary.dcm --points=5000 --init-density=0.01 --cull-density=5e-4 --loss=charbonnier --roi=20 --enable-time --time-min-freq=0.2 --time-max-freq=1.5 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=1000 --split --iters=20000 --out=experiments/output/coronary-old-phase/default

## 结果 (2026-08-25, 20k, thr5e-6)
| 指标 | 值 |
|---|---|
| PSNR | 42.59 |
| SSIM | 0.9904 |
| LPIPS | 0.2286 |
| blur_ratio | 0.623 |
| edge_l1 | 0.0088 |
| splats | 37,256 |

vs 校正相位 (new-phase default): PSNR 44.14 (+1.55), LPIPS 0.2147 (-6%), blur 0.655 (更锐)
