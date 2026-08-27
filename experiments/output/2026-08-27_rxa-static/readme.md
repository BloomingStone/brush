# rxa-static — RXA_chest 纯静态重建 (fit_static 改写版)

## 2026-08-27
- commit: 2fbb271 (fit_static 与 fit_deform 逻辑一致, 删形变场)
- 目的: 验证静态数据上 GS 重建 → 点云导出 volume 是否正常
  (隔离"杂乱点云"是动态/形变问题还是体素化问题)
- 配置: 同 fdksplat_trans 风格 (cyl_r15 + 20k + scene 264 auto)

## 结果 (20k)
| 指标 | 值 |
|---|---|
| PSNR | 36.42 |
| SSIM | 0.968 |
| LPIPS | 0.3819 |
| splats | 18.9k |

## 体积检查 (gs2volume --no-fdk, 326x326x232, 0.927mm)
- 非零 96.7%, 均值(非零) μ=0.0022 (水级), max 0.406 (骨/碘级)
- **相邻切片相关 0.981** — 结构高度连贯 (对比 dsa_cyl15 动态体积 0.86)
- 峰值在中心区域 (x=-7, y=36, z=59) — 解剖合理位置
- 切片预览: `rxa_static/vol/slices.png`

## 结论
- **静态重建的 GS 体积是连贯结构, 不是杂乱点云**。
- 之前的"杂乱点云"问题与动态 (形变) 相关 — 静态数据验证通过,
  说明体素化/导出链路本身正确; 动态情况需用 deform_final.bin 网络权重
  (--ckpt) 且注意相位匹配。

## density_reset
```bash
cargo run -p brush-process --bin fit_static images/RXA_chest.dcm \
    --points=50000 \
    --refine-every=400 \
    --eval-split-every=5 \
    --eval-views=8 \
    --eval-every=1000 \
    --cull-density=0.001 \
    --density-reset=3000 \
    --out=experiments/output/2026-08-27_rxa-static/density_reset/ \
    | tee experiments/output/2026-08-27_rxa-static/density_reset.log 2>&1
```

结果较好， 
[03:35:55] iter 10000 loss=  0.1399 psnr= 34.83 ssim=0.967 lpips=0.3906 visible=23779 splats=23779 (eval 8 views) | grads mean=1.0e-5 rot=1.1e-4 scale=1.0e-4 density=3.4e-5 | pos step≈2.01e-11mm
但不如不进行reset:
[02:59:53] iter 10000 loss=  0.1444 psnr= 34.67 ssim=0.968 lpips=0.3614 visible=46733 splats=46733 (eval 8 views) | grads mean=5.7e-6 rot=5.7e-5 scale=8.9e-5 density=2.9e-5 | pos step≈1.15e-11mm
