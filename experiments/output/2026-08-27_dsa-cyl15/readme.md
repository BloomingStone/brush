# dsa-cyl15 — rotate_dsa_raw 纯 GS 重建 (cyl_r15) + phase0 volume

## 2026-08-27
- 数据: images/rotate_dsa_raw.dcm (合成 DSA: 真实CTA+几何运动+投影, 强度恒定;
  之前结果好: dsa_grad40k LPIPS 0.1184)
- 目的: 用最佳纯 GS 配置 (cyl_r15 圆柱 init, 2026-08-26 init-cylinder 最优)
  重建, 并输出 phase 0 time 0 体积 (gs2volume + deform field phase00) 目视监察。
- 配置: --points=5000 --init-shape=cylinder --init-radius-scale=1.5
  --no-fov-filter + baseline (cull 5e-4, charbonnier, enable-time, 20k)

## ② dsa_cyl15_gamma — rotate_dsa_raw_gamma_preprocessed.dcm
- 修复某些地方灰度值过低变 0 的预处理版本; 同样 cyl_r15 纯 GS 配置。

## 结果
| 配置 | 数据 | LPIPS | PSNR | SSIM | splats | 训练时间 |
|---|---|---|---|---|---|---|
| dsa_cyl15 | rotate_dsa_raw | **0.1297** | 42.09 | 0.990 | 52.4k | ~28min |
| dsa_cyl15_gamma | rotate_dsa_raw_gamma_preprocessed | (进行中) | | | | |

- phase0 time0 体积 (纯 GS, no-fdk, deform field phase00, max|d|=19.9mm):
  `dsa_cyl15/phase0_vol/gs_uns.nrrd` (+nii.gz), gs μ [0,0.157], 相邻切片相关 ~0.9x
