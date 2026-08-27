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
```bash
cd /media/data4/sj/brush-hexplane-1d8eac && \
setsid systemd-run --user --scope -p CPUWeight=100 -p MemoryMax=12G -- \
  env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/fit_deform \
  images/rotate_dsa_raw_gamma_preprocessed.dcm \
  --points=5000 \
  --init-shape=cylinder \
  --init-radius-scale=1.5 \
  --no-fov-filter \
  --cull-density=5e-4 \
  --loss=charbonnier \
  --roi=20 \
  --enable-time \
  --time-min-freq=0.2 \
  --time-max-freq=1.5 \
  --refine-every=400 \
  --eval-split-every=5 \
  --eval-views=8 \
  --eval-every=1000 \
  --split \
  --iters=20000 \
  --out=experiments/output/2026-08-27_dsa-cyl15/dsa_cyl15_gamma \
  > experiments/output/2026-08-27_dsa-cyl15/dsa_cyl15_gamma.log 2>&1 \
  < /dev/null & disown
```

## 结果
| 配置 | 数据 | LPIPS | PSNR | SSIM | splats | 训练时间 |
|---|---|---|---|---|---|---|
| dsa_cyl15 | rotate_dsa_raw | **0.1297** | 42.09 | 0.990 | 52.4k | ~28min |
| dsa_cyl15_gamma | rotate_dsa_raw_gamma_preprocessed | (进行中) | | | | |

- phase0 time0 体积 (纯 GS, no-fdk, deform field phase00, max|d|=19.9mm):
  `dsa_cyl15/phase0_vol/gs_uns.nrrd` (+nii.gz), gs μ [0,0.157], 相邻切片相关 ~0.9x

## ③ dsa_sub_with_phase — images/heart_pig_sub/subtracted_with_phase.dcm
- 剪影(减影)后数据 (只剩造影/动态, 静态背景被减掉)。虽然不一定符合真实
  3D 结构, 跑纯 GS 试试。同样 cyl_r15 配置。

## ④ dsa_sub_signed — 剪影数据 + --signed (有符号高斯)
- 动机: 剪影重建中"前景遮挡却使结构变亮"与 Beer-Lambert 密度积分矛盾;
  有符号高斯 (负密度前景) 可表达"图像变亮 = 低密度积分"。
- 配置: 同 cyl_r15 纯 GS + --signed (无 FDK, opac=MU_WATER·raw 可负,
  raw=±1e-5 init)。

## 最终结果总结
| 配置 | 数据 | LPIPS | PSNR | SSIM | splats | 训练时间 |
|---|---|---|---|---|---|---|
| dsa_cyl15 | rotate_dsa_raw | 0.1297 | 42.09 | 0.990 | 52.4k | ~28min |
| **dsa_cyl15_gamma** | rotate_dsa_raw_gamma_preprocessed | **0.1170** | 45.53 | 0.994 | 45.4k | ~27min |
| dsa_sub_with_phase | heart_pig_sub/subtracted_with_phase | 0.1358 | 42.71 | 0.992 | 65.0k | (暂停, 有遮挡/亮度矛盾伪影) |
| dsa_sub_signed | 同上 + --signed | (11k 时 0.1936, 已暂停) | | | | |

- **gamma 预处理优于 raw**: 0.1170 vs 0.1297 (LPIPS -10%), PSNR +3.4dB。
  (修复低灰度变 0 后, 暗部/低对比结构重建更好)
- phase0 time0 体积 (纯 GS, deform field phase00):
  - dsa_cyl15: `dsa_cyl15/phase0_vol/gs_uns.nrrd` (μ[0,0.157], 相关 0.857)
  - dsa_cyl15_gamma: `dsa_cyl15_gamma/phase0_vol/gs_uns.nrrd` (μ[0,0.085])
- sub 剪影两个实验 (unsigned + signed) 都有建模/优化问题, 暂停, 后续再优化。

## 形变/体素化验证总结 (2026-08-27 晚)
- **verify_gs**: ply + deform_final.bin (网络权重) 精确复现训练 eval (42.09dB)。
  每视图用正确 phase/time; Forward/Backward 渲染无差异; raw logit 激活一致。
- **dump_deform 网格导出有 3 个问题**: ① 轴序 (数据 z-major 但声明
  [nx,ny,nz,3]) ② 缺 d_rotation (只存 3 分量位移) ③ gs2volume 插值索引 bug
  (ix-major 用了错误的 at 索引, 已修 → z-major)。
- **体素化器交叉验证通过**: brush-voxel vs R2Gaussian 单 splat corr=1.0,
  全量 corr=0.995, 比值中位 1.000。差异全来自 Rust 插值 bug。
- **正确做法**: 形变用 deform_final.bin (网络权重, --ckpt, 含旋转),
  不用网格场。phase0 体积: `dsa_cyl15/phase0_vol_net/gs_uns.nrrd`
  (μ[0,0.251], 非零 ~, 相邻切片相关)。
