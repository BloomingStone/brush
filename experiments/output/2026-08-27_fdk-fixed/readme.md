# fdk-fixed — FDK 体积转置修复后的残差 GS 重跑

## 2026-08-27
- commit: a4702f0 (FDK 体积 x<->y 转置修复)
- 动机: 原 FDK 体积是 GT 沿 y=x 反射的镜像 (FDK@A ≈ mirror(GT@90°-A)),
  原 fdksplat 实验 (0.2891) 用了镜像先验 → 用修正体积重跑。
- 配置: 同 fdksplat (cyl_r15 + 有符号残差, 20k, scene=264),
  仅 --fdk-volume 换成 fdk_final_fixed/volume.nii.gz (转置修正)。
- 对照: cyl_r15_ref 0.2199 (无 FDK) / 原 fdksplat 0.2891 (镜像 FDK)。

## 修正方式更正 (重要)
- 之前 fdk_final_fixed/volume.nii.gz 用 --save-volume 存的"转置"体积与原件
  完全相同 (write_nifti_volume 内部 nifti-rs data.t() 抵消了手动转置)。
- 正确修正: fit_deform --fdk-transpose (加载时转置 vol_vec)。已验证
  iter0 pred(≈FDK DRR) 所有帧自身角度 corr>=0.72 (frame2 0.306->0.744)。
- 因此 fdksplat_fixed 那次 (0.2831) 仍是镜像先验 → 需用 --fdk-transpose 重跑。

## 结果 (20k, 修正后)
| 配置 | LPIPS | PSNR | SSIM | splats | 训练时间 |
|---|---|---|---|---|---|
| cyl_r15_ref (纯 GS) | 0.2199 | 44.56 | 0.991 | 46.5k | ~26.6min |
| fdksplat (镜像 FDK) | 0.2891 | 40.10 | 0.972 | 21.0k | ~18.2min |
| **fdksplat_trans (修正 FDK)** | **0.1926** | 42.76 | 0.984 | 10.3k | ~12.9min |

## 结论 (正面结果)
- **修正方向后 FDK 先验 + 有符号残差 GS 以 0.1926 LPIPS 大胜纯 GS 0.2199**
  (+0.027, 相对改善 12%), 且只需 10.3k splats (vs 46.5k, 1/4.5) 和
  ~13min (vs ~27min, 一半)。
- 验证了核心假设: FDK 提供静态解剖 + 残差只修动态 → 点云容量大幅减少。
- 待做: 稀疏权重扫描 (P2c), 残差 init/阈值调优, 更多视角评估确认。

## GS2volume 目视监察 (2026-08-27)
- 工具: gs2volume (brush-voxel 体素化 GS → 与 FDK 相加, XY padding 326,
  spacing 0.927mm, 布局 [y][z][x] x 最快)。
- 输出:
  - fdksplat_trans (signed 残差): gs_signed / fdk_padded / total_signed
    (.nrrd + .nii.gz) + slices_fdksplat_trans.png
    gs μ 范围 [0,0.368], total 相邻切片相关中位 0.965
  - cyl_r15_ref (纯 GS): gs_uns / total_uns, slices_cyl_r15_ref.png
    gs μ 范围 [0,0.036], total 相邻切片相关 0.917
- 观察: 两个 GS 体积相邻切片高度相关 (0.92-0.97) → 有连贯 3D 结构,
  非纯条纹。signed 残差 GS 峰值 μ 高 (0.368, 碘级), 纯 GS 均匀偏低。
