# fdk-opt26 — 用收敛的 27.6dB 优化体积作 FDK 先验

## 2026-08-27
- commit: e8543d9 (TV 梯度修复后 volopt 收敛 27.65dB)
- 动机: fdk_final 先验仅 19.9dB; volopt (DRR 梯度反传 + TV 归一化) 收敛到
  27.6dB 的优化体积 → 作为更强静态先验, 残差 GS 只需修更少。
- 配置: 同 fdksplat_trans (cyl_r15 + 有符号残差, 20k, scene=264),
  --fdk-volume=opt26/volume_refined.nii.gz + --fdk-transpose。
- 对照: fdksplat_trans (fdk_final 19.9dB 先验) 0.1926。
