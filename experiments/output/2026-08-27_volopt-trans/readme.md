# volopt-trans — 体积优化 (DRR 梯度反传) 转置修正重跑

## 2026-08-27
- 动机: 之前 fit_vol* (fit_vol_pad13_tv 等) 用 drr+梯度反传迭代优化体积
  (FDK + padding + TV), 结果差。根因也是 nifti data.t() 转置 → 体积世界
  系镜像, DRR 梯度反传在错误方向优化。用 --volume-transpose 重跑。
- 配置: fit_volume + fdk_final 体积 + --volume-transpose + TV(l1) 约束,
  训练中每 save-val-every 存 8 视角 pred|gt 拼接 NRRD 验证序列。

## 结果
- 转置修正生效: val NRRD 方向正确 (iter0 GT-pred corr 0.857), 水平拼接
  (修复了逐像素交错竖条纹 bug)。
- 但体积优化仍退化: psnr(int) 19.93 -> 14.76 (600 iter), 不收敛。
- 结论: fit_vol* 失败不只是转置 — 动态心脏数据上静态体积优化本质受限
  (体积无法表示运动, 梯度越推越坏)。DRR 梯度反传路线是死胡同。
- 正确路线: fit_deform FDK 先验 + 有符号残差 GS (0.1926) — FDK 作静态
  先验, splat 处理动态, 已验证有效。
