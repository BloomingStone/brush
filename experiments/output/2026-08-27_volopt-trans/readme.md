# volopt-trans — 体积优化 (DRR 梯度反传) 转置修正重跑

## 实验信息
- 日期: 2026-08-27
- commit: e4a3890 (TV 梯度归一化修复) / a2ae1eb (方向诊断)
- 目的: 验证之前 fit_vol* (DRR 梯度反传迭代优化体积) 失败的两个假设:
  (1) nifti data.t() 转置使体积世界系镜像 (2) TV 约束导致的非收敛。

## 运行命令
```bash
./target/release/fit_volume images/pig-data-cor-new-phase.dcm \
  --volume=experiments/output/fdk-residual/fdk_final/volume.nii.gz \
  --meta=.../fdk_final/meta.json --calib=.../fdk_final/calib.json \
  --volume-transpose --iters=3000 --lr=1e-4 --batch=16 --steps=256 \
  --tv=0.05 --tv-type=l1 \
  --save-val-nrrd=.../val --save-val-every=500 --out=...
```

## 结果与根因分析
1. **转置修正生效**: val NRRD 方向正确 (iter0 GT-pred corr 0.857), 并修复了
   验证 NRRD 逐像素交错导致的竖条纹 bug (改为水平拼接)。
2. **loss 上升根因 = TV 梯度尺度 bug**: 
   - 现象: 有 TV=0.05 时 loss 持续上升 (0.036->0.178), PSNR 19.9->14.8;
     无 TV 时 loss 收敛 (0.036->0.0095), PSNR 20.1->25.95 → 反传本身正确。
   - 根因: TV 梯度 O(1)/体素 (未归一化 sum), 数据梯度 O(1e-6) (mean 归一化
     ÷n_pix), 差 ~1e6 倍 → TV 完全淹没数据梯度, 优化器实际在"最小化 TV"
     (抹平体积), 数据 loss 反而上升。
   - 修复: tv_grad / tv_l1_grad 按体素数归一化 (变 per-voxel mean)。
3. **TV 归一化后收敛**: TV l1/l2 300 iter loss 0.036->0.0053, PSNR 20.1->26.05
   (静态上限 ~26dB, 动态心脏数据上静态体积无法再进一步)。

## 结论
- fit_vol* 失败的两个根因均已定位并修复: 转置 (加载时修正) + TV 梯度尺度。
- 修复后 DRR 梯度反传体积优化可收敛到 26dB (静态极限)。
- 但要处理动态 (心脏/冠脉) 仍需 splat 形变: 最佳路线 = fit_deform FDK 先验
  + 有符号残差 GS (0.1926 LPIPS)。

## 后续待做
- (可选) 把收敛的 26dB 体积作为更强的静态先验喂给 fit_deform 残差 GS,
  对比现用 fdk_final (19.9dB) 先验。

