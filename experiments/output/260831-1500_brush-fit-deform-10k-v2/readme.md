# 260831-1500 brush-fit-deform-10k-v2 — 10k 验证 deform 输出 (默认 no-eval)

## 目的
默认 no-eval 模式下验证 10k 步 deform 训练 + deform 场输出质量
(scale 约束 + clone 恢复默认后, fbc885b4)。

## 运行命令 (commit: fbc885b4 + 默认 no-eval)
```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm \
  --points=5000 --iters=10000 --refine-every=400 --save-deform \
  --out=experiments/output/260831-1500_brush-fit-deform-10k-v2
```

## 结果
- 训练时间 ~14m18s (no-eval, 无 metrics.csv/eval 目录 — 默认 no-eval 生效 ✓)
- volume_phase00.nii.gz: 237×237×169 @1mm (图像等中心尺寸), phase=0 形变
  max|d|=50.74mm
- deform_final.bin + deform_field_phase{p:00-07}.nii.gz (8 相位, 5D)

### deform 场位移分布 (nii.gz 网格场)
| phase | mean | p50 | p95 | max |
|---|---|---|---|---|
| 0 | 10.29mm | 6.94 | 33.66 | 87.97 |
| 3 | 9.65mm | 5.93 | 34.81 | 87.45 |
| 7 | 9.79mm | 6.57 | 34.10 | 84.87 |

- 对比 (260831-1300, 5000 步 time 版): mean ~27mm → **10k 仅 phase + scale
  约束后 mean ~10mm, 收敛显著更好** (p95 ~34mm, 位移集中于血管/造影区域)

## 分析
- 默认 no-eval 正确: 产物仅 volume + deform (无 metrics.csv/eval)
- deform 场收敛性改善来自: ①10k 步训练更充分 ②scale 约束 (cap 10mm +
  penalty 0.1) 抑制过细长 splat → 场更物理 ③clone 恢复 (pd 0.02) 更多小
  splat 表达细节

## 后续
- eval 模式 (--eval) 确认最终指标 (此前 10k: PSNR 43.61 / LPIPS 0.1440)
- 检查 max 88mm 位移 (网格边缘/血管区, 是否合理)
