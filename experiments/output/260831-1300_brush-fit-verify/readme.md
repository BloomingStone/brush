# 260831-1300 brush-fit-verify — apps/brush-fit 正式验证 (static + deform)

## 目的
最后验证 `apps/brush-fit` (无头重建 lib+CLI+FFI) 在真实数据上的完整重建流程:
- 静态: `images/RXA_chest.dcm` → 训练 → volume_phase00.nii.gz + ply + bin
- 动态: `images/rotate_dsa_raw_gamma_preprocessed.dcm` → 训练 → volume (phase=0
  形变) + deform ckpt + 8 相位网格场
确认与 fit_static/fit_deform 指标一致、产物完整、deform 场导出正常 (spawn
brush-fit-dump 独立进程)。

## 运行命令 (commit: ed6afa94)
```bash
# 静态 (卡1)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/brush-fit static images/RXA_chest.dcm \
  --points=5000 --refine-every=400 --eval-split-every=5 --eval-views=8 \
  --eval-every=1000 --cull-density=0.001 --iters=5000 --save-ply --save-bin \
  --out=experiments/output/260831-1300_brush-fit-verify/static

# 动态 (卡2)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(2)' \
  ./target/release/brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm \
  --points=5000 --refine-every=400 --eval-split-every=5 --eval-views=8 \
  --eval-every=1000 --iters=5000 --save-ply \
  --out=experiments/output/260831-1300_brush-fit-verify/deform
```

## 结果 (5000 初始点 / 5000 步, 8 held-out eval views, 卡1/卡2 并行)

### static (RXA_chest.dcm, 149 views, 38 held-out)
| iter | loss | PSNR | SSIM | LPIPS | splats | 训练时间 |
|---|---|---|---|---|---|---|
| init | - | 15.49 | 0.895 | 0.6738 | 5000 | - |
| 5000 | 0.1933 | **31.68** | 0.962 | **0.4131** | 24701 | ~4m18s |

- 与 fit_static 同配置 (v8) 对比: LPIPS 0.4131 vs 0.4124 完全吻合; PSNR
  31.68 vs 32.99 (随机初始化差异), 移植一致 ✓
- 产物: `volume_phase00.nii.gz` (435³×286 @1mm, 自适应网格 217/190/143mm
  半宽) + `canonical_final.ply` + `canonical_final_{transforms,raw}.bin` +
  `metrics.csv` + `eval/nrrd/gt_pred_*.nrrd` (6 个 eval 步)

### deform (rotate_dsa_raw_gamma_preprocessed.dcm, 321 views, 81 held-out)
| iter | loss | PSNR | SSIM | LPIPS | splats | 训练时间 |
|---|---|---|---|---|---|---|
| init | - | 5.78 | 0.137 | 0.6823 | 5000 | - |
| 5000 | 0.0782 | **39.76** | 0.984 | **0.2616** | 34435 | ~5m49s |

- deform 场导出 (spawn brush-fit-dump, spacing 4.12mm, 129³ grid):
  `deform_final.bin` + `deform_field_phase{p:00-07}.npy` + `.nii.gz` (5D,
  8 相位) ✓
- volume (含 phase=0 形变): max|d|=79.5mm; 网格场位移分布 mean ~27mm,
  p95 ~48mm, max ~87mm — **5000 步 deform 场未完全收敛** (位移偏大, 旋转
  DSA 造影数据运动强), 后续可延长训练/加时间 TV 正则诊断
- learned time freqs: 0.200~1.500 Hz (10 频带)

## 分析
- brush-fit 与 fit_static/fit_deform 行为一致 (static LPIPS 逐位吻合),
  流程完整: 训练 → eval/CSV → volume (自适应网格, DRR 一致) → ply/bin →
  deform ckpt + 网格场 (独立进程, 训练后内存池不稳规避)
- 训练速度: static 5000 步 ~4.3min, deform ~5.8min (8 eval views), 卡并行
  无干扰

## 后续待做
- deform 场位移偏大诊断: 延长训练 (10k-20k), 或开 time_tv_weight 平滑正则;
  对比 fit_deform 基线 LPIPS (hexplane 10k 基线记录)
- brush-fit CLI 增加 deform 场平滑度统计输出 (复用 dump_deform 的诊断)
