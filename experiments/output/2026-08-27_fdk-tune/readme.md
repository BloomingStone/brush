# fdk-tune — FDK+残差 GS 超参调优 (P4③)

## 实验信息
- 日期: 2026-08-27
- commit: e4a3890 (P2c resid_sparse_weight 接入)
- 目的: 在 fdksplat_trans (0.1926) 基础上扫描 残差稀疏权重 (P2c) 与
  残差初始化密度, 找更优配置。

## 运行命令 (20k, cyl_r15 + FDK 转置修正, scene=264)
```bash
./target/release/fit_deform images/pig-data-cor-new-phase.dcm \
  --points=5000 --init-shape=cylinder --init-radius-scale=1.5 --no-fov-filter \
  --cull-density=5e-4 --loss=charbonnier --roi=20 --enable-time \
  --time-min-freq=0.2 --time-max-freq=1.5 --refine-every=400 \
  --eval-split-every=5 --eval-views=8 --eval-every=1000 --split --iters=20000 \
  [--resid-sparse-weight=W | --fdk-resid-init-density=D] \
  --fdk-volume=.../fdk_final/volume.nii.gz --fdk-meta/--fdk-calib 同目录 \
  --fdk-steps=256 --fdk-transpose --out=...
```

## 结果 (20k)
| 配置 | resid-init | sparse | LPIPS | PSNR | splats |
|---|---|---|---|---|---|
| fdksplat_trans (基线) | 1e-5 | 0 | **0.1926** | 42.76 | 10.3k |
| init1e-4 | 1e-4 | 0 | **0.1921** | 42.71 | 10.5k |
| init1e-3 | 1e-3 | 0 | 0.1945 | 42.66 | 11.9k |
| sparse1e-3 | 1e-5 | 1e-3 | 0.1969 | 42.79 | 10.5k |
| sparse1e-2 | 1e-5 | 1e-2 | 0.1949 | 42.68 | 10.6k |

## 分析
- 调优窗口很平: 所有配置 LPIPS 0.192-0.197, 与基线 0.1926 基本持平。
- init1e-4 边际最优 (0.1921), 但差异 <0.001, 无统计意义。
- L1 残差稀疏先验 (P2c) 未带来提升 (0.1949-0.1969, 略差) — 残差本身已
  足够稀疏 (10.3k splats), 无需额外稀疏惩罚。

## 结论
- fdksplat_trans (0.1926) 是稳定最优配置; 继续在主干上优化 (如更长训练、
  更多视角评估、形变场质量)。

## 后续待做
- 验证 0.1926 的稳健性: 更多 eval 视角 / 更长 iters (30k)。
- 检查形变场质量 (deform_field) 与动态区域重建。

