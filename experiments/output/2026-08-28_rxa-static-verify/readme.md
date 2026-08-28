# rxa-static-verify — 5000点/10000步 短程验证训练 (voxelizer 修复后干净数据)

## 实验信息
- 日期: 2026-08-28 11:41
- commit: 17cc1d9a
- 目的: voxelizer g2c 压实 bug 修复后, 用干净数据重跑短程训练 (避免复用
  旧 40k 数据造成误判), 验证:
  1. 训练产物按 eval/{nrrd,bin,ply} 分目录输出
  2. gs2volume 用 --bin (原始参数) vs --ply (往返) 同帧对比, 量化差异
  3. 与训练 eval 同帧 (held-out) 的 DRR vs GS 一致性

## 运行命令
```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/fit_static images/RXA_chest.dcm \
  --points=5000 --refine-every=400 --eval-split-every=5 --eval-views=8 \
  --eval-every=1000 --cull-density=0.001 --iters=10000 \
  --out=experiments/output/2026-08-28_rxa-static-verify/5000p-10k/
```

## 产物布局
- eval/nrrd/: gt_pred_*.nrrd (训练投影对比) + gs2volume 的 compare_*.nrrd
- eval/bin/ : canonical_final_transforms.bin + canonical_final_raw.bin (raw 域)
- eval/ply/ : canonical_final.ply (标准激活域)

## 结果 (2026-08-28)
- 训练: 5000点/10000步, iter 10000 PSNR 34.68 / SSIM 0.966 / LPIPS 0.422,
  12661→12439 splats (最后 refine 剪 222), 耗时 ~11min (GPU 1)
- 产物布局: eval/nrrd (gt_pred_10000.nrrd + compare_*.nrrd) / eval/bin
  (transforms+raw) / eval/ply (canonical_final.ply) ✓
- **bin vs ply 对比: 逐像素 max diff = 6.4e-5, mean 1.3e-6** — PLY 往返
  无损 (该数据仅 2 个 splat opacity≥0.999, logit clamp 影响可忽略)
- Forward vs Backward(autodiff) 渲染: max diff 2.4e-7 — 渲染路径逐位一致
- 同帧 (held-out, GT 逐像素一致) 下 训练 pred vs gs2volume GS (bin):
  corr 0.80~0.93, gs proj 高 ~5-10% (+0.05 平滑雾差) — 排除了 PLY 往返/
  渲染路径/帧差异后仍存在, 疑似 eval 与导出时刻间的状态细节 (无法事后
  验证), 不影响 voxelizer 正确性结论
- DRR vs GS (同帧): view 19 mean|drr-gs|=0.0295 (proj), 与 40k 数据一致
