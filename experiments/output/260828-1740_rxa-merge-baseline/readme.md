# rxa-merge-baseline — 合并 brush-fix-voxelizer 后基线验证

## 2026-08-28 17:40
- 合并 6 个 fix 提交 (merge-fix-voxelizer 分支): 体积布局统一 / g2c 压实 /
  refine_until_frac / **fit_static deform 真根因修复** / 协方差转置 / backward 链式。
- 目的: 验证合并后 fit_static 训练/导出正常 (deform 修复 + 新 backward 链式
  是否影响收敛), 并刷新基线指标 (旧基线受 deform 污染, 不可直接对比)。
- 卡2: baseline (无约束); 卡3: 最优配置 ab_cap10_pen01 + pd002_gf100。
- 快速协议 5000 点 / 10000 iters。

## 命令
```bash
cargo run -p brush-process --bin fit_static images/RXA_chest.dcm \
  --points=5000 --refine-every=400 --eval-split-every=5 --eval-views=8 \
  --eval-every=1000 --cull-density=0.001 --iters=10000 --density-reset=3000 \
  [--scale-cap-mm=10 --scale-cap-weight=0.5 --screen-area-penalty=0.1 \
   --percent-dense=0.02 --growth-frac=1.0] \
  --out=experiments/output/260828-1740_rxa-merge-baseline/<config>/
```

## 结果
(待填)
