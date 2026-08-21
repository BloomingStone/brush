# Future Plans — 性能与重建精度提升方向 (总结)

数据: heart_pig_stent_with_phase.dcm。详细实验记录见各 2026-08-21 文档。

## 现状基线

| 配置 | PSNR | 耗时 |
|---|---|---|
| 10k, 动态 0.98pct (~16k splats) | 38.90 | ~9min |
| 20k, 600k splats, ROI40+time | 43.17 | 1h52m |
| **20k, thr 1e-5 (~73k splats), ROI20+time+Charbonnier** | ~42.7-43.0 | **~32min** |

## 关键结论

1. **暗边**: 时序中位数帧显示为左上角 collimation 型, 非平滑 vignetting →
   保持 ROI 裁剪, 不做 flat-field 校正。
2. **densify 阈值**: 甜点 `--fixed-grad-thr=1e-5` (~73k splats), 20k 达
   PSNR ~43 (600k 的 43.17 但快 4x)。LPIPS 随 splat 数提升 (5e-6 ~190k
   感知更好)。详见 `2026-08-20-learnable-time-freq-sweep.md`。
3. **损失**: Charbonnier 最佳 (+0.20dB, LPIPS -0.007 vs L1)。默认建议改。
4. **time 条件化**: 呼吸明显数据 +4~5dB; 造影剂猪几乎无增益 (+0.04dB)。
5. **A1 粗运动场**: 随机仿射 -3.6~-4.8dB (坏); 3D 网格 3.3x 慢 (坏);
   零初始化仿射 慢12% + 质量微弱 (中性~略好)。**粗+细分解收益有限**。
   详见 exp/coarse-to-fine branch 中 `2026-08-21-A1-coarse-motion.md`。

## TODO

- [x] P1 densify 阈值扫描 (甜点 1e-5) — 详 `2026-08-20-learnable-time-freq-sweep.md`
- [x] 损失函数实验 (Charbonnier 最优) — 见 `2026-08-21-full20k-roi-time.md` 更新
- [x] A1 粗运动场三方案 — 详 `2026-08-21-A1-coarse-motion.md`
- [ ] **训练后期跳过近零位移 splat 的 hexplane**: 训练一段时间后, hexplane
      贡献位移 ~0 的高斯点 (近似静态) 不再走 deform 路径, 直接按 canonical
      投影 → 减少每步 per-splat 形变开销。随训练进行可用 splat 增多,
      成本递减。需实现: 按 |d_xyz| 阈值 mask 静态点, 只对活跃点跑 deform;
      注意梯度/密度的衔接。
- [x] **batchify / 多视图 batch**: 结论 — GPU 已 ~100% 饱和 (计算受限),
      每个 (phase,time) 形变点云不同需各自 rasterize, batchify 不会提速,
      只可能改善梯度稳定性但每步 ~Nx 代价。当前不做。
- [x] **coarse-to-fine 降分辨率训练** (已实现 --train-res-start/end, 负面):
      低分辨率下固定 densify 阈值 1e-5 使 splat 爆炸 (~397k vs base 73k),
      反而慢 ~2x 且精度受损。需按分辨率缩放阈值, 复杂度高收益不确定。
      详见 `2026-08-21-coarse-to-fine.md`。
- [ ] A2 长训练 + LR 调度 (提速后 30-40k + cosine)
- [ ] A3 HexPlane 分辨率 rs64→128 / rt32→64 / C16→32
- [ ] A4 评估口径: eval 统一裁剪后区域, 修复 LPIPS 可比性
- [ ] A5 deform 时间平滑正则 (相邻时间形变差 L2)
- [ ] f16 混合精度 (参考 dev/f16-mixed 分支)

## 更新 (2026-08-21): pruned≈0 根因 + init_density 扫描

**pruned≈0 根因**: 最终模型密度分布整体饱和在 ~5×水 (中位数 0.0105,
p5=1.6×水), 远高于任何合理 cull 阈值 (2e-3=水级也只剪 2.5%)。原因是
init_density=0.02 (10×水) 把初始球铺成高密度雾, 空气 splat 衰减梯度弱 →
密度永不降到阈值 → 从不被剪。**cull 阈值不是杠杆, init_density 才是。**

**init_density 扫描** (points=5000, 10k, thr 1e-5/Charbonnier/ROI20/time):

| initμ | 速率 | PSNR | LPIPS | splats |
|---|---|---|---|---|
| 0.002 | 19.4 | 40.98 | 0.4835 | 14,829 |
| 0.005 | 18.9 | 40.99 | 0.4786 | 17,946 |
| **0.010** | 18.2 | **41.32** | 0.4730 | 23,143 |
| 0.020 | 16.6 | 41.07 | 0.4657 | 29,851 |

→ **initμ=0.01 为甜点** (PSNR 41.32 历史最佳, 比 30000 点 base 快 ~1.6×)。
降低 initμ 后 pruned 恢复非零 (空气点可衰减被剪)。LPIPS 随 initμ 单调改善
(splat 多感知好)。默认 init_density 已改为 0.01。

## 更新 2 (2026-08-21): 高 prune 阈值扫描 (initμ=0.01, points=5000, 10k)

| cull_density | PSNR | LPIPS | splats | 剪总数/refine |
|---|---|---|---|---|
| 2e-4 (默认) | 41.26 | 0.4719 | 23,245 | 896 / 23 |
| 5e-4 | 41.19 | **0.4658** | 23,790 | 1391 / 39 |
| 1e-3 | 41.27 | 0.4661 | 23,326 | 2313 / 58 |
| 2e-3 | 41.19 | 0.4685 | 21,833 | 4765 / 106 |

→ **initμ 修好后 prune 阈值基本中性**: PSNR 无差异 (41.19-41.27), LPIPS
在 5e-4/1e-3 略好。2e-3 多剪 2× 但质量不掉 → 剪的是真空气废点, 点云更干净。
**建议默认 5e-4 或 1e-3** (LPIPS 最优且剪得足够多)。
