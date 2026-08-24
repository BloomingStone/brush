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

## 下一步改进方向 (2026-08-21, 精度+速度仍不达预期)

### 速度
- [ ] **A. per-step 剖析**: render / deform / loss / optimizer 各占比 (轻量
      计时探针)。确认瓶颈再动手。若 render >70%, 走 B; 若 deform 意外高,
      另有优化点。
- [ ] **B. 修复 coarse-to-fine**: 上次失败根因 = 低分辨率下每 splat 梯度集中
      放大, 固定 densify 阈值 1e-5 使 splat 爆炸 5x (73k→397k)。修复:
      **densify 阈值随分辨率缩放** `thr = thr_full / res²` (res=0.25 →
      ×16 阈值, splat 数回归正常)。预期 2-3x 真实提速。

### 精度
- [ ] **C. 训练时加入 LPIPS/感知损失** (小权重 0.1-0.2): 直接优化最弱指标
      (LPIPS 0.43-0.47)。低分辨率 128² VGG 前向摊薄成本。
- [ ] **D. 提高 HexPlane 分辨率/特征** (rs64→128 / rt32→64 / C16→32):
      形变场表达力。此前同速 LPIPS 0.266→0.242; PSNR 卡 ~43 可能容量不够。
- [ ] **E. deform 时间平滑正则** (相邻时间形变差 L2): 提升 held-out 泛化。

### 执行顺序
1. A 剖析 (确认瓶颈)
2. B 分辨率缩放阈值修复 (2-3x 提速)
3. C LPIPS 训练损失 + D 高分辨率 HexPlane (冲精度)

## 更新 3 (2026-08-21): 4 实验 + 平台期根因 + ROI 分布 + patch c2f
(详见 `2026-08-21-roi-distribution-patch-c2f.md`, 脚本 `tools/roi_metric_dist.py`)

- **4 实验** (最优配置, eval-split=5): RXA_pig 40k = **46.1/0.9942/0.184**;
  rotate_dsa 40k = **43.4/0.9910/0.123**; 20k→40k 仍 +1.2~1.4dB, 未饱和。
- **平台期根因 (确认用户怀疑)**: grad_mean ~5-10k 跌破 densify 阈值 1e-5
  → densify 停 → splat 平台 ~18k; 梯度收缩主要由 LR 线性衰减造成
  (lr_mean 1.9e-5→2e-6)。PSNR 仍涨 → 非硬上限。
- **ROI 分布**: pig 中心窗训练中从"先好"翻转为"最差" (ΔL1 +0.0006@40k),
  全图指标被容易的大多数支配 → SSIM 早饱和 (0.9994@20k)。dsa 相反
  (中心 PSNR 45.2 vs 全图 43.4) → 困难区域因几何而异, 需残差引导自适应。
- [ ] **patch coarse-to-fine**: 全窗 1k 粗形 → 1/2 窗 → 更小窗残差引导采样。
      关键: 残差图=重要性信号(免费) / 覆盖(全窗混 patch) / patch 归一 /
      densify 阈值按面积比缩放 (防提前停) / ROI 分布验证。
- [ ] 廉价验证: 80k 或 warm-restart LR; 固定中心窗 `--roi=54x76` 对照。
