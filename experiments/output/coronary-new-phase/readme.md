# coronary-new-phase

## 2026-08-25 验证正确相位 (158帧/周期, 主动脉瓣计数) 下的重建与形变场
- commit: 4b8fa39
- 数据: images/pig-data-cor-new-phase.dcm (相位已校正, 旧数据相位未同步)
- 目的: 检查正确相位下冠脉模糊是否改善 + 形变场是否合理
- 配置: 默认 (points 5000 / initμ 0.01 / cull 5e-4 / thr 5e-6 / charbonnier / roi20 / enable-time)

### default (2026-08-25 11:xx, 20k, thr5e-6 + deform 导出)
./target/release/fit_deform images/pig-data-cor-new-phase.dcm --points=5000 --init-density=0.01 --cull-density=5e-4 --loss=charbonnier --roi=20 --enable-time --time-min-freq=0.2 --time-max-freq=1.5 --refine-every=400 --eval-split-every=5 --eval-views=8 --eval-every=1000 --split --iters=20000 --out=experiments/output/coronary-new-phase/default

### jitter006 (2026-08-25, 20k, +--time-jitter=0.006)
...同 default + --time-jitter=0.006

### 40k (2026-08-25, 40k)
...同 default --iters=40000

## 结果 (2026-08-25)
| 配置 | PSNR | SSIM | LPIPS | blur_ratio | edge_l1 | splats |
|---|---|---|---|---|---|---|
| **default (新相位)** | **44.14** | 0.9911 | **0.2147** | 0.655 | 0.0075 | 38,376 |
| jitter006 | 44.04 | 0.9910 | 0.2151 | — | — | 37,710 |
| 旧相位 default (对照) | 42.59 | 0.9904 | 0.2286 | 0.623 | 0.0088 | 37,256 |

→ **相位校正显著有效**: PSNR +1.55dB, LPIPS -6%, 边缘更锐 (blur 0.623→0.655)
  且更准 (edge_l1 -15%)。jitter 中性 (正确相位下无益)。
- 形变场: 8 相位 nii.gz (5D [x,y,z,1,3]), 位移随相位变化 59.5% (10.9-19.5mm)
  → 心脏运动被建模, 但绝对锐度仍 ~0.65 (35% 软), 冠脉模糊仍存。

## 形变场诊断 (2026-08-25)
- 结构: 8 相位 nii.gz, 5D [64,64,48,1,3], affine 随 nii (sform_code=2), 与 ASOCA dvf 一致。
- **形变场是高频噪声**: 相邻体素位移差 101-108% (ASOCA 8%), 空间自相关 lag1=-0.3 (ASOCA 0.99)。
- 解剖区位移 p50 6.7mm (合理) 但 p95 26mm, 且解剖区内相邻差 20.6mm (167%) → 空间不连贯。
- **特征频率 ~14mm 周期** (0.07/mm, 高频>1/8mm 为 0%) — 非白噪声, 是 HexPlane 单元
  (4.8mm) 低通 + 学习到的错误周期模式。
- 根因: 每相位投影约束弱 + 无空间平滑约束 → deform 用带限周期模式拟合噪声。
