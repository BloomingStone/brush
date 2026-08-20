# 2026-08-20 HexPlane 融合内核 10k 全量实验 (4 卡)

数据: `images/RXA_pig_with_phase.dcm` (321 训练视图 / 81 held-out, 648x474)
配置: `--iters=10000 --points=15000 --refine-every=400 --eval-split-every=5
--eval-views=8 --eval-every=500 --split`
选卡: `env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'`
进程: `systemd-run --user --scope` (防会话清理杀掉)

## 结果总表

| 卡 | run | deform | densify | splats(end) | PSNR | SSIM | LPIPS | 耗时 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | fit10k_hexplane | hexplane | 固定 1e-6 | 283k | 40.64 | 0.9925 | **0.172** | 29.5min | ✅ |
| 1 | fit10k_hashgrid | hashgrid | 固定 1e-6 | 262k | 27.47* | 0.951 | 0.527* | — | ❌ OOM@3000 |
| 2 | fit10k_hashgrid_dyn | hashgrid | 动态 0.98pct | 14.7k | 34.55 | 0.9864 | 0.331 | 77.6min | ✅ |
| 3 | fit10k_hexplane_dyn | hexplane | 动态 0.98pct | 14k | **41.28** | 0.9914 | 0.266 | 8.5min | ✅ |
| 3 | fit10k_hexplane_dyn_hr | hexplane | 动态 0.98pct | 15k | 41.25 | 0.9921 | 0.242 | 7.8min | ✅ |
| 2 | fit10k_hexplane_ps | hexplane+predict_scaling | 固定 1e-6 | — | — | — | — | — | ❌ 密度坍缩@1200 |

\* hashgrid 基线 iter 3000 时 OOM 崩溃 (未到 10k), 数据为中途值。

## 最终结论

### 1. HexPlane 全面优于 HashGrid (质量 + 速度)
- **同 ~15k splats (动态阈值, 10k 步直接对比)**: hexplane PSNR **41.28** /
  LPIPS 0.266 / **8.5min** vs hashgrid 34.55 / 0.331 / **77.6min** ——
  **+6.7dB, 快 9×**。
- **稠密基线对比**: hexplane 283k splats 干净跑完 10k (40.64dB / LPIPS 0.172,
  29.5min); hashgrid 262k splats 时 GPU **OOM** 崩溃 (仅 27.47dB@3000 步)。
  稠密 hashgrid 每步 ~2s 且内存耗尽, 实际不可用。
- 结论: HexPlane 融合内核 (0.039s/步 @30k) 解决了 hash-grid+MLP 训练过慢的
  原问题, 且作为低频心脏形变的表示质量更高。

### 2. densify 阈值是速度主导因素 (动态百分位 > 固定阈值)
- 固定 1e-6 阈值过激进 → splat 爆炸到 300k (hexplane 29.5min); 动态 0.98
  百分位 → ~14k splats, 快 3.5×, PSNR 反而更高 (41.28 vs 40.64)。
- 但**稠密 splat 明显改善 LPIPS** (0.172 vs 0.266): 更多小 splat 提升感知
  质量, PSNR 略降 —— 这是 quality/speed 的真实权衡, 按需选阈值。

### 3. 高容量 HexPlane 免费提升感知质量
- 动态阈值下 rs128/rt64/C32 (vs rs64/rt32/C16): LPIPS 0.266→0.242, PSNR 持平
  (41.25 vs 41.28), 速度不变 (7.8min)。更大的平面容量捕捉更多细节。

### 4. predict_scaling 变体崩溃 → 佐证质量守恒默认关
- `--predict-scaling` 让形变网络用缩 scale 来降衰减, 密度被优化器一路压低 →
  iter 1200 全量 prune 到 0 splat → matmul m=0 除零崩溃。scale 形变会吸收
  密度职责导致密度坍缩, 默认不预测 d_scaling (保质量) 是正确的。

### 5. 推荐默认配置
- 后端: hexplane (默认), `--deform-backend=hashgrid` 保留作对照
- densify: 动态百分位 (去掉 `--fixed-grad-thr`) —— 更快且 PSNR 不降
- 容量: rs128/rt64/C32 (如预算允许, LPIPS 免费提升)
- `predict_scaling`: 保持默认关
