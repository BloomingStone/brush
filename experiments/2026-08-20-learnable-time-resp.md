# 2026-08-20 可学习时间条件化 (enable-time) — 呼吸运动自动拟合 (hexplane 10k)

数据: `/media/data4/sj/brush/images/heart_pig_stent_with_phase.dcm`
(722 帧 / 30fps / 516x516, 577 训练视图 / 145 held-out, 绕 SI 轴旋转 ±120°,
FOV 集中在心脏区, 无造影剂, 呼吸运动明显)

配置: `--iters=10000 --points=15000 --refine-every=400 --eval-split-every=5
--eval-views=8 --eval-every=500 --split` (动态 densify 阈值, 与
2026-08-20-hexplane-fused-10k 的 fit10k_hexplane_dyn 一致)

| run | deform | time 条件 | splats(end) | PSNR@10k | SSIM | LPIPS | 耗时 |
|---|---|---|---|---|---|---|---|
| fit10k_hex_base | hexplane rs64/rt32/C16 | 无 | 16555 | 33.85 | 0.9769 | 0.4526 | ~8.3min |
| fit10k_hex_time | hexplane rs64/rt32/C16 | **--enable-time** | 15979 | **38.90** | 0.9807 | 0.4434 | ~8.8min |
| fit10k_hash_base | hashgrid | 无 | 15365 | 27.71@1500 | - | - | 中止 |
| fit10k_hash_time | hashgrid | --enable-time | 15358 | 28.05@1500 | - | - | 中止 |

## 关键结论

1. **`--enable-time` 巨大增益**: 同配置下 PSNR 33.85 → 38.90 (**+5.05 dB**),
   SSIM 0.977→0.981, LPIPS 0.453→0.443。且 10k 步仍在上升 (38.23@9k →
   38.90@10k), 未收敛到顶 — 值得跑到 15k/20k 看上限。

2. **机制**: 形变网络额外以真实物理时间 `t = f/fps` 为条件, 通过一个
   **可学习频率的傅里叶编码** (10 个 log 间隔频率, 初始 0.15-3.0 Hz, 可训练)
   自动拟合数据中的呼吸 (~0.83 Hz) 及其它非周期运动 —— 无需预知呼吸频率
   范围。心脏 phase 保持已知圆环轴 (DICOM (0071,1010))。

3. **实测频率**: 训练后打印的 learned time freqs 仍接近初始 log 网格
   (0.150 ... 0.792 ... 3.000) — 初始网格已含 0.792 Hz (≈呼吸 0.83 Hz),
   网络通过 MLP 幅度选择该基函数, 频率本身无需大幅迁移。若想让频率明显
   "迁移", 需更少的基函数或更宽的网格间隔 (可后续验证)。

4. **hashgrid 中止**: 1500 步仅 ~28 dB 且每步 ~0.6s (hexplane ~0.04s),
   用户决定后续不再做 hashgrid 实验。

5. **时间修复**: `FrameTimeVector` 退化为 "0" 的文件 (两个 pig 序列都是)
   现在回退到 `t = f/fps` (parser.rs), 否则 time 恒为 0、time 条件化失效。

## 后续候选
- 跑到 15k/20k 看 --enable-time 上限
- 减少基函数 (如 6 个) 或放宽网格, 观察频率迁移
- 与预处理提取的 resp 信号 (local.properties/refs/extract_breathing.py) 对比/
  联合: resp 绝对位移轴 vs 可学习时间轴
