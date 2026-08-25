# rigid-anchor — 刚性锚定正则扫描 (|mean(d_xyz)|²)

## 2026-08-26 消除形变场规范自由度 (全局平移)
- commit: bf7844d
- 数据: images/pig-data-cor-new-phase.dcm (校正相位)
- 目的: 形变场全场平均位移 16.87mm (规范自由度, canonical 未锚定) 导致
  静态区域 (骨/胸腔) 位移≠0。--rigid-anchor-weight=W 惩罚 |mean(d_xyz)|²,
  强制整体平移留在 canonical, 与 plane-TV 组合扫描。
- 对照: baseline (无TV无锚定) LPIPS 0.2147 @ coronary-new-phase/default;
  plane-tv 0.01 LPIPS 0.2097 @ plane-tv/w001。
- 配置: 同 baseline (points 5000 / initμ 0.01 / cull 5e-4 / thr5e-6 /
  charbonnier / roi20 / enable-time / 20k iters) + anchor/TV 扫描。

## 结果 (2026-08-26, 20k iters, thr5e-6)
| 配置 | PSNR | SSIM | LPIPS | offset | 残余(去均值) |
|---|---|---|---|---|---|
| baseline (复用) | 44.14 | 0.9910 | 0.2147 | 16.87mm | 7.57mm |
| plane-tv 0.01 (复用) | 43.59 | 0.9910 | 0.2097 | 16.05 | 7.58 |
| anchor 1e-3 | 44.52 | 0.9910 | 0.2096 | 2.41 | 10.06 |
| **anchor 1e-3 + TV 0.01** | 43.94 | 0.9910 | **0.2045** | 3.35 | 10.64 |
| anchor 1e-2 | 43.76 | 0.9910 | 0.2145 | 2.80 | 11.40 |
| anchor 1e-2 + TV 0.01 | 43.60 | 0.9910 | 0.2105 | 2.56 | 10.04 |

运行命令: baseline 命令 + --rigid-anchor-weight=W [--plane-tv-weight=0.01]
(cf. coronary-new-phase/default)。4 卡并行, 各 ~25min。

## 分析 (2026-08-26)
- **锚定消除规范自由度**: 全场偏移 16.87 → 2.4-3.4mm (λ=1e-3~1e-2 均有效)。
  形变场不再背负整体刚性平移。
- **LPIPS 最佳 = anchor 1e-3 + TV 0.01 (0.2045)**, 优于 baseline (0.2147)
  与单独 TV (0.2097)。λ 甜点 ~1e-3, 1e-2 开始伤质量。
- 残余空间变化 ~10mm (基线 7.6mm): 中轴切片底部出现强相干 12-15mm 区域
  (疑似膈肌/胸壁呼吸运动, baseline 被 17mm 偏移掩盖)。
- p95 升高 (14→21mm): 锚定后 FOV 边界/角落未约束区位移更集中。
- **训练末尾导出 OOM 已修复**: 训练后设备内存池不稳定, 整网格 forward 的
  autotune 分配 3.3GB 失败。改为 fit_deform 委托独立进程 dump_deform 导出
  (spawn, 新进程设备干净), 输出 deform_field_phase{p:02}.nii.gz/.npy 于
  out/ 根目录, 与 AGENTS.md 约定一致。已验证 exit=0 8 相位完整导出。

## 后续待做
- 可选: 低阶刚性分量锚定 (线性项), 或 λ 细分 (5e-4 / 3e-3)。
- Slicer 目检 a001_tv 形变场底部结构 (膈肌? 噪声?)。
