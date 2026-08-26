# init-cylinder — 初始点云圆柱采样 R 扫描 (验证点漂移猜想)

## 2026-08-26
- commit: dc431dd
- 数据: images/pig-data-cor-new-phase.dcm
- 猜想: 重建时有些点总想往外漂移, 可能在试图重建初始点之外的结构。
  把球形随机点云改为**圆柱形** (匹配锥束 FOV 几何), R 扫描验证。
- 几何: R0 = half_w = 118.6mm (W/2 世界); 高度 = 可视上界
  H = 2·half_h·(1+R/SOD), half_h=84.7, SOD=760。
- **密度恒定**: R0 时 5000 点, 其他 R 点数 = 5000·(R/R0)²·H(R)/H(R0)。
- **关闭 FOV 过滤** (--no-fov-filter): 旋转中视野外点会重新入视野。
- 已知局限: R>1.29R0 (153mm) 超出 deform 网格, 运动外推; canonical 均值
  仍直接训练 (点位置可到那里)。
- 对照: baseline ball (0.2147, coronary-new-phase/default 复用)。

## 配置 (20k, 同 baseline 命令 + init flags)
① cyl_r10  --init-shape=cylinder --init-radius-scale=1.0  --no-fov-filter  (N=5000)
② cyl_r125 --init-shape=cylinder --init-radius-scale=1.25 --no-fov-filter  (N=8090)
③ cyl_r15  --init-shape=cylinder --init-radius-scale=1.5  --no-fov-filter  (N=12020)
④ cyl_r175 --init-shape=cylinder --init-radius-scale=1.75 --no-fov-filter  (N=16863)

## 结果 (2026-08-26, 20k iters)
| 配置 | R(mm) | N | LPIPS | PSNR | SSIM | splats | 漂移>R比例 |
|---|---|---|---|---|---|---|---|
| baseline ball (复用) | 153 | 5000 | 0.2147 | 44.14 | 0.9910 | 38k | — |
| cyl_r10 | 118.6 | 5000 | 0.2171 | 43.62 | 0.9910 | 40k | 24.8% |
| cyl_r125 | 148.3 | 8076 | 0.2020 | 44.51 | 0.9920 | 47k | 14.0% |
| **cyl_r15** | 177.9 | 12009 | **0.2007** | 43.71 | 0.9910 | 52k | 10.4% |
| cyl_r175 | 207.6 | 16863 | 0.2020 | 44.24 | 0.9920 | 52k | 7.6% |

## 分析
- **所有圆柱 R≥1.25R0 显著优于 baseline ball** (0.2007-0.2020 vs 0.2147),
  甜点 ~1.5R0, 1.75R0 平台。
- **漂移猜想部分验证**: 最终点确实漂出初始区 (r10 24.8% 超 R), R 越大漂出比例越低
  (24.8%→7.6%) → 更大/更贴合初始区减少"向外漂移"。
- **混淆因素**: baseline ball 用 FOV 过滤, 圆柱全用 --no-fov-filter。
  圆柱优势可能部分来自关闭过滤 (保留旋转中可入视野的点)。
  → 需控制: ball + --no-fov-filter 跑一个对照隔离形状 vs 过滤。
- R>1.29R0 时超 deform 网格(153)点比例高 (r15 28.5%, r175 37.4%),
  但 LPIPS 未崩 → canonical 均值位置直接训练补偿了形变外推。

## 后续待做
- 控制实验: ball + --no-fov-filter (隔离形状 vs 过滤)。
- 若过滤是主因, 则 baseline 应加 --no-fov-filter 重跑作为新对照。
