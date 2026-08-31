# 260831-1230 backward-jfix — cherry-pick s 双半和 j 索引错位 + quat 梯度修复

## 目的
从 brush-hexplane-1d8eac cherry-pick `4c00a89c` (x-ray backward 链式双 bug),
修复 render-xray 训练质量下降。

## 背景
799e1e66 的 v_cov3D 双半和公式存在 **j 索引错位** (用 j10 代替 j01 等) →
v_cov3d 错误 → quat 梯度被放大 1000 倍 → 训练 rot 梯度异常 (3e-3 vs 修复后
5e-5)、PSNR 下降 (30.04 vs 33)。同时 compute_cov3d_bwd 的 R2 移植 dl_dm 路径
(忽略 R 正交约束) 错误, 改为直接推导:
- v_scale_i = 2·s_i·(Rᵀ·G·R)_ii
- dL/dR = 2·G·R·S², dq_k = Σ dL/dR ⊙ ∂R/∂q_k (标准四元数导数)

## 修改 (cherry-pick d8a02ece)
- crates/brush-xray-bwd/src/kernels/project_bwd.rs: s 公式 (j[a][c]·j[a][d] /
  j[a][c]·j[b][d]+j[b][c]·j[a][d]) + compute_cov3d_bwd 直接推导
- crates/brush-xray-bwd/tests/backward_verify.rs: 新增端到端中心差分验证
  (单各向异性高斯; 修前 quat 差 1000 倍, 修后 11/11 过)
- crates/brush-process/src/bin/gs_sweep.rs: 新增 GS 扫描诊断工具
- tools/ply_scale_anisotropy.py: 保留删除 (fix-voxelizer 已删)
- 适配: backward_verify 的 render_xray 调用去掉 screen_area_penalty 参数
  (该功能不在 fix-voxelizer)

## 验证 (全部通过)
| 项 | 结果 |
|---|---|
| backward_verify 测试 | backward_matches_central_difference ok |
| brush-xray-bwd 全套测试 | 1+1+2+1 全过 (finite_diff / r2_reference / autodiff) |
| 训练 (5000p/5000 步, RXA_chest) | iter 5000: PSNR 32.99 / SSIM 0.962 / LPIPS 0.4124 (修复前 v7: 30.04 / 0.4965, +2.95dB) |
| rot 梯度 | 5.3e-5 (修复前 3.1e-3, 放大 1000 倍现象消失) |
| brush-fit 冒烟 | 300 步静态正常, volume 导出 ✓ |

产物: /tmp/opencode/fit_static_v8 (metrics.csv, eval/nrrd)
