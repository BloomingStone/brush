# 260829-1600 cov-transpose-fix — GS 投影 vs volume+DRR 不一致根因: 协方差变换转置 bug

## 目的
修复 gs2volume --compare-drr 中 **GS 直接投影 (A 中间图) vs 体积 DRR 投影 (A 右侧图)**
不一致。原实验 (2026-08-28) 观察到: alpha=0° 帧一致, 其他角度 mean|drr-gs| 0.06~0.13
(gs_sum 大 6-12%)。

## 调查过程
1. **网格覆盖不足** (主因之一): 原 volume 网格 378×378×186mm (z 半高 93mm ≈ 图像半高),
   splat 中心达 ±236mm, 17.6% 密度质量在网格外被 voxelizer 截断 (其中投影在图像内的
   1.6%)。→ 修复: 网格自适应 splat 数据 (per-axis p99.9(|mean|+3σ), cap 400mm),
   view 19 差异 0.0295→0.0078, gs_sum/drr_sum→1.000。
   - 顺带修复 brush-voxel render 1D dispatch 超 65535 限制 (改 3D grid)。
2. **协方差变换转置 bug** (主因二): brush-xray 的 `cone_geometry` 用
   `cov3 = (R_view·J)ᵀ·Vrk·(R_view·J)` = `Jᵀ·(R_viewᵀ·Vrk·R_view)·J`, 而正确是
   `Jᵀ·(R_view·Vrk·R_viewᵀ)·J` (Σ_cam = R_view·Vrk·R_viewᵀ, 相机系协方差)。
   正位 (AP) 时 R_view ≈ 置换+对角 (对称), 两个公式近似相等 → view 19 一致;
   斜角时 R_view 大旋转 → 屏幕椭圆严重失真 (实测 σ 偏大 6.5 倍) → GS 投影系统性偏差。
   - 验证: 修正公式 vs 逐 splat 精确 ray 数值积分, 屏幕 σ 完全吻合 (10.7/16.6px)。
   - 修复: `cov3 = vrk.congruence(w).transpose_congruence(j)` + backward 链式同步
     (v_cov3D: X=mᵀ·G·m; dl_dj: S_cam·J·D')。
   - 归因过程排除: DRR 步长 (0.25 vs 0.5mm), 体素分辨率 (0.5mm), GS tile 3σ→6σ,
     FOV 雅可比 clamp (1e9), 大 σ splat 剔除 (贡献 40% 差异, 为转置 bug 的表现),
     trilinear 平滑 (DDA 阶梯场结果一致)。

## 运行命令
```bash
# 修复后 compare (自适应网格 742³)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/gs2volume \
  --bin=experiments/output/2026-08-28_rxa-static-verify/5000p-10k/eval/bin/canonical_final \
  --ref-dcm=images/RXA_chest.dcm --eval-split-every=5 --voxel-mm=1.0 --compare-drr \
  --out=/tmp/opencode/gs2vol_fixed

# backward 验证 (200 步短训练, GPU 1)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/fit_static images/RXA_chest.dcm --points=2000 --refine-every=100 \
  --eval-split-every=5 --eval-views=2 --eval-every=100 --iters=200 \
  --out=/tmp/opencode/fit_static_verify
```

## 实验结果 (修复后, 全部 8 视图)
| view | alpha | mean|drr-gs| 修复前 | 修复后 | gs_sum/drr_sum |
|---|---|---|---|---|
| 0 | -99° | 0.1233 | **0.0088** | 0.997 |
| 4 | -78° | 0.1330 | **0.0064** | 1.003 |
| 9 | -51° | 0.1053 | **0.0042** | 1.003 |
| 14 | -24° | 0.0666 | **0.0030** | 1.003 |
| 19 | +3° | 0.0295 | **0.0015** | 1.002 |
| 23 | +25° | 0.0632 | **0.0037** | 1.003 |
| 28 | +52° | 0.0954 | **0.0058** | 1.005 |
| 33 | +79° | 0.1238 | **0.0088** | 1.009 |

- 全部视图 mean|drr-gs| ≤ 0.009 (修复前 0.03~0.13), max ≤ 0.085。
- 短训练验证: loss 0.40→0.34, PSNR 23.2→24.5, 梯度正常, 无 NaN。
- 200 步内 refine 尚未触发 split (grad_thr=-1), 行为与修复前同阶段一致。

## 结果分析
1. **根因**: 屏幕协方差变换转置方向错误 (R_viewᵀ·Vrk·R_view vs R_view·Vrk·R_viewᵀ),
   移植自 R2 的 `Mᵀ·Vrk·M` 约定在 brush 的 `m = R_view·J` (列式 J) 组合下不对称。
2. 该 bug 影响**所有** GS x-ray 渲染 (训练 eval、导出、fit_deform), 斜角视图的
   屏幕椭圆/投影中心分布失真。修复后渲染更接近真实透视投影。
3. **影响**: 已训练模型 (含 2026-08-28 的 10k 实验与之前所有 fit_deform 结果) 的
   渲染数值会变化 (修复前用错误协方差拟合)。需重新训练/评估才能对齐新渲染路径。

## 后续待做
- 用修复后的渲染路径重跑 5000p-10k 或等价训练, 重新评估 PSNR/LPIPS (预期斜角
  eval 更真实, 指标可能变化)。
- fit_deform 系列实验受相同 bug 影响, 按需重训。
- 检查 brush-render (普通 3DGS) 的 cov2d 变换是否也存在同型转置问题 (不在本次范围)。

## 追加 (2026-08-30): backward 链式修复 + 双重 refine bug + 完整验证

### backward 链式 bug (端到端数值验证定位)
修复 forward 后训练不稳定 (rot 梯度爆炸 6.7, loss 反弹) → 逐段数值验证
(单 splat 单像素, 扰动 scale/quat/mean/vrk 与解析链对比) 发现两处 backward bug:

1. **v_cov3D 约定错误**: dL/dVrk 是"独立元素梯度", 需**双半和**展开:
   - dS[c][d] (c≤d) = Σ_{i≤j} v[i][j]·J[c][i]·J[d][j] (对角) / 含交叉项 (非对角)
   - dVrk[a][b] = Σ_{c≤d} dS[c][d]·w[c][a]·w[d][a] (对角) / 交叉 (非对角)
   - 不能用纯矩阵 congruence (对角项权重不同)。数值验证 1e-11。
2. **compute_cov3d_bwd 的 quat 梯度公式错** (R2 移植, 非各向同性时错):
   - 正确: dL/dR = dL/dMᵀ·S (行转置×scale); dq_k = tr(dL/dRᵀ·∂R/∂q_k)
   - 标准四元数导数展开。数值验证 3e-10 (含 dnormvdv4)。

端到端验证: scale err 3.6e-11, quat err 3.0e-10, mean err ~1% (像素 VJP 符号核对)。

### 双重 refine bug (导致 stats 覆盖 + 状态错乱)
`maybe_refine` 里 `refiner.refine()` 被调用两次 (重复行, commit 0306f043b 引入):
第一次实际执行 split (diag: split=393), 第二次 (accumulator 已重置) 返回
split=0 覆盖 stats → 日志恒 "split 0" + optimizer 状态同步基于第二次 update。
修复: 删除重复行 → split 正常 (iter 800: split 2261)。

### 完整验证 (新训练 5000p/5000 步, GPU1, 修复后全部代码)
```
fit_static images/RXA_chest.dcm --points=5000 --refine-every=400 --eval-split-every=5 \
  --eval-views=8 --eval-every=1000 --cull-density=0.001 --iters=5000
```
- iter 5000: loss 0.2243, PSNR 30.04, SSIM 0.957, LPIPS 0.4965, 15140 splats
- splats 增长正常 (5000→15140, iter 800 起 split 2261/次)
- 梯度稳定 (rot ~3e-3 不爆炸)

gs2volume --compare-drr (自适应网格 800³):
- **全部 8 视图 mean|drr-gs| ≤ 0.011 (大部分 0.003)**, gs_sum/drr_sum 差 <0.5%
- view 0: 0.123 → 0.003; view 19: 0.030 → 0.004
- **eval pred vs gs2volume GS (bin 重渲): max diff 1.8e-7** (逐位一致)

产物: v7-verify/ (compare_*.nrrd + fit_metrics.csv)。

## 追加 (2026-08-30): 完整训练产物存档
- `v7-verify/train_output/`: fit_static v7 完整输出 (5000p/5000 步, RXA_chest.dcm, GPU1)
  - `eval/bin/canonical_final_{transforms,raw}.bin`: 最终 splats 原始参数 (供 gs2volume 复用)
  - `eval/ply/canonical_final.ply`: 点云 (激活域)
  - `eval/nrrd/gt_pred_{00000..05000}.nrrd` + `gt_pred_10000_FWD.nrrd`: 训练 eval 投影对比 + 导出时 FWD 重渲
  - `metrics.csv`: 训练曲线 (loss/PSNR/SSIM/LPIPS)
