# rxa-static — RXA_chest 纯静态重建 (fit_static 改写版)

## 2026-08-27
- commit: 2fbb271 (fit_static 与 fit_deform 逻辑一致, 删形变场)
- 目的: 验证静态数据上 GS 重建 → 点云导出 volume 是否正常
  (隔离"杂乱点云"是动态/形变问题还是体素化问题)
- 配置: 同 fdksplat_trans 风格 (cyl_r15 + 20k + scene 264 auto)

## 结果 (20k)
| 指标 | 值 |
|---|---|
| PSNR | 36.42 |
| SSIM | 0.968 |
| LPIPS | 0.3819 |
| splats | 18.9k |

## 体积检查 (gs2volume --no-fdk, 326x326x232, 0.927mm)
- 非零 96.7%, 均值(非零) μ=0.0022 (水级), max 0.406 (骨/碘级)
- **相邻切片相关 0.981** — 结构高度连贯 (对比 dsa_cyl15 动态体积 0.86)
- 峰值在中心区域 (x=-7, y=36, z=59) — 解剖合理位置
- 切片预览: `rxa_static/vol/slices.png`

## 结论
- **静态重建的 GS 体积是连贯结构, 不是杂乱点云**。
- 之前的"杂乱点云"问题与动态 (形变) 相关 — 静态数据验证通过,
  说明体素化/导出链路本身正确; 动态情况需用 deform_final.bin 网络权重
  (--ckpt) 且注意相位匹配。

## density_reset
```bash
cargo run -p brush-process --bin fit_static images/RXA_chest.dcm \
    --points=50000 \
    --refine-every=400 \
    --eval-split-every=5 \
    --eval-views=8 \
    --eval-every=1000 \
    --cull-density=0.001 \
    --density-reset=3000 \
    --out=experiments/output/2026-08-27_rxa-static/density_reset/ \
    | tee experiments/output/2026-08-27_rxa-static/density_reset.log 2>&1
```

结果较好， 
[03:35:55] iter 10000 loss=  0.1399 psnr= 34.83 ssim=0.967 lpips=0.3906 visible=23779 splats=23779 (eval 8 views) | grads mean=1.0e-5 rot=1.1e-4 scale=1.0e-4 density=3.4e-5 | pos step≈2.01e-11mm
但不如不进行reset:
[02:59:53] iter 10000 loss=  0.1444 psnr= 34.67 ssim=0.968 lpips=0.3614 visible=46733 splats=46733 (eval 8 views) | grads mean=5.7e-6 rot=5.7e-5 scale=8.9e-5 density=2.9e-5 | pos step≈1.15e-11mm

## 2026-08-28 — density_reset 40k 全量 (新代码: 合并 a446e0 + 布局/激活统一)
- commit: 39755d05 (exp: 合并 a446e0 — gs2volume 独立尺寸 + DRR 对比 + 步进区间修复)
- 目的: 40k 步全量训练 (之前只跑到 10k), 结束后 gs2volume --ref-dcm 导出
  独立尺寸体积 (XY=1.5×等中心宽=378mm, Z=图高=185mm) + --compare-drr
  对比 DRR vs GS 直接渲染
- 注意: 04:21 曾启动旧代码 debug 训练 (PLY raw 域, 与新 gs2volume 不兼容),
  已杀掉改用 HEAD release 二进制重启 (loss 可忽略)
- 运行命令:
```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(0)' \
  ./target/release/fit_static images/RXA_chest.dcm \
  --points=50000 \
  --refine-every=400 \
  --eval-split-every=5 \
  --eval-views=8 \
  --eval-every=1000 \
  --cull-density=0.001 \
  --iters=40000 \
  --out=experiments/output/2026-08-27_rxa-static/density_reset/
```

## 2026-08-28 二轮 — voxelizer 根因定位与修复 + gs2volume 对比验证
- commit: 39755d05 (基础) + 本轮修复
- 目的: 排查 gs2volume 导出体积的 DRR 与 GS 直接渲染的显著差异

### 根因 (voxelizer bug): preprocess 的 g2c 未压实
- `preprocess_voxel_kernel` 写 `global_from_presort_gid[idx] = idx` (原始索引),
  而 xray 的 project_forward 是 `g2c[atomic_write_id] = global_gid` (压实)。
- 449 个不可见 splat (z 越界 ±93mm) 在 `[0..num_visible)` 切片内留空洞 (0),
  project_visible 读到 global=0 → 把 splat #0 的 lanes 复制进 449 个槽位 →
  render 将 splat #0 累加 ~440 次 → 体积在 (-50.5,-100.5,-26.5) 出现
  1.366 mm⁻¹ 异常团 (真场仅 0.004)。同时索引 ≥ num_visible 的可见 splat
  被切片丢弃 → 贡献缺失。
- 测试未暴露原因: 小测试 (golden/autodiff/consistency) 全部 splat 可见, 无空洞。
- 修复: 对齐 xray 的原子压实写法 (1 行), 已回补单测场景 (见后)。

### 验证 (修复后, 1mm 体素)
- 体积 max 1.3661 → **0.0371** (异常团消失)
- 全量 DRR vs GS: 总缺口 ~14% (亮区损失重) → **根因 = 体积覆盖**: 体积
  378x378x186mm 只含 84.4% splat, 2862 个 (15.6%) 在体积外 (z 超 ±93 或
  r 超 189mm), 其投影缺失。
- **inside-only 对照** (仅体积内 15445 splat): DRR vs GS 总和比率
  **0.988~1.001** (0.2~1.2%), 残差为体素分辨率/截断级 → voxelizer 数学正确。

### 其他修复/改进 (本轮)
- gs2volume --compare-drr: 三子图改为逐行交错 (之前垂直堆叠压扁),
  行序自底向上对齐训练 eval 栈 (上下颠倒修复); 另导出 raw proj NRRD。
- --compare-drr GT 的 gamma 与训练对齐 (dicom_gamma_target=0.5), 不再偏黑。
- fit_static 导出 `canonical_final_transforms.bin` + `canonical_final_raw.bin`
  (raw 域 f32, 无 PLY 激活域往返); gs2volume 支持 `--bin=<prefix>` 输入
  (实测与 PLY 路径结果逐位一致 → PLY 往返无损)。
- gs2volume 新增 --dump-isects 诊断 (中间数组落盘, 用于本次定位)。
- 待办: 若需体积覆盖全部点云, 增大 --extent-z/--extent-xy (或按点云范围
  自动); 0.5mm 体素对照因 212M voxel 过重未跑, 如需高保真 DRR 可分区跑。

## 2026-08-28 三轮 — compare 与训练 gt_pred 的"GS 差异"归因
- 疑问: gt_pred_40000.nrrd 右侧 GS 渲染 vs compare 中间 GS 渲染显著差异,
  是否用了不同训练阶段的 PLY?
- 结论: **不是 PLY/阶段差异** (PLY 即最终 canonical)。
  主因 = **视图帧不同**: 训练 eval 用 held-out 帧 (38 视图 split every 5,
  采样 8 个), compare 原本用 train 帧 i*187/8; 仅 0/70/140 三帧重合。
  已给 gs2volume 加 `--eval-split-every=N` (与训练同 held-out 选择),
  对齐后 GT 逐像素一致 (max diff = 0.0)。
- 残余同帧差异 (~10% proj, corr 0.85~0.90, 中位比值 1.07~1.11): 候选
  (a) 饱和 opacity 的 PLY 往返损失 (raw>17 → logit clamp 13.8, 方向相反),
  (b) iter40000 refine 前/后 27 splat 状态差, (c) 渲染路径微差。
  待 5000点/10000iters 验证训练 (带 .bin 导出) 彻底分离。
