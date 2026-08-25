# AGENTS.md

## 多卡并行启动训练(wgpu / burn)

多卡机器(4× RTX 3090, NVIDIA)。后端是 fork 版 wgpu(ArthurBrussee/js-interop),
**`CUBECL_WGPU_DEVICE` 会被静默忽略,所有卡索引都落到卡 0**。

正确的选卡环境变量是:

```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)' <cmd>
```

- `-u DISPLAY`:去掉 SSH -X 的 X11 display,避免 GL 后端枚举干扰/卡顿。
- `DiscreteGpu(N)`:选第 N 张独立显卡(N = nvidia-smi 的 index,
  `DiscreteGpu(0)`=卡0, `(1)`=卡1, ...)。
- 例: `env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(2)' ./target/release/fit_deform ...`

### 进程持久化

后台训练要用 `systemd-run --user --scope` 起(独立 cgroup,不会被
opencode/bash 工具的会话清理连带杀掉;`setsid nohup & disown` 在工具
abort/timeout 时仍会被杀):

```bash
systemd-run --user --scope -p CPUWeight=100 -p MemoryMax=12G \
  -- env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(0)' \
  ./target/release/fit_deform <dcm> --iters=10000 --out=target/... \
  > target/xxx_run.log 2>&1
```

验证落卡:`nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`
(uuid → 卡 index:`nvidia-smi --query-gpu=index,uuid --format=csv,noheader`)。

### 4 卡并行实验惯例(2026-08-20)

- 卡 0: hexplane 10k 基线

### 实验组织规范 (2026-08-25 起, 强制)

- **输出目录**: `experiments/output/<exp_name>/<config_name>/`
  (不要用 `target/`, 会被 cargo clean 清掉)。
- **日志**: `experiments/output/<exp_name>/<config_name>.log`。
- **每个实验前必须先 git 提交** (提交信息以 `exp/` 开头), 并记录:
  - 日期时间 / 目的 / commit hash / 运行命令
  - 写入 `experiments/output/<exp_name>/readme.md`。
- 形变场导出: `deform_final.bin` + `deform_field_phase{p:02}.nii.gz`
  (5D `[x,y,z,1,3]` 位移场, affine 随 nii 保存; 参考 ASOCA dvf 格式)。
- 实验结束后，将前一个实验的结果（包括实验目的，运行命令，实验结果，结果分
  析，后续待做）记录在 experiments/ 中，并使用 amend 提交到此前提交过的实
  验 commit 中

## fit_deform 训练要点

- HexPlane 是默认 deform 后端(`--deform-backend=hashgrid` 切回),融合 cubecl
  内核 ~10x 加速。
- `predict_scaling` 默认关(保质量形变: 不预测 d_scaling, 积分吸收守恒)。
- 减缓 splat 增长: 默认 5k 初始点 / growth_frac 0.25 / percent_dense 0.0003 /
  cull_density 5e-4 / max_splats 300k / **fixed-grad-thr 5e-6** (~40k splats)。

## 评估口径 (2026-08-21 起)

- **LPIPS 是主要指标**(最符合人眼观感; PSNR/SSIM 在边缘处饱和失真,
  SSIM 早期就 0.999+ 不动)。对比实验看 LPIPS 为主, PSNR/SSIM 为辅。
- 边缘清晰度辅助指标 (`tools/edge_blur_analysis.py`): `blur_ratio` =
  pred/GT 梯度幅度比 (<1 越糊), `edge/bg_l1` = 边缘像素误差 vs 背景。
- 数据诊断: `tools/roi_metric_dist.py`(ROI 分布), `tools/gt_gradmap.py`
  (GT 梯度/噪声), `tools/data_consistency.py`(帧统计/相位调制), 
  `tools/edge_blur_map.py`(空间模糊热图)。
