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
- 卡 1: hashgrid 10k 对照
- 卡 2/3: variants(如 `--predict-scaling`、动态 densify 阈值)
- 每跑一次 eval 自动写 `metrics.csv` + `gt_pred_*.nrrd`;实验记录放 `experiments/`。

## fit_deform 训练要点

- HexPlane 是默认 deform 后端(`--deform-backend=hashgrid` 切回),融合 cubecl
  内核 ~10x 加速。
- `predict_scaling` 默认关(保质量形变: 不预测 d_scaling, 积分吸收守恒)。
- 减缓 splat 增长: 默认 15k 初始点 / growth_frac 0.15 / percent_dense 0.0003 /
  cull_density 2e-4 / max_splats 300k。
