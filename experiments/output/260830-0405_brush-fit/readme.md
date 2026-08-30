# 260830-0405 brush-fit — 无头重建 lib+CLI+FFI (重构 fit_static/fit_deform)

## 目的
在 apps/ 下新增 `brush-fit` crate (lib + bins 薄包装, crate-type=[rlib,cdylib]),
重新实现 fit_static / fit_deform (剔除 FDK): 输入 DICOM 路径 + config
(struct/CLI/JSON), 输出必须的 phase=0 volume (.nii.gz) + 可选 PLY 点云 /
deform 网络权重 / 网格形变场。

## 复用评估结论
- brush-app (viewer) / brush-cli (TrainStreamConfig 16 字段子集) / brush-c
  (5 字段 FFI) 均无法直接复用 → 新 crate 直接依赖 crates/ 底层
  (XRayTrainer/XRayRefineConfig/SceneLoader/load_dataset/voxelize_forward),
  训练循环照 fit_*.rs 权威实现移植。
- 关键教训: xray_stream.rs 是 fit_*.rs 的简化分叉 (Ball init / refine 3 字段),
  继续增强 = 第三套训练入口 → 新 crate 不做第三套。

## 结构
```
apps/brush-fit/
├── Cargo.toml      # rlib + cdylib + [[bin]] brush-fit / brush-fit-dump
└── src/
    ├── config.rs   # FitConfig (serde snake_case, 86 字段, 模式敏感字段 Option→模式默认)
    ├── data.rs     # dcm 加载 + 相机几何 + scene_extent + init region
    ├── train.rs    # run_static/run_deform: 训练循环 + eval/CSV + 进度回调
    ├── export.rs   # volume nii.gz (自适应网格+deform phase0) / ply / bin / deform ckpt+spawn
    ├── ffi.rs      # brush_fit_run(config_json, errbuf) / brush_fit_set_progress_cb
    └── bins/       # main.rs (clap static|deform 子命令), dump_deform.rs
```

## 关键决策
1. **deform 网格场导出 = spawn 独立进程 brush-fit-dump**: 训练后内存池状态
   不稳定 (整网格 matmul autotune OOM / memory_manage 断言), 进程内导出
   失败; 独立进程干净设备成功 (与 fit_deform spawn dump_deform 行为一致,
   含 DSD 线程断言 panic 噪音但不影响完成)。BRUSH_FIT_DUMP_BIN 可覆盖路径。
2. **不做 memory_cleanup** (训练后调用会触发 cubecl 池断言)。
3. 分批 forward (64³) 反而失败 (多次 autotune), 整网格一次 forward 成功。
4. serde snake_case (JSON 惯例); CLI 参数 kebab-case 不变。
5. save_eval (默认 true) / save_ply (默认 false) / save_deform (默认 true,
   deform 模式) / save_bin (默认 false)。

## 验证 (全部通过)
| 项 | 命令 | 结果 |
|---|---|---|
| 静态冒烟 | brush-fit static images/RXA_chest.dcm --points=500 --iters=500 | volume_phase00.nii.gz 286³×212 (0~0.0089), ply, bin ✓ |
| 动态冒烟 | brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm --points=500 --iters=500 | deform_final.bin + 8 相位 npy/nii.gz + volume (deform phase0, max|d|=6mm) ✓ |
| 移植一致性 | fit_static vs brush-fit 同配置 500 步 | init psnr 14.45 完全一致; iter 500: 24.13 vs 24.29 ✓ |
| DRR vs GS | gs2volume --compare-drr (brush-fit bin) | 全视图 mean|drr-gs| ≤ 0.004 ✓ |
| C FFI | python ctypes brush_fit_run(JSON) | rc=0, 产物齐全, 配置全生效 ✓ |

## 产物
- 冒烟产物: /tmp/opencode/brushfit_smoke{1,7} (静态/动态), ffi_smoke2
- 后续待做: 长训练质量验证 (5000p/10k 步), deform 场质量诊断, brush-c
  风格的完整 C 头文件/示例。
