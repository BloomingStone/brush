# brush-fit — 无头 X-ray 重建 (lib + CLI + C FFI)

重新实现 `fit_static` / `fit_deform` 的无头重建入口 (FDK 相关剔除)。输入
DICOM 路径 + 配置, 输出必须的 **phase=0 volume (.nii.gz)**, 可选 PLY 点云 /
deform 网络权重 / 网格形变场 / 原始参数 .bin。

## 构建

```bash
cargo build --release -p brush-fit
# 产物: target/release/brush-fit (CLI), brush-fit-dump (deform 场导出子进程),
#        libbrush_fit.so (C 动态库)
```

多卡机器选卡 (AGENTS.md, `CUBECL_WGPU_DEVICE` 会被忽略):

```bash
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)' ./target/release/brush-fit ...
```

## CLI 用法

```bash
# 静态重建
./target/release/brush-fit static images/RXA_chest.dcm \
  --points=5000 --iters=5000 --refine-every=400 \
  --eval-split-every=5 --eval-views=8 --eval-every=1000 \
  --fixed-grad-thr=5e-6 --cull-density=0.001 --save-ply --save-bin \
  --out=target/fit_static

# 动态重建 (deform 网络 + 心动相位)
./target/release/brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm \
  --points=5000 --iters=10000 --refine-every=400 --eval-split-every=5 \
  --eval-views=8 --eval-every=1000 --out=target/fit_deform
```

`--help` 查看全部参数 (与 fit_static/fit_deform CLI 参数一一对应; FDK 相关
参数已剔除)。模式敏感字段 (init_shape / init_radius_scale / init_density /
eval_every / percent_dense 等) 缺省按模式默认, 与 fit_*.rs 一致。

## 输出

```
<out>/
├── volume_phase00.nii.gz          # 必须: phase=0 volume (3D float32, sform
│                                  #   affine; deform 模式先过 phase=0 形变场)
├── metrics.csv                    # 训练指标 (loss/PSNR/SSIM/LPIPS/梯度)
├── eval/nrrd/gt_pred_*.nrrd       # eval GT|pred 拼接栈 (--no-save-eval 关闭)
├── canonical_final.ply            # --save-ply: 点云 (激活域)
├── canonical_final_{transforms,raw}.bin  # --save-bin: 原始参数 (gs2volume --bin= 复用)
├── deform_final.bin               # --save-deform (deform 模式, 默认开): 网络权重
└── deform_field_phase{p:02}.{npy,nii.gz} # 8 相位网格形变场 (行主序 [nx,ny,nz,3]
                                         #   + 5D nii.gz 同 ASOCA dvf)
```

## Rust lib

```rust
use brush_fit::{FitConfig, FitMode, run_static, run_deform};

let cfg = FitConfig::default();
// ... 修改字段 (serde snake_case) ...
let outcome = run_static(cfg).await?;   // 或 run_deform
println!("volume: {}", outcome.volume_phase0.display());
```

`FitConfig` 全部字段 (86 个) serde 可序列化; `FitConfig::default()` 给出与
fit_static/fit_deform CLI 一致的默认值。带进度回调版本:
`run_static_with_progress(cfg, Box::new(|p| ...))` (`FitProgress::Step/Done`)。

## C FFI (libbrush_fit.so)

```c
// config 为 FitConfig 的 JSON (snake_case), 同步阻塞, 返回 0=成功
int brush_fit_run(const char* config_json, char* errbuf, size_t errbuf_len);
// 可选: 注册进度回调 extern "C" void cb(uint32_t iter, uint32_t total,
//       float loss, void* user_data) — 训练开始前调用
void brush_fit_set_progress_cb(brush_fit_progress_cb cb, void* user_data);
// 生成示例 config JSON (写入 errbuf), 返回 0
int brush_fit_example_config(char* errbuf, size_t errbuf_len);
```

Python 示例:

```python
import ctypes, json
lib = ctypes.CDLL('./target/release/libbrush_fit.so')
lib.brush_fit_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
cfg = dict(dcm='images/RXA_chest.dcm', out='/tmp/fit', mode='static',
           points=5000, iters=5000, save_ply=True)
err = ctypes.create_string_buffer(4096)
rc = lib.brush_fit_run(json.dumps(cfg).encode(), err, 4096)
```

## 注意事项

- **deform 网格场导出走独立进程 `brush-fit-dump`**: 训练后 GPU 内存池状态
  不稳定 (整网格 matmul autotune OOM / memory_manage 断言), 独立进程干净
  设备导出稳定 (与 fit_deform spawn dump_deform 同行为, 日志中 DSD 线程
  断言 panic 噪音无害)。子进程路径: 当前 exe 同目录; 可
  `BRUSH_FIT_DUMP_BIN` 覆盖; 仅用 lib (嵌入) 时需确保该二进制可寻。
- 不依赖 brush-process / brush-app / brush-cli / brush-c (训练核心在 crates/
  底层, 训练循环移植自 fit_static / fit_deform 权威实现)。
- 评估口径: LPIPS 为主 (PSNR/SSIM 边缘饱和), 见 AGENTS.md。
