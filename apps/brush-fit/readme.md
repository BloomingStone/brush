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

### Windows (DLL) 构建

```bash
# 原生构建 (MSVC 工具链, rustup 默认 host 即 msvc; 无需交叉工具链)
cargo build --release -p brush-fit
# 产物 (target\release\):
#   brush_fit.dll + brush_fit.dll.lib   — C ABI 动态库 + MSVC 导入库
#   brush-fit-cli.exe / brush-fit-dump.exe — CLI + deform 场导出子进程
#     (bin 名与 lib 名在 Windows 大小写不敏感文件系统冲突, cargo 自动加 -cli)
#   brush_fit.pdb                       — 调试符号
```

- **C 调用方**: MSVC `cl my_prog.c /I apps/brush-fit/include /link brush_fit.dll.lib`;
  MinGW/GCC `gcc my_prog.c -I apps/brush-fit/include -L target/release -lbrush_fit`
  (自动找 brush_fit.dll; 也可用 dlltool 生成 .a)。头文件已带 `BRUSH_FIT_API`
  (dllimport/dllexport) 宏。
- **GPU 后端**: wgpu 在 Windows 优先 DX12 (Vulkan/GL 可选), 无需 DISPLAY
  处理; 选卡仍用 `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'`。
- **deform 场导出**: `brush-fit-dump.exe` 需与宿主程序同目录 (或 PATH /
  `BRUSH_FIT_DUMP_BIN`), 与 Linux 行为一致。
- 若需交叉编译 (Linux → Windows): 需 MSVC link.exe (cargo-xwin) 或
  mingw-w64; 依赖树大 (burn/wgpu/naga), 强烈建议 Windows 原生构建。

## CLI 用法

```bash
# 静态重建 (默认 no-eval; 加 --eval 开验证集评估)
./target/release/brush-fit static images/RXA_chest.dcm \
  --points=5000 --iters=5000 --refine-every=400 \
  --eval --eval-split-every=5 --eval-views=8 --eval-every=1000 \
  --fixed-grad-thr=5e-6 --cull-density=0.001 --save-ply --save-bin \
  --out=target/fit_static

# 动态重建 10k 参考命令 (deform 网络 + 心动相位, 仅 phase; no-eval ~10min)
env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)' \
  ./target/release/brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm \
  --points=5000 --iters=10000 --refine-every=400 \
  --save-deform --out=experiments/output/260831-1500_brush-fit-deform-10k-v2
```

**eval 默认关闭** (`--eval` 显式开启; 不 eval 时无 metrics.csv / eval 目录,
省 VGG 推理与 readback)。`--help` 查看全部参数 (与 fit_static/fit_deform CLI
参数一一对应; FDK 与 time 相关参数已剔除)。scale 约束默认启用 (ab_cap10_pen01:
`--screen-area-penalty=0.1 --scale-cap-mm=10 --scale-cap-weight=0.5`,
细长条抑制); `--scale-aniso-weight` 各向异性正则默认关。模式敏感字段 (init_shape / init_radius_scale /
init_density / eval_every / percent_dense 等) 缺省按模式默认, 与 fit_*.rs
一致。形变网络仅以心动 phase 为条件 (time 输入已从 CLI/config 移除, 后端
time 硬编码 0)。

## 输出

```
<out>/
├── volume_phase00.nii.gz          # 必须: phase=0 volume (3D float32, sform
│                                  #   affine; deform 模式先过 phase=0 形变场)
│                                  #   网格 = 输入 DCM 图像的等中心尺寸
│                                  #   (XY 全宽 = 图宽, Z 全高 = 图高)
├── metrics.csv                    # 训练指标 (loss/PSNR/SSIM/LPIPS/梯度; 仅 --eval)
├── eval/nrrd/gt_pred_*.nrrd       # eval GT|pred 拼接栈 (仅 --eval; --no-save-eval 关闭)
├── canonical_final.ply            # --save-ply: 点云 (激活域)
├── canonical_final_{transforms,raw}.bin  # --save-bin: 原始参数 (gs2volume --bin= 复用)
├── deform_final.bin               # --save-deform (deform 模式, 默认关): 网络权重
└── deform_field_phase{p:02}.nii.gz # --save-deform: 8 相位网格形变场 (5D
                                    #   [x,y,z,1,3] 同 ASOCA dvf, affine 随 nii)
                                    #   (仅 nii.gz, 不导出 npy)
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

## C ABI (libbrush_fit.so)

构建产物 `target/release/libbrush_fit.so` (cdylib), 头文件
`apps/brush-fit/include/brush_fit.h`。三个导出符号:

```c
// config 为 FitConfig 的 JSON (snake_case, 缺失字段用默认), 同步阻塞, 返回 0=成功
int brush_fit_run(const char* config_json, char* errbuf, size_t errbuf_len);
// 注册进度回调 (训练开始前调用): iter=0,total=0 表示结束
void brush_fit_set_progress_cb(brush_fit_progress_cb cb, void* user_data);
// 生成示例 config JSON (全部字段+默认值, 写入 errbuf), 返回 0
int brush_fit_example_config(char* errbuf, size_t errbuf_len);
```

C 程序编译链接:

```bash
gcc my_prog.c -I apps/brush-fit/include -L target/release -lbrush_fit \
    -Wl,-rpath,$PWD/target/release -o my_prog
```

C 示例:

```c
#include <stdio.h>
#include "brush_fit.h"

static int steps = 0;
static void on_progress(uint32_t iter, uint32_t total, float loss, void* ud) {
    (void)ud; steps++;
    if (iter == 0 && total == 0) printf("[done]\n");
    else printf("step %u/%u loss=%.4f\n", iter, total, loss);
}
int main(void) {
    char err[4096];
    brush_fit_set_progress_cb(on_progress, NULL);
    const char* cfg =
        "{\"dcm\":\"images/RXA_chest.dcm\",\"out\":\"/tmp/fit\","
        "\"mode\":\"static\",\"points\":200,\"iters\":200}";
    int rc = brush_fit_run(cfg, err, sizeof err);
    if (rc) fprintf(stderr, "err: %s\n", err);
    return rc;
}
```

Python 示例 (ctypes):

```python
import ctypes, json
lib = ctypes.CDLL('./target/release/libbrush_fit.so')
lib.brush_fit_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
cfg = dict(dcm='images/RXA_chest.dcm', out='/tmp/fit', mode='static',
           points=5000, iters=5000, save_ply=True)
err = ctypes.create_string_buffer(4096)
rc = lib.brush_fit_run(json.dumps(cfg).encode(), err, 4096)
```

注意:
- 进度回调的 `loss` 在默认 no-eval 下为 NaN (loss 只读回于 eval 步)。
- 需 GPU 环境 (选卡见构建节); 多进程/多线程同时 `brush_fit_run` 不保证安全
  (全局 wgpu 设备 + 一次训练)。

## 注意事项

- **deform 网格场导出走独立进程 `brush-fit-dump`** (仅 nii.gz, 默认不导出;
  `--save-deform` 开启): 训练后 GPU 内存池状态
  不稳定 (整网格 matmul autotune OOM / memory_manage 断言), 独立进程干净
  设备导出稳定 (与 fit_deform spawn dump_deform 同行为, 日志中 DSD 线程
  断言 panic 噪音无害)。子进程路径: 当前 exe 同目录; 可
  `BRUSH_FIT_DUMP_BIN` 覆盖; 仅用 lib (嵌入) 时需确保该二进制可寻。
- 不依赖 brush-process / brush-app / brush-cli / brush-c (训练核心在 crates/
  底层, 训练循环移植自 fit_static / fit_deform 权威实现)。
- 评估口径: LPIPS 为主 (PSNR/SSIM 边缘饱和), 见 AGENTS.md。
