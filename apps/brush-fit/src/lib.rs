//! brush-fit: 无头 X-ray 重建 (lib + CLI + C FFI)。
//!
//! 重新实现 `fit_static` / `fit_deform` (FDK 相关剔除): 输入 DICOM 路径 +
//! [`config::FitConfig`], 输出必须的 phase=0 volume (.nii.gz), 可选 PLY 点云
//! / deform 网络权重 / 网格形变场。
//!
//! - Rust: [`run_static`] / [`run_deform`]
//! - C 动态库: [`ffi::brush_fit_run`] (config JSON → block_on)
//! - CLI: `brush-fit static|deform <dcm> [options]`

pub mod config;
pub mod data;
pub mod export;
pub mod train;

mod ffi;

pub use config::{FitConfig, FitMode};
pub use train::{FitOutcome, FitProgress, ProgressFn, run_deform, run_static};

use burn_wgpu::{RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi};

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

/// 初始化 burn wgpu 后端 (默认设备; 多卡机器请用
/// `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'` 环境变量选卡)。
pub async fn burn_init_setup() -> WgpuDevice {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    WgpuDevice::DefaultDevice
}
