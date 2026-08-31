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

use burn_wgpu::{
    RuntimeOptions, WgpuDevice,
    graphics::{AutoGraphicsApi, Dx12, GraphicsApi, Metal, OpenGl, Vulkan},
};

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

/// 图形后端候选 (按序尝试; 失败自动降级)。
#[derive(Clone)]
enum ApiKind {
    Dx12,
    Vulkan,
    OpenGl,
    Metal,
    Auto,
}

impl ApiKind {
    fn name(&self) -> &'static str {
        match self {
            Self::Dx12 => "dx12",
            Self::Vulkan => "vulkan",
            Self::OpenGl => "opengl",
            Self::Metal => "metal",
            Self::Auto => "auto",
        }
    }
    async fn init(&self) -> WgpuDevice {
        match self {
            Self::Dx12 => init::<Dx12>().await,
            Self::Vulkan => init::<Vulkan>().await,
            Self::OpenGl => init::<OpenGl>().await,
            Self::Metal => init::<Metal>().await,
            Self::Auto => init::<AutoGraphicsApi>().await,
        }
    }
}

/// 初始化 burn wgpu 后端 (默认设备; 多卡机器请用
/// `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(N)'` 环境变量选卡)。
///
/// 图形 API 按序尝试、失败自动降级:
/// - Windows: `dx12 → vulkan` (DX12 是原生后端; 无 DX12 设备时自动降级
///   Vulkan — 即使只有软件驱动 (lavapipe) 也能跑; GL 后端 cubecl 支持差,
///   不默认尝试, 需显式 BRUSH_FIT_GRAPHICS_API=opengl);
/// - macOS: `metal`;
/// - 其他 (Linux): `vulkan → opengl`;
/// - `BRUSH_FIT_GRAPHICS_API=dx12|vulkan|opengl|metal|auto` 或逗号分隔列表
///   (如 `dx12,vulkan`) 可显式指定/限定候选。
pub async fn burn_init_setup() -> WgpuDevice {
    let candidates: Vec<ApiKind> = match std::env::var("BRUSH_FIT_GRAPHICS_API") {
        Ok(v) => v
            .split(',')
            .map(|s| match s.trim().to_ascii_lowercase().as_str() {
                "dx12" | "d3d12" => ApiKind::Dx12,
                "vulkan" => ApiKind::Vulkan,
                "opengl" | "gl" => ApiKind::OpenGl,
                "metal" => ApiKind::Metal,
                "auto" => ApiKind::Auto,
                other => {
                    log::warn!("未知 BRUSH_FIT_GRAPHICS_API 值 '{other}', 忽略");
                    ApiKind::Auto
                }
            })
            .collect(),
        Err(_) => {
            if cfg!(target_os = "windows") {
                vec![ApiKind::Dx12, ApiKind::Vulkan]
            } else if cfg!(target_os = "macos") {
                vec![ApiKind::Metal]
            } else {
                vec![ApiKind::Vulkan, ApiKind::OpenGl]
            }
        }
    };

    let mut last_err = None;
    for api in &candidates {
        // cubecl-wgpu 在适配器枚举失败时直接 panic (expect), async 上下文
        // 无法 catch (嵌套 runtime 会 panic "Cannot start a runtime from
        // within a runtime") — 每次尝试放到独立线程 (线程内建 current_thread
        // runtime), join 捕获 panic, 失败继续下一个后端。
        let handle = {
            let api = api.clone();
            std::thread::spawn(move || {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("temporary runtime")
                    .block_on(api.init())
            })
        };
        match handle.join() {
            Ok(dev) => {
                log::info!("wgpu 后端初始化成功: {}", api.name());
                return dev;
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| e.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_owned());
                log::warn!("wgpu 后端 {} 初始化失败: {msg}", api.name());
                last_err = Some(format!("{}: {msg}", api.name()));
            }
        }
    }
    panic!(
        "所有 wgpu 图形后端初始化失败 ({}) — 检查 GPU 驱动, 或设置 \
         BRUSH_FIT_GRAPHICS_API=dx12,vulkan,opengl 指定后端",
        last_err.unwrap_or_else(|| "无候选".to_owned())
    );
}

async fn init<G: GraphicsApi>() -> WgpuDevice {
    burn_wgpu::init_setup_async::<G>(&WgpuDevice::DefaultDevice, burn_options()).await;
    WgpuDevice::DefaultDevice
}
