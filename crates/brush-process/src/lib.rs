pub mod args_file;
pub mod config;
pub mod message;
pub mod slot;
pub mod train_stream;
pub mod volume_layout;
pub mod xray_stream;

pub use brush_vfs::DataSource;

use burn_wgpu::{
    AutoCompiler, RuntimeOptions, WgpuDevice,
    graphics::{AutoGraphicsApi, GraphicsApi},
};
use wgpu::{Adapter, Device, Queue};

use std::future::Future;
use std::pin::{Pin, pin};

use anyhow::Error;
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::SendNotWasm;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::WgpuRuntime;
use tokio_stream::{Stream, StreamExt};

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

pub async fn burn_init_setup() -> WgpuDevice {
    // 图形后端按序尝试、失败自动降级 (cubecl-wgpu 枚举不到适配器时直接
    // panic, 无法在 async 上下文 catch → 独立 current_thread runtime +
    // catch_unwind 隔离每次尝试):
    //   Windows: dx12 → vulkan → opengl; macOS: metal; 其他: vulkan → opengl
    // BRUSH_FIT_GRAPHICS_API 可指定单个或逗号列表 (dx12,vulkan,opengl,...)。
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
                Self::Dx12 => init_setup_gpu::<burn_wgpu::graphics::Dx12>().await,
                Self::Vulkan => init_setup_gpu::<burn_wgpu::graphics::Vulkan>().await,
                Self::OpenGl => init_setup_gpu::<burn_wgpu::graphics::OpenGl>().await,
                Self::Metal => init_setup_gpu::<burn_wgpu::graphics::Metal>().await,
                Self::Auto => init_setup_gpu::<AutoGraphicsApi>().await,
            }
        }
    }
    use ApiKind as A;
    let candidates: Vec<ApiKind> = match std::env::var("BRUSH_FIT_GRAPHICS_API") {
        Ok(v) => v
            .split(',')
            .map(|s| match s.trim().to_ascii_lowercase().as_str() {
                "dx12" | "d3d12" => A::Dx12,
                "vulkan" => A::Vulkan,
                "opengl" | "gl" => A::OpenGl,
                "metal" => A::Metal,
                "auto" => A::Auto,
                other => {
                    log::warn!("未知 BRUSH_FIT_GRAPHICS_API 值 '{other}', 忽略");
                    A::Auto
                }
            })
            .collect(),
        Err(_) => {
            if cfg!(target_os = "windows") {
                vec![A::Dx12, A::Vulkan, A::OpenGl]
            } else if cfg!(target_os = "macos") {
                vec![A::Metal]
            } else {
                vec![A::Vulkan, A::OpenGl]
            }
        }
    };
    let mut last_err = None;
    for api in &candidates {
        // 独立线程 + current_thread runtime 隔离每次尝试 (async 上下文嵌套
        // runtime 会 panic; join 捕获 cubecl-wgpu 的适配器枚举 panic)。
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

async fn init_setup_gpu<G: burn_wgpu::graphics::GraphicsApi>() -> WgpuDevice {
    burn_wgpu::init_setup_async::<G>(&WgpuDevice::DefaultDevice, burn_options()).await;
    connect_device(WgpuDevice::DefaultDevice);
    WgpuDevice::DefaultDevice
}

/// Initialize Burn with a wgpu setup the host already owns. Useful when
/// integrating with an existing wgpu/WebGPU application that wants to share
/// its device with Brush so tensor buffers can flow back into the host's
/// render pipeline without copies.
pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    connect_device(burn.clone());
    burn
}

use crate::{
    message::ProcessMessage,
    slot::{Slot, SlotSender},
    train_stream::train_stream,
};

pub trait ProcessStream: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}
impl<T> ProcessStream for T where T: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}

pub struct RunningProcess {
    pub stream: Pin<Box<dyn ProcessStream>>,
    pub splat_view: Slot<Splats>,
}

/// Convenience alias for the emitter `try_fn_stream` hands us inside
/// the producer body — `try_fn_stream` itself drives the state
/// machine, so this is just the channel for `emit(msg).await`.
pub(crate) type Emitter = TryStreamEmitter<ProcessMessage, Error>;

use tokio::sync::SetOnce;

static DEVICE: SetOnce<WgpuDevice> = SetOnce::const_new();

pub(crate) fn connect_device(device: WgpuDevice) {
    // Idempotent: a JS host can call `init()` and `init_existing()`, or a
    // dev-mode double-mount can re-run setup. Re-registering the same device
    // is fine; we only care that *some* device wins the race.
    let _ = DEVICE.set(device);
}

pub async fn wait_for_device() -> &'static WgpuDevice {
    DEVICE.wait().await
}

/// Create a running process from a datasource and args.
///
/// The `config_fn` callback receives the initial config (loaded from
/// args.txt if present, otherwise defaults) and returns the final
/// config to use. This allows the caller to modify or override
/// settings as needed.
pub fn create_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>> + SendNotWasm,
>(
    source: DataSource,
    config_fn: Fun,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();

    let stream =
        try_fn_stream(
            |emitter| async move { run_process(source, config_fn, &emitter, splat_tx).await },
        );

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
    }
}

async fn run_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>>,
>(
    source: DataSource,
    config_fn: Fun,
    emitter: &Emitter,
    splat_view: SlotSender<Splats>,
) -> Result<(), Error> {
    log::info!("Starting process with source {source:?}");
    emitter.emit(ProcessMessage::NewProcess).await;

    let vfs = source.clone().into_vfs().await?;
    let vfs_counts = vfs.file_count();

    if vfs_counts == 0 {
        return Err(anyhow::anyhow!("No files found."));
    }

    let ply_count = vfs.files_with_extension("ply").count();

    log::info!(
        "Mounted VFS with {} files. (plys: {})",
        vfs.file_count(),
        ply_count
    );

    let is_training = vfs_counts != ply_count;

    // Emit source info - just the display name
    let paths: Vec<_> = vfs.file_paths().collect();
    let source_name = if let Some(base_path) = vfs.base_path() {
        base_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(if is_training { "dataset" } else { "file" })
            .to_owned()
    } else if paths.len() == 1 {
        paths[0]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.ply")
            .to_owned()
    } else {
        format!("{} files", paths.len())
    };

    let base_path = vfs.base_path();

    // Load initial config from args.txt via VFS if present
    let initial_config = args_file::load_config_from_vfs(&vfs).await;

    emitter
        .emit(ProcessMessage::StartLoading {
            name: source_name,
            source,
            training: is_training,
            base_path,
        })
        .await;

    if !is_training {
        let wgpu_device = wait_for_device().await;
        let device: burn::tensor::Device = wgpu_device.clone().into();
        let mut paths: Vec<_> = vfs.file_paths().collect();
        alphanumeric_sort::sort_path_slice(&mut paths);
        let client = WgpuRuntime::<AutoCompiler>::client(wgpu_device);
        let total_frames = paths.len() as u32;

        for (frame, path) in paths.iter().enumerate() {
            log::info!("Loading single ply file");

            let mut splat_stream = pin!(brush_serde::stream_splat_from_ply(
                vfs.reader_at_path(path).await?,
                None,
                true,
            ));

            while let Some(message) = splat_stream.next().await {
                let message = message?;

                let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
                let splats = message.data.into_splats(&device, mode);

                // As loading concatenates splats each time, memory usage tends to accumulate a lot
                // over time. Clear out memory after each step to prevent this buildup.
                client.memory_cleanup();

                // For the first frame of a new file, clear existing frames
                if frame == 0 {
                    splat_view.clear();
                }

                // Capture stats before moving splats
                let num_splats = splats.num_splats();
                let sh_degree = splats.sh_degree();
                splat_view.set(frame, splats);

                emitter
                    .emit(ProcessMessage::SplatsUpdated {
                        up_axis: message.meta.up_axis,
                        frame: frame as u32,
                        total_frames,
                        num_splats,
                        sh_degree,
                    })
                    .await;
            }
        }

        emitter.emit(ProcessMessage::DoneLoading).await;
    } else {
        // Pass initial config (from args.txt or defaults) to the callback.
        // Returning `None` from `config_fn` aborts cleanly without
        // surfacing as an error.
        let base_config = initial_config.unwrap_or_default();
        let Some(config) = config_fn(base_config).await else {
            log::info!("config_fn returned None — aborting before training");
            return Ok(());
        };
        train_stream(vfs, config, emitter, splat_view).await?;
    };

    Ok(())
}
