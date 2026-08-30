//! brush-fit CLI 薄包装: 解析参数 → 构造 FitConfig → block_on 运行。
//!
//! 用法:
//!   brush-fit static  images/RXA_chest.dcm  --iters=5000 --out=target/fit
//!   brush-fit deform images/rotate_dsa_raw_gamma_preprocessed.dcm \
//!       --iters=10000 --points=30000 --out=target/fitd

use std::path::PathBuf;

use brush_fit::config::{FitConfig, FitMode};
use clap::{Parser, Subcommand};

/// 子命令选择模式, 其余参数共享 (deform 参数在 static 模式忽略)。
#[derive(Subcommand)]
enum Command {
    /// 静态重建: 无 deform, canonical splats 直接渲染。
    Static(CommonArgs),
    /// 相位驱动动态重建: deform 网络 + 心动相位条件。
    Deform(CommonArgs),
}

#[derive(clap::Args)]
struct CommonArgs {
    /// DICOM 序列文件路径。
    #[arg(value_name = "DCM")]
    dcm: PathBuf,
    /// 输出目录。
    #[arg(long, default_value = "target/fit")]
    out: PathBuf,

    // ---- 数据预处理 ----
    /// 自动 gamma 目标灰度 (0.5 = 中灰; 默认开)。
    #[arg(long)]
    gamma_target: Option<f32>,
    /// 关闭自动 gamma。
    #[arg(long)]
    no_gamma: bool,
    /// ROI: no | N | x0,y0,w,h。
    #[arg(long, default_value = "20")]
    roi: String,
    /// 场景半径 (mm); 缺省自动。
    #[arg(long)]
    scene_extent: Option<f32>,
    /// 加载最大分辨率。
    #[arg(long, default_value_t = 1920)]
    max_resolution: u32,
    /// 关闭初始化 FOV 过滤。
    #[arg(long)]
    no_fov_filter: bool,

    // ---- 初始化 ----
    #[arg(long, default_value_t = 5_000)]
    points: u32,
    /// ball | cylinder (缺省: static=cylinder, deform=ball)。
    #[arg(long)]
    init_shape: Option<String>,
    #[arg(long)]
    init_radius_scale: Option<f32>,
    #[arg(long, default_value_t = 5_000)]
    init_density_base: u32,
    #[arg(long, default_value_t = 0.0)]
    init_height_factor: f32,
    #[arg(long)]
    init_density: Option<f32>,

    // ---- 优化 / 学习率 ----
    #[arg(long, default_value_t = 10_000)]
    iters: u32,
    #[arg(long, default_value_t = 2e-5)]
    lr_mean: f64,
    #[arg(long, default_value_t = 2e-6)]
    lr_mean_end: f64,
    #[arg(long, default_value_t = 5e-3)]
    lr_scale: f64,
    #[arg(long, default_value_t = 0.012)]
    lr_opac: f64,
    #[arg(long, default_value_t = 1e-3)]
    lr_deform: f64,
    #[arg(long, default_value_t = 1e-4)]
    lr_deform_end: f64,
    #[arg(long)]
    cosine_lr: bool,

    // ---- 形变网络 ----
    #[arg(long)]
    no_ast: bool,
    #[arg(long, default_value_t = 300)]
    warm_up: u32,
    /// hexplane | hashgrid。
    #[arg(long, default_value = "hexplane")]
    deform_backend: String,
    #[arg(long)]
    predict_scaling: bool,
    #[arg(long)]
    no_time: bool,
    #[arg(long, default_value_t = 10)]
    time_freqs: usize,
    #[arg(long, default_value_t = 0.2)]
    time_min_freq: f32,
    #[arg(long, default_value_t = 1.5)]
    time_max_freq: f32,
    #[arg(long, default_value_t = 0.0)]
    time_jitter: f32,
    #[arg(long, default_value_t = 0.0)]
    time_tv_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    time_tv_dp: f32,
    #[arg(long, default_value_t = 0.0125)]
    time_tv_dt: f32,
    #[arg(long, default_value_t = 1024)]
    time_tv_sample: usize,

    // ---- HexPlane ----
    #[arg(long, default_value_t = 64)]
    hex_res: u32,
    #[arg(long, default_value_t = 32)]
    hex_time_res: u32,
    #[arg(long, default_value_t = 16)]
    hex_features: usize,
    #[arg(long, default_value_t = 128)]
    hex_mlp_width: usize,
    #[arg(long, default_value_t = 2)]
    hex_mlp_layers: usize,
    #[arg(long, default_value_t = 0.0)]
    plane_tv_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    rigid_anchor_weight: f32,

    // ---- 密度控制 / refine ----
    #[arg(long, default_value_t = 0.25)]
    growth_frac: f32,
    #[arg(long, default_value_t = 400)]
    refine_every: u32,
    #[arg(long, default_value_t = 300_000)]
    max_splats: u32,
    #[arg(long, default_value_t = 0.9)]
    refine_until_frac: f32,
    #[arg(long)]
    fixed_grad_thr: Option<f32>,
    #[arg(long)]
    dyn_grad_percentile: Option<f32>,
    #[arg(long)]
    no_split: bool,
    #[arg(long)]
    percent_dense: Option<f32>,
    #[arg(long)]
    split_scale: Option<f32>,
    #[arg(long)]
    bound_factor: Option<f32>,
    #[arg(long)]
    cull_density: Option<f32>,
    #[arg(long)]
    max_screen_size: Option<f32>,
    #[arg(long)]
    cull_contribution: bool,
    #[arg(long, default_value_t = 0.05)]
    cull_percentile: f32,
    #[arg(long, default_value_t = 1e-3)]
    cull_floor: f32,
    #[arg(long, default_value_t = 0)]
    min_splats: u32,
    #[arg(long, default_value_t = 0)]
    density_reset: u32,

    // ---- 损失 ----
    #[arg(long, default_value = "charbonnier")]
    loss: String,
    #[arg(long, default_value_t = 1e-3)]
    loss_eps: f32,
    #[arg(long, default_value_t = 0.1)]
    loss_delta: f32,
    #[arg(long, default_value_t = 1.0)]
    proj_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    proj_ssim_weight: f32,
    #[arg(long, default_value_t = 0.5)]
    multiscale_weight: f32,
    #[arg(long, default_value_t = 0.5)]
    window_weight: f32,
    #[arg(long, default_value_t = 0.0)]
    grad_weight: f32,
    #[arg(long, default_value_t = 3_000)]
    grad_ramp_from: u32,
    #[arg(long, default_value_t = 0)]
    grad_ramp_to: u32,
    #[arg(long, default_value_t = 0.03)]
    grad_edge_scale: f32,

    // ---- 分阶段呼吸场 ----
    #[arg(long, default_value_t = 0)]
    respi_after: u32,
    #[arg(long)]
    no_respi_freeze: bool,

    // ---- 评估与输出 ----
    #[arg(long)]
    eval_every: Option<u32>,
    #[arg(long)]
    eval_split_every: Option<usize>,
    #[arg(long, default_value_t = 8)]
    eval_views: usize,
    /// 关闭 eval GT|pred NRRD 保存。
    #[arg(long)]
    no_save_eval: bool,
    /// 导出 canonical PLY 点云。
    #[arg(long)]
    save_ply: bool,
    /// 关闭 deform 权重/网格场导出 (deform 模式)。
    #[arg(long)]
    no_save_deform: bool,
    /// 导出原始参数 .bin (供 gs2volume 复用)。
    #[arg(long)]
    save_bin: bool,
    /// 指标 CSV ("off" 关闭)。
    #[arg(long)]
    log_csv: Option<PathBuf>,

    // ---- volume 导出 ----
    /// 体素尺寸 mm。
    #[arg(long, default_value_t = 1.0)]
    voxel_mm: f32,
    /// 显式 XY / Z 范围 mm (缺省自适应 splat bbox)。
    #[arg(long)]
    extent_xy: Option<f32>,
    #[arg(long)]
    extent_z: Option<f32>,
    /// deform 网格场相位数。
    #[arg(long, default_value_t = 8)]
    n_deform_phases: u32,
}

impl CommonArgs {
    fn into_config(self, mode: FitMode) -> FitConfig {
        FitConfig {
            mode,
            dcm: self.dcm,
            out: self.out,
            gamma_target: self.gamma_target,
            no_gamma: self.no_gamma,
            roi: self.roi,
            scene_extent: self.scene_extent,
            max_resolution: self.max_resolution,
            fov_filter: !self.no_fov_filter,
            points: self.points,
            init_shape: self.init_shape,
            init_radius_scale: self.init_radius_scale,
            init_density_base: self.init_density_base,
            init_height_factor: self.init_height_factor,
            init_density: self.init_density,
            iters: self.iters,
            lr_mean: self.lr_mean,
            lr_mean_end: self.lr_mean_end,
            lr_scale: self.lr_scale,
            lr_opac: self.lr_opac,
            lr_deform: self.lr_deform,
            lr_deform_end: self.lr_deform_end,
            cosine_lr: self.cosine_lr,
            no_ast: self.no_ast,
            warm_up: self.warm_up,
            deform_backend: self.deform_backend,
            predict_scaling: self.predict_scaling,
            enable_time: true,
            no_time: self.no_time,
            time_freqs: self.time_freqs,
            time_min_freq: self.time_min_freq,
            time_max_freq: self.time_max_freq,
            time_jitter: self.time_jitter,
            time_tv_weight: self.time_tv_weight,
            time_tv_dp: self.time_tv_dp,
            time_tv_dt: self.time_tv_dt,
            time_tv_sample: self.time_tv_sample,
            hex_res: self.hex_res,
            hex_time_res: self.hex_time_res,
            hex_features: self.hex_features,
            hex_mlp_width: self.hex_mlp_width,
            hex_mlp_layers: self.hex_mlp_layers,
            plane_tv_weight: self.plane_tv_weight,
            rigid_anchor_weight: self.rigid_anchor_weight,
            growth_frac: self.growth_frac,
            refine_every: self.refine_every,
            max_splats: self.max_splats,
            refine_until_frac: self.refine_until_frac,
            fixed_grad_thr: self.fixed_grad_thr,
            dyn_grad_percentile: self.dyn_grad_percentile,
            split: !self.no_split,
            percent_dense: self.percent_dense,
            split_scale: self.split_scale,
            bound_factor: self.bound_factor,
            cull_density: self.cull_density,
            max_screen_size: self.max_screen_size,
            cull_contribution: self.cull_contribution,
            cull_percentile: self.cull_percentile,
            cull_floor: self.cull_floor,
            min_splats: self.min_splats,
            density_reset: self.density_reset,
            loss: self.loss,
            loss_eps: self.loss_eps,
            loss_delta: self.loss_delta,
            proj_weight: self.proj_weight,
            proj_ssim_weight: self.proj_ssim_weight,
            multiscale_weight: self.multiscale_weight,
            window_weight: self.window_weight,
            grad_weight: self.grad_weight,
            grad_ramp_from: self.grad_ramp_from,
            grad_ramp_to: self.grad_ramp_to,
            grad_edge_scale: self.grad_edge_scale,
            respi_after: self.respi_after,
            no_respi_freeze: self.no_respi_freeze,
            eval_every: self.eval_every,
            eval_split_every: self.eval_split_every,
            eval_views: self.eval_views,
            save_eval: !self.no_save_eval,
            save_ply: self.save_ply,
            save_deform: !self.no_save_deform,
            save_bin: self.save_bin,
            log_csv: self.log_csv,
            voxel_mm: self.voxel_mm,
            extent_xy: self.extent_xy,
            extent_z: self.extent_z,
            n_deform_phases: self.n_deform_phases,
        }
    }
}

#[derive(Parser)]
#[command(name = "brush-fit", about = "无头 X-ray 重建 (fit_static / fit_deform)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = match cli.command {
        Command::Static(args) => args.into_config(FitMode::Static),
        Command::Deform(args) => args.into_config(FitMode::Deform),
    };

    // NVIDIA 550.76 驱动 bug: [vkps] 线程在正常析构时可 segfault, 跳过析构。
    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime")
        .block_on(async {
            match cfg.mode {
                FitMode::Static => brush_fit::run_static(cfg).await,
                FitMode::Deform => brush_fit::run_deform(cfg).await,
            }
        });

    if result.is_ok() {
        use std::io::Write as _;
        std::io::stdout().flush().ok();
        std::io::stderr().flush().ok();
        std::process::exit(0);
    }
    result.map(|_| ())
}
