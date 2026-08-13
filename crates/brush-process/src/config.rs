use clap::{Args, Parser, ValueEnum};
use serde::{Deserialize, Serialize};

/// On-disk format for X-ray eval intensity images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum XRayEvalFormat {
    /// Lossless 16-bit grayscale PNG (65536 levels — matches uint16 DICOM).
    #[default]
    Png16,
    /// Lossless float32 NRRD (no quantization, for quantitative analysis).
    Nrrd,
}

#[derive(Clone, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ProcessConfig {
    /// Random seed.
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,
    /// Iteration to resume from
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,
    /// Eval every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "1000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub eval_every: u32,
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    pub eval_save_to_disk: bool,
    /// Export every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "5000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub export_every: u32,
    /// Location to put exported files. Supports {dataset} interpolation for the dataset
    /// folder name. Path is relative to the dataset's parent directory (or CWD if unavailable).
    /// Use "./{dataset}/" to export inside the dataset folder.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./{dataset}_exports/"
    )]
    pub export_path: String,
    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "export_{iter}.ply"
    )]
    pub export_name: String,
    /// Use the X-ray deform-GS training path (requires a DICOM source).
    #[arg(long, help_heading = "X-ray options", default_value = "false")]
    pub xray: bool,
    /// Number of canonical splats to initialize for X-ray training.
    #[arg(
        long,
        help_heading = "X-ray options",
        default_value = "20000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub xray_num_points: u32,
    /// Radius of the random-init splat ball and the deform-net coordinate
    /// scale, in mm (C-arm isocenter distance for typical DSA runs).
    #[arg(
        long,
        help_heading = "X-ray options",
        default_value = "200",
        value_parser = clap::value_parser!(f32)
    )]
    pub xray_scene_extent: f32,
    /// Warm-up steps before the deform network / AST noise kick in.
    #[arg(
        long,
        help_heading = "X-ray options",
        default_value = "300"
    )]
    pub xray_warm_up: u32,
    /// Enable AST (asynchronous time) noise on the phase conditioning input.
    #[arg(long, help_heading = "X-ray options", default_value = "true")]
    pub xray_enable_ast: bool,
    /// Static X-ray reconstruction: no deform field and no phase conditioning.
    /// Trains the canonical splats directly against the multi-angle DICOM
    /// projections (for static rotational scans such as `RXA_brain.dcm`).
    #[arg(long, help_heading = "X-ray options", default_value = "false")]
    pub xray_static: bool,
    /// How often (in steps) to run X-ray density control.
    #[arg(
        long,
        help_heading = "X-ray options",
        default_value = "300",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub xray_refine_every: u32,
    /// Number of views sampled per X-ray eval round (spread evenly over the
    /// DICOM sequence to cover different angles / phases).
    #[arg(
        long,
        help_heading = "X-ray options",
        default_value = "8",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub xray_eval_views: u32,
    /// On-disk format for X-ray eval intensity images (`--eval-save-to-disk`).
    #[arg(long, help_heading = "X-ray options", default_value = "png16")]
    pub xray_eval_format: XRayEvalFormat,
    /// Opacity (density) logit learning rate. The Beer-Lambert path integral
    /// `proj` saturates (`clamp(proj, 14)`), so a too-high opacity LR blows
    /// the density past the visible band → all-black images and ~zero
    /// gradients. Lower values train more stably on dense static scans.
    #[arg(long, help_heading = "X-ray options", default_value = "0.003")]
    pub xray_lr_opac: f64,
    /// Start learning rate for splat means.
    #[arg(long, help_heading = "X-ray options", default_value = "2e-5")]
    pub xray_lr_mean: f64,
    /// End (decayed) learning rate for splat means. With exponential decay
    /// over `total_iters`, keep this high enough that position still updates
    /// late in training (a too-low end LR stalls the scan once opacity
    /// stabilizes).
    #[arg(long, help_heading = "X-ray options", default_value = "2e-6")]
    pub xray_lr_mean_end: f64,
    /// Start learning rate for splat log-scales.
    #[arg(long, help_heading = "X-ray options", default_value = "5e-3")]
    pub xray_lr_scale: f64,
    /// Start learning rate for splat rotations.
    #[arg(long, help_heading = "X-ray options", default_value = "2e-3")]
    pub xray_lr_rotation: f64,
    /// Prune splats whose density (`sigmoid(raw_opacity)`) is below this
    /// (mm⁻¹). Physical `μ_water ≈ 0.002`, so a threshold above that culls
    /// meaningful low-density (soft-tissue) splats during early training.
    #[arg(long, help_heading = "X-ray options", default_value = "1e-5")]
    pub xray_cull_density: f64,
    /// Periodically reset all splat densities to `μ_water` (0 disables).
    /// Disabled by default — resets destroy the learned Beer-Lambert
    /// attenuation field and make PSNR dive.
    #[arg(long, help_heading = "X-ray options", default_value = "0")]
    pub xray_density_reset_interval: u32,
    /// Start learning rate for the deform network (deform mode only).
    #[arg(long, help_heading = "X-ray options", default_value = "1e-3")]
    pub xray_lr_deform: f64,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainStreamConfig {
    #[clap(flatten)]
    #[serde(flatten)]
    pub train_config: brush_train::config::TrainConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub model_config: brush_dataset::config::ModelConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub load_config: brush_dataset::config::LoadDatasetConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub rerun_config: brush_rerun::RerunConfig,
}

impl Default for TrainStreamConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
