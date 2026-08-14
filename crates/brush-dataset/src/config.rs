use brush_render::AlphaMode;
use clap::Args;
use serde::{Deserialize, Serialize};

/// Default Cache budget for packed scene batches. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "6GiB";
#[cfg(target_family = "wasm")]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "2GiB";

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(
        long,
        help_heading = "Model Options",
        default_value = "3",
        value_parser = clap::value_parser!(u32).range(0..=4)
    )]
    pub sh_degree: u32,
}

/// C-arm orientation for DICOM X-ray datasets. `AP` = source in front of the
/// patient (beam anterior → posterior), `PA` = source behind the patient
/// (beam posterior → anterior).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum XRayOrientation {
    /// Anterior-posterior: source at `+Y` (patient front).
    Ap,
    /// Posterior-anterior: source at `-Y` (patient back). Default.
    #[default]
    Pa,
}

/// How raw uint16 DICOM pixels are mapped to the `[0, 1]` training range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum DicomNormalization {
    /// Global min-max over all frames. Preserves absolute intensity ratios;
    /// crushes contrast when a bright outlier dominates the range.
    #[default]
    Minmax,
    /// Clip to the `[p1, p99]` percentiles then scale. Robust to outliers;
    /// recommended for low-dynamic-range / spike-heavy scans (e.g.
    /// `RXA_brain.dcm`).
    Percentile,
}

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LoadDatasetConfig {
    /// Max nr. of frames of dataset to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load.
    #[arg(long, help_heading = "Dataset Options", default_value = "1920")]
    pub max_resolution: u32,
    /// Create an eval dataset by selecting every nth image
    #[arg(long, help_heading = "Dataset Options")]
    pub eval_split_every: Option<usize>,
    /// Load only every nth frame
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_points: Option<u32>,
    /// Whether to interpret an alpha channel (or masks) as transparency or masking.
    #[arg(long, help_heading = "Dataset Options")]
    pub alpha_mode: Option<AlphaMode>,
    /// C-arm orientation for DICOM X-ray datasets. Not stored in the DICOM
    /// header (the converter writes no such tag), so it must be configured.
    #[arg(long, help_heading = "Dataset Options", default_value = "pa")]
    pub dicom_orientation: XRayOrientation,
    /// How raw uint16 DICOM pixels are normalized to `[0, 1]`.
    #[arg(long, help_heading = "Dataset Options", default_value = "minmax")]
    pub dicom_normalization: DicomNormalization,
    /// Explicit gamma correction applied to the normalized `[0,1]` pixels
    /// (`out = in^γ`). `γ < 1` brightens dark parts. When set, it wins over
    /// `dicom_gamma_target`.
    #[arg(long, help_heading = "Dataset Options")]
    pub dicom_gamma: Option<f32>,
    /// Auto-compute gamma so the global intensity median maps to this target
    /// gray level (e.g. `0.5` pulls a low-signal scan up to mid-gray). Ignored
    /// when `dicom_gamma` is set explicitly. `None` disables auto-gamma.
    #[arg(long, help_heading = "Dataset Options")]
    pub dicom_gamma_target: Option<f32>,
    /// Max size of the cache for frames of the dataset, larger values usually improve performance for large datasets at the cost of more memory usage, can be e.g. 6G, 6000M, 6000MiB, 6000MB
    #[arg(long, help_heading = "Dataset Options", default_value = DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE, value_parser = parse_size)]
    pub max_scene_batch_cache_size: u64,
}

fn parse_size(s: &str) -> Result<u64, parse_size::Error> {
    parse_size::parse_size(s)
}
