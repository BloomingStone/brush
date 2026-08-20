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
    /// Crop the detector image to an ROI. Both the pixels **and** the camera
    /// intrinsics (fov / principal point) are adjusted, so the projection
    /// stays exact — used to trim FOV edge artifacts (e.g. dark vignetting)
    /// or to reconstruct a local region of interest. Accepts:
    ///   - `no` — no crop
    ///   - a single number `N` — symmetric crop: remove `N` px from all four
    ///     edges (inset)
    ///   - `x0,y0,w,h` — explicit rectangle (pixels, relative to full image)
    #[arg(long, help_heading = "Dataset Options", default_value = "20", value_parser = parse_roi)]
    pub roi: RoiSpec,
    /// Max size of the cache for frames of the dataset, larger values usually improve performance for large datasets at the cost of more memory usage, can be e.g. 6G, 6000M, 6000MiB, 6000MB
    #[arg(long, help_heading = "Dataset Options", default_value = DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE, value_parser = parse_size)]
    pub max_scene_batch_cache_size: u64,
}

fn parse_size(s: &str) -> Result<u64, parse_size::Error> {
    parse_size::parse_size(s)
}

/// ROI crop specification: disabled, symmetric edge inset (pixels), or an
/// explicit rectangle `[x0, y0, w, h]` (pixels).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoiSpec {
    /// No crop (full detector image).
    #[default]
    None,
    /// Symmetric crop: remove `inset` px from all four edges.
    Inset(u32),
    /// Explicit rectangle `[x0, y0, w, h]` in pixels.
    Rect([u32; 4]),
}

impl RoiSpec {
    /// Resolve against the full image size `[w, h]` into a concrete rectangle
    /// `[x0, y0, w, h]` (clamped to the image).
    pub fn resolve(self, full_w: u32, full_h: u32) -> Option<[u32; 4]> {
        match self {
            RoiSpec::None => None,
            RoiSpec::Inset(inset) => {
                let x0 = inset.min(full_w / 2);
                let y0 = inset.min(full_h / 2);
                Some([x0, y0, (full_w - 2 * x0).max(1), (full_h - 2 * y0).max(1)])
            }
            RoiSpec::Rect([x0, y0, w, h]) => {
                let x0 = x0.min(full_w);
                let y0 = y0.min(full_h);
                Some([x0, y0, (w.min(full_w - x0)).max(1), (h.min(full_h - y0)).max(1)])
            }
        }
    }
}

/// Parse `no` / `N` / `x0,y0,w,h` into a [`RoiSpec`].
pub fn parse_roi(s: &str) -> Result<RoiSpec, String> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("no") || t.eq_ignore_ascii_case("none") || t.is_empty() {
        return Ok(RoiSpec::None);
    }
    let parts: Vec<u32> = t
        .split(',')
        .map(|p| {
            p.trim()
                .parse::<u32>()
                .map_err(|e| format!("invalid ROI value '{p}': {e}"))
        })
        .collect::<Result<_, _>>()?;
    match parts.len() {
        1 => Ok(RoiSpec::Inset(parts[0])),
        4 => Ok(RoiSpec::Rect([parts[0], parts[1], parts[2], parts[3]])),
        n => Err(format!(
            "ROI expects `no`, a single inset `N`, or `x0,y0,w,h` — got {n} values"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roi_accepts_no_inset_rect() {
        assert_eq!(parse_roi("no").unwrap(), RoiSpec::None);
        assert_eq!(parse_roi(" 20 ").unwrap(), RoiSpec::Inset(20));
        assert_eq!(parse_roi("20,30,400,300").unwrap(), RoiSpec::Rect([20, 30, 400, 300]));
        assert!(parse_roi("1,2").is_err());
        assert!(parse_roi("x").is_err());
    }

    #[test]
    fn inset_resolves_against_image_size() {
        // 516x516, inset 20 → (20,20) 476x476
        assert_eq!(RoiSpec::Inset(20).resolve(516, 516), Some([20, 20, 476, 476]));
        // Rect clamped to image
        assert_eq!(RoiSpec::Rect([500, 0, 100, 10]).resolve(516, 516), Some([500, 0, 16, 10]));
        // Inset larger than half the image → clamped to a 1px-wide center band
        assert_eq!(RoiSpec::Inset(1000).resolve(100, 100), Some([50, 50, 1, 1]));
        assert_eq!(RoiSpec::None.resolve(516, 516), None);
    }
}
