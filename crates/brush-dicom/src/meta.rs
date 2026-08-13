use serde::{Deserialize, Serialize};

/// C-arm geometry extracted from the DICOM header.
///
/// Mirrors the Python `CArmGeometry` in
/// `GS-dev-contrast-flow/internal/dataparsers/xray_dataparser/meta.py`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CArmGeometry {
    /// Source-to-detector distance in mm (`DistanceSourceToDetector`,
    /// tag `(0018,1110)`).
    pub sdd: f64,
    /// Source-to-object (isocenter) distance in mm
    /// (`DistanceSourceToPatient`, tag `(0018,1111)`).
    pub sod: f64,
    /// Detector height in pixels (`Rows`, tag `(0028,0010)`).
    pub height: u32,
    /// Detector width in pixels (`Columns`, tag `(0028,0011)`).
    pub width: u32,
    /// Pixel size along x at the detector plane in mm (`ImagerPixelSpacing`,
    /// tag `(0018,1164)`).
    pub delx: f64,
    /// Pixel size along y at the detector plane in mm.
    pub dely: f64,
    /// Principal-point x offset in mm. Not stored in the DICOM header; kept
    /// for parity with the JSON meta (defaults to 0).
    pub x0: f64,
    /// Principal-point y offset in mm (defaults to 0).
    pub y0: f64,
}

/// Per-frame information derived from the DICOM header.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameInfo {
    /// Zero-based frame index.
    pub frame: u32,
    /// Cumulative time in seconds (from `FrameTimeVector`, tag `(0018,1065)`).
    pub time_s: f64,
    /// Cardiac phase in `[0, 1]` (private tag `(0071,1010)`).
    pub phase: f64,
    /// Primary rotation angle (around SI axis) in degrees
    /// (`PositionerPrimaryAngleIncrement`, tag `(0018,1520)`).
    pub alpha_degree: f64,
    /// Secondary rotation angle (around RL axis) in degrees
    /// (`PositionerSecondaryAngleIncrement`, tag `(0018,1521)`).
    pub beta_degree: f64,
}

/// Full metadata extracted from a DICOM header, equivalent to the Python
/// `XRayMeta` fields the cameras/dataset builders consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DicomMeta {
    /// Number of frames in the DICOM sequence (`NumberOfFrames`,
    /// tag `(0028,0008)`).
    pub num_frames: u32,
    /// C-arm geometry.
    pub geometry: CArmGeometry,
    /// Nominal frame rate (frames per second), from `CineRate`
    /// (tag `(0018,0040)`) or `RecommendedDisplayFrameRate` (tag
    /// `(0008,2144)`).
    pub fps: f64,
    /// Per-frame data. Length equals `num_frames`.
    pub frames: Vec<FrameInfo>,
}

impl DicomMeta {
    /// Per-frame primary rotation angles in radians.
    pub fn alphas_radians(&self) -> Vec<f64> {
        self.frames
            .iter()
            .map(|f| f.alpha_degree.to_radians())
            .collect()
    }

    /// Per-frame secondary rotation angles in radians.
    pub fn betas_radians(&self) -> Vec<f64> {
        self.frames
            .iter()
            .map(|f| f.beta_degree.to_radians())
            .collect()
    }

    /// Per-frame cardiac phases.
    pub fn phase_array(&self) -> Vec<f64> {
        self.frames.iter().map(|f| f.phase).collect()
    }

    /// Per-frame times in seconds.
    pub fn time_array(&self) -> Vec<f64> {
        self.frames.iter().map(|f| f.time_s).collect()
    }
}
