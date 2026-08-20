//! DICOM header parsing for the multi-frame XA rotational-DSA files produced
//! by `local.properties/refs/convert_tif_to_dicom.py`.
//!
//! Extracts everything the camera / dataset builders need: C-arm geometry,
//! frame timing, per-frame positioner angles, and the per-frame cardiac
//! phase stored in the private tag `(0071, 1010)`.

use std::io::Cursor;

use dicom_core::value::ConvertValueError;
use dicom_core::{PrimitiveValue, Tag};
use dicom_object::file::{OpenFileOptions, ReadPreamble};
use dicom_object::DefaultDicomObject;

use crate::meta::{CArmGeometry, DicomMeta, FrameInfo};

/// Tags used by the rotational-DSA files we target.
pub(crate) mod tags {
    use dicom_core::Tag;

    pub const DISTANCE_SOURCE_TO_DETECTOR: Tag = Tag(0x0018, 0x1110);
    pub const DISTANCE_SOURCE_TO_PATIENT: Tag = Tag(0x0018, 0x1111);
    pub const IMAGER_PIXEL_SPACING: Tag = Tag(0x0018, 0x1164);
    pub const PIXEL_SPACING: Tag = Tag(0x0028, 0x0030);

    pub const CINE_RATE: Tag = Tag(0x0018, 0x0040);
    pub const RECOMMENDED_DISPLAY_FRAME_RATE: Tag = Tag(0x0008, 0x2144);
    pub const FRAME_TIME_VECTOR: Tag = Tag(0x0018, 0x1065);

    pub const POSITIONER_PRIMARY_ANGLE: Tag = Tag(0x0018, 0x1510);
    pub const POSITIONER_SECONDARY_ANGLE: Tag = Tag(0x0018, 0x1511);
    pub const POSITIONER_PRIMARY_ANGLE_INCREMENT: Tag = Tag(0x0018, 0x1520);
    pub const POSITIONER_SECONDARY_ANGLE_INCREMENT: Tag = Tag(0x0018, 0x1521);

    pub const ROWS: Tag = Tag(0x0028, 0x0010);
    pub const COLUMNS: Tag = Tag(0x0028, 0x0011);
    pub const NUMBER_OF_FRAMES: Tag = Tag(0x0028, 0x0008);
    pub const SAMPLES_PER_PIXEL: Tag = Tag(0x0028, 0x0002);
    pub const BITS_ALLOCATED: Tag = Tag(0x0028, 0x0100);
    pub const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);

    /// Private cardiac-phase array. The private creator
    /// (`YOUR_INSTITUTION_PHASE_1.0`) is reserved at `(0071, 0010)` and the
    /// data element (`VR = FL`, one `f32` per frame) lives at `(0071, 1010)`.
    pub const PHASE_ARRAY: Tag = Tag(0x0071, 0x1010);
}

/// Errors produced while parsing a DICOM header.
#[derive(Debug, thiserror::Error)]
pub enum DicomError {
    /// The DICOM stream could not be decoded.
    #[error("failed to parse DICOM stream: {0}")]
    Object(#[from] dicom_object::ReadError),
    /// A required element is missing from the data set.
    #[error("missing required DICOM element {tag:?}")]
    MissingTag { tag: Tag },
    /// An element that should hold a primitive value does not.
    #[error("element {tag:?} is not a primitive value")]
    NonPrimitive { tag: Tag },
    /// A primitive value could not be converted to the requested type.
    #[error("element {tag:?} has an incompatible value: {source}")]
    Value {
        tag: Tag,
        #[source]
        source: ConvertValueError,
    },
    /// A value has the wrong shape / multiplicity.
    #[error("element {tag:?} has malformed value: {reason}")]
    Malformed { tag: Tag, reason: String },
}

/// Parse the DICOM header from an in-memory byte buffer. The standard
/// 128-byte preamble (if present) is skipped automatically.
pub fn parse_dicom(bytes: &[u8]) -> Result<DicomMeta, DicomError> {
    parse_from_object(&parse_object(bytes)?)
}

/// Parse the raw bytes into a full in-memory DICOM object (header + pixel
/// data). Shared by the header parser and the pixel extractor.
pub(crate) fn parse_object(bytes: &[u8]) -> Result<DefaultDicomObject, DicomError> {
    Ok(OpenFileOptions::new()
        .read_preamble(ReadPreamble::Always)
        .from_reader(Cursor::new(bytes))?)
}

fn parse_from_object(obj: &DefaultDicomObject) -> Result<DicomMeta, DicomError> {
    let num_frames = read_u32(obj, tags::NUMBER_OF_FRAMES)?;
    let geometry = read_geometry(obj)?;
    let fps = read_fps(obj)?;
    let frames = read_frames(obj, num_frames, fps)?;

    Ok(DicomMeta {
        num_frames,
        geometry,
        fps,
        frames,
    })
}

// ---------------------------------------------------------------------------
// Low-level element readers
// ---------------------------------------------------------------------------

pub(crate) fn primitive(
    obj: &DefaultDicomObject,
    tag: Tag,
) -> Result<&PrimitiveValue, DicomError> {
    let el = obj
        .element(tag)
        .map_err(|_| DicomError::MissingTag { tag })?;
    el.value()
        .primitive()
        .ok_or(DicomError::NonPrimitive { tag })
}

pub(crate) fn read_f64(obj: &DefaultDicomObject, tag: Tag) -> Result<f64, DicomError> {
    primitive(obj, tag)?
        .to_float64()
        .map_err(|source| DicomError::Value { tag, source })
}

pub(crate) fn read_f64_multi(obj: &DefaultDicomObject, tag: Tag) -> Result<Vec<f64>, DicomError> {
    primitive(obj, tag)?
        .to_multi_float64()
        .map_err(|source| DicomError::Value { tag, source })
}

pub(crate) fn read_u32(obj: &DefaultDicomObject, tag: Tag) -> Result<u32, DicomError> {
    primitive(obj, tag)?
        .to_int()
        .map_err(|source| DicomError::Value { tag, source })
}

pub(crate) fn read_u16(obj: &DefaultDicomObject, tag: Tag) -> Result<u16, DicomError> {
    primitive(obj, tag)?
        .to_int()
        .map_err(|source| DicomError::Value { tag, source })
}

// ---------------------------------------------------------------------------
// Header field extraction
// ---------------------------------------------------------------------------

fn read_geometry(obj: &DefaultDicomObject) -> Result<CArmGeometry, DicomError> {
    let sdd = read_f64(obj, tags::DISTANCE_SOURCE_TO_DETECTOR)?;
    let sod = read_f64(obj, tags::DISTANCE_SOURCE_TO_PATIENT)?;
    let height = read_u32(obj, tags::ROWS)?;
    let width = read_u32(obj, tags::COLUMNS)?;

    // `ImagerPixelSpacing` is measured at the detector plane and matches the
    // `delx / dely` used for `fx = sdd / delx`. `PixelSpacing` is the physical
    // (isocenter) spacing and is only a fallback.
    let (delx, dely) = match read_f64_multi(obj, tags::IMAGER_PIXEL_SPACING) {
        Ok(spacing) if spacing.len() == 2 => (spacing[0], spacing[1]),
        _ => {
            let spacing = read_f64_multi(obj, tags::PIXEL_SPACING)?;
            match spacing.as_slice() {
                [x, y] => (*x, *y),
                _ => {
                    return Err(DicomError::Malformed {
                        tag: tags::PIXEL_SPACING,
                        reason: format!("expected 2 values, got {}", spacing.len()),
                    })
                }
            }
        }
    };

    Ok(CArmGeometry {
        sdd,
        sod,
        height,
        width,
        delx,
        dely,
        x0: 0.0,
        y0: 0.0,
    })
}

fn read_fps(obj: &DefaultDicomObject) -> Result<f64, DicomError> {
    read_f64(obj, tags::CINE_RATE).or_else(|_| read_f64(obj, tags::RECOMMENDED_DISPLAY_FRAME_RATE))
}

/// Per-frame alpha / beta in degrees and cumulative time, plus the cardiac
/// phase from the private tag.
fn read_frames(
    obj: &DefaultDicomObject,
    num_frames: u32,
    fps: f64,
) -> Result<Vec<FrameInfo>, DicomError> {
    let n = num_frames as usize;

    // Per-frame positioner angles: prefer the multi-value increment arrays,
    // fall back to the (constant) starting angles.
    let primary_inc = read_f64_multi(obj, tags::POSITIONER_PRIMARY_ANGLE_INCREMENT).ok();
    let secondary_inc = read_f64_multi(obj, tags::POSITIONER_SECONDARY_ANGLE_INCREMENT).ok();
    let primary_start = read_f64(obj, tags::POSITIONER_PRIMARY_ANGLE).ok();
    let secondary_start = read_f64(obj, tags::POSITIONER_SECONDARY_ANGLE).ok();

    let angle_at = |arr: &Option<Vec<f64>>, start: Option<f64>, i: usize| -> f64 {
        arr.as_ref()
            .and_then(|a| a.get(i).copied())
            .or(start)
            .unwrap_or(0.0)
    };

    // Per-frame times: `FrameTimeVector` holds per-frame intervals in ms; the
    // time of frame i is the cumulative sum of intervals [0, i). When absent —
    // or degenerate (too few entries / all zero, e.g. some Neusoft XA files
    // store a single "0") — fall back to uniform `i / fps` (the acquisition
    // is uniformly sampled, so real time = frame_index / fps).
    let frame_times_ms = read_f64_multi(obj, tags::FRAME_TIME_VECTOR).ok();
    let valid_ftv = frame_times_ms.as_ref().is_some_and(|t| {
        t.len() >= n.saturating_sub(1) && t.iter().any(|v| v.abs() > 1e-9)
    });
    let times_s: Vec<f64> = if let Some(times) = frame_times_ms.filter(|_| valid_ftv) {
        let mut cum = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &iv in times.iter().take(n) {
            cum.push(acc);
            acc += iv / 1000.0;
        }
        // FrameTimeVector may legally hold n-1 entries; pad the last frame.
        cum.resize(n, acc);
        cum
    } else {
        (0..n).map(|i| i as f64 / fps).collect()
    };

    // Cardiac phase array (private tag, one f32 per frame). Missing or
    // malformed phase degrades gracefully to all-zero (like the Python
    // project's viewer path).
    let phases = read_phase_array(obj, n);

    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        frames.push(FrameInfo {
            frame: i as u32,
            time_s: times_s[i],
            phase: phases[i],
            alpha_degree: angle_at(&primary_inc, primary_start, i),
            beta_degree: angle_at(&secondary_inc, secondary_start, i),
        });
    }

    Ok(frames)
}

/// Read the per-frame cardiac phase array from the private tag `(0071,1010)`,
/// trimming or zero-padding to `n` frames (mirrors the converter script).
fn read_phase_array(obj: &DefaultDicomObject, n: usize) -> Vec<f64> {
    let mut out = read_f64_multi(obj, tags::PHASE_ARRAY).unwrap_or_default();
    out.resize(n, 0.0); // resize truncates when longer, pads with 0 when shorter
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::extract_pixel_data;
    use std::path::PathBuf;

    /// Path to the example rotational-DSA file (workspace `images/` dir).
    /// Skipped when the file is absent so `cargo test -p brush-dicom` works
    /// without the data checkout.
    fn example_dcm() -> Option<Vec<u8>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../images/rotate_dsa_raw.dcm");
        if !path.exists() {
            eprintln!("skipping: example DICOM not found at {path:?}");
            return None;
        }
        std::fs::read(path).ok()
    }

    fn parse_example() -> Option<(DicomMeta, crate::pixel::PixelData)> {
        let bytes = example_dcm()?;
        let meta = parse_dicom(&bytes).expect("parse header");
        let pixels = extract_pixel_data(&bytes).expect("parse pixels");
        Some((meta, pixels))
    }

    #[test]
    fn parses_geometry_from_example() {
        let Some((meta, _)) = parse_example() else {
            return;
        };
        let g = meta.geometry;
        assert_eq!(g.sdd, 1200.0);
        assert_eq!(g.sod, 760.0);
        assert_eq!(g.height, 474);
        assert_eq!(g.width, 648);
        assert!((g.delx - 0.616).abs() < 1e-9, "delx = {}", g.delx);
        assert!((g.dely - 0.616).abs() < 1e-9, "dely = {}", g.dely);
        assert_eq!(meta.num_frames, 402);
        assert!((meta.fps - 80.0).abs() < 1e-9, "fps = {}", meta.fps);
    }

    #[test]
    fn parses_frames_from_example() {
        let Some((meta, _)) = parse_example() else {
            return;
        };
        assert_eq!(meta.frames.len(), 402);

        let f0 = meta.frames[0];
        assert_eq!(f0.frame, 0);
        assert_eq!(f0.time_s, 0.0);
        assert_eq!(f0.phase, 0.0);
        assert!((f0.alpha_degree - (-120.0)).abs() < 1e-9, "alpha0 = {}", f0.alpha_degree);
        assert!((f0.beta_degree - 0.0).abs() < 1e-9);

        let f1 = meta.frames[1];
        assert!((f1.time_s - 0.0125).abs() < 1e-9, "time1 = {}", f1.time_s);
        // The converter writes angles rounded to 2 decimals.
        assert!((f1.alpha_degree - (-119.4)).abs() < 1e-6, "alpha1 = {}", f1.alpha_degree);

        let last = meta.frames[401];
        assert!((last.time_s - 401.0 * 0.0125).abs() < 1e-9, "last time = {}", last.time_s);
        // Converter rounds each angle to 2 decimals; frame 401 alpha =
        // round(-120 + 401 * angular_velocity / fps, 2) = 119.4.
        assert!((last.alpha_degree - 119.4).abs() < 1e-6,
            "last alpha = {}", last.alpha_degree);
    }

    #[test]
    fn extracts_pixels_from_example() {
        let Some((_, pixels)) = parse_example() else {
            return;
        };
        assert_eq!((pixels.num_frames, pixels.height, pixels.width), (402, 474, 648));
        assert_eq!(pixels.values.len(), 402 * 474 * 648);
        assert_eq!(pixels.frame(0).len(), 474 * 648);

        // Sanity: raw pixel range is plausible for the synthetic DSA data.
        let min = pixels.values.iter().copied().min().unwrap();
        let max = pixels.values.iter().copied().max().unwrap();
        assert!(max > min, "expected varying pixels, got [{min}, {max}]");
    }

    #[test]
    fn normalize_pixels_from_example() {
        let Some((_, pixels)) = parse_example() else {
            return;
        };
        let norm = crate::pixel::normalize_01(&pixels.values);
        assert_eq!(norm.len(), pixels.values.len());
        let min = norm.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = norm.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= 0.0 && max <= 1.0, "norm range [{min}, {max}]");
        assert!((max - 1.0).abs() < 1e-6, "expected max to reach 1, got {max}");
    }
}
