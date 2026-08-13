//! Pixel-data extraction + normalization for the multi-frame XA DICOM files.
//!
//! The files we target are uncompressed 16-bit `MONOCHROME2` with
//! `SamplesPerPixel == 1`, so the raw `PixelData` bytes can be reinterpreted
//! directly (little-endian, matching the Explicit VR Little Endian transfer
//! syntax) without a pixel-data codec.

use dicom_object::DefaultDicomObject;

use crate::parser::{DicomError, parse_object, tags};

/// Multi-frame grayscale pixel data.
#[derive(Debug, Clone, PartialEq)]
pub struct PixelData {
    /// Raw pixel values, `[num_frames, height, width]` row-major.
    pub values: Vec<u16>,
    /// Number of frames.
    pub num_frames: u32,
    /// Rows per frame.
    pub height: u32,
    /// Columns per frame.
    pub width: u32,
}

impl PixelData {
    /// Total number of pixels per frame.
    pub fn frame_len(&self) -> usize {
        (self.height * self.width) as usize
    }

    /// Frame `f` as a flat `[height * width]` slice.
    pub fn frame(&self, f: usize) -> &[u16] {
        let start = f * self.frame_len();
        &self.values[start..start + self.frame_len()]
    }

    /// Shape as `(frames, height, width)`.
    pub fn shape(&self) -> (u32, u32, u32) {
        (self.num_frames, self.height, self.width)
    }
}

/// Extract the multi-frame `uint16` pixel data from an in-memory DICOM file.
pub fn extract_pixel_data(bytes: &[u8]) -> Result<PixelData, DicomError> {
    let obj = parse_object(bytes)?;
    extract_from_object(&obj)
}

pub(crate) fn extract_from_object(
    obj: &DefaultDicomObject,
) -> Result<PixelData, DicomError> {
    let num_frames = crate::parser::read_u32(obj, tags::NUMBER_OF_FRAMES)?;
    let height = crate::parser::read_u32(obj, tags::ROWS)?;
    let width = crate::parser::read_u32(obj, tags::COLUMNS)?;

    let samples_per_pixel = crate::parser::read_u16(obj, tags::SAMPLES_PER_PIXEL)?;
    if samples_per_pixel != 1 {
        return Err(DicomError::Malformed {
            tag: tags::SAMPLES_PER_PIXEL,
            reason: format!("expected 1 (MONOCHROME), got {samples_per_pixel}"),
        });
    }
    let bits_allocated = crate::parser::read_u16(obj, tags::BITS_ALLOCATED)?;
    if bits_allocated != 16 {
        return Err(DicomError::Malformed {
            tag: tags::BITS_ALLOCATED,
            reason: format!("expected 16-bit pixels, got {bits_allocated}"),
        });
    }

    let el = obj
        .element(tags::PIXEL_DATA)
        .map_err(|_| DicomError::MissingTag { tag: tags::PIXEL_DATA })?;
    let bytes = el
        .value()
        .primitive()
        .ok_or(DicomError::NonPrimitive { tag: tags::PIXEL_DATA })?
        .to_bytes();

    let expected = num_frames as usize * height as usize * width as usize * 2;
    if bytes.len() < expected {
        return Err(DicomError::Malformed {
            tag: tags::PIXEL_DATA,
            reason: format!(
                "expected at least {expected} bytes for {num_frames}x{height}x{width} u16, got {}",
                bytes.len()
            ),
        });
    }

    let mut values = Vec::with_capacity(expected / 2);
    for chunk in bytes[..expected].chunks_exact(2) {
        values.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }

    Ok(PixelData {
        values,
        num_frames,
        height,
        width,
    })
}

/// Normalize raw pixel values to `[0, 1]` via min-max scaling.
///
/// The rendered X-ray projection is mapped to the same range (via
/// `exp(-clamp(proj))`), so the two sides differ only by an unknown linear
/// transform, which min-max normalisation discards — matching the Python
/// reconstruction project's convention. A constant image maps to all zeros.
pub fn normalize_01(values: &[u16]) -> Vec<f32> {
    let mut min = u16::MAX;
    let mut max = 0u16;
    for &v in values {
        min = min.min(v);
        max = max.max(v);
    }
    let range = (max as f32) - (min as f32);
    values
        .iter()
        .map(|&v| {
            if range <= 0.0 {
                0.0
            } else {
                (v as f32 - min as f32) / range
            }
        })
        .collect()
}

/// Normalize raw pixel values to `[0, 1]` by clipping to the `[lo, hi]`
/// percentiles of the intensity distribution, then linear min-max scaling
/// within that window.
///
/// A single bright outlier (metal / bone spike) inflates the global max and
/// crushes everything else under plain min-max (e.g. `RXA_brain.dcm` has raw
/// values 52–1305 → min-max maps the median to ≈0.014). Percentile scaling
/// keeps the clinically relevant soft-tissue/iodine range visible and gives
/// the Beer-Lambert loss a better-conditioned target.
pub fn normalize_01_percentile(values: &[u16], lo: f32, hi: f32) -> Vec<f32> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sorted: Vec<u16> = values.to_vec();
    sorted.sort_unstable();
    let idx = |p: f32| ((p * (n as f32 - 1.0)).round() as usize).min(n - 1);
    let lo_v = sorted[idx(lo)] as f32;
    let hi_v = sorted[idx(hi)] as f32;
    let range = hi_v - lo_v;
    values
        .iter()
        .map(|&v| {
            let x = v as f32;
            if range <= 0.0 {
                0.0
            } else {
                ((x - lo_v) / range).clamp(0.0, 1.0)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_01_uses_min_max() {
        let vals = [10u16, 20, 30, 40];
        let out = normalize_01(&vals);
        assert_eq!(out, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn normalize_01_constant_image_is_zero() {
        let out = normalize_01(&[5u16, 5, 5]);
        assert_eq!(out, vec![0.0; 3]);
    }
}
