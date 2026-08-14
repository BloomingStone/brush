//! X-ray evaluation: render sampled views against the normalized DICOM GT and
//! compute grayscale PSNR / SSIM, plus lossless on-disk storage of the
//! intensity images.
//!
//! # Storage formats
//!
//! The training data is `uint16` DICOM normalized to `[0, 1]` floats. Saving a
//! plain 8-bit PNG would quantize to 256 levels and lose most of the dynamic
//! range, so we offer two lossless options:
//!
//! - [`save_gray_png16`]: **16-bit grayscale PNG** — 65536 levels, matching
//!   the original `uint16` DICOM range. One file per frame; easy to view,
//!   diff, or assemble into a video. Default.
//! - [`save_gray_nrrd_f32`]: **NRRD with `float32` raw data** — zero
//!   quantization; the predicted intensity stays an exact float. Best for
//!   quantitative analysis, at the cost of a non-image viewer.

use std::path::Path;

use anyhow::Result;
use burn::tensor::TensorData;

/// One evaluated X-ray view.
#[derive(Debug, Clone)]
pub struct XRayEvalSample {
    /// Predicted intensity `[H, W]` f32 in `[0, 1]`.
    pub pred: TensorData,
    /// Normalized GT `[H, W]` f32 in `[0, 1]`.
    pub gt: TensorData,
    pub psnr: f32,
    pub ssim: f32,
}

/// Write a `[H, W]` f32 image (in `[0, 1]`) as a lossless 16-bit grayscale PNG
/// (`Luma<u16>`), scaled to the full `uint16` range.
pub fn save_gray_png16(path: &Path, data: &TensorData) -> Result<()> {
    let [h, w] = [data.shape[0], data.shape[1]];
    let f32_buf = data.as_slice::<f32>().expect("gray f32 buffer");
    let u16_buf: Vec<u16> = f32_buf
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
        .collect();
    let img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_raw(
        w as u32,
        h as u32,
        u16_buf,
    )
    .expect("failed to build Gray16 image");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(path)?;
    Ok(())
}

/// Write a `[H, W]` f32 image as a **lossless `float32` NRRD** (raw,
/// little-endian). No quantization. The header is hand-written — NRRD is a
/// trivial text header + binary payload, so no extra dependency is needed.
///
/// NRRD viewers (3D Slicer, `ParaView`, `ITK`, `napari`) follow a spatial
/// convention where the first row is the **bottom** of the image (origin at
/// the lower-left). Since our tensor is stored top-to-bottom, the payload is
/// **flipped vertically** so the image displays upright in those tools.
pub fn save_gray_nrrd_f32(path: &Path, data: &TensorData) -> Result<()> {
    let [h, w] = [data.shape[0], data.shape[1]];
    let f32_buf = data.as_slice::<f32>().expect("gray f32 buffer");
    let mut payload = Vec::with_capacity(f32_buf.len() * 4);
    for row in (0..h).rev() {
        for v in &f32_buf[row * w..(row + 1) * w] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    let header = format!(
        "NRRD0004\n\
         # Brush X-ray eval (float32, normalized [0,1])\n\
         type: float\n\
         dimension: 2\n\
         sizes: {w} {h}\n\
         encoding: raw\n\
         endian: little\n\
         \n"
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = Vec::with_capacity(header.len() + payload.len());
    file.extend_from_slice(header.as_bytes());
    file.append(&mut payload);
    std::fs::write(path, file)?;
    Ok(())
}

/// Write an `[N, H, W]` f32 stack as a lossless `float32` NRRD volume
/// (`dimension: 3`, sizes `W H N`, z = view index). Each z-slice is flipped
/// vertically so it displays upright in NRRD viewers — same convention as
/// [`save_gray_nrrd_f32`]. Useful to batch-browse every eval view of one
/// iteration in 3D Slicer / ParaView / napari.
pub fn save_gray_nrrd_f32_stack(path: &Path, data: &TensorData) -> Result<()> {
    let [n, h, w] = [data.shape[0], data.shape[1], data.shape[2]];
    let f32_buf = data.as_slice::<f32>().expect("gray f32 buffer");
    let mut payload = Vec::with_capacity(f32_buf.len() * 4);
    for z in 0..n {
        let slice = &f32_buf[z * h * w..(z + 1) * h * w];
        for row in (0..h).rev() {
            for v in &slice[row * w..(row + 1) * w] {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let header = format!(
        "NRRD0004\n\
         # Brush X-ray eval (float32, normalized [0,1]), z = view index\n\
         type: float\n\
         dimension: 3\n\
         sizes: {w} {h} {n}\n\
         encoding: raw\n\
         endian: little\n\
         \n"
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = Vec::with_capacity(header.len() + payload.len());
    file.extend_from_slice(header.as_bytes());
    file.append(&mut payload);
    std::fs::write(path, file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray16_png_roundtrip_preserves_16bit_range() {
        // A gradient spanning the full uint16 range must survive PNG16 intact.
        let h = 8usize;
        let w = 8usize;
        let vals: Vec<f32> = (0..h * w)
            .map(|i| (i as f32 / (h * w - 1) as f32).clamp(0.0, 1.0))
            .collect();
        let data = TensorData::new(vals.clone(), [h, w]);

        let dir = std::env::temp_dir().join("brush_xray_eval_test_png16");
        let path = dir.join("grad16.png");
        save_gray_png16(&path, &data).expect("save png16");

        let img = image::open(&path).expect("open png16");
        assert_eq!(img.color(), image::ColorType::L16, "must stay 16-bit gray");
        let gray = img.to_luma16();
        // Compare against the same (v * 65535).round() mapping.
        for (i, v) in vals.iter().enumerate() {
            let expected = (v * 65535.0 + 0.5) as u16;
            let got = gray.get_pixel((i % w) as u32, (i / w) as u32)[0];
            assert_eq!(
                got, expected,
                "pixel {i}: expected {expected}, got {got} (8-bit would collapse this)"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn nrrd_f32_roundtrip_preserves_exact_float() {
        let h = 4usize;
        let w = 6usize;
        // Deliberately use values that cannot be represented in uint16
        // quantization (fractional sub-65536 steps).
        let vals: Vec<f32> = (0..h * w)
            .map(|i| {
                let base = i as f32 / 17.0;
                (base - base.trunc()).max(1e-6) // fractional part in (0,1)
            })
            .collect();
        let data = TensorData::new(vals.clone(), [h, w]);

        let dir = std::env::temp_dir().join("brush_xray_eval_test_nrrd");
        let path = dir.join("f32.nrrd");
        save_gray_nrrd_f32(&path, &data).expect("save nrrd");

        let bytes = std::fs::read(&path).expect("read nrrd");
        // Header ends at the first blank line; everything after is raw binary.
        let head_end = bytes
            .windows(2)
            .position(|win| win == b"\n\n")
            .expect("header terminator")
            + 2;
        let head = &bytes[..head_end];
        let head_str = std::str::from_utf8(head).expect("utf8 header");
        assert!(head_str.starts_with("NRRD0004"), "valid NRRD magic");
        assert!(head_str.contains("type: float"));

        let payload = &bytes[head_end..];
        assert_eq!(payload.len(), h * w * 4, "raw payload size");
        let stored: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // save_gray_nrrd_f32 flips vertically (NRRD origin is the lower-left),
        // so un-flip before comparing against the original top-to-bottom data.
        let mut got = vec![0.0_f32; h * w];
        for row in 0..h {
            let src = &stored[(h - 1 - row) * w..(h - row) * w];
            got[row * w..(row + 1) * w].copy_from_slice(src);
        }
        assert_eq!(got, vals, "float32 must roundtrip exactly (bit-exact)");

        let _ = std::fs::remove_dir_all(dir);
    }
}
