//! DICOM (multi-frame XA) dataset loader.
//!
//! Loads a rotational-DSA `.dcm` file, parses the header (C-arm geometry,
//! per-frame positioner angles, timing, cardiac phase) and the multi-frame
//! `uint16` pixel data, normalizes the pixels to `[0, 1]`, and builds a
//! [`Dataset`] of X-ray [`SceneView`]s (per-frame camera + time + phase +
//! grayscale GT).
//!
//! Camera construction follows `local.properties/refs/get_proj_matric.py`
//! (the authoritative reference for the projection geometry):
//! `m_c2w = R_zxy(alpha, beta, 0) @ T(0, ±sod, 0) @ reorient`, then
//! `m_w2c = inv(m_c2w)`; `reorient = reorient_zup @ R_conv` with the
//! COLMAP/GS camera convention (+X right, +Y down, +Z forward).

use std::path::PathBuf;
use std::sync::Arc;

use brush_dicom::meta::{CArmGeometry, DicomMeta};
use brush_dicom::pixel::{normalize_01, normalize_01_percentile};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_vfs::BrushVfs;
use glam::{DMat3, DMat4, DVec4, Quat, Vec2, Vec3};
use tokio::io::AsyncReadExt;

use crate::Dataset;
use crate::config::{LoadDatasetConfig, XRayOrientation};
use crate::formats::DatasetLoadResult;
use crate::scene::{GrayImage, LoadImage, SceneView};

use super::FormatError;

pub(crate) async fn load_dataset(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
) -> Option<Result<DatasetLoadResult, FormatError>> {
    let path = vfs.files_with_extension("dcm").next()?;
    Some(load_dataset_inner(vfs, load_args, path).await)
}

async fn load_dataset_inner(
    vfs: Arc<BrushVfs>,
    load_args: &LoadDatasetConfig,
    path: PathBuf,
) -> Result<DatasetLoadResult, FormatError> {
    log::info!("Loading DICOM dataset from {path:?}");

    let mut reader = vfs.reader_at_path(&path).await?;
    let mut bytes = vec![];
    reader.read_to_end(&mut bytes).await?;

    let meta = brush_dicom::parse_dicom(&bytes).map_err(|e| {
        FormatError::InvalidFormat(format!("failed to parse DICOM header from {path:?}: {e}"))
    })?;
    let pixels = brush_dicom::extract_pixel_data(&bytes).map_err(|e| {
        FormatError::InvalidFormat(format!("failed to extract DICOM pixels from {path:?}: {e}"))
    })?;

    let orientation = load_args.dicom_orientation;
    // Resolve the ROI spec (no / inset / rect) against the full image size.
    let roi_rect = load_args.roi.resolve(pixels.width, pixels.height);
    let cameras = build_cameras(&meta, orientation, roi_rect);

    // Normalize all frames with one shared transform so relative intensity is
    // preserved across frames. Min-max matches the Python project's
    // convention; percentile clipping is robust to bright outliers (use it
    // for low-dynamic-range / spike-heavy scans such as RXA_brain.dcm).
    let mut normalized = match load_args.dicom_normalization {
        crate::config::DicomNormalization::Minmax => normalize_01(&pixels.values),
        crate::config::DicomNormalization::Percentile => {
            normalize_01_percentile(&pixels.values, 0.01, 0.99)
        }
    };
    // Gamma correction: brighten dark parts so the loss target has real
    // contrast. Explicit `dicom_gamma` wins; otherwise auto-compute gamma so
    // the global intensity median maps to `dicom_gamma_target` (e.g. 0.5).
    // Since the model renders `exp(-proj)`, a gamma'd target equals scaling
    // every density by γ — matching the init μ to the target keeps the
    // initial gray level aligned.
    let gamma = load_args.dicom_gamma.or_else(|| {
        load_args.dicom_gamma_target.map(|target| {
            let mut med: Vec<f32> = normalized
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            med.sort_by(|a, b| a.total_cmp(b));
            let m = med.get(med.len() / 2).copied().unwrap_or(0.0);
            if m > 1e-4 && target > 0.0 {
                (target.ln() / m.ln()).clamp(0.05, 1.0)
            } else {
                1.0
            }
        })
    });
    if let Some(g) = gamma {
        for v in &mut normalized {
            *v = v.powf(g);
        }
    }
    let frame_len = pixels.frame_len();

    // ROI crop (pixels): trim FOV edge artifacts (dark vignetting) or focus a
    // local region. Both the pixels and the camera intrinsics (fov / principal
    // point) are adjusted so the projection stays exact.
    let (crop_w, crop_h, crop_x0, crop_y0) = match roi_rect {
        Some([x0, y0, w, h]) => {
            println!(
                "ROI crop: full {}x{} -> ({x0},{y0}) {}x{}",
                pixels.width,
                pixels.height,
                w,
                h
            );
            (w as usize, h as usize, x0 as usize, y0 as usize)
        }
        None => (pixels.width as usize, pixels.height as usize, 0, 0),
    };
    if crop_w != pixels.width as usize || crop_h != pixels.height as usize {
        // Row-major [nf, H, W] → [nf, h, w]: copy the sub-rectangle per row.
        let n_frames = pixels.num_frames as usize;
        let full_w = pixels.width as usize;
        let cropped = Vec::with_capacity(n_frames * crop_w * crop_h);
        let mut cropped = cropped;
        for f in 0..n_frames {
            let start = f * frame_len;
            for r in crop_y0..crop_y0 + crop_h {
                let row_start = start + r * full_w + crop_x0;
                cropped.extend_from_slice(&normalized[row_start..row_start + crop_w]);
            }
        }
        normalized = cropped;
    }

    // The RGB `LoadImage` is unused for X-ray views (the scene loader skips
    // decoding it); point it at the source file so the path stays meaningful.
    let image = LoadImage::new(
        vfs,
        path.clone(),
        None,
        load_args.max_resolution,
        None,
    );

    let step = load_args.subsample_frames.unwrap_or(1) as usize;
    let max_frames = load_args.max_frames.unwrap_or(usize::MAX);

    let mut views = Vec::new();
    for i in (0..pixels.num_frames as usize).step_by(step).take(max_frames) {
        let frame = &meta.frames[i];
        let start = i * crop_w * crop_h;
        let gray = GrayImage::new(
            normalized[start..start + crop_w * crop_h].to_vec(),
            crop_w as u32,
            crop_h as u32,
        );
        views.push(SceneView::xray(
            cameras[i],
            image.clone(),
            frame.time_s as f32,
            frame.phase as f32,
            gray,
        ));
    }

    let (train_views, eval_views) = crate::formats::split_eval_every(views, load_args.eval_split_every);
    let dataset = Dataset::from_views(train_views, eval_views);

    Ok(DatasetLoadResult {
        init_splat: None,
        dataset,
        warnings: vec![],
        gamma,
    })
}

// ---------------------------------------------------------------------------
// Camera construction (get_proj_matric.py, camera convention "gs"/COLMAP)
// ---------------------------------------------------------------------------

/// Rotation matrix about a single axis, matching `get_proj_matric.py`'s
/// `_axis_angle_rotation` (row-major math convention).
fn axis_angle_rotation(axis: u8, angle: f64) -> DMat3 {
    match axis {
        b'X' => DMat3::from_rotation_x(angle),
        b'Y' => DMat3::from_rotation_y(angle),
        b'Z' => DMat3::from_rotation_z(angle),
        _ => unreachable!("axis must be X, Y or Z"),
    }
}

/// Euler-angle rotation matrix. For convention `"ZXY"` and angles
/// `[alpha, beta, gamma]`: `R = Rz(alpha) @ Rx(beta) @ Ry(gamma)` — internal
/// (intrinsic) rotation about the rotated axes.
fn euler_angles_to_matrix(angles: [f64; 3], convention: [u8; 3]) -> DMat3 {
    axis_angle_rotation(convention[0], angles[0])
        * axis_angle_rotation(convention[1], angles[1])
        * axis_angle_rotation(convention[2], angles[2])
}

/// World → camera reorientation matrix `reorient = reorient_zup @ R_conv` for
/// the COLMAP/GS camera convention (+X right, +Y down, +Z forward).
///
/// Values derived from `get_proj_matric.py`:
/// - AP: `reorient_zup` rows `[right=-X, front=-Y, up=+Z]` →
///   `[[-1,0,0],[0,0,-1],[0,-1,0]]`
/// - PA: `reorient_zup = I` with `R_conv["gs"]` →
///   `[[1,0,0],[0,0,1],[0,-1,0]]`
///
/// `from_cols_array_2d` takes columns, so the row-major matrices above are
/// passed column-by-column.
fn reorientation(orientation: XRayOrientation) -> DMat3 {
    match orientation {
        XRayOrientation::Ap => DMat3::from_cols_array_2d(&[
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ]),
        XRayOrientation::Pa => DMat3::from_cols_array_2d(&[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]),
    }
}

/// Camera-to-world matrix for one frame:
/// `m_c2w = R_zxy(alpha, beta, 0) @ T(0, ±sod, 0) @ reorient`.
fn camera_to_world(alpha_rad: f64, beta_rad: f64, sod: f64, orientation: XRayOrientation) -> DMat4 {
    let rotation = euler_angles_to_matrix([alpha_rad, beta_rad, 0.0], *b"ZXY");
    let mut m_rotation = DMat4::from_mat3(rotation);

    let sod_sign = match orientation {
        XRayOrientation::Ap => 1.0, // source in front of the patient (+Y)
        XRayOrientation::Pa => -1.0, // source behind the patient (-Y)
    };
    let mut m_translation = DMat4::IDENTITY;
    m_translation.w_axis = DVec4::new(0.0, sod_sign * sod, 0.0, 1.0);

    let m_reorient = DMat4::from_mat3(reorientation(orientation));

    // m_c2w = m_rotation @ m_translation @ m_reorient
    m_rotation *= m_translation;
    m_rotation *= m_reorient;
    m_rotation
}

/// Build a brush [`Camera`] for one DICOM frame, following
/// `get_proj_matric.py::get_view_matrix` (world-to-camera inverse of the
/// c2w above). The camera sits at the X-ray source and looks toward the
/// isocenter.
///
/// Public so the forward-projection integration test can build the same
/// cameras the loader produces.
pub fn build_camera(
    alpha_rad: f64,
    beta_rad: f64,
    sod: f64,
    orientation: XRayOrientation,
    geom: &CArmGeometry,
) -> Camera {
    build_camera_roi(alpha_rad, beta_rad, sod, orientation, geom, None)
}

/// [`build_camera`] with an optional ROI crop `[x0, y0, w, h]` (pixels).
///
/// Cropping the detector keeps the focal length (`fx = sdd / delx`) intact and
/// re-derives the field of view from the cropped dimensions, and shifts the
/// normalized principal point so the isocenter stays at the same physical
/// detector location: `center_uv = ((W/2 - x0)/w, (H/2 - y0)/h)`.
pub fn build_camera_roi(
    alpha_rad: f64,
    beta_rad: f64,
    sod: f64,
    orientation: XRayOrientation,
    geom: &CArmGeometry,
    roi: Option<[u32; 4]>,
) -> Camera {
    let m_c2w = camera_to_world(alpha_rad, beta_rad, sod, orientation);
    let (_, rotation, position) = m_c2w.to_scale_rotation_translation();
    let position = Vec3::new(position.x as f32, position.y as f32, position.z as f32);
    let rotation = Quat::from_xyzw(
        rotation.x as f32,
        rotation.y as f32,
        rotation.z as f32,
        rotation.w as f32,
    );

    let (w, h, x0, y0) = match roi {
        Some([x0, y0, w, h]) => {
            let x0 = x0.min(geom.width);
            let y0 = y0.min(geom.height);
            let w = w.clamp(1, geom.width - x0);
            let h = h.clamp(1, geom.height - y0);
            (w as f64, h as f64, x0 as f64, y0 as f64)
        }
        None => (
            geom.width as f64,
            geom.height as f64,
            0.0,
            0.0,
        ),
    };

    let fx = geom.sdd / geom.delx; // pixels
    let fy = geom.sdd / geom.dely;
    // Same focal length as the full detector → the cropped FOV only covers
    // the sub-rectangle of the detector.
    let fov_x = 2.0 * ((w * 0.5) / fx).atan();
    let fov_y = 2.0 * ((h * 0.5) / fy).atan();
    // Isocenter principal point in cropped (normalized) coordinates.
    let center_uv = Vec2::new(
        ((geom.width as f64 * 0.5 - x0) / w) as f32,
        ((geom.height as f64 * 0.5 - y0) / h) as f32,
    );

    Camera::new(
        position,
        rotation,
        fov_x,
        fov_y,
        center_uv,
        CameraModel::Pinhole,
    )
}

/// Build cameras for every frame of the DICOM sequence.
fn build_cameras(
    meta: &DicomMeta,
    orientation: XRayOrientation,
    roi: Option<[u32; 4]>,
) -> Vec<Camera> {
    let alphas = meta.alphas_radians();
    let betas = meta.betas_radians();
    (0..meta.num_frames as usize)
        .map(|i| {
            build_camera_roi(
                alphas[i],
                betas[i],
                meta.geometry.sod,
                orientation,
                &meta.geometry,
                roi,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Decisive geometry check: project known world points with the brush
    /// camera (AP, alpha=beta=0, the `rotate_dsa` geometry) and compare with
    /// the Python reference `get_proj_matric.py` (gs/COLMAP convention).
    ///
    /// Reference (from the Python script, AP + gs):
    ///   fx=fy=1948.05, cx=324, cy=237 (Python uses W/2; brush uses (W-1)/2)
    ///   (0,0,0)     → (324,   237)
    ///   (100,0,0)   → (67.7,  237)
    ///   (0,100,0)   → (324,   237)
    ///   (0,0,100)   → (324,  -19.3)   (head, above FOV)
    ///   (-100,0,0)  → (580.3, 237)
    ///   (0,0,-100)  → (324,   493.3)  (feet, below FOV)
    #[test]
    fn projection_matches_python_reference_ap() {
        let geom = CArmGeometry {
            sdd: 1200.0,
            sod: 760.0,
            height: 474,
            width: 648,
            delx: 0.616,
            dely: 0.616,
            x0: 0.0,
            y0: 0.0,
        };
        let cam = build_camera(0.0, 0.0, 760.0, XRayOrientation::Ap, &geom);
        let img_size = glam::uvec2(648, 474);
        let w2l = cam.world_to_local();
        let focal = cam.focal(img_size);
        // Same principal point as the renderer (R2 ndc2Pix: (S-1)/2).
        let center = glam::vec2((648.0 - 1.0) * 0.5, (474.0 - 1.0) * 0.5);

        let pts = [
            ([0.0f32, 0.0, 0.0], (324.0, 237.0)),
            ([100.0, 0.0, 0.0], (67.7, 237.0)),
            ([0.0, 100.0, 0.0], (324.0, 237.0)),
            ([0.0, 0.0, 100.0], (324.0, -19.3)),
            ([-100.0, 0.0, 0.0], (580.3, 237.0)),
            ([0.0, 0.0, -100.0], (324.0, 493.3)),
        ];
        for (p, (ref_u, ref_v)) in pts {
            let p_w = glam::Vec3::new(p[0], p[1], p[2]);
            let p_c = w2l.transform_point3(p_w);
            let u = focal.x * p_c.x / p_c.z + center.x;
            let v = focal.y * p_c.y / p_c.z + center.y;
            let tol = 1.0;
            assert!(
                (u - ref_u).abs() < tol && (v - ref_v).abs() < tol,
                "P_w={p_w:?} P_c={p_c:?}: uv=({u:.2},{v:.2}) expected ({ref_u:.1},{ref_v:.1})"
            );
        }
    }

    #[test]
    fn camera_position_matches_sod() {
        let geom = CArmGeometry {
            sdd: 1200.0,
            sod: 760.0,
            height: 474,
            width: 648,
            delx: 0.616,
            dely: 0.616,
            x0: 0.0,
            y0: 0.0,
        };
        let ap = build_camera(0.0, 0.0, 760.0, XRayOrientation::Ap, &geom);
        // AP: source in front of the patient, at +Y.
        assert!((ap.position - glam::Vec3::new(0.0, 760.0, 0.0)).length() < 1e-3);
        // PA: source behind the patient, at -Y.
        let pa = build_camera(0.0, 0.0, 760.0, XRayOrientation::Pa, &geom);
        assert!((pa.position - glam::Vec3::new(0.0, -760.0, 0.0)).length() < 1e-3);
    }

    /// Same as [`projection_matches_python_reference_ap`] but for a rotated
    /// C-arm (alpha=30°, beta=0°). Python reference (corrected
    /// `m_c2w = R @ T @ reorient`):
    ///   (0,0,0)    → (324.0, 237.0)
    ///   (100,0,0)  → (115.7, 237.0)
    ///   (0,100,0)  → (179.4, 237.0)
    ///   (0,0,100)  → (324.0, -19.3)
    ///   (-100,0,0) → (561.6, 237.0)
    #[test]
    fn projection_matches_python_reference_rotated() {
        let geom = CArmGeometry {
            sdd: 1200.0,
            sod: 760.0,
            height: 474,
            width: 648,
            delx: 0.616,
            dely: 0.616,
            x0: 0.0,
            y0: 0.0,
        };
        let cam = build_camera(30.0_f64.to_radians(), 0.0, 760.0, XRayOrientation::Ap, &geom);
        let img_size = glam::uvec2(648, 474);
        let w2l = cam.world_to_local();
        let focal = cam.focal(img_size);
        let center = glam::vec2((648.0 - 1.0) * 0.5, (474.0 - 1.0) * 0.5);

        let pts = [
            ([0.0f32, 0.0, 0.0], (324.0, 237.0)),
            ([100.0, 0.0, 0.0], (115.7, 237.0)),
            ([0.0, 100.0, 0.0], (179.4, 237.0)),
            ([0.0, 0.0, 100.0], (324.0, -19.3)),
            ([-100.0, 0.0, 0.0], (561.6, 237.0)),
        ];
        for (p, (ref_u, ref_v)) in pts {
            let p_c = w2l.transform_point3(glam::Vec3::new(p[0], p[1], p[2]));
            let u = focal.x * p_c.x / p_c.z + center.x;
            let v = focal.y * p_c.y / p_c.z + center.y;
            let tol = 1.0;
            assert!(
                (u - ref_u).abs() < tol && (v - ref_v).abs() < tol,
                "P={p:?} P_c={p_c:?}: uv=({u:.2},{v:.2}) expected ({ref_u:.1},{ref_v:.1})"
            );
        }
    }

    #[allow(clippy::print_stderr)]
    fn example_meta() -> Option<(DicomMeta, brush_dicom::pixel::PixelData)> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../images/rotate_dsa_raw.dcm");
        if !path.exists() {
            eprintln!("skipping: example DICOM not found at {path:?}");
            return None;
        }
        let bytes = std::fs::read(path).ok()?;
        Some((
            brush_dicom::parse_dicom(&bytes).expect("parse header"),
            brush_dicom::extract_pixel_data(&bytes).expect("parse pixels"),
        ))
    }

    #[cfg(not(target_family = "wasm"))]
    #[allow(clippy::print_stderr)]
    #[tokio::test]
    async fn loads_example_dataset_end_to_end() {
        let images_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../images");
        if !images_dir.join("rotate_dsa_raw.dcm").exists() {
            eprintln!("skipping: example DICOM not found under {images_dir:?}");
            return;
        }
        let vfs = Arc::new(
            brush_vfs::BrushVfs::from_path(&images_dir)
                .await
                .expect("construct vfs"),
        );
        let load_args = LoadDatasetConfig {
            max_frames: None,
            max_resolution: 1920,
            eval_split_every: None,
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: None,
            dicom_orientation: XRayOrientation::Ap,
            dicom_normalization: crate::config::DicomNormalization::Minmax,
            dicom_gamma: None,
            dicom_gamma_target: None,
            roi: crate::config::RoiSpec::None,
            max_scene_batch_cache_size: 1 << 30,
        };

        // Explicitly target `rotate_dsa_raw.dcm`: the `images/` dir now also
        // contains other DICOM scans (RXA_brain.dcm, RXA_static_coronary.dcm),
        // so `load_dataset`'s `files_with_extension("dcm").next()` would pick
        // whichever file the VFS happens to visit first.
        let path = images_dir.join("rotate_dsa_raw.dcm");
        let result = load_dataset_inner(vfs, &load_args, path)
            .await
            .expect("load dataset");
        let dataset = result.dataset;
        assert_eq!(dataset.train.views.len(), 402, "expected 402 frames");

        let first = &dataset.train.views[0];
        let gray = first.gray_image.as_ref().expect("gray image");
        assert_eq!((gray.width, gray.height), (648, 474));
        assert_eq!(first.time, 0.0);
        assert_eq!(first.phase, 0.0);
        assert!((first.camera.position.length() - 760.0).abs() < 1e-1);

        let last = dataset.train.views.last().unwrap();
        assert!(last.phase > 0.0, "phase should advance with time");
        assert!(last.time > first.time);
        assert!((last.camera.position.length() - 760.0).abs() < 1e-1);
    }

    #[test]
    fn camera_math_matches_proj_matric_reference() {
        // Hand-derived from get_proj_matric.py for AP, alpha = -120°,
        // beta = 0°, sod = 760 (see the module docs). The source sits on the
        // +Y side and looks at the isocenter.
        let geom = CArmGeometry {
            sdd: 1200.0,
            sod: 760.0,
            height: 474,
            width: 648,
            delx: 0.616,
            dely: 0.616,
            x0: 0.0,
            y0: 0.0,
        };
        let cam = build_camera(
            (-120.0_f64).to_radians(),
            0.0,
            760.0,
            XRayOrientation::Ap,
            &geom,
        );

        // position = translation of m_c2w = (0.866·760, -0.5·760, 0)
        let p = cam.position;
        assert!((p.x - 0.8660254_f32 * 760.0).abs() < 1e-3, "pos.x = {}", p.x);
        assert!((p.y + 380.0).abs() < 1e-3, "pos.y = {}", p.y);
        assert!(p.z.abs() < 1e-3, "pos.z = {}", p.z);

        // Source-to-isocenter distance must equal SOD.
        assert!((p.length() - 760.0).abs() < 1e-2, "dist = {}", p.length());

        // w2c (brush world_to_local) must equal inv(m_c2w):
        // row-major [[0.5, 0.866, 0, 0], [0, 0, -1, 0], [-0.866, 0.5, 0, 760],
        //            [0, 0, 0, 1]]. `to_cols_array_2d` returns columns, so
        // `m[col][row]`.
        let w2c = cam.world_to_local();
        let m = glam::Mat4::from(w2c).to_cols_array_2d();
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-3;
        // col 0 = (0.5, 0, -0.866, 0), col 1 = (0.866, 0, 0.5, 0),
        // col 2 = (0, -1, 0, 0), col 3 = (0, 0, 760, 1)
        assert!(approx(m[0][0], 0.5) && approx(m[0][1], 0.0) && approx(m[0][2], -0.8660254) && approx(m[0][3], 0.0));
        assert!(approx(m[1][0], 0.8660254) && approx(m[1][1], 0.0) && approx(m[1][2], 0.5) && approx(m[1][3], 0.0));
        assert!(approx(m[2][0], 0.0) && approx(m[2][1], -1.0) && approx(m[2][2], 0.0) && approx(m[2][3], 0.0));
        assert!(approx(m[3][0], 0.0) && approx(m[3][1], 0.0) && approx(m[3][2], 760.0) && approx(m[3][3], 1.0));

        // fov_x = 2·atan((W/2)/fx), fx = sdd/delx = 1948.05
        let fx = 1200.0_f64 / 0.616_f64;
        let expected_fov = 2.0 * ((648.0_f64 / 2.0) / fx).atan();
        assert!((cam.fov_x - expected_fov).abs() < 1e-9, "fov_x = {}", cam.fov_x);
        assert!((cam.fov_y - 2.0 * ((474.0_f64 / 2.0) / fx).atan()).abs() < 1e-9);
    }

    #[test]
    fn builds_cameras_for_all_frames() {
        let Some((meta, _)) = example_meta() else {
            return;
        };
        let cams = build_cameras(&meta, XRayOrientation::Ap, None);
        assert_eq!(cams.len(), 402);

        // Every camera must be valid and the same distance from isocenter.
        for cam in &cams {
            assert!(cam.is_valid());
            assert!((cam.position.length() - 760.0).abs() < 1e-1);
        }

        // First and last frames differ in alpha → different rotation.
        assert!((cams[0].rotation - cams[401].rotation).length() > 1e-3);
    }

    /// ROI crop must keep the projection exact: the isocenter lands on the
    /// same *physical* detector pixel, and the focal length is preserved
    /// (only fov / principal point change).
    #[test]
    fn roi_crop_keeps_projection_exact() {
        let geom = CArmGeometry {
            sdd: 1200.0,
            sod: 760.0,
            height: 474,
            width: 648,
            delx: 0.616,
            dely: 0.616,
            x0: 0.0,
            y0: 0.0,
        };
        let roi = [20u32, 30, 300, 200];
        let cam = build_camera(0.0, 0.0, 760.0, XRayOrientation::Ap, &geom);
        let cam_c = build_camera_roi(0.0, 0.0, 760.0, XRayOrientation::Ap, &geom, Some(roi));
        let [x0, y0, w, h] = roi;

        // Focal length (px) preserved: fov_to_focal(fov, cropped_px) == sdd/delx.
        use brush_render::camera::fov_to_focal;
        let fx = geom.sdd / geom.delx;
        assert!(
            (fov_to_focal(cam_c.fov_x, w, &CameraModel::Pinhole) - fx).abs() < 1e-6,
            "focal x not preserved: {} vs {}",
            fov_to_focal(cam_c.fov_x, w, &CameraModel::Pinhole),
            fx
        );
        assert!(
            (fov_to_focal(cam_c.fov_y, h, &CameraModel::Pinhole) - fx).abs() < 1e-6,
            "focal y not preserved"
        );

        // Isocenter projects to the same physical detector pixel: full camera
        // → (W/2, H/2); cropped camera → (W/2 - x0, H/2 - y0) in cropped px.
        let w2l = cam_c.world_to_local();
        let p_c = w2l.transform_point3(glam::Vec3::ZERO);
        let focal = cam_c.focal(glam::uvec2(w, h));
        let u = focal.x * p_c.x / p_c.z + cam_c.center_uv.x * w as f32;
        let v = focal.y * p_c.y / p_c.z + cam_c.center_uv.y * h as f32;
        assert!(
            (u - (geom.width as f32 / 2.0 - x0 as f32)).abs() < 0.5
                && (v - (geom.height as f32 / 2.0 - y0 as f32)).abs() < 0.5,
            "isocenter uv = ({u:.1},{v:.1}), expected ({},{})",
            geom.width as f32 / 2.0 - x0 as f32,
            geom.height as f32 / 2.0 - y0 as f32
        );
    }
}

// The real-data forward-projection integration check lives in
// `tests/coronary_projection.rs` (it exercises the public `build_camera`
// against the static-coronary ground-truth fixtures).
