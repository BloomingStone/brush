//! GPU integration test: render the coronary vessel centerline as Gaussian
//! splats through brush's **real** X-ray rasterizer backend (cube kernels on
//! wgpu) and measure the overlap of the rendered projection with the 2D
//! ground-truth label masks.
//!
//! This exercises the whole forward render path (not just the projection
//! math like `brush-dataset/tests/coronary_projection.rs`):
//!   1. `brush_dataset::build_camera` — DICOM C-arm camera (AP / COLMAP).
//!   2. centerline `.xyz` → [`brush_xray::XRaySplats`] (means + unit quats +
//!      isotropic σ=1.5 mm log-scales + iodinated-vessel density μ).
//!   3. `brush_xray::render_xray_forward` — project + cov2d + rasterize into
//!      a cone-beam density image `[512, 512]`.
//!   4. threshold the density map and measure coverage / precision vs the
//!      `label_*.png` vessel masks (white = vessel).
//!
//! Fixtures live in `tests/data/coronary/` (embedded at compile time).
//! Requires a GPU backend (wgpu). Random pixels would only hit ~3.8% of the
//! mask, so high coverage is a strong geometric check of the renderer.
//!
//! Visual output (for eyeballing the projection) is written to
//! `target/coronary_render_overlap/` before the assertions, so it is
//! available even on failure:
//! - `render_{frame:03}.png` — the rendered density projection as a gray
//!   X-ray-style intensity image `exp(-proj)`.
//! - `overlay_{frame:03}.png` — label mask (green) + rendered foreground
//!   (red); overlap is yellow.

use brush_dataset::build_camera;
use brush_dataset::config::XRayOrientation;
use brush_dicom::meta::CArmGeometry;
use brush_xray::{XRaySplats, render_xray_forward};
use burn::tensor::Device;
use glam::{uvec2, Vec3};
use std::path::Path;

/// Inverse sigmoid → raw-opacity logit for a target density `μ` (mm⁻¹).
fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

#[tokio::test]
#[allow(clippy::print_stdout)]
async fn coronary_centerline_splats_render_overlap_label() {
    // -- backend ---------------------------------------------------------
    let wgpu_device = brush_cube::test_helpers::test_device().await;
    let device = Device::from(wgpu_device);

    // -- geometry (static_coronary_LCA / rotate_dsa.json) ----------------
    let geom = CArmGeometry {
        sdd: 1200.0,
        sod: 760.0,
        height: 512,
        width: 512,
        delx: 0.368,
        dely: 0.368,
        x0: 0.0,
        y0: 0.0,
    };
    let img_size = uvec2(512, 512);

    // -- centerline points (world RAS mm) --------------------------------
    let xyz = include_str!("data/coronary/central_line_world.xyz");
    let mut pts: Vec<Vec3> = Vec::new();
    for line in xyz.lines() {
        let mut it = line.split_whitespace();
        let x: f32 = it.next().expect("x").parse().expect("parse x");
        let y: f32 = it.next().expect("y").parse().expect("parse y");
        let z: f32 = it.next().expect("z").parse().expect("parse z");
        pts.push(Vec3::new(x, y, z));
    }
    assert_eq!(pts.len(), 704, "centerline should have 704 points");

    // -- centerline -> splats ---------------------------------------------
    // The centerline is the vessel axis (label→centerline NN p90 ≈ 1.7 mm,
    // i.e. the vessel radius). Give every splat an isotropic Gaussian
    // σ = 0.7 mm (3σ ≈ 2.1 mm, just covering the ~1.7 mm vessel radius) so
    // the projected footprint matches the vessel width — a σ of 1.5 mm (3σ =
    // 4.5 mm) overflowed the vessel and dropped precision to ~0.27. Density
    // is that of an iodinated vessel (μ = 0.02 mm⁻¹).
    const MU: f32 = 0.02; // mm⁻¹
    let log_scale = 0.7f32.ln();
    let raw_opac_logit = inverse_sigmoid(MU);
    let mut means = Vec::with_capacity(pts.len() * 3);
    let mut rots = Vec::with_capacity(pts.len() * 4);
    let mut log_scales = Vec::with_capacity(pts.len() * 3);
    let mut raw_opacs = Vec::with_capacity(pts.len());
    for p in &pts {
        means.extend_from_slice(&[p.x, p.y, p.z]);
        rots.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]); // identity rotation
        log_scales.extend_from_slice(&[log_scale, log_scale, log_scale]);
        raw_opacs.push(raw_opac_logit);
    }
    let splats = XRaySplats::from_raw(means, rots, log_scales, raw_opacs, &device);

    // -- frames to check (frame index, alpha degrees; beta = 0) -----------
    let frames: [(usize, f64); 5] = [
        (0, 30.0),
        (30, 105.0),
        (60, 180.0),
        (90, 255.0),
        (119, 327.5),
    ];
    let label_pngs: [&[u8]; 5] = [
        include_bytes!("data/coronary/label_000.png"),
        include_bytes!("data/coronary/label_030.png"),
        include_bytes!("data/coronary/label_060.png"),
        include_bytes!("data/coronary/label_090.png"),
        include_bytes!("data/coronary/label_119.png"),
    ];

    // Beer-Lambert density-projection threshold (raw `proj`, not exp(-proj)).
    // One splat contributes μ·σ·√(2π) ≈ 0.075 to `proj`; anything meaningful
    // is far above this floor.
    const PROJ_THRESHOLD: f32 = 0.01;

    // Output dir for visual inspection (git-ignored `target/`).
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/coronary_render_overlap");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    for ((frame, alpha_deg), png) in frames.iter().zip(label_pngs.iter()) {
        let cam =
            build_camera(alpha_deg.to_radians(), 0.0, 760.0, XRayOrientation::Ap, &geom);

        // Real renderer: project → cov2d → depth sort → additive rasterize.
        let proj = render_xray_forward(&splats, &cam, img_size, 1.0).await;
        let data = proj
            .into_data_async()
            .await
            .expect("proj readback")
            .to_vec::<f32>()
            .expect("f32 vec");
        assert_eq!(data.len(), 512 * 512);

        let img = image::load_from_memory(png)
            .expect("decode label PNG")
            .into_luma8();
        assert_eq!((img.width(), img.height()), (512, 512));

        // --- visual output (saved before assertions) ---------------------
        // X-ray-style gray intensity `exp(-proj)`: vessels absorb → dark.
        let gray = image::GrayImage::from_fn(512, 512, |x, y| {
            let p = data[(y * 512 + x) as usize];
            let intensity = (-p.clamp(0.001, 14.0)).exp();
            image::Luma([(intensity * 255.0).clamp(0.0, 255.0) as u8])
        });
        let render_path = out_dir.join(format!("render_{frame:03}.png"));
        gray.save(&render_path).expect("save render PNG");

        // Overlay: green = label vessel, red = rendered foreground,
        // yellow = overlap.
        let rgb = image::RgbImage::from_fn(512, 512, |x, y| {
            let is_label = img.get_pixel(x, y)[0] > 127;
            let is_rendered = data[(y * 512 + x) as usize] > PROJ_THRESHOLD;
            image::Rgb([if is_rendered { 255 } else { 0 }, if is_label { 255 } else { 0 }, 0])
        });
        let overlay_path = out_dir.join(format!("overlay_{frame:03}.png"));
        rgb.save(&overlay_path).expect("save overlay PNG");
        println!("saved {}", overlay_path.display());

        let mut label_fg = 0usize;
        let mut hit = 0usize; // label_fg ∩ rendered_fg
        let mut rendered_fg = 0usize;
        for y in 0..512usize {
            for x in 0..512usize {
                let is_label = img.get_pixel(x as u32, y as u32)[0] > 127;
                let is_rendered = data[y * 512 + x] > PROJ_THRESHOLD;
                if is_label {
                    label_fg += 1;
                    if is_rendered {
                        hit += 1;
                    }
                }
                if is_rendered {
                    rendered_fg += 1;
                }
            }
        }
        assert!(label_fg > 0, "frame {frame}: empty label mask");

        // coverage = recall of the vessel mask (should be ~1: the centerline
        // splats project onto the vessel). precision = fraction of rendered
        // pixels that are inside the vessel (centerline is a subset of the
        // vessel, but σ=1.5mm splats spill slightly past the 1.7mm radius).
        let coverage = hit as f64 / label_fg as f64;
        let precision = hit as f64 / rendered_fg as f64;
        println!(
            "frame {frame:3} alpha={alpha_deg:6.1}°: label_fg={label_fg:5} \
             rendered_fg={rendered_fg:5} coverage={coverage:.3} precision={precision:.3}"
        );

        assert!(
            coverage >= 0.90,
            "frame {frame} (alpha={alpha_deg}°): vessel-mask coverage {coverage:.3} < 0.90",
        );
        assert!(
            precision >= 0.50,
            "frame {frame} (alpha={alpha_deg}°): rendered precision {precision:.3} < 0.50",
        );
    }
}
