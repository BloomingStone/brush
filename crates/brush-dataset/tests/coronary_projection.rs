//! Forward-projection integration test on real static-coronary data.
//!
//! Verifies the whole forward chain end-to-end: brush's DICOM camera
//! construction ([`build_camera`], COLMAP/GS convention) + the renderer's
//! pinhole projection against ground truth from the coronary rotational-DSA
//! scan. This is an integration (not unit) test because it exercises the
//! public [`build_camera`] with real data fixtures.
//!
//! Fixtures live in `tests/data/coronary/` (embedded at compile time via
//! `include_str!` / `include_bytes!`, so the test is self-contained and does
//! not depend on the workspace `images/` directory):
//! - `central_line_world.xyz`: coronary vessel centerline in world RAS mm,
//!   generated from `central_line.npz`
//!   (`local.properties/refs/convert_central_line_to_xyz.py`) and verified to
//!   lie inside `coronary_label.nii.gz` (nearest-neighbour distance 0 mm for
//!   100% of points).
//! - `label_{000,030,060,090,119}.png`: 2D ground-truth label projections
//!   (white = vessel) for frames 0/30/60/90/119.
//!
//! Reference truth (geometry + angles) from `rotate_dsa.json`. Python
//! reference (`get_proj_matric.py`, `camera_convention="colmap"`): every frame
//! projects all 704 centerline points inside the 512×512 image with 96.6–98.2%
//! on the white vessel mask (random pixels would only hit ~3.8%, since the
//! mask covers 3.78% of the image). We assert ≥95% to leave margin for the
//! ±0.5 px principal-point convention.

use brush_dataset::build_camera;
use brush_dataset::config::XRayOrientation;
use brush_dicom::meta::CArmGeometry;
use glam::{uvec2, vec2, Vec3};

#[test]
fn coronary_central_line_projection_overlaps_label() {
    // static_coronary_LCA geometry (rotate_dsa.json: sod/sdd/delx).
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

    // Centerline (world RAS mm), embedded at compile time.
    let xyz = include_str!("data/coronary/central_line_world.xyz");
    let mut pts = Vec::new();
    for line in xyz.lines() {
        let mut it = line.split_whitespace();
        let x: f32 = it.next().expect("x").parse().expect("parse x");
        let y: f32 = it.next().expect("y").parse().expect("parse y");
        let z: f32 = it.next().expect("z").parse().expect("parse z");
        pts.push(Vec3::new(x, y, z));
    }
    assert_eq!(pts.len(), 704, "centerline should have 704 points");

    let img_size = uvec2(512, 512);
    // R2 ndc2Pix principal-point convention (S-1)/2, as the renderer.
    let center = vec2((512.0 - 1.0) * 0.5, (512.0 - 1.0) * 0.5);

    // (frame index, alpha degrees); beta is 0 for every frame. Reference
    // truth from `rotate_dsa.json` frames 0/30/60/90/119.
    let frames: [(usize, f64); 5] = [
        (0, 30.0),
        (30, 105.0),
        (60, 180.0),
        (90, 255.0),
        (119, 327.5),
    ];
    // 2D ground-truth label projections, embedded as bytes so the test is
    // fully self-contained (no runtime dependency on the images/ dir).
    let label_pngs: [&[u8]; 5] = [
        include_bytes!("data/coronary/label_000.png"),
        include_bytes!("data/coronary/label_030.png"),
        include_bytes!("data/coronary/label_060.png"),
        include_bytes!("data/coronary/label_090.png"),
        include_bytes!("data/coronary/label_119.png"),
    ];

    for ((frame, alpha_deg), png) in frames.iter().zip(label_pngs.iter()) {
        let cam = build_camera(alpha_deg.to_radians(), 0.0, 760.0, XRayOrientation::Ap, &geom);
        let w2l = cam.world_to_local();
        let focal = cam.focal(img_size);

        let img = image::load_from_memory(png)
            .expect("decode label PNG")
            .into_luma8();
        assert_eq!((img.width(), img.height()), (512, 512));

        let mut n_in = 0usize;
        let mut n_hit = 0usize;
        for p in &pts {
            let p_c = w2l.transform_point3(*p);
            if p_c.z <= 0.0 {
                continue;
            }
            let u = focal.x * p_c.x / p_c.z + center.x;
            let v = focal.y * p_c.y / p_c.z + center.y;
            if !(0.0..512.0).contains(&u) || !(0.0..512.0).contains(&v) {
                continue;
            }
            n_in += 1;
            if img.get_pixel(u as u32, v as u32)[0] > 127 {
                n_hit += 1;
            }
        }

        let frac = n_hit as f64 / n_in as f64;
        assert!(
            frac >= 0.95,
            "frame {frame} (alpha={alpha_deg}°): {n_hit}/{n_in} centerline \
             projections on the vessel mask ({:.1}%), expected ≥95%",
            frac * 100.0
        );
    }
}
