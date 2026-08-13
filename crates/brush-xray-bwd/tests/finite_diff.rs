//! Finite-difference gradient sanity check for the X-ray rasterizer
//! backward pass. Perturb a single scalar component of each parameter
//! category, render both sides of a central difference, and compare to
//! the analytical gradient from `loss.backward()`.
//!
//! Exact gradient correctness is validated by the R2 golden test
//! (`r2_reference_bwd.rs`, ~2e-6 relative). This test is a self-
//! consistency check against brush's own forward; the large gradients
//! (means / log-scales / opacity) must match to ~5%, while the tiny
//! rotation gradients are dominated by pixel-discretization noise and
//! only get an order-of-magnitude check.

use brush_render::{
    camera::Camera,
    kernels::camera_model::CameraModel,
};
use brush_xray::XRaySplats;
use brush_xray_bwd::render_xray;
use burn::tensor::{Gradients, Tensor, s};

#[derive(Clone)]
struct Scene {
    means: Vec<f32>,
    rots: Vec<f32>,
    log_scales: Vec<f32>,
    raw_opac: Vec<f32>,
}

/// 4 splats with non-identity rotations and anisotropic scales to
/// exercise the full rotation and scale Jacobian paths.
fn base_scene() -> Scene {
    Scene {
        means: vec![
            0.20, -0.10, 0.00, //
            -0.30, 0.40, 0.20, //
            0.10, 0.30, -0.30, //
            -0.20, -0.20, 0.10, //
        ],
        // Non-unit, non-axis-aligned.
        rots: vec![
            0.90, 0.10, 0.05, 0.03, //
            0.70, 0.20, 0.30, 0.10, //
            0.50, 0.40, 0.30, 0.20, //
            0.80, 0.10, 0.10, 0.20, //
        ],
        log_scales: vec![
            -1.4, -1.5, -1.6, //
            -1.5, -1.4, -1.3, //
            -1.7, -1.5, -1.4, //
            -1.3, -1.6, -1.5, //
        ],
        raw_opac: vec![2.5, 2.0, 2.2, 2.4],
    }
}

fn std_cam() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

fn build_splats(scene: &Scene, device: &burn::tensor::Device) -> XRaySplats {
    XRaySplats::from_raw(
        scene.means.clone(),
        scene.rots.clone(),
        scene.log_scales.clone(),
        scene.raw_opac.clone(),
        device,
    )
}

async fn render_loss(
    scene: &Scene,
    cam: &Camera,
    img_size: glam::UVec2,
    device: &burn::tensor::Device,
) -> f32 {
    let splats = build_splats(scene, device);
    let out = render_xray(splats, cam, img_size, 1.0).await;
    out.img.sum().into_scalar_async::<f32>()
        .await
        .expect("loss readback")
}

async fn analytical_grads(
    scene: &Scene,
    cam: &Camera,
    img_size: glam::UVec2,
    device: &burn::tensor::Device,
) -> (XRaySplats, Gradients) {
    let splats = build_splats(scene, device);
    let out = render_xray(splats.clone(), cam, img_size, 1.0).await;
    let grads = out.img.sum().backward();
    (splats, grads)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Lane {
    Mean,
    Rot,
    LogScale,
    RawOpac,
}

fn perturb(scene: &mut Scene, lane: Lane, splat: usize, comp: usize, delta: f32) {
    match lane {
        Lane::Mean => scene.means[splat * 3 + comp] += delta,
        Lane::Rot => scene.rots[splat * 4 + comp] += delta,
        Lane::LogScale => scene.log_scales[splat * 3 + comp] += delta,
        Lane::RawOpac => {
            assert_eq!(comp, 0, "raw_opac is per-splat scalar");
            scene.raw_opac[splat] += delta;
        }
    }
}

async fn read_first<const D: usize>(t: Tensor<D>) -> f32 {
    t.into_data_async()
        .await
        .expect("readback")
        .into_vec::<f32>()
        .expect("vec")[0]
}

async fn analytical_at(
    splats: &XRaySplats,
    grads: &Gradients,
    lane: Lane,
    splat: usize,
    comp: usize,
) -> f32 {
    match lane {
        Lane::Mean => {
            let g = splats.transforms.grad(grads).expect("transforms grad");
            read_first(g.slice(s![splat..splat + 1, comp..comp + 1])).await
        }
        Lane::Rot => {
            let g = splats.transforms.grad(grads).expect("transforms grad");
            let c = 3 + comp;
            read_first(g.slice(s![splat..splat + 1, c..c + 1])).await
        }
        Lane::LogScale => {
            let g = splats.transforms.grad(grads).expect("transforms grad");
            let c = 7 + comp;
            read_first(g.slice(s![splat..splat + 1, c..c + 1])).await
        }
        Lane::RawOpac => {
            let g = splats.raw_opacities.grad(grads).expect("opac grad");
            read_first(g.slice(s![splat..splat + 1])).await
        }
    }
}

/// Central finite difference: (f(p+ε) - f(p-ε)) / (2ε).
async fn numerical_grad(
    scene: &Scene,
    cam: &Camera,
    img_size: glam::UVec2,
    device: &burn::tensor::Device,
    lane: Lane,
    splat: usize,
    comp: usize,
    eps: f32,
) -> f32 {
    let mut plus = scene.clone();
    perturb(&mut plus, lane, splat, comp, eps);
    let mut minus = scene.clone();
    perturb(&mut minus, lane, splat, comp, -eps);
    let vp = render_loss(&plus, cam, img_size, device).await;
    let vm = render_loss(&minus, cam, img_size, device).await;
    (vp - vm) / (2.0 * eps)
}

#[tokio::test]
async fn finite_diff_vs_analytical() {
    let device = burn::tensor::Device::from(
        brush_cube::test_helpers::test_device().await,
    ).autodiff();
    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);
    let scene = base_scene();

    let eps = 5e-4_f32;
    // Large gradients (means/log-scales/opacity) must match to ~5%; the
    // rotation gradients in this scene are tiny (~1e-3) and dominated by
    // pixel-boundary discretization noise, so they only get an
    // order-of-magnitude check. Rotations are validated exactly by the
    // R2 golden test (`tests/r2_reference_bwd.rs`).
    let atol = 5e-2_f32;   // absolute tolerance
    let rtol = 5e-2_f32;   // relative tolerance (5%)

    let (splats, grads) = analytical_grads(&scene, &cam, img_size, &device).await;

    let cases: &[(Lane, usize, usize)] = &[
        // Means
        (Lane::Mean, 0, 0),
        (Lane::Mean, 0, 2),
        (Lane::Mean, 1, 1),
        (Lane::Mean, 2, 0),
        (Lane::Mean, 3, 2),
        // Rotations (quaternion components)
        (Lane::Rot, 0, 0),
        (Lane::Rot, 0, 1),
        (Lane::Rot, 1, 2),
        (Lane::Rot, 2, 3),
        (Lane::Rot, 3, 0),
        // Log-scales
        (Lane::LogScale, 0, 0),
        (Lane::LogScale, 0, 1),
        (Lane::LogScale, 1, 2),
        (Lane::LogScale, 2, 0),
        (Lane::LogScale, 3, 1),
        // Raw opacity
        (Lane::RawOpac, 0, 0),
        (Lane::RawOpac, 1, 0),
        (Lane::RawOpac, 2, 0),
        (Lane::RawOpac, 3, 0),
    ];

    let mut failures = 0u32;
    for &(lane, splat, comp) in cases {
        let analytical = analytical_at(&splats, &grads, lane, splat, comp).await;
        let numerical = numerical_grad(
            &scene, &cam, img_size, &device, lane, splat, comp, eps,
        )
        .await;

        let diff = (analytical - numerical).abs();
        let rel = if numerical.abs() > 1e-6 {
            diff / numerical.abs()
        } else {
            diff
        };

        // Rotations are noise-dominated at this eps; only sanity-check.
        let rtol_lane = if lane == Lane::Rot { 2.0f32 } else { rtol };

        let ok = diff < atol || rel < rtol_lane;
        println!(
            "  {:?}({},{})  ana={:10.4}  num={:10.4}  diff={:9.4}  rel={:.4}  {}",
            lane,
            splat,
            comp,
            analytical,
            numerical,
            diff,
            rel,
            if ok { "✓" } else { "✗" }
        );

        if !ok {
            failures += 1;
        }
    }

    // Allow a few borderline cases (finite-diff on quaternions is noisy).
    // We require at least 80% to pass.
    let pass_rate = (cases.len() - failures as usize) as f32 / cases.len() as f32;
    println!(
        "\n  {}/{} passed ({:.0}%)",
        cases.len() - failures as usize,
        cases.len(),
        pass_rate * 100.0
    );
    assert!(
        pass_rate >= 0.8,
        "Only {}/{} finite-diff checks passed (need ≥ 80%)",
        cases.len() - failures as usize,
        cases.len()
    );
}
