//! Independent backward verification: single large anisotropic Gaussian,
//! end-to-end central-difference gradients of a moment loss vs the analytic
//! backward from `brush_xray_bwd::render_xray`.
//!
//! Why a separate test: the old `finite_diff` only checked *self-consistency*
//! of backward vs forward (a buggy forward passes it). Here we use a single
//! big, well-resolved splat so pixel-discretization noise is small, and a
//! moment loss `Σ proj·(x²+y²)` that has a clean nonzero gradient w.r.t.
//! scale / quat / mean / opacity.

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_xray::{XRaySplats, render_xray_forward};
use brush_xray_bwd::render_xray;
use burn::tensor::{Gradients, Tensor, s};

const IMG: u32 = 128;
const FOCAL: f64 = 1948.0;

fn cam() -> Camera {
    // Camera at +Z looking at origin (local +Z toward origin).
    let pos = glam::vec3(0.0, 0.0, 760.0);
    let dir = -pos.normalize();
    let up = glam::Vec3::Y;
    let right = up.cross(dir).normalize();
    let up2 = dir.cross(right);
    let rot = glam::Quat::from_mat3(&glam::Mat3::from_cols(right, up2, dir));
    let r_pix = (IMG as f64) / 2.0;
    let fov = 2.0 * (r_pix / FOCAL).atan();
    Camera::new(pos, rot, fov, fov, glam::vec2(0.5, 0.5), CameraModel::Pinhole)
}

/// One anisotropic splat, off-axis and obliquely oriented (long axis along
/// world +Z + tilted). Big enough that the blob spans many pixels.
struct Scene {
    means: [f32; 3],
    rots: [f32; 4], // w,x,y,z (unnormalized ok — code normalizes)
    log_scales: [f32; 3],
    raw_opac: f32,
}

fn base_scene() -> Scene {
    Scene {
        means: [20.0, -10.0, 0.0],
        rots: {
            // Non-unit quat rotating +X onto (0, 1, 1)/√2, scaled by 2 to
            // exercise the normalize-VJP (dnormvdv4) 1/||q|| factor.
            let d = glam::vec3(0.0, 1.0, 1.0).normalize();
            let q = glam::Quat::from_rotation_arc(glam::Vec3::X, d);
            [2.0 * q.w, 2.0 * q.x, 2.0 * q.y, 2.0 * q.z]
        },
        log_scales: [2.0f32.ln(), 0.8f32.ln(), 0.5f32.ln()], // 7.4×2.2×1.6 mm
        raw_opac: 40.0,
    }
}

fn build(scene: &Scene, device: &burn::tensor::Device) -> XRaySplats {
    XRaySplats::from_raw(
        scene.means.to_vec(),
        scene.rots.to_vec(),
        scene.log_scales.to_vec(),
        vec![scene.raw_opac],
        device,
    )
}

/// Moment loss `Σ proj·(x² + y²)` about the image center (pixels 0..IMG).
async fn moment_loss(scene: &Scene, device: &burn::tensor::Device) -> f32 {
    let splats = build(scene, device);
    let out = render_xray_forward(&splats, &cam(), glam::uvec2(IMG, IMG), 1.0).await;
    // Weight = (x - cx)² + (y - cy)² over the image grid.
    let (w, h) = (IMG as usize, IMG as usize);
    let cx = (IMG as f32 - 1.0) * 0.5;
    let cy = (IMG as f32 - 1.0) * 0.5;
    let mut wt = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            wt.push(dx * dx + dy * dy);
        }
    }
    let wten = Tensor::<2>::from_data(burn::tensor::TensorData::new(wt, [h, w]), device);
    let loss = out.mul(wten).sum();
    loss.into_scalar_async().await.expect("loss")
}

/// Analytic backward of the same moment loss.
async fn analytic_grad(scene: &Scene, device: &burn::tensor::Device) -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let splats = build(scene, device);
    let out = render_xray(splats.clone(), &cam(), glam::uvec2(IMG, IMG), 1.0, false, 0.0).await;
    let (w, h) = (IMG as usize, IMG as usize);
    let cx = (IMG as f32 - 1.0) * 0.5;
    let cy = (IMG as f32 - 1.0) * 0.5;
    let mut wt = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            wt.push(dx * dx + dy * dy);
        }
    }
    let wten = Tensor::<2>::from_data(burn::tensor::TensorData::new(wt, [h, w]), device);
    let grads = out.img.mul(wten).sum().backward();

    let read = async move |t: Tensor<1>| -> f32 {
        t.into_scalar_async().await.expect("g")
    };
    let tg = splats.transforms.grad(&grads).expect("transforms grad");
    let dmean = tg.clone().slice(s![0, 0..3]).squeeze_dim::<1>(0);
    let dquat = tg.clone().slice(s![0, 3..7]).squeeze_dim::<1>(0);
    let dscale = tg.slice(s![0, 7..10]).squeeze_dim::<1>(0);
    // quat grads are w.r.t. the RAW (unnormalized) quat? Compute the
    // numerical side with the same unnormalized quat so they compare.
    let dmean: Vec<f32> = dmean.to_data_async().await.expect("d").to_vec().unwrap();
    let dquat: Vec<f32> = dquat.to_data_async().await.expect("d").to_vec().unwrap();
    let dscale: Vec<f32> = dscale.to_data_async().await.expect("d").to_vec().unwrap();
    let dopac = splats.raw_opacities.grad(&grads).expect("opac grad");
    let dopac = read(dopac.slice([0])).await;
    (dmean, dquat, dscale, dopac)
}

fn perturb_scene(scene: &Scene, lane: usize, comp: usize, delta: f32) -> Scene {
    let mut s = Scene { ..*scene };
    match lane {
        0 => s.means[comp] += delta,
        1 => s.rots[comp] += delta,
        2 => s.log_scales[comp] += delta,
        3 => s.raw_opac += delta,
        _ => unreachable!(),
    }
    s
}

async fn num_grad(scene: &Scene, lane: usize, comp: usize, eps: f32, device: &burn::tensor::Device) -> f32 {
    let plus = moment_loss(&perturb_scene(scene, lane, comp, eps), device).await;
    let minus = moment_loss(&perturb_scene(scene, lane, comp, -eps), device).await;
    (plus - minus) / (2.0 * eps)
}

#[tokio::test]
async fn backward_matches_central_difference() {
    let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
    let adevice = device.clone().autodiff();
    let scene = base_scene();
    let eps = 8e-3f32;

    let (dmean, dquat, dscale, dopac) = analytic_grad(&scene, &adevice).await;

    let mut failures = 0usize;
    let mut total = 0usize;

    // Means
    for c in 0..3 {
        let ana = dmean[c];
        let num = num_grad(&scene, 0, c, eps, &device).await;
        total += 1;
        let rel = (ana - num).abs() / num.abs().max(1e-9);
        let ok = rel < 0.1 || (ana - num).abs() < 0.05;
        println!("  mean[{c}]  ana={ana:10.4} num={num:10.4} rel={rel:.3} {}", if ok { "✓" } else { "✗" });
        if !ok { failures += 1; }
    }
    // Scales
    for c in 0..3 {
        let ana = dscale[c];
        let num = num_grad(&scene, 2, c, eps, &device).await;
        total += 1;
        let rel = (ana - num).abs() / num.abs().max(1e-9);
        let ok = rel < 0.1 || (ana - num).abs() < 0.05;
        println!("  scale[{c}] ana={ana:10.4} num={num:10.4} rel={rel:.3} {}", if ok { "✓" } else { "✗" });
        if !ok { failures += 1; }
    }
    // Quat (unnormalized w,x,y,z)
    for c in 0..4 {
        let ana = dquat[c];
        let num = num_grad(&scene, 1, c, eps, &device).await;
        total += 1;
        let rel = (ana - num).abs() / num.abs().max(1e-9);
        let ok = rel < 0.15 || (ana - num).abs() < 0.05;
        println!("  quat[{c}]  ana={ana:10.4} num={num:10.4} rel={rel:.3} {}", if ok { "✓" } else { "✗" });
        if !ok { failures += 1; }
    }
    // Opacity
    {
        let ana = dopac;
        let num = num_grad(&scene, 3, 0, eps, &device).await;
        total += 1;
        let rel = (ana - num).abs() / num.abs().max(1e-9);
        let ok = rel < 0.1 || (ana - num).abs() < 0.05;
        println!("  opac    ana={ana:10.4} num={num:10.4} rel={rel:.3} {}", if ok { "✓" } else { "✗" });
        if !ok { failures += 1; }
    }

    println!("\n  {}/{} passed", total - failures, total);
    assert!(
        failures == 0,
        "{failures}/{total} backward gradient checks failed (quat/scale chain suspect)"
    );
}
