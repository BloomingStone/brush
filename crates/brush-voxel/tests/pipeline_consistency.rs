//! Pipeline-consistency test: `brush-xray` (splats → 2D) must match
//! `brush-voxel` → `brush-drr` (splats → density volume → 2D) for the same
//! gaussian point cloud.
//!
//! Density convention: both pipelines consume the **same raw logits** and
//! apply the same in-kernel activation `μ = MU_WATER·silu(raw)` (brush-xray
//! unsigned mode == brush-voxel unsigned mode), so the comparison is
//! activation-exact — no host-side preprocessing on either side.
//!
//! Layout: both the voxelizer output and the DRR kernel are x-major
//! (`flat(x,y,z) = x*(ny*nz) + y*nz + z`), so the volume feeds the DRR
//! with zero transpose. The mathematical equivalence holds in the limit of
//! fine voxels / march steps and small splats; the tolerances absorb the
//! trilinear discretization, the 3σ truncation (~1% tails) and the
//! rasterizer's plane-parallel approximation.


use brush_cube::MU_WATER;
use brush_drr::{DrrOps, DrrSettings};
use brush_render::burn_glue::{unwrap_wgpu_float, wrap_wgpu_float};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_voxel::{VoxelSettings, voxelize_forward};
use brush_xray::{XRaySplats, render_xray_forward};
use burn::tensor::Device;

/// Deterministic scene (CPU RNG, same style as the R2 reference generators).
/// Returns (means, quats, log_scales, raw_opacities).
fn scene(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rng = |s: u64| {
        // xorshift64 — deterministic, no external deps.
        let mut x = s | 1;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32
        }
    };
    let mut r_m = rng(seed);
    let mut r_s = rng(seed + 0x9E3779B97F4A7C15);
    let mut r_q = rng(seed + 0xC2B2AE3D27D4EB4F);
    let mut r_o = rng(seed + 0x165667B19E3779F9);

    let mut means = Vec::with_capacity(n * 3);
    let mut log_scales = Vec::with_capacity(n * 3);
    let mut quats = Vec::with_capacity(n * 4);
    let mut raw_opac = Vec::with_capacity(n);
    for _ in 0..n {
        // positions inside ±8mm (volume half-extent 20mm; 3σ ≤ 6mm).
        for _ in 0..3 {
            means.push((r_m() * 2.0 - 1.0) * 8.0);
        }
        // σ ∈ [0.74, 2.0]mm — ≥2.4 voxels at dV=0.3125mm (trilinear ok).
        for _ in 0..3 {
            log_scales.push((r_s() * 1.0 - 0.3).ln().exp().ln()); // placeholder, replaced below
        }
        let mut q = [r_q() * 2.0 - 1.0; 4];
        let nrm = (q.iter().map(|v| v * v).sum::<f32>()).sqrt().max(1e-6);
        for v in q.iter_mut() {
            *v /= nrm;
        }
        quats.extend_from_slice(&q);
        // raw ∈ [1,4] → μ = MU_WATER·silu(raw) ∈ [1.46e-3, 7.9e-3] mm⁻¹.
        raw_opac.push(1.0 + r_o() * 3.0);
    }
    // log_scales: σ = exp(ls) ∈ [0.74, 2.0].
    let ls = |r: f32| (0.74f32 * (2.0f32 / 0.74).powf(r)).ln();
    for i in 0..n {
        for a in 0..3 {
            log_scales[i * 3 + a] = ls(r_s());
        }
    }
    (means, quats, log_scales, raw_opac)
}

fn std_cam() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -100.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

/// Per-pixel relative-error stats on pixels with `|ref| > floor`.
/// Returns (mean_rel, p99_rel, max_rel, max_abs, max_ref).
fn rel_stats(ours: &[f32], reference: &[f32], floor_frac: f32) -> (f32, f32, f32, f32, f32) {
    let max_ref = reference.iter().cloned().fold(0.0f32, f32::max);
    let floor = floor_frac * max_ref;
    let mut rels: Vec<f32> = Vec::new();
    let mut max_abs = 0.0f32;
    for (a, b) in ours.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        if b.abs() > floor {
            rels.push(d / b.abs());
        }
    }
    rels.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = rels.len().max(1);
    let mean = rels.iter().sum::<f32>() / n as f32;
    let p99 = rels[((n as f32 * 0.99) as usize).min(n - 1)];
    let max = rels.last().copied().unwrap_or(0.0);
    (mean, p99, max, max_abs, max_ref)
}

#[tokio::test]
async fn xray_matches_voxel_drr() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();

    let (means, quats, log_scales, raw_opac) = scene(64, 1234);

    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);

    // ---- Pipeline A: brush-xray (raw logits, unsigned → μ) ----
    let splats_xray = XRaySplats::from_raw(means.clone(), quats.clone(), log_scales.clone(), raw_opac.clone(), &device);
    let proj_xray = render_xray_forward(&splats_xray, &cam, img_size, 1.0).await;
    let img_a = proj_xray.to_data_async().await.expect("readback");
    let a: Vec<f32> = img_a.as_slice::<f32>().expect("f32").to_vec();

    // ---- Pipeline B: brush-voxel (Preactivated μ) → brush-drr ----
    const NV: u32 = 128;
    const EXTENT: f32 = 40.0; // mm → dV = 0.3125mm
    let settings = VoxelSettings::new(
        glam::uvec3(NV, NV, NV),
        glam::vec3(EXTENT, EXTENT, EXTENT),
        glam::Vec3::ZERO,
    );
    let splats_vox = XRaySplats::from_raw(means, quats, log_scales, raw_opac, &device);
    let vol = voxelize_forward(&splats_vox, &settings).await; // [NV,NV,NV], x-major

    let drr_settings = DrrSettings::new(
        &cam,
        img_size.x,
        img_size.y,
        NV,
        NV,
        NV,
        512, // steps; dt = 2·rx/steps = 0.078mm ≪ σ
        0.5 * EXTENT,
        0.5 * EXTENT,
        0.5 * EXTENT,
        1.0, // scale
        0.0, // bias
    );
    let vol_ft = unwrap_wgpu_float(vol);
    let proj = <brush_cube::MainBackend as DrrOps>::drr_forward(&drr_settings, vol_ft).await;
    let proj = wrap_wgpu_float::<2>(proj);
    let img_b = proj.to_data_async().await.expect("readback");
    let b: Vec<f32> = img_b.as_slice::<f32>().expect("f32").to_vec();

    assert_eq!(a.len(), b.len(), "image sizes must match");

    // Sanity: both projections carry comparable total density.
    let sum_a: f32 = a.iter().sum();
    let sum_b: f32 = b.iter().sum();
    println!("xray sum={sum_a:.4}  voxel+drr sum={sum_b:.4}  ratio={:.4}", sum_b / sum_a.max(1e-12));

    let (mean, p99, max, max_abs, max_ref) = rel_stats(&b, &a, 0.02);
    println!("xray vs voxel+drr: mean_rel={mean:.4} p99_rel={p99:.4} max_rel={max:.4} max_abs={max_abs:.6} (max_ref={max_ref:.6})");

    // Tolerances absorb: voxel trilinear discretization (dV/σ ≥ 2.4 voxels
    // → ≤ ~2% near peaks), 3σ truncation tails (~1%), rasterizer
    // plane-parallel approximation (small for σ ≤ 2mm at SOD=100mm).
    assert!(mean < 0.02, "mean relative error {mean} too large");
    assert!(p99 < 0.08, "p99 relative error {p99} too large");
    assert!(max < 0.15, "max relative error {max} too large");
    assert!((sum_b - sum_a).abs() / sum_a < 0.10, "total density drift too large");
}

/// Anchor test: a single isotropic splat at the volume center. Checks the
/// voxelizer reproduces the amplitude (volume peak ≈ μ) and the DRR peak
/// matches the xray peak (both sample the same principal ray).
#[tokio::test]
async fn single_splat_anchor() {
    let device: Device = brush_cube::test_helpers::test_device().await.into();

    const MU_TARGET: f32 = 0.004; // mm⁻¹ (activated density target)
    const SIGMA: f32 = 1.5; // mm
    // raw logit so that MU_WATER·silu(raw) = MU_TARGET — fed to BOTH
    // pipelines; each activates in-kernel.
    let raw = brush_cube::inverse_silu(MU_TARGET / MU_WATER);

    let means = vec![0.0f32, 0.0, 0.0];
    let quats = vec![1.0f32, 0.0, 0.0, 0.0];
    let log_scales = vec![SIGMA.ln(), SIGMA.ln(), SIGMA.ln()];

    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);

    // ---- xray ----
    let splats_x = XRaySplats::from_raw(means.clone(), quats.clone(), log_scales.clone(), vec![raw], &device);
    let img_a = render_xray_forward(&splats_x, &cam, img_size, 1.0)
        .await
        .to_data_async()
        .await
        .expect("readback");
    let a: Vec<f32> = img_a.as_slice::<f32>().expect("f32").to_vec();

    // ---- voxel (preactivated μ) + drr ----
    const NV: u32 = 128;
    const EXTENT: f32 = 40.0;
    let settings = VoxelSettings::new(
        glam::uvec3(NV, NV, NV),
        glam::vec3(EXTENT, EXTENT, EXTENT),
        glam::Vec3::ZERO,
    );
    let splats_v = XRaySplats::from_raw(means, quats, log_scales, vec![raw], &device);
    let vol = voxelize_forward(&splats_v, &settings).await;
    let vol_data = vol.to_data_async().await.expect("readback");
    let vol_vec: Vec<f32> = vol_data.as_slice::<f32>().expect("f32").to_vec();

    // Amplitude anchor: the splat sits at the origin; the nearest voxel
    // centers are 0.125mm away (dV/2) → value ≈ μ·exp(-½·(0.125/1.5)²).
    let expected_peak = MU_TARGET * (-0.5 * (0.125 / SIGMA).powi(2)).exp();
    let peak = vol_vec.iter().cloned().fold(0.0f32, f32::max);
    let peak_err = (peak - expected_peak).abs() / expected_peak;
    println!("voxel peak={peak:.6} expected≈{expected_peak:.6} rel_err={peak_err:.4}");
    assert!(peak_err < 0.02, "voxel amplitude anchor failed: {peak_err}");

    // ---- DRR ----
    let drr_settings = DrrSettings::new(
        &cam,
        img_size.x,
        img_size.y,
        NV,
        NV,
        NV,
        512,
        0.5 * EXTENT,
        0.5 * EXTENT,
        0.5 * EXTENT,
        1.0,
        0.0,
    );
    let vol_ft = unwrap_wgpu_float(vol);
    let proj = <brush_cube::MainBackend as DrrOps>::drr_forward(&drr_settings, vol_ft).await;
    let proj = wrap_wgpu_float::<2>(proj);
    let img_b = proj.to_data_async().await.expect("readback");
    let b: Vec<f32> = img_b.as_slice::<f32>().expect("f32").to_vec();

    // The splat sits on the principal ray: the DRR ray through the central
    // pixels integrates the gaussian along z → ~μ·σ·√(2π), and the xray
    // pixel reports ρ·mu·exp(power) at integer pixels ±0.5px off-center.
    // Both must agree to the rasterizer-approximation tolerance.
    let (mean, p99, max, max_abs, max_ref) = rel_stats(&b, &a, 0.02);
    println!("single splat: mean_rel={mean:.4} p99={p99:.4} max={max:.4} max_abs={max_abs:.6} (max_ref={max_ref:.6})");
    assert!(mean < 0.05, "single-splat mean rel error {mean}");
    assert!(max < 0.12, "single-splat max rel error {max}");
}
