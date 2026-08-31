//! 目视验证: 10 个各向异性高斯 splat 沿 C 臂 alpha/beta 扫描的多视角投影。
//!
//! 构造一条圆弧血管样路径, 每个 splat 的拉长轴沿路径切线 (模拟血管走向),
//! 另加几个沿固定世界轴拉长的对照 splat。用 `build_camera` 按
//! alpha ∈ [-90°, 90°] (181 帧) × beta ∈ [-90°, 90°] (181 帧) 渲染。
//!
//! 输出 (均写入 <out> 前缀, NRRD 多帧, z 轴 = 帧索引, 帧内垂直翻转显示正立):
//! - `<out>_alpha.nrrd`  [181, H, W]: alpha 扫描, beta=0
//! - `<out>_beta.nrrd`   [181, H, W]: beta 扫描, alpha=0
//! - `<out>_montage.png`: 完整网格缩略图 (可选目视)
//!
//! 用法: gs_sweep <out_prefix>
use std::path::PathBuf;

use brush_cube::test_helpers::test_device;
use brush_dataset::config::XRayOrientation;
use brush_dataset::{build_camera};
use brush_dicom::meta::CArmGeometry;
use brush_train::xray_eval::save_gray_nrrd_f32_stack;
use brush_xray::{XRaySplats, render_xray_forward};
use burn::tensor::TensorData;
use clap::Parser;

/// RXA_chest C-arm geometry.
fn arm_geom() -> CArmGeometry {
    CArmGeometry {
        sdd: 1200.0,
        sod: 760.0,
        height: 256,
        width: 256,
        delx: 0.616,
        dely: 0.616,
        x0: 0.0,
        y0: 0.0,
    }
}

const IMG: u32 = 256;
const STEPS: usize = 181;

/// Quaternion (w,x,y,z) rotating +X onto `dir`.
fn quat_align(dir: glam::Vec3) -> [f32; 4] {
    let d = dir.normalize();
    let x = glam::Vec3::X;
    let dot = x.dot(d);
    let q = if dot > 0.9999 {
        glam::Quat::IDENTITY
    } else if dot < -0.9999 {
        glam::Quat::from_axis_angle(glam::Vec3::Y, std::f32::consts::PI)
    } else {
        let axis = x.cross(d).normalize();
        glam::Quat::from_axis_angle(axis, dot.acos())
    };
    [q.w, q.x, q.y, q.z]
}

/// Build 10 splats: 8 along an arc (tangent-elongated, like a vessel) +
/// 2 axis-aligned controls (X-elongated and Z-elongated).
fn build_splats(device: &burn::tensor::Device) -> XRaySplats {
    let mut means = Vec::new();
    let mut rots = Vec::new();
    let mut scales = Vec::new();

    // Arc in the Y=0 plane: radius 40mm, angle 0..180° (semicircle in X-Z).
    let r = 40.0f32;
    for i in 0..8 {
        let a = std::f32::consts::PI * (i as f32) / 7.0;
        let p = glam::vec3(r * a.cos(), 0.0, r * a.sin());
        // Tangent = derivative of (r cos a, 0, r sin a) = (-sin a, 0, cos a).
        let t = glam::vec3(-a.sin(), 0.0, a.cos());
        means.extend_from_slice(&[p.x, p.y, p.z]);
        rots.extend_from_slice(&quat_align(t));
        // Long axis 6mm along tangent, 1.2mm × 1.2mm cross-section.
        scales.extend_from_slice(&[6.0f32.ln(), 1.2f32.ln(), 1.2f32.ln()]);
    }
    // Control 1: X-elongated at a distinct spot.
    means.extend_from_slice(&[-55.0, 0.0, -55.0]);
    rots.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    scales.extend_from_slice(&[6.0f32.ln(), 1.2f32.ln(), 1.2f32.ln()]);
    // Control 2: Z-elongated.
    means.extend_from_slice(&[-55.0, 0.0, 55.0]);
    rots.extend_from_slice(&quat_align(glam::Vec3::Z));
    scales.extend_from_slice(&[6.0f32.ln(), 1.2f32.ln(), 1.2f32.ln()]);

    let opac: Vec<f32> = (0..10).map(|_| 25.0f32).collect();
    XRaySplats::from_raw(means, rots, scales, opac, device)
}

/// Render one frame and return the raw proj image `[H, W]` f32 (line integral).
async fn render_frame(
    splats: &XRaySplats,
    alpha: f64,
    beta: f64,
    geom: &CArmGeometry,
) -> Vec<f32> {
    let cam = build_camera(alpha, beta, geom.sod, XRayOrientation::Ap, geom);
    let proj = render_xray_forward(splats, &cam, glam::uvec2(IMG, IMG), 1.0).await;
    proj.to_data_async().await.expect("read").to_vec().unwrap()
}

/// Pack a `[N, H, W]` f32 stack into NRRD (z = frame index), values windowed
/// by `proj/window` then clipped — keep raw f32 but window for visibility.
fn write_stack_nrrd(path: &PathBuf, stack: &[f32], n: usize, h: usize, w: usize) -> anyhow::Result<()> {
    // Reuse the standard stack writer (expects [n, h, w] TensorData).
    let data = TensorData::new(stack.to_vec(), [n, h, w]);
    save_gray_nrrd_f32_stack(path, &data)?;
    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Output prefix (e.g. /tmp/sweep) — appends _alpha.nrrd / _beta.nrrd / _montage.png.
    out: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device: burn::tensor::Device = test_device().await.into();
    let splats = build_splats(&device);
    let geom = arm_geom();
    let (w, h) = (IMG as usize, IMG as usize);
    let deg = std::f64::consts::PI / 180.0;

    let alphas: Vec<f64> = (-90..=90).map(|d| d as f64 * deg).collect();
    let betas: Vec<f64> = (-90..=90).map(|d| d as f64 * deg).collect();

    // ---- alpha sweep (beta=0) -----------------------------------------
    let mut alpha_stack = Vec::with_capacity(STEPS * h * w);
    for a in &alphas {
        alpha_stack.extend(render_frame(&splats, *a, 0.0, &geom).await);
    }
    write_stack_nrrd(&args.out.with_file_name(format!(
        "{}_alpha.nrrd",
        args.out.file_name().unwrap().to_string_lossy()
    )), &alpha_stack, STEPS, h, w)?;

    // ---- beta sweep (alpha=0) -----------------------------------------
    let mut beta_stack = Vec::with_capacity(STEPS * h * w);
    for b in &betas {
        beta_stack.extend(render_frame(&splats, 0.0, *b, &geom).await);
    }
    write_stack_nrrd(&args.out.with_file_name(format!(
        "{}_beta.nrrd",
        args.out.file_name().unwrap().to_string_lossy()
    )), &beta_stack, STEPS, h, w)?;

    println!("saved NRRD stacks under {}", args.out.display());
    println!(
        "  {}_alpha.nrrd: [alpha={STEPS}, {h}, {w}], beta=0",
        args.out.file_name().unwrap().to_string_lossy()
    );
    println!(
        "  {}_beta.nrrd:  [beta={STEPS}, {h}, {w}], alpha=0",
        args.out.file_name().unwrap().to_string_lossy()
    );
    Ok(())
}