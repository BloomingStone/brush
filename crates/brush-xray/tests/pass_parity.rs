//! Forward vs Backward pass image equality: the training eval renders via
//! `XRayPass::Backward` (bwd bookkeeping), gs2volume via `XRayPass::Forward`.
//! The images must be identical; this guards against drift between the two.

use brush_cube::MainBackendBase;
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_xray::{XRayOps, XRayPass};
use burn::backend::ops::FloatTensorOps;
use burn::tensor::TensorData;
use burn_wgpu::WgpuDevice;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[tokio::test]
async fn forward_and_backward_passes_render_identically() {
    let device: WgpuDevice = brush_cube::test_helpers::test_device().await;

    // 大点云 + 部分不可见 splat (覆盖真实场景的 compact/hole 路径)。
    let n = 4096usize;
    let mut transforms = Vec::with_capacity(n * 10);
    let mut raw = Vec::with_capacity(n);
    for i in 0..n {
        let x = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
        let y = ((i * 104729) % 1000) as f32 / 500.0 - 1.0;
        let z = if i % 3 == 0 { 20.0 } else { ((i * 1299721) % 1000) as f32 / 500.0 - 1.0 };
        transforms.extend_from_slice(&[x, y, z, 1.0, 0.0, 0.0, 0.0, -0.3, -0.5, -0.7]);
        raw.push(1.0 + 0.001 * (i % 500) as f32);
    }
    let t = MainBackendBase::float_from_data(
        TensorData::new::<f32, _>(transforms, [n, 10]),
        &device,
    );
    let r = MainBackendBase::float_from_data(TensorData::new::<f32, _>(raw, [n]), &device);

    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );
    let img = glam::uvec2(128, 128);

    let fwd = MainBackendBase::render_xray(&cam, img, t.clone(), r.clone(), 1.0, false, XRayPass::Forward).await;
    let bwd = MainBackendBase::render_xray(&cam, img, t, r, 1.0, false, XRayPass::Backward).await;

    let a = MainBackendBase::float_into_data(fwd.out_img).await.unwrap();
    let b = MainBackendBase::float_into_data(bwd.out_img).await.unwrap();
    let a: Vec<f32> = a.as_slice::<f32>().unwrap().to_vec();
    let b: Vec<f32> = b.as_slice::<f32>().unwrap().to_vec();
    let d = max_diff(&a, &b);
    println!("forward vs backward pass: max diff = {d:.6e}");
    assert!(d < 1e-4, "Forward/Backward pass images drifted: {d}");
}
