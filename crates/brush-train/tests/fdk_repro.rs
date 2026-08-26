//! Minimal repro: does `drr_forward` → `wrap_wgpu_float` → `into_data_async`
//! corrupt the cubecl/wgpu memory manager when run in a loop (the pattern the
//! trainer's per-step FDK DRR readback uses)?

use brush_drr::{DrrOps, DrrSettings};
use brush_render::burn_glue::{unwrap_wgpu_float, wrap_wgpu_float};
use brush_render::camera::Camera;
use burn::tensor::{Tensor, TensorData};

fn camera() -> Camera {
    Camera::new(
        glam::vec3(0.0, 760.0, 0.0),
        glam::Quat::IDENTITY,
        0.5,
        0.4,
        glam::vec2(0.5, 0.5),
        brush_render::kernels::camera_model::CameraModel::Pinhole,
    )
}

#[tokio::test]
async fn drr_wrap_readback_loop() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let vol_x = 64usize;
    let vol_y = 64usize;
    let vol_z = 48usize;
    let mut v: Vec<f32> = (0..vol_x * vol_y * vol_z)
        .map(|i| ((i % 7) as f32) * 0.001)
        .collect();
    let volume = Tensor::<3>::from_data(
        TensorData::new::<f32, _>(v.clone(), [vol_x, vol_y, vol_z]),
        &device,
    );
    let cam = camera();
    let img = glam::uvec2(128, 128);
    let settings = DrrSettings::new(&cam, img.x, img.y, vol_x as u32, vol_y as u32, vol_z as u32, 128, 118.6, 118.6, 84.7, 1.0, 0.0);

    // Pattern A: wrap + readback (what xray_train::fdk.drr_for does).
    for i in 0..100 {
        let ft = unwrap_wgpu_float(volume.clone());
        let out = <brush_cube::MainBackend as DrrOps>::drr_forward(&settings, ft).await;
        let t = wrap_wgpu_float::<2>(out);
        let data = t.into_data_async().await.expect("readback");
        let s: f32 = data.as_slice::<f32>().unwrap().iter().sum();
        if i % 25 == 0 {
            println!("A[{i}] sum={s:.3}");
        }
    }

    // Pattern B: wrap + tensor op (what fit_volume does).
    let out = <brush_cube::MainBackend as DrrOps>::drr_forward(
        &settings,
        unwrap_wgpu_float(volume.clone()),
    )
    .await;
    let t = wrap_wgpu_float::<2>(out);
    let s: f32 = t.sum().into_scalar::<f32>();
    println!("B sum={s:.3}");

    // Pattern C: full trainer pattern — autodiff render (signed) + per-step
    // FDK DRR (wrap + readback + from_data to AD) + loss.backward().
    use brush_xray::XRaySplats;
    use brush_xray_bwd::render_xray;
    let device_ad = device.clone().autodiff();
    let n = 128usize;
    let means: Vec<f32> = (0..n * 3)
        .map(|i| match i % 3 {
            0 => (i as f32 * 0.0007) - 0.05,
            1 => (i as f32 * 0.0009) - 0.05,
            _ => (i as f32 * 0.0005) - 0.05,
        })
        .collect();
    let rots: Vec<f32> = (0..n * 4).map(|_| 1.0).collect();
    let log_scales: Vec<f32> = (0..n * 3).map(|_| -1.5).collect();
    let raw_opac: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.005 } else { -0.005 }).collect();
    let splats = XRaySplats::from_raw(means, rots, log_scales, raw_opac, &device_ad);
    for i in 0..100 {
        let out = render_xray(splats.clone(), &cam, img, 1.0, true).await;
        let mut proj = out.img;
        let ft = unwrap_wgpu_float(volume.clone());
        let drr = <brush_cube::MainBackend as DrrOps>::drr_forward(&settings, ft).await;
        let wrapped = wrap_wgpu_float::<2>(drr);
        let data = wrapped.into_data_async().await.expect("drr readback");
        let fdk = Tensor::<2>::from_data(data, &device_ad);
        proj = proj.add(fdk);
        let intensity = (-proj.clamp(1e-3, 14.0)).exp();
        let loss = intensity.mean();
        loss.backward();
        if i % 25 == 0 {
            println!("C[{i}] loss={}", loss.into_scalar::<f32>());
        }
    }
    println!("OK");
}

/// Pattern D: the actual FdkPrior::drr_for (custom AD op, no readback) in a
/// full render + backward loop — the real trainer path.
#[tokio::test]
async fn fdk_prior_custom_op_loop() {
    let device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let device_ad = device.clone().autodiff();
    let vol_x = 64usize;
    let vol_y = 64usize;
    let vol_z = 48usize;
    let v: Vec<f32> = (0..vol_x * vol_y * vol_z)
        .map(|i| ((i % 7) as f32) * 0.001)
        .collect();
    let prior = brush_train::fdk_prior::FdkPrior::new(
        v,
        vol_x,
        vol_y,
        vol_z,
        118.6,
        118.6,
        84.7,
        128,
        1.0,
        0.0,
        &device_ad,
    );
    use brush_xray::XRaySplats;
    use brush_xray_bwd::render_xray;
    let cam = camera();
    let img = glam::uvec2(128, 128);
    let n = 128usize;
    let means: Vec<f32> = (0..n * 3)
        .map(|i| match i % 3 {
            0 => (i as f32 * 0.0007) - 0.05,
            1 => (i as f32 * 0.0009) - 0.05,
            _ => (i as f32 * 0.0005) - 0.05,
        })
        .collect();
    let rots: Vec<f32> = (0..n * 4).map(|_| 1.0).collect();
    let log_scales: Vec<f32> = (0..n * 3).map(|_| -1.5).collect();
    let raw_opac: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.005 } else { -0.005 }).collect();
    let splats = XRaySplats::from_raw(means, rots, log_scales, raw_opac, &device_ad);
    for i in 0..200 {
        let out = render_xray(splats.clone(), &cam, img, 1.0, true).await;
        let mut proj = out.img;
        let fdk = prior.drr_for(&cam, img).await;
        proj = proj.add(fdk);
        let intensity = (-proj.clamp(1e-3, 14.0)).exp();
        let loss = intensity.mean();
        loss.backward();
        if i % 50 == 0 {
            println!("D[{i}] loss={}", loss.into_scalar::<f32>());
        }
    }
    println!("OK");
}
