//! End-to-end autodiff test: `brush_xray_bwd::render_xray` produces an
//! image that backprops through burn's autodiff graph to `transforms`
//! and `raw_opacities`, and those grads match the raw
//! `xray_bwd_pipeline` (already validated against R2-Gaussian).

use brush_cube::MainBackendBase;
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use burn::backend::ops::FloatTensorOps;
use burn::tensor::TensorData;
use brush_xray::XRaySplats;
use brush_xray_bwd::render_xray;
use brush_xray_bwd::xray_bwd_pipeline;
use burn_wgpu::WgpuDevice;

#[tokio::test]
async fn autodiff_grads_match_pipeline() {
    let base: WgpuDevice = brush_cube::test_helpers::test_device().await;
    let device = burn::tensor::Device::from(base.clone()).autodiff();

    // Deterministic scene (identical to the R2 golden reference: camera at
    // (0,0,-5), identity rot, 64x64, 32 splats, equal scales, sigmoid opac).
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );
    let img_size = glam::uvec2(64, 64);

    let n = 32usize;
    let mut means = vec![0.0f32; n * 3];
    let mut g = 1234u64;
    for v in means.iter_mut() {
        g = g.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((g >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    }
    // Anisotropic scales + random rotations (same LCG, continued) so the
    // rotation-sensitive covariance path is exercised.
    let mut rots = vec![0.0f32; n * 4];
    let mut log_scales = vec![0.0f32; n * 3];
    for v in rots.iter_mut().chain(log_scales.iter_mut()) {
        g = g.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((g >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    }
    let raw_opac = vec![2.0f32; n];

    let splats = XRaySplats::from_raw(
        means.clone(),
        rots.clone(),
        log_scales.clone(),
        raw_opac.clone(),
        &device,
    );

    // Differentiable forward.
    let out = render_xray(splats.clone(), &cam, img_size, 1.0).await;
    assert_eq!(out.img.dims(), [64, 64]);
    let loss = out.img.sum();
    let grads = loss.backward();

    let t_grad = splats.transforms.grad(&grads).expect("transforms grad");
    let t_data = t_grad.into_data_async().await.expect("t grad read");
    let ad_v_transforms: Vec<f32> = t_data.into_vec::<f32>().unwrap();
    let o_grad = splats.raw_opacities.grad(&grads).expect("opac grad");
    let o_data = o_grad.into_data_async().await.expect("o grad read");
    let ad_v_raw: Vec<f32> = o_data.into_vec::<f32>().unwrap();

    // Reference: run the raw pipeline with dL/dout = 1 (L = sum(C)).
    let mut transforms_data = Vec::with_capacity(n * 10);
    for i in 0..n {
        transforms_data.extend_from_slice(&means[i * 3..i * 3 + 3]);
        transforms_data.extend_from_slice(&rots[i * 4..i * 4 + 4]);
        transforms_data.extend_from_slice(&log_scales[i * 3..i * 3 + 3]);
    }
    let transforms_ft = MainBackendBase::float_from_data(
        TensorData::new::<f32, _>(transforms_data, [n, 10]),
        &base,
    );
    let raw_opac_ft =
        MainBackendBase::float_from_data(TensorData::new::<f32, _>(raw_opac, [n]), &base);
    let v_output_ft =
        MainBackendBase::float_from_data(TensorData::ones::<f32, _>([64, 64]), &base);

    let ref_grads = xray_bwd_pipeline(&cam, img_size, transforms_ft, raw_opac_ft, 1.0, v_output_ft)
        .await;
    let ref_t = MainBackendBase::float_into_data(ref_grads.v_transforms)
        .await
        .expect("ref v_transforms");
    let ref_t = ref_t.as_slice::<f32>().unwrap();
    let ref_o = MainBackendBase::float_into_data(ref_grads.v_raw_opac)
        .await
        .expect("ref v_raw_opac");
    let ref_o = ref_o.as_slice::<f32>().unwrap();

    // Both `ad_v_transforms` (grad of the [N,10] transforms param) and the
    // reference `ref_t` are interleaved `[N,10]` = means(3)+quats(4)+
    // log_scales(3) per splat — compare them directly.
    let rel_means = max_rel_diff(
        &ad_v_transforms[..n * 3],
        &ref_t[..n * 3],
        "ad dL/dmeans",
    );
    let rel_log = max_rel_diff(
        &ad_v_transforms[n * 7..n * 10],
        &ref_t[n * 7..n * 10],
        "ad dL/dlog_scales",
    );
    let rel_opac = max_rel_diff(&ad_v_raw, ref_o, "ad dL/draw_opac");
    let ad_quats: Vec<f32> =
        ad_v_transforms.chunks_exact(10).flat_map(|c| c[3..7].to_vec()).collect();
    let ref_quats: Vec<f32> =
        ref_t.chunks_exact(10).flat_map(|c| c[3..7].to_vec()).collect();
    let rel_quats = max_rel_diff(&ad_quats, &ref_quats, "ad dL/dquats");

    assert!(rel_means < 1.0e-2, "ad dL/dmeans rel {rel_means}");
    assert!(rel_log < 1.0e-2, "ad dL/dlog_scales rel {rel_log}");
    assert!(rel_opac < 1.0e-2, "ad dL/draw_opac rel {rel_opac}");
    assert!(rel_quats < 1.0e-2, "ad dL/dquats rel {rel_quats}");
}

fn max_rel_diff(ours: &[f32], ref_: &[f32], label: &str) -> f32 {
    let mut max_abs_ref = 0.0f32;
    let mut max_diff = 0.0f32;
    for (a, b) in ours.iter().zip(ref_.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        if b.abs() > max_abs_ref {
            max_abs_ref = b.abs();
        }
    }
    let rel = max_diff / f32::max(1.0e-4f32, max_abs_ref);
    println!("{label}: max_abs_diff={max_diff:.6} max_abs_ref={max_abs_ref:.6} rel={rel:.6}");
    rel
}
