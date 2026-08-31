//! Golden-reference backward comparison against R2-Gaussian's cone-beam
//! rasterizer (CUDA). The reference grads are `dL/dC` with `L = C.sum()`
//! w.r.t. (means, log_scales, raw_opac, quats) — see
//! `brush-xray/test_cases/generate_reference.py`.
//!
//! R2's forward and backward are internally consistent (both use the same
//! GLM column-major `J`); its `dL_dcov3D` is the symmetric-vector gradient
//! of the world covariance, and its `dL_dM` is the symmetric-adjoint
//! `(Vrk·M)·(D with diagonal doubled)`. Brush's `compute_cov2d_cone_bwd`
//! reproduces both conventions exactly, so this golden test passes for
//! non-identity quaternions and anisotropic scales.

use brush_cube::MainBackendBase;
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use burn::backend::ops::FloatTensorOps;
use burn::tensor::TensorData;
use brush_xray_bwd::xray_bwd_pipeline;
use safetensors::SafeTensors;
use burn_wgpu::WgpuDevice;


const REFERENCE: &[u8] =
    include_bytes!("../../brush-xray/test_cases/r2_xray_reference.safetensors");

fn f32_from_u8(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_f32(tensors: &SafeTensors, name: &str) -> Vec<f32> {
    let t = tensors.tensor(name).expect(name);
    assert_eq!(t.dtype(), safetensors::Dtype::F32, "{name} dtype");
    f32_from_u8(t.data())
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

#[tokio::test]
async fn matches_r2_cone_beam_backward() {
    let device: WgpuDevice = brush_cube::test_helpers::test_device().await;
    let tensors = SafeTensors::deserialize(REFERENCE).expect("deserialize reference");

    let means = read_f32(&tensors, "means");
    let log_scales = read_f32(&tensors, "log_scales");
    let quats = read_f32(&tensors, "quats");
    let raw_opac = read_f32(&tensors, "raw_opac");
    let n = means.len() / 3;

    // packed [N,10] = means(3) + quats(4) + log_scales(3).
    let mut transforms_data = Vec::with_capacity(n * 10);
    for i in 0..n {
        transforms_data.extend_from_slice(&means[i * 3..i * 3 + 3]);
        transforms_data.extend_from_slice(&quats[i * 4..i * 4 + 4]);
        transforms_data.extend_from_slice(&log_scales[i * 3..i * 3 + 3]);
    }

    let transforms_ft = MainBackendBase::float_from_data(
        TensorData::new::<f32, _>(transforms_data, [n, 10]),
        &device,
    );
    let raw_opac_ft = MainBackendBase::float_from_data(
        TensorData::new::<f32, _>(raw_opac.clone(), [n]),
        &device,
    );
    let v_output_ft =
        MainBackendBase::float_from_data(TensorData::ones::<f32, _>([64, 64]), &device);

    let cam_pos = read_f32(&tensors, "cam_pos");
    let fov_x = read_f32(&tensors, "fov_x")[0] as f64;
    let fov_y = read_f32(&tensors, "fov_y")[0] as f64;
    let cam = Camera::new(
        glam::vec3(cam_pos[0], cam_pos[1], cam_pos[2]),
        glam::Quat::IDENTITY,
        fov_x,
        fov_y,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );
    let img_size = glam::uvec2(64, 64);

    let grads = xray_bwd_pipeline(&cam, img_size, transforms_ft, raw_opac_ft, 1.0, 0.0, v_output_ft)
        .await;

    let v_transforms = MainBackendBase::float_into_data(grads.v_transforms)
        .await
        .expect("read v_transforms");
    let ours_t = v_transforms.as_slice::<f32>().expect("f32").to_vec();
    let v_raw = MainBackendBase::float_into_data(grads.v_raw_opac)
        .await
        .expect("read v_raw_opac");
    let ours_op = v_raw.as_slice::<f32>().expect("f32").to_vec();
    let ref_means = read_f32(&tensors, "grad_means");
    let ref_log_scales = read_f32(&tensors, "grad_log_scales");
    let ref_raw_opac = read_f32(&tensors, "grad_raw_opac");
    let ref_quats = read_f32(&tensors, "grad_quats");

    // `v_transforms` is interleaved `[N,10]` = means(3)+quats(4)+log_scales(3)
    // per splat; de-interleave into the reference's blocked layout.
    let ours_means: Vec<f32> = ours_t.chunks_exact(10).flat_map(|c| c[0..3].to_vec()).collect();
    let ours_quats: Vec<f32> = ours_t.chunks_exact(10).flat_map(|c| c[3..7].to_vec()).collect();
    let ours_log_scales: Vec<f32> =
        ours_t.chunks_exact(10).flat_map(|c| c[7..10].to_vec()).collect();

    let r_means = max_rel_diff(&ours_means, &ref_means, "dL/dmeans");
    let r_log_scales = max_rel_diff(&ours_log_scales, &ref_log_scales, "dL/dlog_scales");
    let r_opac = max_rel_diff(&ours_op, &ref_raw_opac, "dL/draw_opac");
    // With anisotropic scales + random rotations the quat grads are now
    // non-zero, so a plain relative tolerance applies (like the others).
    let r_quats = max_rel_diff(&ours_quats, &ref_quats, "dL/dquats");

    assert!(r_means < 1.0e-2, "dL/dmeans rel diff {r_means}");
    assert!(r_log_scales < 1.0e-2, "dL/dlog_scales rel diff {r_log_scales}");
    assert!(r_opac < 1.0e-2, "dL/draw_opac rel diff {r_opac}");
    assert!(r_quats < 1.0e-2, "dL/dquats rel diff {r_quats}");
}
