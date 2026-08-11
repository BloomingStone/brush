//! Golden-reference forward comparison against R2-Gaussian's voxelizer
//! (CUDA). The reference volume is `fields` `[nVoxel_x, nVoxel_y, nVoxel_z]`
//! — see `brush-voxel/test_cases/generate_reference.py`.

use brush_cube::MainBackendBase;
use burn::backend::ops::FloatTensorOps;
use burn::tensor::TensorData;
use brush_voxel::{VoxelOps, VoxelPass, VoxelSettings};
use safetensors::SafeTensors;
use burn_wgpu::WgpuDevice;

const REFERENCE: &[u8] =
    include_bytes!("../test_cases/r2_voxel_reference.safetensors");

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

fn read_i32(tensors: &SafeTensors, name: &str) -> Vec<i32> {
    let t = tensors.tensor(name).expect(name);
    assert_eq!(t.dtype(), safetensors::Dtype::I32, "{name} dtype");
    t.data()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::test]
async fn matches_r2_voxel_forward() {
    let device: WgpuDevice = brush_cube::test_helpers::test_device().await;
    let tensors = SafeTensors::deserialize(REFERENCE).expect("deserialize reference");

    let means = read_f32(&tensors, "means");
    let log_scales = read_f32(&tensors, "log_scales");
    let quats = read_f32(&tensors, "quats");
    let raw_opac = read_f32(&tensors, "raw_opac");
    let ref_vol = read_f32(&tensors, "out_vol");
    let n_voxel = read_i32(&tensors, "nVoxel");
    let s_voxel = read_f32(&tensors, "sVoxel");
    let center = read_f32(&tensors, "center");

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

    let settings = VoxelSettings::new(
        glam::uvec3(n_voxel[0] as u32, n_voxel[1] as u32, n_voxel[2] as u32),
        glam::vec3(s_voxel[0], s_voxel[1], s_voxel[2]),
        glam::vec3(center[0], center[1], center[2]),
    );

    let out = MainBackendBase::voxelize(&settings, transforms_ft, raw_opac_ft, VoxelPass::Forward)
        .await;

    let ours = MainBackendBase::float_into_data(out.out_volume)
        .await
        .expect("read out_volume");
    let ours = ours.as_slice::<f32>().expect("f32").to_vec();

    assert_eq!(ours.len(), ref_vol.len(), "volume element count");

    let mut max_abs_diff = 0.0f32;
    let mut max_abs_ref = 0.0f32;
    for (a, b) in ours.iter().zip(ref_vol.iter()) {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        if b.abs() > max_abs_ref {
            max_abs_ref = b.abs();
        }
    }
    let rel = max_abs_diff / f32::max(1.0e-6f32, max_abs_ref);
    println!(
        "voxel forward: max_abs_diff={max_abs_diff:.6} max_abs_ref={max_abs_ref:.6} rel={rel:.6}"
    );

    // 2e-4 tolerance like the rasterizer golden test.
    assert!(rel < 2.0e-4, "voxel forward rel diff {rel}");
}
