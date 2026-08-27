//! Regression test for the g2c compaction bug (2026-08-28): the preprocess
//! kernel used to write `global_from_presort_gid[idx] = idx` at the ORIGINAL
//! index, leaving holes at invisible splats' positions. The `[0..num_visible)`
//! slice then contained zeros that `project_visible` resolved to splat 0,
//! replaying splat 0's lanes into hundreds of compact slots — a ~440× density
//! spike at splat 0's location. All earlier tests had every splat visible, so
//! the identity mapping was exact and the bug never surfaced.
//!
//! Here half the splats sit outside the grid; the volume must equal the one
//! produced from the in-grid splats alone.

use brush_cube::MainBackendBase;
use burn::backend::ops::FloatTensorOps;
use burn::tensor::TensorData;
use brush_voxel::{VoxelOps, VoxelPass, VoxelSettings};
use burn_wgpu::WgpuDevice;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[tokio::test]
async fn invisible_splats_do_not_replay_first_visible() {
    let device: WgpuDevice = brush_cube::test_helpers::test_device().await;

    let n = 64usize;
    // Half the splats inside the grid (z in [-1,1]), half far outside (z=+50).
    let mut means = Vec::with_capacity(n * 3);
    for i in 0..n {
        let z = if i % 2 == 0 { (i as f32 - 32.0) * 0.1 } else { 50.0 };
        means.extend_from_slice(&[(i % 7) as f32 * 0.3 - 1.0, (i % 5) as f32 * 0.3 - 1.0, z]);
    }
    let mut rots = vec![0.0f32; n * 4];
    for i in 0..n {
        rots[i * 4] = 1.0;
    }
    let log_scales: Vec<f32> = vec![-0.5; n * 3]; // σ = 0.6mm
    // 不同 raw 值: 若 splat 0 被重放, 其振幅会出现在错误位置。
    let raw: Vec<f32> = (0..n).map(|i| 0.5 + 0.02 * i as f32).collect();

    let settings = VoxelSettings::new(
        glam::uvec3(32, 32, 32),
        glam::vec3(4.0, 4.0, 4.0), // z ∈ [-2,2] → 半数 splat (z=50) 在网格外
        glam::Vec3::ZERO,
    );

    let voxelize_set = |subset: Option<&[usize]>| {
        let idxs: Vec<usize> = match subset {
            Some(s) => s.to_vec(),
            None => (0..n).collect(),
        };
        let mut t = Vec::with_capacity(idxs.len() * 10);
        let mut r = Vec::with_capacity(idxs.len());
        for &i in &idxs {
            t.extend_from_slice(&means[i * 3..i * 3 + 3]);
            t.extend_from_slice(&rots[i * 4..i * 4 + 4]);
            t.extend_from_slice(&log_scales[i * 3..i * 3 + 3]);
            r.push(raw[i]);
        }
        let t_ft = MainBackendBase::float_from_data(
            TensorData::new::<f32, _>(t, [idxs.len(), 10]),
            &device,
        );
        let r_ft = MainBackendBase::float_from_data(
            TensorData::new::<f32, _>(r, [idxs.len()]),
            &device,
        );
        MainBackendBase::voxelize(&settings, t_ft, r_ft, VoxelPass::Forward)
    };

    let inside: Vec<usize> = (0..n).filter(|&i| i % 2 == 0).collect();

    let full = voxelize_set(None).await;
    let sub = voxelize_set(Some(&inside)).await;
    let out_vol = MainBackendBase::float_into_data(full.out_volume).await.unwrap();
    let sub_vol = MainBackendBase::float_into_data(sub.out_volume).await.unwrap();
    let a: Vec<f32> = out_vol.as_slice::<f32>().unwrap().to_vec();
    let b: Vec<f32> = sub_vol.as_slice::<f32>().unwrap().to_vec();

    let d = max_diff(&a, &b);
    println!("full vs inside-only volume max diff = {d:.6}");
    assert!(
        d < 1e-4,
        "outside splats must not contribute (g2c holes replay bug): diff {d}"
    );

    // 且 num_visible 应为 inside 数量。
    assert_eq!(full.aux.num_visible as usize, inside.len(), "num_visible");
}
