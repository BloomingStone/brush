//! Backward of the 3D additive density-volume render: per-voxel, accumulate
//! gradients w.r.t. each contributing splat's packed lanes
//! (`point_vol` 3, `conic` 6, `opacity`). Mirrors R2 `renderCUDA_voxel`
//! backward. The mean grad carries R2's `dVoxel` compensation factor.

#![allow(non_snake_case)]

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::atomic::AtomicAddF32;
use super::helpers::{
    BLOCK3D_X, BLOCK3D_Y, BLOCK3D_Z, MIN_ALPHA, VOXEL_LANES, voxel_id_from_xyz,
};
use super::types::VoxelUniforms;

pub const BWD_LANES: u32 = 10;

#[cube(launch)]
pub fn render_voxel_bwd_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    cube_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    n_contrib: &Tensor<u32>,
    v_volume: &Tensor<f32>,
    v_combined: &mut Tensor<Atomic<A::Storage>>,
    u: VoxelUniforms,
) {
    let gx = CUBE_POS_X;
    let gy = CUBE_POS_Y;
    let gz = CUBE_POS_Z;
    let workgroup_id = gx + gy * CUBE_COUNT_X + gz * CUBE_COUNT_X * CUBE_COUNT_Y;

    let local = UNIT_POS;
    let vx_local = local % BLOCK3D_X;
    let vy_local = (local / BLOCK3D_X) % BLOCK3D_Y;
    let vz_local = local / (BLOCK3D_X * BLOCK3D_Y);

    let vx = gx * BLOCK3D_X + vx_local;
    let vy = gy * BLOCK3D_Y + vy_local;
    let vz = gz * BLOCK3D_Z + vz_local;

    let inside = vx < u.n_voxel_x && vy < u.n_voxel_y && vz < u.n_voxel_z;
    let voxel_id = voxel_id_from_xyz(vx, vy, vz, u);

    let range_lo = cube_offsets[(workgroup_id * 2u32) as usize];
    let last = select(inside, n_contrib[voxel_id as usize], range_lo);
    let v_pix = select(inside, v_volume[voxel_id as usize], 0.0f32);

    let voxelf_x = vx as f32 + 0.5f32;
    let voxelf_y = vy as f32 + 0.5f32;
    let voxelf_z = vz as f32 + 0.5f32;

    let mut t = range_lo;
    while t < last {
        let compact_gid = compact_gid_from_isect[t as usize];
        let b = (compact_gid * VOXEL_LANES) as usize;
        let px = projected[b];
        let py = projected[b + 1];
        let pz = projected[b + 2];
        let conic_a = projected[b + 3];
        let conic_b = projected[b + 4];
        let conic_c = projected[b + 5];
        let conic_d = projected[b + 6];
        let conic_e = projected[b + 7];
        let conic_f = projected[b + 8];
        let opa = projected[b + 9];

        let d_x = px - voxelf_x;
        let d_y = py - voxelf_y;
        let d_z = pz - voxelf_z;
        let power = -0.5f32 * (conic_a * d_x * d_x + conic_d * d_y * d_y + conic_f * d_z * d_z)
            - conic_b * d_x * d_y
            - conic_c * d_x * d_z
            - conic_e * d_y * d_z;

        if power <= 0.0f32 {
            let g = f32::exp(power);
            // |·| gate mirrors the forward's signed accumulate.
            if f32::abs(opa * g) >= MIN_ALPHA {
                let dL_dalpha = v_pix;
                let dL_dg = opa * dL_dalpha;
                let gdx = g * d_x;
                let gdy = g * d_y;
                let gdz = g * d_z;
                let dg_ddelx = -conic_a * gdx - conic_b * gdy - conic_c * gdz;
                let dg_ddely = -conic_d * gdy - conic_b * gdx - conic_e * gdz;
                let dg_ddelz = -conic_f * gdz - conic_c * gdx - conic_e * gdy;

                let vbase = (compact_gid * BWD_LANES) as usize;
                A::add(&v_combined[vbase], dL_dg * dg_ddelx * u.d_voxel_x);
                A::add(&v_combined[vbase + 1], dL_dg * dg_ddely * u.d_voxel_y);
                A::add(&v_combined[vbase + 2], dL_dg * dg_ddelz * u.d_voxel_z);

                A::add(&v_combined[vbase + 3], -0.5f32 * gdx * d_x * dL_dg);
                A::add(&v_combined[vbase + 4], -(gdx * d_y * dL_dg));
                A::add(&v_combined[vbase + 5], -(gdx * d_z * dL_dg));
                A::add(&v_combined[vbase + 6], -0.5f32 * gdy * d_y * dL_dg);
                A::add(&v_combined[vbase + 7], -(gdy * d_z * dL_dg));
                A::add(&v_combined[vbase + 8], -0.5f32 * gdz * d_z * dL_dg);

                A::add(&v_combined[vbase + 9], g * dL_dalpha);
            }
        }

        t += 1u32;
    }
}
