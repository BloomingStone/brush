//! 3D additive density-volume render: one workgroup per cube (8×8×8
//! voxels), each thread accumulates `alpha = opac·exp(power)` over the
//! cube's gaussian range. Mirrors R2 `renderCUDA_voxel`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{
    BLOCK3D_X, BLOCK3D_Y, BLOCK3D_Z, MIN_ALPHA, VOXEL_LANES, voxel_id_from_xyz,
};
use super::types::VoxelUniforms;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn render_voxel_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    cube_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    n_contrib: &mut Tensor<u32>,
    out_volume: &mut Tensor<f32>,
    u: VoxelUniforms,
    #[comptime] bwd_info: bool,
) {
    let workgroup_id = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let gx = workgroup_id % u.grid_x;
    let gy = (workgroup_id / u.grid_x) % u.grid_y;
    let gz = workgroup_id / (u.grid_x * u.grid_y);

    let local = UNIT_POS;
    let vx_local = local % BLOCK3D_X;
    let vy_local = (local / BLOCK3D_X) % BLOCK3D_Y;
    let vz_local = local / (BLOCK3D_X * BLOCK3D_Y);

    let vx = gx * BLOCK3D_X + vx_local;
    let vy = gy * BLOCK3D_Y + vy_local;
    let vz = gz * BLOCK3D_Z + vz_local;

    let inside = vx < u.n_voxel_x && vy < u.n_voxel_y && vz < u.n_voxel_z;
    let voxel_id = voxel_id_from_xyz(vx, vy, vz, u);

    // Range of gaussians mapped to this cube.
    let range_lo = cube_offsets[(workgroup_id * 2u32) as usize];
    let range_hi = cube_offsets[(workgroup_id * 2u32 + 1u32) as usize];

    let voxelf_x = vx as f32 + 0.5f32;
    let voxelf_y = vy as f32 + 0.5f32;
    let voxelf_z = vz as f32 + 0.5f32;

    let mut acc = 0.0f32;
    let mut last_contributor = range_lo;
    let mut t = range_lo;
    while t < range_hi {
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
            let alpha = opa * g;
            if alpha >= MIN_ALPHA {
                acc += alpha;
                // Global isect index one-past-last-contributor (matches the
                // rasterizer's `n_contrib` semantics).
                last_contributor = t + 1u32;
            }
        }

        t += 1u32;
    }

    if inside {
        out_volume[voxel_id as usize] = acc;
        if comptime![bwd_info] {
            n_contrib[voxel_id as usize] = last_contributor;
        }
    }
}
