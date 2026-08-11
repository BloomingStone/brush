//! Map per-splat cube counts to per-intersection `(cube_id, compact_gid)`
//! pairs for the voxelizer. Mirrors R2's `duplicateWithKeys` (the 32-bit
//! cube_id sort key — depth is not needed since the additive render is
//! order-independent).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::Vec3A;

use super::helpers::{VOXEL_LANES, get_cube_bbox};
use super::types::VoxelUniforms;

pub const WG_SIZE: u32 = 256;

#[cube]
fn read_point_vol_radius(
    projected: &Tensor<f32>,
    compact_gid: u32,
) -> (Vec3A, Vec3A) {
    let b = (compact_gid * VOXEL_LANES) as usize;
    (
        Vec3A::new(projected[b], projected[b + 1], projected[b + 2]),
        Vec3A::new(projected[b + 10], projected[b + 11], projected[b + 12]),
    )
}

#[cube(launch)]
pub fn map_cubes_voxel_kernel(
    projected: &Tensor<f32>,
    splat_cum_hit_counts: &Tensor<u32>,
    cube_id_from_isect: &mut Tensor<u32>,
    compact_gid_from_isect: &mut Tensor<u32>,
    u: VoxelUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let (point_vol, radius) = read_point_vol_radius(projected, compact_gid);
    let bb = get_cube_bbox(point_vol, radius, u);

    // Inclusive prefix sum: base = cum[compact_gid - 1] (or 0 for first).
    let prev_idx = max(compact_gid, 1u32) - 1u32;
    let base_isect_id = select(
        compact_gid == 0u32,
        0u32,
        splat_cum_hit_counts[prev_idx as usize],
    );
    let pf_count = splat_cum_hit_counts[compact_gid as usize] - base_isect_id;
    let local_count = bb.cube_volume();
    let writable = min(local_count, pf_count);

    let sentinel_cube_id = u.num_cubes;

    let bb_w = bb.max_x - bb.min_x;
    let bb_wh = (bb.max_y - bb.min_y) * bb_w;
    let num_cubes_bbox = (bb.max_z - bb.min_z) * bb_wh;
    let mut num_cubes_hit = 0u32;
    for cube_idx in 0u32..num_cubes_bbox {
        let tx = (cube_idx % bb_w) + bb.min_x;
        let ty = (cube_idx / bb_w) % (bb.max_y - bb.min_y) + bb.min_y;
        let tz = (cube_idx / bb_wh) + bb.min_z;
        if num_cubes_hit < writable {
            let cube_id = tz * u.grid_y * u.grid_x + ty * u.grid_x + tx;
            let isect_id = base_isect_id + num_cubes_hit;
            cube_id_from_isect[isect_id as usize] = cube_id;
            compact_gid_from_isect[isect_id as usize] = compact_gid;
            num_cubes_hit += 1u32;
        }
    }

    // Pad leftover budget with sentinel cube ids so no slot is uninitialised.
    for pad_idx in writable..pf_count {
        let isect_id = base_isect_id + pad_idx;
        cube_id_from_isect[isect_id as usize] = sentinel_cube_id;
        compact_gid_from_isect[isect_id as usize] = 0u32;
    }
}
