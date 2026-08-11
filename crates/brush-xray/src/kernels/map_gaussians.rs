//! Map per-splat tile counts to per-intersection (tile_id, compact_gid)
//! pairs for the X-ray rasterizer. Tiling is radius-based (3σ pixel
//! bbox → tile range), matching `project_forward_xray`'s count exactly.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{count_tiles, get_tile_bbox_xray, read_packed};
use super::types::XRayProjectUniforms;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
pub fn map_gaussians_xray_kernel(
    projected: &Tensor<f32>,
    splat_cum_hit_counts: &Tensor<u32>,
    tile_id_from_isect: &mut Tensor<u32>,
    compact_gid_from_isect: &mut Tensor<u32>,
    u: XRayProjectUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let (xy_x, xy_y, _conic, _opac, _mu, radius) = read_packed(projected, compact_gid);
    let bb = get_tile_bbox_xray(xy_x, xy_y, radius, u);

    // Inclusive prefix sum: use cum[compact_gid - 1] as base (or 0 for first).
    let prev_idx = max(compact_gid, 1u32) - 1u32;
    let base_isect_id = select(
        compact_gid == 0u32,
        0u32,
        splat_cum_hit_counts[prev_idx as usize],
    );
    let pf_count = splat_cum_hit_counts[compact_gid as usize] - base_isect_id;
    let local_count = count_tiles(bb);
    let writable = min(local_count, pf_count);

    let sentinel_tile_id = u.tile_bw * u.tile_bh;

    let bb_w = bb.max_x - bb.min_x;
    let num_tiles_bbox = (bb.max_y - bb.min_y) * bb_w;
    let mut num_tiles_hit = 0u32;
    for tile_idx in 0u32..num_tiles_bbox {
        let tx = (tile_idx % bb_w) + bb.min_x;
        let ty = (tile_idx / bb_w) + bb.min_y;
        if num_tiles_hit < writable {
            let tile_id = tx + ty * u.tile_bw;
            let isect_id = base_isect_id + num_tiles_hit;
            tile_id_from_isect[isect_id as usize] = tile_id;
            compact_gid_from_isect[isect_id as usize] = compact_gid;
            num_tiles_hit += 1u32;
        }
    }

    // Pad leftover budget with sentinel rows so no slot is uninitialised.
    for pad_idx in writable..pf_count {
        let isect_id = base_isect_id + pad_idx;
        tile_id_from_isect[isect_id as usize] = sentinel_tile_id;
        compact_gid_from_isect[isect_id as usize] = 0u32;
    }
}
