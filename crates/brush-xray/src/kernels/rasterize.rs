//! Tile-based additive X-ray rasterizer.
//!
//! One workgroup of `TILE_SIZE` threads per tile, each thread processes
//! a single pixel. Splats are loaded into workgroup-shared memory in
//! batches and accumulated additively: `acc += opacity·mu·exp(power)`.
//! No transmittance, no early termination (order-independent density
//! summation). Single-channel output.
//!
//! `bwd_info` additionally writes a per-pixel contributor count
//! (`n_contrib`, the last isect index this pixel consumed) and shrinks
//! `tile_offsets[tile*2+1]` to "one past the last useful isect" so the
//! backward kernel's outer loop ends early.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::Sym2;

use super::helpers::{
    MIN_ALPHA, TILE_SIZE, TILE_WIDTH, XRAY_LANES, XRAY_LANES_USIZE, pixel_from_global,
};
use super::types::XRayRasterizeUniforms;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn rasterize_xray_kernel(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &mut Tensor<u32>,
    projected: &Tensor<f32>,
    out_img: &mut Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    visible: &mut Tensor<f32>,
    n_contrib: &mut Tensor<u32>,
    u: XRayRasterizeUniforms,
    #[comptime] bwd_info: bool,
) {
    let global_id = ABSOLUTE_POS as u32;
    let (pix_x, pix_y) = pixel_from_global(global_id, u.tile_bw);
    let pix_id = pix_x + pix_y * u.img_w;
    let pixel_coord_x = pix_x as f32;
    let pixel_coord_y = pix_y as f32;
    let tile_loc_x = pix_x / TILE_WIDTH;
    let tile_loc_y = pix_y / TILE_WIDTH;
    let tile_id = tile_loc_x + tile_loc_y * u.tile_bw;
    let inside = pix_x < u.img_w && pix_y < u.img_h;

    let mut local_batch = Shared::new_slice((TILE_SIZE * XRAY_LANES) as usize);
    let mut load_gid =
        Shared::new_slice(comptime![if bwd_info { TILE_SIZE } else { 1u32 }] as usize);
    let num_done_atomic = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let max_useful_isect = Shared::<[Atomic<u32>]>::new_slice(1usize);
    let mut range = Shared::new_slice(2usize);

    let local_idx = UNIT_POS;
    if local_idx == 0u32 {
        range[0] = tile_offsets[(tile_id * 2u32) as usize];
        range[1] = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
        Atomic::store(&num_done_atomic[0], 0u32);
    }

    let range_lo = workgroup_uniform_load(&range[0]);
    let range_hi = workgroup_uniform_load(&range[1]);

    if comptime![bwd_info] && local_idx == 0u32 {
        Atomic::store(&max_useful_isect[0], range_lo);
    }

    let mut acc = 0.0f32;
    let done = !inside;
    let mut last_useful_isect = range_lo;

    if done {
        Atomic::fetch_add(&num_done_atomic[0], 1u32);
    }
    sync_cube();

    let mut batch_start = range_lo;
    while batch_start < range_hi {
        if workgroup_uniform_load_atomic(&num_done_atomic[0]) >= TILE_SIZE {
            break;
        }
        let remaining = min(TILE_SIZE, range_hi - batch_start);
        let load_isect_id = batch_start + local_idx;
        let mut compact_gid = 0u32;
        if local_idx < remaining {
            compact_gid = compact_gid_from_isect[load_isect_id as usize];
        }
        if local_idx < remaining {
            let src_base = (compact_gid * XRAY_LANES) as usize;
            let dst_base = (local_idx * XRAY_LANES) as usize;
            #[unroll]
            for lane in 0..XRAY_LANES_USIZE {
                local_batch[dst_base + lane] = projected[src_base + lane];
            }
            if comptime![bwd_info] {
                load_gid[local_idx as usize] = global_from_compact_gid[compact_gid as usize];
            }
        }
        sync_cube();

        let mut t = 0u32;
        while !done && t < remaining {
            let dst_base = (t * XRAY_LANES) as usize;
            let xy_x = local_batch[dst_base];
            let xy_y = local_batch[dst_base + 1];
            let conic = Sym2 {
                c00: local_batch[dst_base + 2],
                c01: local_batch[dst_base + 3],
                c11: local_batch[dst_base + 4],
            };
            let opac = local_batch[dst_base + 5];
            let mu = local_batch[dst_base + 6];

            let d_x = xy_x - pixel_coord_x;
            let d_y = xy_y - pixel_coord_y;
            let power =
                -0.5f32 * (conic.c00 * d_x * d_x + conic.c11 * d_y * d_y) - conic.c01 * d_x * d_y;
            if power <= 0.0f32 {
                let alpha = opac * mu * f32::exp(power);
                if alpha >= MIN_ALPHA {
                    acc += alpha;
                    last_useful_isect = batch_start + t + 1u32;
                    if comptime![bwd_info] {
                        visible[load_gid[t as usize] as usize] = 1.0f32;
                    }
                }
            }
            t += 1u32;
        }
        batch_start += TILE_SIZE;
    }

    if inside {
        out_img[pix_id as usize] = acc;
        if comptime![bwd_info] {
            n_contrib[pix_id as usize] = last_useful_isect;
        }
    }

    if comptime![bwd_info] {
        Atomic::fetch_max(&max_useful_isect[0], last_useful_isect);
        sync_cube();
        if local_idx == 0u32 {
            tile_offsets[(tile_id * 2u32 + 1u32) as usize] = Atomic::load(&max_useful_isect[0]);
        }
    }
}
