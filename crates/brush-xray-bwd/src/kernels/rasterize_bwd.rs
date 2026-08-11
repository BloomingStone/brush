//! Backward of the additive X-ray rasterize: per-pixel, accumulate
//! gradients w.r.t. each contributing splat's packed lanes
//! (`xy`, `conic`, `opacity`, `mu`).
//!
//! Since `C = Σ alpha` with `alpha = opac·mu·exp(power)`, the per-pixel
//! `dL/dalpha = dL/dC`. Mirrors R2-Gaussian's `renderCUDA` backward
//! (single channel, pixel-space xy — no ndc2Pix compensation factor).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::Sym2;
use brush_xray::kernels::helpers::{
    MIN_ALPHA, TILE_WIDTH, XRAY_LANES, pixel_from_global,
};
use brush_xray::kernels::types::XRayRasterizeUniforms;

use super::atomic::AtomicAddF32;

#[cube(launch)]
pub fn rasterize_xray_bwd_kernel<A: AtomicAddF32>(
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    projected: &Tensor<f32>,
    v_output: &Tensor<f32>,
    n_contrib: &Tensor<u32>,
    v_splats: &mut Tensor<Atomic<A::Storage>>,
    u: XRayRasterizeUniforms,
) {
    let global_id = ABSOLUTE_POS as u32;
    let (pix_x, pix_y) = pixel_from_global(global_id, u.tile_bw);
    let pix_id = pix_x + pix_y * u.img_w;
    let tile_id = (pix_x / TILE_WIDTH) + (pix_y / TILE_WIDTH) * u.tile_bw;
    let inside = pix_x < u.img_w && pix_y < u.img_h;
    let pixel_coord_x = pix_x as f32;
    let pixel_coord_y = pix_y as f32;

    let range_lo = tile_offsets[(tile_id * 2u32) as usize];
    let range_hi = tile_offsets[(tile_id * 2u32 + 1u32) as usize];
    let last_contrib = select(inside, n_contrib[pix_id as usize], range_lo);
    let v_pix = select(inside, v_output[pix_id as usize], 0.0f32);

    // Splats in [range_lo, last_contrib) were eligible for this pixel;
    // the per-splat threshold check below skips non-contributing ones.
    let mut t = range_lo;
    while t < last_contrib && t < range_hi {
        let compact_gid = compact_gid_from_isect[t as usize];
        let b = (compact_gid * XRAY_LANES) as usize;
        let xy_x = projected[b];
        let xy_y = projected[b + 1];
        let conic = Sym2 {
            c00: projected[b + 2],
            c01: projected[b + 3],
            c11: projected[b + 4],
        };
        let opac = projected[b + 5];
        let mu = projected[b + 6];

        let d_x = xy_x - pixel_coord_x;
        let d_y = xy_y - pixel_coord_y;
        let power = -0.5f32 * (conic.c00 * d_x * d_x + conic.c11 * d_y * d_y)
            - conic.c01 * d_x * d_y;
        if power <= 0.0f32 {
            let g = f32::exp(power);
            let alpha = opac * mu * g;
            if alpha >= MIN_ALPHA {
                let dL_dalpha = v_pix;
                let dL_dg = opac * mu * dL_dalpha;
                let gdx = g * d_x;
                let gdy = g * d_y;
                let dg_ddelx = -gdx * conic.c00 - gdy * conic.c01;
                let dg_ddely = -gdy * conic.c11 - gdx * conic.c01;

                A::add(&v_splats[b], dL_dg * dg_ddelx);
                A::add(&v_splats[b + 1], dL_dg * dg_ddely);
                A::add(&v_splats[b + 2], dL_dg * (-0.5f32 * gdx * d_x));
                A::add(&v_splats[b + 3], dL_dg * (-gdx * d_y));
                A::add(&v_splats[b + 4], dL_dg * (-0.5f32 * gdy * d_y));
                A::add(&v_splats[b + 5], mu * g * dL_dalpha);
                A::add(&v_splats[b + 6], opac * g * dL_dalpha);
            }
        }
        t += 1u32;
    }
}
