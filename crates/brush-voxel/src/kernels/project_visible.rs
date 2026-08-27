//! Per-visible-splat packing: recompute and pack the 12 voxel lanes
//! (`point_vol` 3, `conic` 6, `opacity`, `radius` 3) into
//! `projected_splats` (compact-indexed).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{MU_WATER, Vec3A, silu};

use super::helpers::{
    VOXEL_LANES, read_quat, read_scale_mod, read_scale_raw, voxel_geometry,
};
use super::types::VoxelUniforms;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
pub fn project_visible_voxel_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    projected: &mut Tensor<f32>,
    u: VoxelUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];
    let base = (global_gid * 10u32) as usize;

    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale_mod(transforms, base, u.scale_modifier);
    let scale_raw = read_scale_raw(transforms, base);
    let quat = read_quat(transforms, base).normalize();

    let (point_vol, inv_a, inv_b, inv_c, inv_d, inv_e, inv_f, radius, _valid) =
        voxel_geometry(mean, scale, scale_raw, quat, u);
    // Density activation, identical to brush-xray's `signed_opac` flag so
    // the voxelizer consumes the same raw logits as the rasterizer:
    //   unsigned: opac = MU_WATER · silu(raw) (≥ 0)
    //   signed:   opac = MU_WATER · raw (may be negative, FDK residual)
    let raw = raw_opacities[global_gid as usize];
    let opac = select(
        u.signed_opac != 0u32,
        MU_WATER * raw,
        MU_WATER * silu(raw),
    );

    let dst = (compact_gid * VOXEL_LANES) as usize;
    projected[dst] = point_vol.x();
    projected[dst + 1] = point_vol.y();
    projected[dst + 2] = point_vol.z();
    projected[dst + 3] = inv_a;
    projected[dst + 4] = inv_b;
    projected[dst + 5] = inv_c;
    projected[dst + 6] = inv_d;
    projected[dst + 7] = inv_e;
    projected[dst + 8] = inv_f;
    projected[dst + 9] = opac;
    projected[dst + 10] = radius.x();
    projected[dst + 11] = radius.y();
    projected[dst + 12] = radius.z();
}
