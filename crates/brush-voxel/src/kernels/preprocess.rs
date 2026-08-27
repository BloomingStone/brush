//! Per-splat preprocess for the voxelizer: compute the voxel-space
//! geometry (point_vol, 3D conic, radius, cube bbox, cube count) and
//! mark visibility. Mirrors R2 `preprocessCUDA_voxel` + `duplicate`
//! bookkeeping.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::Vec3A;

use super::helpers::{get_cube_bbox, read_quat, read_scale_mod, read_scale_raw, voxel_geometry};
use super::types::VoxelUniforms;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
pub fn preprocess_voxel_kernel(
    transforms: &Tensor<f32>,
    global_from_presort_gid: &mut Tensor<u32>,
    num_visible_buf: &mut Tensor<Atomic<u32>>,
    intersect_counts: &mut Tensor<u32>,
    num_intersections_buf: &mut Tensor<Atomic<u32>>,
    u: VoxelUniforms,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= u.total_splats {
        terminate!();
    }

    let base = (idx * 10u32) as usize;
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale_mod(transforms, base, u.scale_modifier);
    let scale_raw = read_scale_raw(transforms, base);
    let quat = read_quat(transforms, base).normalize();

    let (point_vol, _inv_a, _inv_b, _inv_c, _inv_d, _inv_e, _inv_f, radius, valid) =
        voxel_geometry(mean, scale, scale_raw, quat, u);

    // Cull: bbox must intersect the [0, nVoxel]^3 domain and the
    // covariance must be non-singular.
    let in_x =
        point_vol.x() + radius.x() >= 0.0f32 && point_vol.x() - radius.x() <= u.n_voxel_x as f32;
    let in_y =
        point_vol.y() + radius.y() >= 0.0f32 && point_vol.y() - radius.y() <= u.n_voxel_y as f32;
    let in_z =
        point_vol.z() + radius.z() >= 0.0f32 && point_vol.z() - radius.z() <= u.n_voxel_z as f32;

    let bb = get_cube_bbox(point_vol, radius, u);
    let count = bb.cube_volume();

    let visible = valid && in_x && in_y && in_z && count > 0u32;

    if visible {
        // Compact write via the atomic counter (NOT the original index):
        // invisible splats leave holes in [0, n) otherwise, and the
        // [0..num_visible) slice would pick up zeros that map every hole to
        // splat 0 in `project_visible` (replaying splat 0's lanes N times).
        // Mirrors brush-xray's `project_forward`.
        let write_id = Atomic::fetch_add(&num_visible_buf[0], 1u32);
        global_from_presort_gid[write_id as usize] = idx;
        intersect_counts[idx as usize] = count;
        Atomic::fetch_add(&num_intersections_buf[0], count);
    } else {
        intersect_counts[idx as usize] = 0u32;
    }
}
