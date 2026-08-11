//! Per-visible-splat projection: recompute and pack the 8 density lanes
//! (`xy`, `conic`, `opacity`, `mu`, `radius`) into `projected_splats`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::sigmoid;

use super::helpers::{
    XRAY_LANES, compute_radius, cone_cov2d_mu, project_xy, read_mean_viewspace_xray,
    read_quat_unorm_xray, read_scale_xray,
};
use super::types::XRayProjectUniforms;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_visible_xray_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    projected: &mut Tensor<f32>,
    u: XRayProjectUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];
    let base = (global_gid * 10u32) as usize;

    let mean_c = read_mean_viewspace_xray(transforms, base, u);
    let scale = read_scale_xray(transforms, base, u.scale_modifier);
    let quat = read_quat_unorm_xray(transforms, base).normalize();

    let (conic, mu, cov3) = cone_cov2d_mu(mean_c, scale, quat, u);
    let opac = sigmoid(raw_opacities[global_gid as usize]);
    let (xy_x, xy_y) = project_xy(mean_c, u);
    let radius = compute_radius(cov3);

    let dst = (compact_gid * XRAY_LANES) as usize;
    projected[dst] = xy_x;
    projected[dst + 1] = xy_y;
    projected[dst + 2] = conic.c00;
    projected[dst + 3] = conic.c01;
    projected[dst + 4] = conic.c11;
    projected[dst + 5] = opac;
    projected[dst + 6] = mu;
    projected[dst + 7] = radius;
}
