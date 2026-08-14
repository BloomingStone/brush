//! Project & cull pass for the cone-beam X-ray rasterizer.
//!
//! Single visibility gate: near cull + finite checks, then cone-beam
//! cov2d + `mu`, pixel position, screen-space radius and per-tile count.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{is_finite_f32, MU_WATER, silu};

use super::helpers::{
    compute_radius, cone_cov2d_mu, count_tiles, get_tile_bbox_xray, project_xy,
    read_mean_viewspace_xray, read_quat_unorm_xray, read_scale_xray,
};
use super::types::XRayProjectUniforms;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_forward_xray_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_presort_gid: &mut Tensor<u32>,
    depths: &mut Tensor<f32>,
    num_visible: &mut Tensor<Atomic<u32>>,
    intersect_counts: &mut Tensor<u32>,
    num_intersections: &mut Tensor<Atomic<u32>>,
    max_radius: &mut Tensor<f32>,
    u: XRayProjectUniforms,
) {
    let global_gid = ABSOLUTE_POS as u32;
    if global_gid >= u.total_splats {
        terminate!();
    }

    // means(3) + quats(4) + log_scales(3)
    let base = (global_gid * 10u32) as usize;

    let mean_c = read_mean_viewspace_xray(transforms, base, u);
    if !(mean_c.is_finite() && mean_c.z() <= 1.0e10f32) {
        terminate!();
    }
    if mean_c.z() < 0.01f32 {
        terminate!();
    }

    let scale = read_scale_xray(transforms, base, u.scale_modifier);
    if !scale.is_finite() {
        terminate!();
    }

    let quat_unorm = read_quat_unorm_xray(transforms, base);
    let qnorm_sq = quat_unorm.dot(quat_unorm);
    if !(qnorm_sq >= 1.0e-6f32 && is_finite_f32(qnorm_sq)) {
        terminate!();
    }

    let raw_opac = raw_opacities[global_gid as usize];
    if !is_finite_f32(raw_opac) {
        terminate!();
    }

    let quat = quat_unorm.normalize();
    let (conic, _mu, cov3) = cone_cov2d_mu(mean_c, scale, quat, u);
    if !conic.is_finite() {
        terminate!();
    }

    let opac = MU_WATER * silu(raw_opac); // exp6: silu 代替 softplus
    // Physical-density floor: μ_water ≈ 0.002 mm⁻¹ is a perfectly meaningful
    // Beer-Lambert path-integral contribution (0.002 × 200 mm ≈ 0.4 optical
    // depth ≈ exp(-0.4) ≈ 0.67 intensity) even though it sits below the RGB
    // opacity-visibility threshold of 1/255 ≈ 0.0039. Gate on the same tiny
    // per-splat alpha floor as the rasterizer (MIN_ALPHA) instead, so low-μ
    // soft-tissue/water backgrounds are actually projected.
    //
    // Activated density = MU_WATER · softplus(raw): μ stays in the
    // water→iodine band (a stray large logit gives at most ~0.002·raw instead
    // of the unbounded sigmoid → 1 mm⁻¹ that caused black-blob artifacts).
    if !(opac >= 1.0e-5f32) {
        terminate!();
    }

    let (xy_x, xy_y) = project_xy(mean_c, u);
    let radius = compute_radius(cov3);
    let bb = get_tile_bbox_xray(xy_x, xy_y, radius, u);
    let num_tiles_hit = count_tiles(bb);

    intersect_counts[global_gid as usize] = num_tiles_hit;
    Atomic::fetch_add(&num_intersections[0], num_tiles_hit);

    // Screen-space radius (pixels), normalized — used for diagnostics/aux.
    let img_w_f = u.img_w as f32;
    let img_h_f = u.img_h as f32;
    max_radius[global_gid as usize] = f32::max(radius / img_w_f, radius / img_h_f);

    let write_id = Atomic::fetch_add(&num_visible[0], 1u32);
    global_from_presort_gid[write_id as usize] = global_gid;
    depths[write_id as usize] = mean_c.z();
}
