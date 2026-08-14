//! X-ray rasterizer cube helpers: cone-beam covariance + `mu`, tiling,
//! and packed-lane readers.

use brush_cube::{Mat3, Quat, Sym2, Sym3, Vec3A, TileBbox, compute_cov3d};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::types::XRayProjectUniforms;

/// Tile width / size (matches brush-render's rasterizer).
pub const TILE_WIDTH: u32 = 16;
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

/// `f32` lanes per projected splat (density projection variant):
///   0:xy_x, 1:xy_y, 2:conic_c00, 3:conic_c01, 4:conic_c11,
///   5:opacity, 6:mu, 7:radius_px.
pub const XRAY_LANES: u32 = 8;
pub const XRAY_LANES_USIZE: usize = XRAY_LANES as usize;

/// Alpha floor for density contribution (R2 `1e-5`).
pub const MIN_ALPHA: f32 = 1.0e-5f32;

/// Full cone-beam geometry. Returns `(conic2, mu, cov3, j, m, tx, ty, txtz, tytz)`:
/// - `conic2` — inverse of the 2x2 top-left block of `cov3`.
/// - `mu` — `sqrt(2π·circ/diamond)`, the along-ray integration factor.
/// - `cov3` — full `Mᵀ·Vrk·M` covariance.
/// - `j`, `m` — the 3x3 Jacobian and `W·J`.
/// - `tx`, `ty` — the clamped camera-space coordinates.
/// - `txtz`, `tytz` — the pre-clamp normalized coords (backward needs them
///   for the `x/y_grad_mul` multipliers).
///
/// Shared verbatim by the forward (`cone_cov2d_mu`) and the backward so the
/// two never drift.
#[cube]
#[allow(clippy::approx_constant)]
pub fn cone_geometry(
    mean_c: Vec3A,
    scale: Vec3A,
    quat: Quat,
    u: XRayProjectUniforms,
) -> (Sym2, f32, Sym3, Mat3, Mat3, f32, f32, f32, f32) {
    let vrk = compute_cov3d(scale, quat);

    let inv_tz = 1.0f32 / mean_c.z();
    let txtz = mean_c.x() * inv_tz;
    let tytz = mean_c.y() * inv_tz;
    let tx = clamp(txtz, u.lim_neg_x, u.lim_pos_x) * mean_c.z();
    let ty = clamp(tytz, u.lim_neg_y, u.lim_pos_y) * mean_c.z();
    let l = f32::sqrt(tx * tx + ty * ty + mean_c.z() * mean_c.z());
    let inv_tz2 = inv_tz * inv_tz;

    // J (3x3, columns as R2-Gaussian):
    //   col0 = (fx/tz, 0, -fx·tx/tz²)
    //   col1 = (0, fy/tz, -fy·ty/tz²)
    //   col2 = (tx/l, ty/l, tz/l)      <- ray direction (for mu)
    let j = Mat3::from_cols(
        Vec3A::new(u.focal_x * inv_tz, 0.0f32, -u.focal_x * tx * inv_tz2),
        Vec3A::new(0.0f32, u.focal_y * inv_tz, -u.focal_y * ty * inv_tz2),
        Vec3A::new(tx / l, ty / l, mean_c.z() / l),
    );
    let m = u.view_rotation().mul_mat3(j);
    let cov3 = vrk.transpose_congruence(m);

    let conic = Sym2 {
        c00: cov3.c00,
        c01: cov3.c01,
        c11: cov3.c11,
    }
    .inverse();

    let diamond = cov3.c00 * cov3.c11 - cov3.c01 * cov3.c01;
    let circ = cov3.c00 * (cov3.c11 * cov3.c22 - cov3.c12 * cov3.c12)
        - cov3.c01 * (cov3.c01 * cov3.c22 - cov3.c12 * cov3.c02)
        + cov3.c02 * (cov3.c01 * cov3.c12 - cov3.c11 * cov3.c02);
    let two_pi = 6.28318530717958647692f32;
    let mu_sq = select(diamond != 0.0f32, two_pi * circ / diamond, -1.0f32);
    let mu = select(mu_sq > 0.0f32, f32::sqrt(mu_sq), 0.0f32);

    (conic, mu, cov3, j, m, tx, ty, txtz, tytz)
}

/// Cone-beam 2D covariance + integration factor `mu`, following
/// R2-Gaussian `computeCov2D` mode=1. `mean_c` must be the camera-space
/// mean. Returns `(conic2, mu, cov3)`.
#[cube]
pub fn cone_cov2d_mu(
    mean_c: Vec3A,
    scale: Vec3A,
    quat: Quat,
    u: XRayProjectUniforms,
) -> (Sym2, f32, Sym3) {
    let (conic, mu, cov3, _j, _m, _tx, _ty, _txtz, _tytz) =
        cone_geometry(mean_c, scale, quat, u);
    (conic, mu, cov3)
}

/// Screen-space radius (pixels): `ceil(3·σ)` from the 2x2 covariance's
/// max eigenvalue (R2-Gaussian `preprocessCUDA`).
#[cube]
pub fn compute_radius(cov3: Sym3) -> f32 {
    let mid = 0.5f32 * (cov3.c00 + cov3.c11);
    let det = cov3.c00 * cov3.c11 - cov3.c01 * cov3.c01;
    let rad_sq = f32::max(mid * mid - det, 0.1f32);
    let lambda1 = mid + f32::sqrt(rad_sq);
    let lambda2 = mid - f32::sqrt(rad_sq);
    f32::ceil(3.0f32 * f32::sqrt(f32::max(lambda1, lambda2)))
}

/// Tile-range bbox (inclusive min, exclusive max in tile coords) from a
/// pixel-space center + radius, replicating R2-Gaussian's `getRect`
/// exactly:
/// - `min = clamp(floor((p - r) / BLOCK), 0, grid)`
/// - `max = clamp(floor((p + r + BLOCK - 1) / BLOCK), 0, grid)`
#[cube]
pub fn get_tile_bbox_xray(xy_x: f32, xy_y: f32, radius: f32, u: XRayProjectUniforms) -> TileBbox {
    let grid_x = u.tile_bw as f32;
    let grid_y = u.tile_bh as f32;
    let b = TILE_WIDTH as f32;

    let min_x_f = f32::max(0.0f32, f32::floor((xy_x - radius) / b));
    let min_x = f32::min(grid_x, min_x_f) as u32;
    let min_y_f = f32::max(0.0f32, f32::floor((xy_y - radius) / b));
    let min_y = f32::min(grid_y, min_y_f) as u32;

    let max_x_f = f32::max(0.0f32, f32::floor((xy_x + radius + b - 1.0f32) / b));
    let max_x = f32::min(grid_x, max_x_f) as u32;
    let max_y_f = f32::max(0.0f32, f32::floor((xy_y + radius + b - 1.0f32) / b));
    let max_y = f32::min(grid_y, max_y_f) as u32;

    TileBbox {
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

/// Number of tiles covered by a tile bbox.
#[cube]
pub fn count_tiles(bb: TileBbox) -> u32 {
    (bb.max_y - bb.min_y) * (bb.max_x - bb.min_x)
}

/// Pixel-space projection of a camera-space mean (pinhole, integer
/// pixel-center convention matching R2 `ndc2Pix`).
#[cube]
pub fn project_xy(mean_c: Vec3A, u: XRayProjectUniforms) -> (f32, f32) {
    let inv_z = 1.0f32 / mean_c.z();
    (
        u.focal_x * mean_c.x() * inv_z + u.cx,
        u.focal_y * mean_c.y() * inv_z + u.cy,
    )
}

/// Read the world mean and transform to camera space.
#[cube]
pub fn read_mean_viewspace_xray(transforms: &Tensor<f32>, base: usize, u: XRayProjectUniforms) -> Vec3A {
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    u.world_to_cam(mean)
}

/// `exp(clamp(log_scale, lo, hi)) · scale_modifier` — **exp5 experiment**:
/// standard-3DGS `exp` activation for X-ray scale instead of softplus.
///
/// Rationale: softplus bottoms out at 1 mm (gradient → 0 as raw → -∞), so
/// fine structures can never shrink below 1 mm; `exp` has no positive floor
/// and its log-space gradient is uniform (`d exp/d raw = exp = σ`), matching
/// the backward kernel's existing `dL/draw = dL/dσ·σ` chain rule. The hard
/// clamp replaces softplus's linear tail as the blow-up guard for the
/// additive path integral (`hi = ln 1000 ≈ 6.9` mm; oversized splats are
/// still caught by the refine prune).
#[cube]
pub fn read_scale_xray(transforms: &Tensor<f32>, base: usize, mod_: f32) -> Vec3A {
    let hi = 6.907755f32; // ln(1000 mm)
    Vec3A::new(
        f32::exp(clamp(transforms[base + 7], -20.0f32, hi)) * mod_,
        f32::exp(clamp(transforms[base + 8], -20.0f32, hi)) * mod_,
        f32::exp(clamp(transforms[base + 9], -20.0f32, hi)) * mod_,
    )
}

/// Un-normalized quaternion from the packed transforms.
#[cube]
pub fn read_quat_unorm_xray(transforms: &Tensor<f32>, base: usize) -> Quat {
    Quat::new(
        transforms[base + 3],
        transforms[base + 4],
        transforms[base + 5],
        transforms[base + 6],
    )
}

/// (xy_x, xy_y) from a global linear pixel id using plain row-major tiles.
#[cube]
pub fn pixel_from_global(global_id: u32, tile_bw: u32) -> (u32, u32) {
    let tile_id = global_id / TILE_SIZE;
    let within = global_id % TILE_SIZE;
    let tile_x = tile_id % tile_bw;
    let tile_y = tile_id / tile_bw;
    (
        tile_x * TILE_WIDTH + within % TILE_WIDTH,
        tile_y * TILE_WIDTH + within / TILE_WIDTH,
    )
}

/// Packed-lane readers (density projection layout).
#[cube]
pub fn read_packed(projected: &Tensor<f32>, idx: u32) -> (f32, f32, Sym2, f32, f32, f32) {
    let b = (idx * XRAY_LANES) as usize;
    (
        projected[b],
        projected[b + 1],
        Sym2 {
            c00: projected[b + 2],
            c01: projected[b + 3],
            c11: projected[b + 4],
        },
        projected[b + 5],
        projected[b + 6],
        projected[b + 7],
    )
}
