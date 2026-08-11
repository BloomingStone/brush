//! Voxel-geometry helpers: voxel-space covariance, 3D conic inverse,
//! normalized voxel coordinates, 3σ radius, and the 3D cube bounding box
//! — mirroring R2-Gaussian's `cuda_voxelizer` `preprocessCUDA_voxel`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{Quat, Vec3A, compute_cov3d};

use super::types::VoxelUniforms;

pub const BLOCK3D_X: u32 = 8;
pub const BLOCK3D_Y: u32 = 8;
pub const BLOCK3D_Z: u32 = 8;
pub const BLOCK3D_SIZE: u32 = BLOCK3D_X * BLOCK3D_Y * BLOCK3D_Z;
/// Packed lanes per visible splat: point_vol(3) + conic(6) + opacity(1) + radius(3).
pub const VOXEL_LANES: u32 = 13;
pub const VOXEL_LANES_USIZE: usize = VOXEL_LANES as usize;
pub const MIN_ALPHA: f32 = 1.0e-6f32;

/// 3D cube bounding box (inclusive min, exclusive max) in cube units,
/// mirroring R2 `getCube`.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct VoxelBbox {
    pub min_x: u32,
    pub min_y: u32,
    pub min_z: u32,
    pub max_x: u32,
    pub max_y: u32,
    pub max_z: u32,
}

impl VoxelBbox {
    /// Number of cubes in the bbox (== `tiles_touched`).
    pub fn volume(self) -> u32 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)
    }
}

#[cube]
impl VoxelBbox {
    /// Cube-side volume (== `tiles_touched`).
    pub fn cube_volume(self) -> u32 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)
    }
}

/// R2 `getCube`: `min = clamp((p - r)/BLOCK3D, 0, grid)`,
/// `max = clamp((p + r + BLOCK3D - 1)/BLOCK3D, 0, grid)` (floor division).
#[cube]
pub fn get_cube_bbox(point_vol: Vec3A, radius: Vec3A, u: VoxelUniforms) -> VoxelBbox {
    let b_x = BLOCK3D_X as f32;
    let b_y = BLOCK3D_Y as f32;
    let b_z = BLOCK3D_Z as f32;

    let min_x = f32::min(
        u.grid_x as f32,
        f32::max(0.0f32, f32::floor((point_vol.x() - radius.x()) / b_x)),
    ) as u32;
    let min_y = f32::min(
        u.grid_y as f32,
        f32::max(0.0f32, f32::floor((point_vol.y() - radius.y()) / b_y)),
    ) as u32;
    let min_z = f32::min(
        u.grid_z as f32,
        f32::max(0.0f32, f32::floor((point_vol.z() - radius.z()) / b_z)),
    ) as u32;

    let max_x = f32::min(
        u.grid_x as f32,
        f32::max(0.0f32, f32::floor((point_vol.x() + radius.x() + b_x - 1.0f32) / b_x)),
    ) as u32;
    let max_y = f32::min(
        u.grid_y as f32,
        f32::max(0.0f32, f32::floor((point_vol.y() + radius.y() + b_y - 1.0f32) / b_y)),
    ) as u32;
    let max_z = f32::min(
        u.grid_z as f32,
        f32::max(0.0f32, f32::floor((point_vol.z() + radius.z() + b_z - 1.0f32) / b_z)),
    ) as u32;

    VoxelBbox {
        min_x,
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
    }
}

/// Compute the voxel-space geometry for a gaussian:
/// returns `(point_vol, inv_a, inv_b, inv_c, inv_d, inv_e, inv_f, radius,
/// valid)` where `inv_*` is the 3D conic (inverse of the voxel-space
/// covariance, stored upper-triangle), `radius` is the 3σ per-axis radius
/// in voxel units, and `valid` is false iff the covariance is singular
/// (det == 0). Mirrors R2 `preprocessCUDA_voxel`.
///
/// `scale` must already include `scale_modifier` (like the rasterizer);
/// the radius however uses the RAW (unmodified) scale, exactly like R2.
#[cube]
pub fn voxel_geometry(
    mean: Vec3A,
    scale: Vec3A,
    scale_raw: Vec3A,
    quat: Quat,
    u: VoxelUniforms,
) -> (Vec3A, f32, f32, f32, f32, f32, f32, Vec3A, bool) {
    let vrk = compute_cov3d(scale, quat);

    // cov_voxel = M^T · Vrk · M with M = diag(1/dVoxel). Since M is
    // diagonal, cov_voxel[i][j] = vrk[i][j] · (1/dVoxel_i) · (1/dVoxel_j).
    let ix = u.inv_d_voxel_x;
    let iy = u.inv_d_voxel_y;
    let iz = u.inv_d_voxel_z;
    let hata = vrk.c00 * ix * ix;
    let hatb = vrk.c01 * ix * iy;
    let hatc = vrk.c02 * ix * iz;
    let hatd = vrk.c11 * iy * iy;
    let hate = vrk.c12 * iy * iz;
    let hatf = vrk.c22 * iz * iz;

    let det = hata * hatd * hatf + 2.0f32 * hatb * hatc * hate - hata * hate * hate
        - hatf * hatb * hatb - hatd * hatc * hatc;
    let valid = det != 0.0f32;
    let det_inv = select(valid, 1.0f32 / det, 0.0f32);

    let inv_a = (hatd * hatf - hate * hate) * det_inv;
    let inv_b = (hatc * hate - hatb * hatf) * det_inv;
    let inv_c = (hatb * hate - hatc * hatd) * det_inv;
    let inv_d = (hata * hatf - hatc * hatc) * det_inv;
    let inv_e = (hatb * hatc - hata * hate) * det_inv;
    let inv_f = (hata * hatd - hatb * hatb) * det_inv;

    let max_scale = f32::max(scale_raw.x(), f32::max(scale_raw.y(), scale_raw.z()));
    let radius = Vec3A::new(
        f32::ceil(3.0f32 * max_scale * u.inv_d_voxel_x),
        f32::ceil(3.0f32 * max_scale * u.inv_d_voxel_y),
        f32::ceil(3.0f32 * max_scale * u.inv_d_voxel_z),
    );

    // point_vol = (mean - center + sVoxel/2) / dVoxel.
    let point_vol = Vec3A::new(
        (mean.x() - u.center_x + 0.5f32 * u.s_voxel_x) * u.inv_d_voxel_x,
        (mean.y() - u.center_y + 0.5f32 * u.s_voxel_y) * u.inv_d_voxel_y,
        (mean.z() - u.center_z + 0.5f32 * u.s_voxel_z) * u.inv_d_voxel_z,
    );

    (point_vol, inv_a, inv_b, inv_c, inv_d, inv_e, inv_f, radius, valid)
}

/// Read the raw (unmodified) scale `exp(log_scales)` from the packed
/// transforms (used for the 3σ radius, exactly like R2).
#[cube]
pub fn read_scale_raw(transforms: &Tensor<f32>, base: usize) -> Vec3A {
    Vec3A::new(
        f32::exp(transforms[base + 7]),
        f32::exp(transforms[base + 8]),
        f32::exp(transforms[base + 9]),
    )
}

/// Read the scale `exp(log_scales) · scale_modifier`.
#[cube]
pub fn read_scale_mod(transforms: &Tensor<f32>, base: usize, scale_modifier: f32) -> Vec3A {
    Vec3A::new(
        f32::exp(transforms[base + 7]) * scale_modifier,
        f32::exp(transforms[base + 8]) * scale_modifier,
        f32::exp(transforms[base + 9]) * scale_modifier,
    )
}

/// Read the quaternion `(w,x,y,z)` from the packed transforms. Callers
/// normalize it (`.normalize()`) before use — mirroring the rasterizer so
/// the voxelizer is consistent with R2's calling convention (the caller
/// normalizes), and the backward applies the corresponding `dnormvdv4`
/// VJP.
#[cube]
pub fn read_quat(transforms: &Tensor<f32>, base: usize) -> Quat {
    Quat::new(
        transforms[base + 3],
        transforms[base + 4],
        transforms[base + 5],
        transforms[base + 6],
    )
}

/// Voxel index (x-major): `id = nVoxel_z·nVoxel_y·x + nVoxel_z·y + z`.
#[cube]
pub fn voxel_id_from_xyz(x: u32, y: u32, z: u32, u: VoxelUniforms) -> u32 {
    u.n_voxel_z * u.n_voxel_y * x + u.n_voxel_z * y + z
}
