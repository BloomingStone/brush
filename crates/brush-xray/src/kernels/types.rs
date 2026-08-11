//! Cube-side uniform structs for the X-ray rasterizer kernels.

use brush_cube::{Mat3, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

/// Project/uniforms for the cone-beam X-ray rasterizer. Carries the 3x4
/// view matrix (world→cam, column-major), cone-beam focal lengths and
/// Jacobian clamp limits, principal point, and image/grid dims.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct XRayProjectUniforms {
    // 3x4 view matrix, column-major. `vm{i}_*` is column i.
    pub vm0_x: f32,
    pub vm0_y: f32,
    pub vm0_z: f32,
    pub vm1_x: f32,
    pub vm1_y: f32,
    pub vm1_z: f32,
    pub vm2_x: f32,
    pub vm2_y: f32,
    pub vm2_z: f32,
    pub vm3_x: f32,
    pub vm3_y: f32,
    pub vm3_z: f32,
    pub focal_x: f32,
    pub focal_y: f32,
    /// Jacobian clamp limits, `±1.3·tan(fov/2)` for the cone beam.
    pub lim_pos_x: f32,
    pub lim_pos_y: f32,
    pub lim_neg_x: f32,
    pub lim_neg_y: f32,
    /// Principal point in pixels.
    pub cx: f32,
    pub cy: f32,
    pub img_w: u32,
    pub img_h: u32,
    pub tile_bw: u32,
    pub tile_bh: u32,
    pub total_splats: u32,
    pub num_visible: u32,
    /// Linear scale multiplier applied before covariance (R2 `scale_modifier`).
    pub scale_modifier: f32,
}

#[cube]
impl XRayProjectUniforms {
    /// Top-left 3x3 of the world-to-cam viewmat.
    pub fn view_rotation(self) -> Mat3 {
        Mat3 {
            c0_x: self.vm0_x,
            c0_y: self.vm0_y,
            c0_z: self.vm0_z,
            c1_x: self.vm1_x,
            c1_y: self.vm1_y,
            c1_z: self.vm1_z,
            c2_x: self.vm2_x,
            c2_y: self.vm2_y,
            c2_z: self.vm2_z,
        }
    }

    /// Translation column of the world-to-cam viewmat.
    pub fn view_translation(self) -> Vec3A {
        Vec3A::new(self.vm3_x, self.vm3_y, self.vm3_z)
    }

    /// World → camera space.
    pub fn world_to_cam(self, mean: Vec3A) -> Vec3A {
        self.view_rotation().mul_vec3(mean).add(self.view_translation())
    }
}

/// Rasterize-pass uniforms (additive, no background).
#[derive(CubeLaunch, CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct XRayRasterizeUniforms {
    pub tile_bw: u32,
    pub img_w: u32,
    pub img_h: u32,
}
