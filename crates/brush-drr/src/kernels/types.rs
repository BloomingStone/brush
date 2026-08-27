//! Cube-side uniform structs for the DRR ray-marcher.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// Cube launch uniforms: camera + volume geometry for one DRR projection.
/// A `[H,W]` pixel grid, each pixel marches `steps` rays from `sod - r`
/// to `sod + r` through an **anisotropic** `[vol_x, vol_y, vol_z]` volume
/// spanning `[-rx, rx] x [-ry, ry] x [-rz, rz]` in world coordinates.
/// Memory layout `idx(x,y,z) = x*(vol_y*vol_z) + y*vol_z + z` (x slowest,
/// z fastest — x-major, matches brush-voxel output / R2 fields).
#[derive(CubeLaunch, CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct DrrUniforms {
    pub img_w: u32,
    pub img_h: u32,
    pub vol_x: u32,
    pub vol_y: u32,
    pub vol_z: u32,
    pub steps: u32,
    /// Pinhole focal length (px).
    pub fx: f32,
    pub fy: f32,
    /// Principal point (px).
    pub cx: f32,
    pub cy: f32,
    /// Camera position (world).
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    /// Camera rotation columns (world-from-local), 9 floats.
    pub rot_c0_x: f32,
    pub rot_c0_y: f32,
    pub rot_c0_z: f32,
    pub rot_c1_x: f32,
    pub rot_c1_y: f32,
    pub rot_c1_z: f32,
    pub rot_c2_x: f32,
    pub rot_c2_y: f32,
    pub rot_c2_z: f32,
    /// Source-to-object distance (world mm).
    pub sod: f32,
    /// Volume half extents (world mm) per axis.
    pub rx: f32,
    pub ry: f32,
    pub rz: f32,
    /// `1/voxel = vol_axis / (2*half_axis)` per axis.
    pub inv_dx: f32,
    pub inv_dy: f32,
    pub inv_dz: f32,
    /// Output scale/bias: `proj = scale * integral + bias`.
    pub scale: f32,
    pub bias: f32,
}

impl DrrUniforms {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_w: u32,
        img_h: u32,
        vol_x: u32,
        vol_y: u32,
        vol_z: u32,
        steps: u32,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        cam_x: f32,
        cam_y: f32,
        cam_z: f32,
        rot_c0_x: f32,
        rot_c0_y: f32,
        rot_c0_z: f32,
        rot_c1_x: f32,
        rot_c1_y: f32,
        rot_c1_z: f32,
        rot_c2_x: f32,
        rot_c2_y: f32,
        rot_c2_z: f32,
        sod: f32,
        rx: f32,
        ry: f32,
        rz: f32,
        inv_dx: f32,
        inv_dy: f32,
        inv_dz: f32,
        scale: f32,
        bias: f32,
    ) -> Self {
        Self {
            img_w,
            img_h,
            vol_x,
            vol_y,
            vol_z,
            steps,
            fx,
            fy,
            cx,
            cy,
            cam_x,
            cam_y,
            cam_z,
            rot_c0_x,
            rot_c0_y,
            rot_c0_z,
            rot_c1_x,
            rot_c1_y,
            rot_c1_z,
            rot_c2_x,
            rot_c2_y,
            rot_c2_z,
            sod,
            rx,
            ry,
            rz,
            inv_dx,
            inv_dy,
            inv_dz,
            scale,
            bias,
        }
    }
}
