//! Cube-side uniform structs for the DRR ray-marcher.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// Cube launch uniforms: camera + volume geometry for one DRR projection.
/// A `[H,W]` pixel grid, each pixel marches `steps` rays from `sod - half_r`
/// to `sod + half_r` through a `[vol,vol,vol]` volume spanning
/// `[-half_r, half_r]^3` in world coordinates.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct DrrUniforms {
    pub img_w: u32,
    pub img_h: u32,
    pub vol: u32,
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
    /// Volume half extent (world mm) — grid spans [-half_r, half_r]^3.
    pub half_r: f32,
    /// `1/voxel_size = vol / (2*half_r)`.
    pub inv_delta: f32,
    /// Output scale/bias: `proj = scale * integral + bias`.
    pub scale: f32,
    pub bias: f32,
}

impl DrrUniforms {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_w: u32,
        img_h: u32,
        vol: u32,
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
        half_r: f32,
        inv_delta: f32,
        scale: f32,
        bias: f32,
    ) -> Self {
        Self {
            img_w,
            img_h,
            vol,
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
            half_r,
            inv_delta,
            scale,
            bias,
        }
    }
}
