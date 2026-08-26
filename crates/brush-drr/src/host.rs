//! Host-side DRR settings: build the cube launch object from a camera +
//! volume geometry. Carried (as a plain struct) across the backend
//! boundary.

use brush_render::camera::Camera;

use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use crate::kernels::types::DrrUniformsLaunch;

/// Geometry for one DRR projection (a specific camera + volume).
#[derive(Debug, Clone, Copy)]
pub struct DrrSettings {
    pub img_w: u32,
    pub img_h: u32,
    /// Volume grid size (cubic).
    pub vol: u32,
    /// Ray-march samples per pixel.
    pub steps: u32,
    /// Volume half extent (world mm) — grid spans `[-half_r, half_r]^3`.
    pub half_r: f32,
    /// Output calibration: `proj = scale * integral + bias`.
    pub scale: f32,
    pub bias: f32,
    // Camera intrinsics (px).
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    // Camera extrinsics (world).
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    /// World-from-local rotation columns (9 floats).
    pub rot0: [f32; 3],
    pub rot1: [f32; 3],
    pub rot2: [f32; 3],
    pub sod: f32,
}

impl DrrSettings {
    /// Build from a camera + image size. `half_r` is the volume half extent,
    /// `scale`/`bias` the LSQ calibration of the FDK prior.
    pub fn new(
        cam: &Camera,
        img_w: u32,
        img_h: u32,
        vol: u32,
        steps: u32,
        half_r: f32,
        scale: f32,
        bias: f32,
    ) -> Self {
        let size = glam::uvec2(img_w, img_h);
        let f = cam.focal(size);
        let c = cam.center(size);
        let p = cam.position;
        let m = glam::Mat3::from_quat(cam.rotation).to_cols_array();
        Self {
            img_w,
            img_h,
            vol,
            steps,
            half_r,
            scale,
            bias,
            fx: f.x,
            fy: f.y,
            cx: c.x,
            cy: c.y,
            cam_x: p.x,
            cam_y: p.y,
            cam_z: p.z,
            rot0: [m[0], m[1], m[2]],
            rot1: [m[3], m[4], m[5]],
            rot2: [m[6], m[7], m[8]],
            sod: p.length(),
        }
    }

    pub fn inv_delta(&self) -> f32 {
        self.vol as f32 / (2.0 * self.half_r)
    }

    /// Build the cube-side launch arg.
    pub fn to_launch_object(&self) -> DrrUniformsLaunch<WgpuRuntime> {
        DrrUniformsLaunch::new(
            self.img_w,
            self.img_h,
            self.vol,
            self.steps,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.cam_x,
            self.cam_y,
            self.cam_z,
            self.rot0[0],
            self.rot0[1],
            self.rot0[2],
            self.rot1[0],
            self.rot1[1],
            self.rot1[2],
            self.rot2[0],
            self.rot2[1],
            self.rot2[2],
            self.sod,
            self.half_r,
            self.inv_delta(),
            self.scale,
            self.bias,
        )
    }
}
