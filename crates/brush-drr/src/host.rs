//! Host-side DRR settings: build the cube launch object from a camera +
//! volume geometry. Carried (as a plain struct) across the backend
//! boundary.

use brush_render::camera::Camera;

use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use crate::kernels::types::DrrUniformsLaunch;

/// Geometry for one DRR projection (a specific camera + anisotropic volume).
#[derive(Debug, Clone, Copy)]
pub struct DrrSettings {
    pub img_w: u32,
    pub img_h: u32,
    /// Volume grid size per axis.
    pub vol_x: u32,
    pub vol_y: u32,
    pub vol_z: u32,
    /// Ray-march samples per pixel.
    pub steps: u32,
    /// Volume half extents (world mm).
    pub rx: f32,
    pub ry: f32,
    pub rz: f32,
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
    /// Build from a camera + image size + anisotropic volume geometry.
    pub fn new(
        cam: &Camera,
        img_w: u32,
        img_h: u32,
        vol_x: u32,
        vol_y: u32,
        vol_z: u32,
        steps: u32,
        rx: f32,
        ry: f32,
        rz: f32,
        scale: f32,
        bias: f32,
    ) -> Self {
        let size = glam::uvec2(img_w, img_h);
        let f = cam.focal(size);
        // Principal point, pixel-center-at-integer convention: `(S-1)` span
        // matches brush-xray's R2 `ndc2Pix` (center at (S-1)/2 for
        // center_uv=0.5), so both pipelines sample the same rays. Using
        // `camera.center()` (`center_uv·S`) would offset every ray by half
        // a pixel from the rasterizer's.
        let c = glam::vec2(cam.center_uv.x * (img_w - 1) as f32, cam.center_uv.y * (img_h - 1) as f32);
        let p = cam.position;
        let m = glam::Mat3::from_quat(cam.rotation).to_cols_array();
        Self {
            img_w,
            img_h,
            vol_x,
            vol_y,
            vol_z,
            steps,
            rx,
            ry,
            rz,
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

    fn inv_delta(&self, vol: u32, r: f32) -> f32 {
        vol as f32 / (2.0 * r)
    }

    /// Build the cube-side launch arg.
    pub fn to_launch_object(&self) -> DrrUniformsLaunch<WgpuRuntime> {
        DrrUniformsLaunch::new(
            self.img_w,
            self.img_h,
            self.vol_x,
            self.vol_y,
            self.vol_z,
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
            self.rx,
            self.ry,
            self.rz,
            self.inv_delta(self.vol_x, self.rx),
            self.inv_delta(self.vol_y, self.ry),
            self.inv_delta(self.vol_z, self.rz),
            self.scale,
            self.bias,
        )
    }
}
