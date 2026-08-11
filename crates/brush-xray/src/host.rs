//! Host-side X-ray uniforms: build the cube launch object from a brush
//! pinhole `Camera`. Also carried (as a plain struct) across the backend
//! boundary / backward state.

use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use brush_render::camera::Camera;

use crate::kernels::helpers::TILE_WIDTH;
use crate::kernels::types::XRayProjectUniformsLaunch;

/// Host mirror of the cube-side `XRayProjectUniforms`. Holds everything
/// needed to rebuild the launch object for the backward pass.
#[derive(Debug, Clone, Copy)]
pub struct XRayProjectUniforms {
    /// 3x4 view matrix, column-major (`to_cols_array_2d`).
    pub viewmat: [[f32; 4]; 4],
    /// Cone-beam focal lengths (pixels).
    pub focal: glam::Vec2,
    /// Principal point (pixels).
    pub center: glam::Vec2,
    /// `[lim_pos_x, lim_pos_y, lim_neg_x, lim_neg_y]` — Jacobian clamp
    /// limits, `±1.3·tan(fov/2)` for the cone beam.
    /// When calculating the derivative of the projection (backpropagation), 
    /// it's used to prevent numerical overflow. 
    /// Apply the same clamping range to all Splat.
    pub lims: [f32; 4],
    pub img_size: glam::UVec2,
    pub tile_bounds: glam::UVec2,
    pub total_splats: u32,
    /// Number of visible splats (for backward bookkeeping). mutably updated 
    /// by the forward pass in CPU, and used to allocate the backward buffers. 
    pub num_visible: u32,
    /// Linear scale multiplier applied to all splats before covariance 
    /// (R2-Gaussian `scale_modifier`).
    pub scale_modifier: f32,
}

impl XRayProjectUniforms {
    /// Build from a pinhole `Camera` + image size. Only cone-beam
    /// projection is supported; the parallel-beam mode of R2-Gaussian is
    /// intentionally not implemented.
    pub fn from_camera(
        camera: &Camera,
        img_size: glam::UVec2,
        total_splats: u32,
        scale_modifier: f32,
    ) -> Self {
        let viewmat = glam::Mat4::from(camera.world_to_local()).to_cols_array_2d();
        let focal = camera.focal(img_size);
        // R2-Gaussian's `ndc2Pix` is `((ndc+1)·S - 1)·0.5`, so the effective
        // principal point sits at `(S-1)/2` (pixel-corner convention) — NOT at
        // `S/2`. The cone-beam cov2d is principal-point-independent; only the
        // tiling center / pixel sampling use this, and they must match R2.
        let center = glam::vec2(
            (img_size.x as f32 - 1.0) * 0.5,
            (img_size.y as f32 - 1.0) * 0.5,
        );

        let tan_fovx = (camera.fov_x as f32 * 0.5).tan();
        let tan_fovy = (camera.fov_y as f32 * 0.5).tan();
        let lim = 1.3f32;

        Self {
            viewmat,
            focal,
            center,
            lims: [
                lim * tan_fovx,
                lim * tan_fovy,
                -lim * tan_fovx,
                -lim * tan_fovy,
            ],
            img_size,
            tile_bounds: glam::uvec2(
                img_size.x.div_ceil(TILE_WIDTH),
                img_size.y.div_ceil(TILE_WIDTH),
            ),
            total_splats,
            num_visible: 0,
            scale_modifier,
        }
    }

    /// Build the cube-side launch arg.
    pub fn to_launch_object(&self) -> XRayProjectUniformsLaunch<WgpuRuntime> {
        XRayProjectUniformsLaunch::new(
            self.viewmat[0][0],
            self.viewmat[0][1],
            self.viewmat[0][2],
            self.viewmat[1][0],
            self.viewmat[1][1],
            self.viewmat[1][2],
            self.viewmat[2][0],
            self.viewmat[2][1],
            self.viewmat[2][2],
            self.viewmat[3][0],
            self.viewmat[3][1],
            self.viewmat[3][2],
            self.focal.x,
            self.focal.y,
            self.lims[0],
            self.lims[1],
            self.lims[2],
            self.lims[3],
            self.center.x,
            self.center.y,
            self.img_size.x,
            self.img_size.y,
            self.tile_bounds.x,
            self.tile_bounds.y,
            self.total_splats,
            self.num_visible,
            self.scale_modifier,
        )
    }
}
