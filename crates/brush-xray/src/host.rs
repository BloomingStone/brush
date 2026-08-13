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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the cube-side view matrix built by [`XRayProjectUniforms`]
    /// reproduces the Python `get_proj_matric.py` reference projection for
    /// an AP C-arm at alpha=beta=0 (gs/COLMAP convention, `rotate_dsa`
    /// geometry: SDD=1200, SOD=760, 648x474, delx=0.616).
    ///
    /// Python reference (cx=W/2=324, cy=H/2=237; brush uses (S-1)/2 → ±0.5 px):
    ///   (0,0,0)    → (324,   237)
    ///   (100,0,0)  → (67.7,  237)
    ///   (0,100,0)  → (324,   237)
    ///   (0,0,100)  → (324,  -19.3)
    ///   (-100,0,0) → (580.3, 237)
    ///   (0,0,-100) → (324,   493.3)
    #[test]
    fn from_camera_viewmat_matches_python_reference() {
        // Reorient(AP, gs) = [[-1,0,0],[0,0,-1],[0,-1,0]] (row-major), which is
        // symmetric, so passing it as columns gives the same matrix.
        let rotation = glam::DQuat::from_mat3(&glam::DMat3::from_cols_array_2d(&[
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ]));
        // FOVs in RADIANS (brush Camera convention): fx=fy=1200/0.616=1948.05,
        // fov_x=2·atan(324/1948.05), fov_y=2·atan(237/1948.05).
        let cam = Camera::new(
            glam::Vec3::new(0.0, 760.0, 0.0),
            glam::Quat::from_xyzw(
                rotation.x as f32,
                rotation.y as f32,
                rotation.z as f32,
                rotation.w as f32,
            ),
            2.0 * (324.0_f64 / (1200.0 / 0.616)).atan(),
            2.0 * (237.0_f64 / (1200.0 / 0.616)).atan(),
            glam::Vec2::splat(0.5),
            brush_render::kernels::camera_model::CameraModel::Pinhole,
        );
        let img = glam::uvec2(648, 474);
        let u = XRayProjectUniforms::from_camera(&cam, img, 1000, 1.0);

        // Column-major viewmat.
        let v = |c: usize, r: usize| u.viewmat[c][r];
        let w2c = |p: glam::Vec3| -> glam::Vec3 {
            glam::Vec3::new(
                v(0, 0) * p.x + v(1, 0) * p.y + v(2, 0) * p.z + v(3, 0),
                v(0, 1) * p.x + v(1, 1) * p.y + v(2, 1) * p.z + v(3, 1),
                v(0, 2) * p.x + v(1, 2) * p.y + v(2, 2) * p.z + v(3, 2),
            )
        };

        let pts = [
            ([0.0f32, 0.0, 0.0], (324.0, 237.0)),
            ([100.0, 0.0, 0.0], (67.7, 237.0)),
            ([0.0, 100.0, 0.0], (324.0, 237.0)),
            ([0.0, 0.0, 100.0], (324.0, -19.3)),
            ([-100.0, 0.0, 0.0], (580.3, 237.0)),
            ([0.0, 0.0, -100.0], (324.0, 493.3)),
        ];
        for (p, (ref_u, ref_v)) in pts {
            let p_c = w2c(glam::Vec3::new(p[0], p[1], p[2]));
            let uu = u.focal.x * p_c.x / p_c.z + u.center.x;
            let vv = u.focal.y * p_c.y / p_c.z + u.center.y;
            let tol = 1.0;
            assert!(
                (uu - ref_u).abs() < tol && (vv - ref_v).abs() < tol,
                "P={p:?} P_c={p_c:?}: uv=({uu:.2},{vv:.2}) expected ({ref_u:.1},{ref_v:.1})"
            );
        }
    }
}
