//! Host-side voxelizer uniforms: build the cube launch object from a
//! [`VoxelSettings`]. Carried (as a plain struct) across the backend
//! boundary / backward state.

use burn_cubecl::cubecl::wgpu::WgpuRuntime;

use crate::kernels::helpers::BLOCK3D_X;
use crate::kernels::types::VoxelUniformsLaunch;
use crate::settings::VoxelSettings;

/// Host mirror of the cube-side `VoxelUniforms`.
#[derive(Debug, Clone, Copy)]
pub struct VoxelUniformsHost {
    pub n_voxel: glam::UVec3,
    /// `1/dVoxel = nVoxel/sVoxel` per axis.
    pub inv_d_voxel: glam::Vec3,
    /// `dSoxel = sVoxel/nVoxel` per axis.
    pub d_voxel: glam::Vec3,
    pub s_voxel: glam::Vec3,
    pub center: glam::Vec3,
    /// Cube-grid dims: `ceil(nVoxel/BLOCK3D)` per axis.
    pub grid: glam::UVec3,
    pub num_cubes: u32,
    pub total_splats: u32,
    pub num_visible: u32,
    pub num_intersections: u32,
    pub scale_modifier: f32,
}

impl VoxelUniformsHost {
    pub fn from_settings(settings: &VoxelSettings, total_splats: u32) -> Self {
        let d_voxel = settings.d_voxel();
        let inv_d_voxel = settings.n_voxel.as_vec3() / settings.s_voxel;
        let grid = glam::uvec3(
            settings.n_voxel.x.div_ceil(BLOCK3D_X),
            settings.n_voxel.y.div_ceil(BLOCK3D_X),
            settings.n_voxel.z.div_ceil(BLOCK3D_X),
        );
        let num_cubes = grid.x * grid.y * grid.z;
        Self {
            n_voxel: settings.n_voxel,
            inv_d_voxel,
            d_voxel,
            s_voxel: settings.s_voxel,
            center: settings.center,
            grid,
            num_cubes,
            total_splats,
            num_visible: 0,
            num_intersections: 0,
            scale_modifier: settings.scale_modifier,
        }
    }

    /// Build the cube-side launch arg.
    pub fn to_launch_object(&self) -> VoxelUniformsLaunch<WgpuRuntime> {
        VoxelUniformsLaunch::new(
            self.n_voxel.x,
            self.n_voxel.y,
            self.n_voxel.z,
            self.inv_d_voxel.x,
            self.inv_d_voxel.y,
            self.inv_d_voxel.z,
            self.d_voxel.x,
            self.d_voxel.y,
            self.d_voxel.z,
            self.s_voxel.x,
            self.s_voxel.y,
            self.s_voxel.z,
            self.center.x,
            self.center.y,
            self.center.z,
            self.grid.x,
            self.grid.y,
            self.grid.z,
            self.num_cubes,
            self.total_splats,
            self.num_visible,
            self.scale_modifier,
        )
    }
}
