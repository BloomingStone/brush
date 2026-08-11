//! Cube-side uniform structs for the voxelizer.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

use super::helpers::{BLOCK3D_X, BLOCK3D_Y, BLOCK3D_Z};

/// Cube launch uniforms for the 3D voxelizer. Carries the grid geometry
/// (voxel counts, spacing, center) plus the cube-grid dims.
#[derive(CubeLaunch, CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct VoxelUniforms {
    pub n_voxel_x: u32,
    pub n_voxel_y: u32,
    pub n_voxel_z: u32,
    /// `1/dVoxel = nVoxel/sVoxel` per axis.
    pub inv_d_voxel_x: f32,
    pub inv_d_voxel_y: f32,
    pub inv_d_voxel_z: f32,
    /// `dSoxel = sVoxel/nVoxel` per axis.
    pub d_voxel_x: f32,
    pub d_voxel_y: f32,
    pub d_voxel_z: f32,
    pub s_voxel_x: f32,
    pub s_voxel_y: f32,
    pub s_voxel_z: f32,
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    /// Cube-grid dims: `ceil(nVoxel / BLOCK3D)` per axis.
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub num_cubes: u32,
    pub total_splats: u32,
    pub num_visible: u32,
    pub scale_modifier: f32,
}

impl VoxelUniforms {
    /// Build from host values (used by the launch helpers).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_voxel_x: u32,
        n_voxel_y: u32,
        n_voxel_z: u32,
        inv_d_voxel_x: f32,
        inv_d_voxel_y: f32,
        inv_d_voxel_z: f32,
        d_voxel_x: f32,
        d_voxel_y: f32,
        d_voxel_z: f32,
        s_voxel_x: f32,
        s_voxel_y: f32,
        s_voxel_z: f32,
        center_x: f32,
        center_y: f32,
        center_z: f32,
        total_splats: u32,
        num_visible: u32,
        scale_modifier: f32,
    ) -> Self {
        let grid_x = n_voxel_x.div_ceil(BLOCK3D_X);
        let grid_y = n_voxel_y.div_ceil(BLOCK3D_Y);
        let grid_z = n_voxel_z.div_ceil(BLOCK3D_Z);
        let num_cubes = grid_x * grid_y * grid_z;
        Self {
            n_voxel_x,
            n_voxel_y,
            n_voxel_z,
            inv_d_voxel_x,
            inv_d_voxel_y,
            inv_d_voxel_z,
            d_voxel_x,
            d_voxel_y,
            d_voxel_z,
            s_voxel_x,
            s_voxel_y,
            s_voxel_z,
            center_x,
            center_y,
            center_z,
            grid_x,
            grid_y,
            grid_z,
            num_cubes,
            total_splats,
            num_visible,
            scale_modifier,
        }
    }
}
