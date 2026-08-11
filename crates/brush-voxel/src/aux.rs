//! Voxelizer output containers (backend primitives + host aux).

use burn::backend::{
    Backend, ExtensionType,
    tensor::{FloatTensor, IntTensor},
};

use crate::host::VoxelUniformsHost;

/// Internal voxelize output used by kernel impls. Holds backend primitives.
#[derive(Debug, Clone, ExtensionType)]
pub struct VoxelOutput<B: Backend> {
    /// Single-channel density volume `[nVoxel_x, nVoxel_y, nVoxel_z]` f32.
    pub out_volume: FloatTensor<B>,
    #[extension_type]
    pub aux: VoxelAuxInner<B>,
    /// Sparse `[num_visible, VOXEL_LANES]` packed splats (compact-indexed).
    pub projected_splats: FloatTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    /// Uniforms needed by the backward pass.
    pub uniforms: VoxelUniformsHost,
    pub global_from_compact_gid: IntTensor<B>,
}

/// Internal aux struct holding backend primitives.
#[derive(Debug, Clone, ExtensionType)]
pub struct VoxelAuxInner<B: Backend> {
    pub num_visible: u32,
    pub num_intersections: u32,
    /// Per-cube `[start, end)` offsets, `[num_cubes, 2]`.
    pub cube_offsets: IntTensor<B>,
    /// Per-voxel contributor count (backward bookkeeping), `[N]`.
    pub n_contrib: IntTensor<B>,
    pub n_voxel: glam::UVec3,
}
