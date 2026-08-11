//! Voxel grid settings for the 3D density-volume voxelizer.
//!
//! Mirrors R2-Gaussian's `GaussianVoxelizationSettings`: the volume spans
//! `center ± sVoxel/2` in world units, split into `nVoxel` voxels per
//! axis. No camera — the voxelizer only needs splat params + the grid.

use serde::{Deserialize, Serialize};

/// Voxel grid configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct VoxelSettings {
    /// Number of voxels along each axis `(nVoxel_x, nVoxel_y, nVoxel_z)`.
    pub n_voxel: glam::UVec3,
    /// Total voxelised extent along each axis `(sVoxel_x, ...)` — the
    /// volume spans `center ± sVoxel/2`.
    pub s_voxel: glam::Vec3,
    /// Volume center in world units.
    pub center: glam::Vec3,
    /// Scale modifier (like the rasterizer's `scale_modifier`).
    pub scale_modifier: f32,
}

impl VoxelSettings {
    pub fn new(n_voxel: glam::UVec3, s_voxel: glam::Vec3, center: glam::Vec3) -> Self {
        Self {
            n_voxel,
            s_voxel,
            center,
            scale_modifier: 1.0,
        }
    }

    /// World-space size of a single voxel along each axis (`sVoxel/nVoxel`).
    pub fn d_voxel(&self) -> glam::Vec3 {
        self.s_voxel / self.n_voxel.as_vec3()
    }
}

