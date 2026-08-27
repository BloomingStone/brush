//! 3D density-volume voxelizer for Gaussian splats.
//!
//! Port of R2-Gaussian's `cuda_voxelizer` (voxelization of
//! `xray-gaussian-rasterization-voxelization`) from CUDA to
//! backend-agnostic cubecl kernels. Independent of any camera: each
//! voxel accumulates `opacity · exp(power)` over the gaussians whose 3σ
//! bbox overlaps its cube, where `power` uses the 3D conic in voxel
//! space. The `raw_opacities` input carries **raw density logits**, and
//! the kernel applies the same activation as brush-xray (`MU_WATER·silu`
//! unsigned / `MU_WATER·raw` signed via [`VoxelSettings::signed_opac`]) —
//! all brush backends share the "raw logits in, kernel activates"
//! convention. Forward + backward live in this crate (mirroring
//! `brush-xray` / `brush-xray-bwd` but without a projection step).

use brush_cube::MainBackend as Wgpu;
use burn::backend::Backend;
use burn::backend::tensor::FloatTensor;
use brush_render::burn_glue::{unwrap_wgpu_float, wrap_wgpu_float};
use brush_xray::XRaySplats;
use burn::tensor::Tensor;

pub use crate::aux::{VoxelAuxInner, VoxelOutput};
pub use crate::backward::{VoxelBwdOps, VoxelRasterizeGrads, VoxelSplatGrads, voxelize_bwd_pipeline};
pub use crate::burn_glue::voxelize;
pub use crate::host::VoxelUniformsHost;
pub use crate::settings::VoxelSettings;

pub mod aux;
pub mod backward;
pub mod burn_glue;
pub mod host;
#[doc(hidden)]
pub mod kernels;
pub mod pipeline;
pub mod settings;

/// Forward/backward pass for the voxelizer.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VoxelPass {
    /// Forward only — inference / eval. No backward bookkeeping.
    #[default]
    Forward,
    /// Forward + backward bookkeeping (training): per-voxel contributor
    /// counts.
    Backward,
}

impl VoxelPass {
    pub const fn bwd_info(self) -> bool {
        matches!(self, Self::Backward)
    }
}

/// Trait for the 3D density-volume voxelizer.
///
/// A single call performs: preprocess (voxel cov + 3D conic + radius +
/// cube count) → readback → cube bin → 3D additive render.
#[burn::backend::backend_extension(Wgpu)]
pub trait VoxelOps: Backend {
    fn voxelize(
        settings: &VoxelSettings,
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        pass: VoxelPass,
    ) -> impl Future<Output = VoxelOutput<Self>>;
}

/// Forward-only, high-level voxelization of `splats` into a density
/// volume `[nVoxel_x, nVoxel_y, nVoxel_z]`. Uses `VoxelPass::Forward`
/// (no backward bookkeeping). Non-differentiable — see the
/// `burn_glue::voxelize` path for autodiff.
///
/// Takes [`XRaySplats`] (transforms + raw opacity logits only): like the
/// X-ray projection, voxelization is a pure density field and ignores SH.
/// The kernel activates the density (`MU_WATER·silu(raw)`, or
/// `MU_WATER·raw` when `signed_opac`), exactly like brush-xray.
pub async fn voxelize_forward(
    splats: &XRaySplats,
    settings: &VoxelSettings,
) -> Tensor<3> {
    let transforms = unwrap_wgpu_float(splats.transforms.val());
    let raw_opac = unwrap_wgpu_float(splats.raw_opacities.val());
    let out = <Wgpu as VoxelOps>::voxelize(settings, transforms, raw_opac, VoxelPass::Forward)
        .await;
    wrap_wgpu_float::<3>(out.out_volume)
}

