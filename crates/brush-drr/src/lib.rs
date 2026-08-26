//! Differentiable DRR (digitally reconstructed radiograph): cone-beam line
//! integral through a 3D density volume via cubecl ray-marching kernels.
//!
//! Analogue of DiffDRR / nanodrr for the burn/cubecl stack. A `[H,W]`
//! pixel grid marches `steps` rays through a `[vol,vol,vol]` volume
//! (`[-half_r, half_r]^3`, world coords), trilinearly sampling and
//! accumulating `∫μ dl`; the output is `proj = scale * integral + bias`
//! (calibrated optical depth). The backward scatters the projection
//! gradient back into the volume (atomic add), so the volume can be
//! refined with gradient descent against the DICOM projections.
//!
//! `DrrOps` is implemented for the raw `MainBackendBase` (kernel launch)
//! and for the fusion `MainBackend` (resolve → run base → bind the result
//! back into the fusion stream), mirroring `brush-voxel`/`brush-xray`.
//! This keeps the volume + Adam + loss **on GPU** with no CPU round-trips.

use burn::backend::Backend;
use burn::backend::tensor::FloatTensor;
use brush_cube::{MainBackend, MainBackendBase};
use burn_fusion::Fusion;

pub use crate::host::DrrSettings;

pub mod burn_glue;
pub mod host;
pub mod kernels;
pub mod pipeline;

/// Forward + backward DRR ops. Implemented for [`MainBackendBase`] (raw
/// cubecl kernels) and [`MainBackend`] (fusion glue).
pub trait DrrOps: Backend {
    /// Ray-march `volume` ([vol,vol,vol]) to a `[H*W]` projection.
    fn drr_forward(
        settings: &DrrSettings,
        volume: FloatTensor<Self>,
    ) -> impl Future<Output = FloatTensor<Self>>;

    /// Scatter `v_proj` ([H*W]) back into the volume gradient ([vol³]).
    fn drr_backward(
        settings: &DrrSettings,
        v_proj: FloatTensor<Self>,
    ) -> impl Future<Output = FloatTensor<Self>>;
}

impl DrrOps for Fusion<MainBackendBase> {
    fn drr_forward(
        settings: &DrrSettings,
        volume: FloatTensor<Self>,
    ) -> impl Future<Output = FloatTensor<Self>> {
        burn_glue::drr_forward_fused(settings, volume)
    }

    fn drr_backward(
        settings: &DrrSettings,
        v_proj: FloatTensor<Self>,
    ) -> impl Future<Output = FloatTensor<Self>> {
        burn_glue::drr_backward_fused(settings, v_proj)
    }
}

/// Convenience aliases for the active backends.
pub type DrrMainBackend = MainBackend;
pub type DrrMainBackendBase = MainBackendBase;
