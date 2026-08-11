//! Cone-beam X-ray (volume-rendering) rasterizer for Gaussian splats.
//!
//! Port of R2-Gaussian's `xray-gaussian-rasterization-voxelization`
//! rasterizer (mode=1 cone beam) from CUDA to backend-agnostic cubecl
//! kernels. Output is a single-channel density projection: each pixel
//! accumulates `opacity · mu · exp(power)` (additive, no transmittance),
//! where `mu` is the along-ray integration factor (Eq. 7 of the
//! R2-Gaussian paper).
//!
//! Reuses brush's pinhole `Camera` directly — only the cone-beam
//! projection is supported (the parallel-beam mode 0 of R2-Gaussian is
//! intentionally dropped). The differentiable path lives in
//! `brush-xray-bwd`.

use brush_cube::MainBackend as Wgpu;
use burn::backend::Backend;
use burn::backend::tensor::FloatTensor;
use brush_render::burn_glue::{unwrap_wgpu_float, wrap_wgpu_float};
use brush_render::camera::Camera;
use burn::tensor::Tensor;

pub use crate::aux::{XRayRenderAuxInner, XRayRenderOutput};
pub use crate::host::XRayProjectUniforms as XRayProjectUniformsHost;
pub use crate::splats::XRaySplats;

pub mod aux;
pub mod burn_glue;
pub mod host;
#[doc(hidden)]
pub mod kernels;
pub mod pipeline;
pub mod splats;

/// Forward/backward pass for the X-ray rasterizer.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum XRayPass {
    /// Forward only — inference / eval. No backward bookkeeping.
    #[default]
    Forward,
    /// Forward + backward bookkeeping (training): per-pixel contributor
    /// counts, `visible` flags, tile-range shrinking.
    Backward,
}

impl XRayPass {
    pub const fn bwd_info(self) -> bool {
        matches!(self, Self::Backward)
    }
}

/// Trait for the cone-beam X-ray rendering pipeline.
///
/// A single call performs: project (cull + cov2d + mu) → readback →
/// depth sort → tile bin → additive rasterize. Mirrors
/// [`brush_render::SplatOps::render`] but for the single-channel X-ray
/// projection.
#[burn::backend::backend_extension(Wgpu)]
pub trait XRayOps: Backend {
    #[allow(clippy::too_many_arguments)]
    fn render_xray(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        scale_modifier: f32,
        pass: XRayPass,
    ) -> impl Future<Output = XRayRenderOutput<Self>>;
}

/// Forward-only, high-level X-ray projection of `splats` to a single
/// channel density image `[H, W]`. Uses `XRayPass::Forward` (no backward
/// bookkeeping). Non-differentiable — see `brush-xray-bwd` for the
/// autodiff path.
///
/// Takes [`XRaySplats`] (transforms + raw opacity only) rather than the
/// color-capable [`Splats`](brush_render::gaussian_splats::Splats): the
/// X-ray projection is a pure density field and ignores SH entirely.
pub async fn render_xray_forward(
    splats: &XRaySplats,
    camera: &Camera,
    img_size: glam::UVec2,
    scale_modifier: f32,
) -> Tensor<2> {
    let transforms = unwrap_wgpu_float(splats.transforms.val());
    let raw_opac = unwrap_wgpu_float(splats.raw_opacities.val());
    let out = <Wgpu as XRayOps>::render_xray(
        camera,
        img_size,
        transforms,
        raw_opac,
        scale_modifier,
        XRayPass::Forward,
    )
    .await;
    wrap_wgpu_float::<2>(out.out_img)
}
