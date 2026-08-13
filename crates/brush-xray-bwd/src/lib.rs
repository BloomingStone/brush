//! Differentiable cone-beam X-ray rasterization: backward kernels +
//! autodiff wiring, mirroring `brush-render-bwd`.

pub mod burn_glue;
pub mod kernels;
mod render_bwd;

pub use burn_glue::{XRayRenderDiffOutput, render_xray};
pub use render_bwd::{
    XRayRasterizeGrads, XRaySplatBwdOps, XRaySplatGrads, xray_bwd_pipeline,
};

use brush_render::burn_glue::lift_to_autodiff;
use brush_xray::XRaySplats;
use burn::module::Param;

/// Lift an [`XRaySplats`] to the autodiff backend (equivalent to
/// [`brush_render_bwd::lift_splats_to_autodiff`] for the X-ray type). Splats
/// live on the inner (non-autodiff) device between steps; each training step
/// lifts them, renders, then strips back via `.valid()`.
pub fn lift_xray_splats_to_autodiff(splats: XRaySplats) -> XRaySplats {
    let (transforms_id, transforms, _) = splats.transforms.consume();
    let (raw_opacity_id, raw_opacity, _) = splats.raw_opacities.consume();
    XRaySplats {
        transforms: Param::initialized(
            transforms_id,
            lift_to_autodiff(transforms).require_grad(),
        ),
        raw_opacities: Param::initialized(
            raw_opacity_id,
            lift_to_autodiff(raw_opacity).require_grad(),
        ),
    }
}
