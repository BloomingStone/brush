//! Differentiable cone-beam X-ray rasterization: backward kernels +
//! autodiff wiring, mirroring `brush-render-bwd`.

pub mod burn_glue;
pub mod kernels;
mod render_bwd;

pub use burn_glue::render_xray;
pub use render_bwd::{
    XRayRasterizeGrads, XRaySplatBwdOps, XRaySplatGrads, xray_bwd_pipeline,
};
