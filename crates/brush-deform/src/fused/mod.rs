//! Fused cubecl kernels for the HexPlane encoder (Stage 2).
//!
//! The pure-burn `hex_plane::HexPlane::forward_pure` is kept as the
//! reference implementation; `HexPlane::forward` dispatches to the fused
//! path (one forward kernel + one backward kernel with atomic scatter)
//! unless `HexPlaneConfig::fused` is disabled.

pub mod atomic;
pub mod burn_glue;
pub mod kernels;
pub mod ops;

pub use burn_glue::hex_plane_query_ad;
pub use ops::HexPlaneFusedOps;
