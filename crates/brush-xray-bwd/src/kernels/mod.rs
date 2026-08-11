//! Per-splat backward kernels for the cone-beam X-ray rasterizer.

#![allow(
    non_snake_case,
    clippy::doc_markdown,
    clippy::manual_div_ceil,
    clippy::manual_range_contains,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::excessive_precision,
    clippy::should_implement_trait,
    clippy::similar_names
)]

pub mod atomic;
pub mod project_bwd;
pub mod rasterize_bwd;
