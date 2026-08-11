//! Voxelizer kernels (forward + backward).

#![allow(
    non_snake_case,
    clippy::doc_markdown,
    clippy::manual_div_ceil,
    clippy::manual_range_contains,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::excessive_precision,
    clippy::too_many_arguments
)]

pub mod atomic;
pub mod helpers;
pub mod map_cubes;
pub mod preprocess;
pub mod preprocess_bwd;
pub mod project_visible;
pub mod render;
pub mod render_bwd;
pub mod types;
