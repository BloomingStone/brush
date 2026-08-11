use burn::backend::{
    Backend, ExtensionType,
    tensor::{FloatTensor, IntTensor},
};

use crate::XRayProjectUniformsHost;

/// Internal render output used by kernel impls. Holds backend primitives.
#[derive(Debug, Clone, ExtensionType)]
pub struct XRayRenderOutput<B: Backend> {
    /// Single-channel density projection, `[H, W]` f32.
    pub out_img: FloatTensor<B>,
    #[extension_type]
    pub aux: XRayRenderAuxInner<B>,
    /// Sparse `[num_visible, XRAY_LANES]` packed splats (compact-indexed).
    pub projected_splats: FloatTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    /// Uniforms needed by the backward pass.
    pub uniforms: XRayProjectUniformsHost,
    pub global_from_compact_gid: IntTensor<B>,
}

/// Internal aux struct holding backend primitives. Used by the kernel
/// pipeline and the backward registration.
#[derive(Debug, Clone, ExtensionType)]
pub struct XRayRenderAuxInner<B: Backend> {
    pub num_visible: u32,
    pub num_intersections: u32,
    pub visible: FloatTensor<B>,
    /// Per-splat maximum screen-space radius in pixels (global-gid
    /// indexed). Zero for culled / invisible splats.
    pub max_radius: FloatTensor<B>,
    pub tile_offsets: IntTensor<B>,
    /// Per-pixel last contributing isect (bwd only; dummy size 1 else).
    pub n_contrib: IntTensor<B>,
    pub img_size: glam::UVec2,
}
