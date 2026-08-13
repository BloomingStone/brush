//! Backward pass for the cone-beam X-ray rasterizer: trait definition +
//! `MainBackendBase` implementation.

use brush_cube::{MainBackendBase, calc_cube_count_1d};
use brush_render::camera::Camera;
use burn::backend::TensorMetadata;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::tensor::FloatDType;
use burn_cubecl::cubecl::CubeCount;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::features::AtomicUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, Type};
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use glam::uvec2;
use tracing::trace_span;

use crate::kernels;
use brush_xray::XRayOps;
use brush_xray::kernels::helpers::{TILE_SIZE, TILE_WIDTH, XRAY_LANES_USIZE};
use brush_xray::kernels::types::XRayRasterizeUniformsLaunch;
use brush_xray::XRayProjectUniformsHost;

/// Intermediate gradients from the X-ray rasterize backward.
/// Sparse `[num_visible, XRAY_LANES]`, compact-indexed: slots 0..6 are
/// xy(2) / conic(3) / opacity / mu.
#[derive(Debug, Clone)]
pub struct XRayRasterizeGrads<B: burn::backend::Backend> {
    pub v_combined: FloatTensor<B>,
}

/// Final gradients w.r.t. the splat inputs.
#[derive(Debug, Clone)]
pub struct XRaySplatGrads<B: burn::backend::Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
    /// Dense per-splat viewspace (mean2D) gradient norm, used for
    /// densification by the density controller.
    pub v_refine_weight: FloatTensor<B>,
}

/// Backward pass trait mirroring [`XRayOps`].
pub trait XRaySplatBwdOps: XRayOps {
    /// Additive render backward. Returns sparse `v_combined`
    /// `[num_visible, XRAY_LANES]` indexed by `compact_gid`.
    #[allow(clippy::too_many_arguments)]
    fn rasterize_xray_bwd(
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        n_contrib: IntTensor<Self>,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
    ) -> XRayRasterizeGrads<Self>;

    /// Projection + covariance backward. Reads sparse `v_combined`,
    /// writes dense outputs (scatter in kernel).
    #[allow(clippy::too_many_arguments)]
    fn project_xray_bwd(
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        uniforms: XRayProjectUniformsHost,
        v_combined: FloatTensor<Self>,
    ) -> XRaySplatGrads<Self>;
}

impl XRaySplatBwdOps for MainBackendBase {
    fn rasterize_xray_bwd(
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        n_contrib: IntTensor<Self>,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
    ) -> XRayRasterizeGrads<Self> {
        let _span = trace_span!("rasterize_xray_bwd").entered();

        let device = projected_splats.device.clone();
        let num_visible = projected_splats.shape()[0].max(1);
        let client = projected_splats.client.clone();

        let v_combined =
            Self::float_zeros([num_visible, XRAY_LANES_USIZE].into(), &device, FloatDType::F32);

        let tile_bounds = uvec2(
            img_size.x.div_ceil(TILE_WIDTH),
            img_size.y.div_ceil(TILE_WIDTH),
        );

        let hard_floats = client
            .properties()
            .atomic_type_usage(Type::atomic(Type::scalar(ElemType::Float(FloatKind::F32))))
            .contains(AtomicUsage::Add);

        let cube_count = CubeCount::Static(tile_bounds.x, tile_bounds.y, 1);
        let cube_dim = CubeDim::new_1d(TILE_SIZE);
        let uniforms = XRayRasterizeUniformsLaunch::new(tile_bounds.x, img_size.x, img_size.y);

        trace_span!("XRayRasterizeBackwards").in_scope(|| {
            use crate::kernels::atomic::{CasAtomicAdd, HfAtomicAdd};
            use crate::kernels::rasterize_bwd::rasterize_xray_bwd_kernel;
            if hard_floats {
                rasterize_xray_bwd_kernel::launch::<HfAtomicAdd, WgpuRuntime>(
                    &client,
                    cube_count,
                    cube_dim,
                    compact_gid_from_isect.into_tensor_arg(),
                    tile_offsets.into_tensor_arg(),
                    projected_splats.into_tensor_arg(),
                    v_output.into_tensor_arg(),
                    n_contrib.into_tensor_arg(),
                    v_combined.clone().into_tensor_arg(),
                    uniforms,
                );
            } else {
                rasterize_xray_bwd_kernel::launch::<CasAtomicAdd, WgpuRuntime>(
                    &client,
                    cube_count,
                    cube_dim,
                    compact_gid_from_isect.into_tensor_arg(),
                    tile_offsets.into_tensor_arg(),
                    projected_splats.into_tensor_arg(),
                    v_output.into_tensor_arg(),
                    n_contrib.into_tensor_arg(),
                    v_combined.clone().into_tensor_arg(),
                    uniforms,
                );
            }
        });

        XRayRasterizeGrads { v_combined }
    }

    fn project_xray_bwd(
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        uniforms: XRayProjectUniformsHost,
        v_combined: FloatTensor<Self>,
    ) -> XRaySplatGrads<Self> {
        let _span = trace_span!("project_xray_bwd").entered();

        let transforms = into_contiguous(transforms);
        let raw_opacities = into_contiguous(raw_opacities);

        let device = transforms.device.clone();
        let num_points = transforms.shape()[0];
        let client = transforms.client.clone();

        let v_transforms =
            Self::float_zeros([num_points, 10].into(), &device, FloatDType::F32);
        let v_raw_opac = Self::float_zeros([num_points].into(), &device, FloatDType::F32);
        let v_refine_weight = Self::float_zeros([num_points].into(), &device, FloatDType::F32);

        let num_visible = uniforms.num_visible;
        let launch_uniforms = uniforms.to_launch_object();

        trace_span!("XRayProjectBackwards").in_scope(|| {
            kernels::project_bwd::project_xray_bwd_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::project_bwd::WG_SIZE),
                CubeDim::new_1d(kernels::project_bwd::WG_SIZE),
                transforms.into_tensor_arg(),
                raw_opacities.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_combined.into_tensor_arg(),
                v_transforms.clone().into_tensor_arg(),
                v_raw_opac.clone().into_tensor_arg(),
                v_refine_weight.clone().into_tensor_arg(),
                launch_uniforms,
            );
        });

        XRaySplatGrads {
            v_transforms,
            v_raw_opac,
            v_refine_weight,
        }
    }
}

/// Convenience: run forward (Backward pass) then both backward stages for
/// a given upstream gradient, returning the dense splat gradients.
/// Used by tests and by the autodiff wiring.
#[allow(clippy::too_many_arguments)]
pub async fn xray_bwd_pipeline(
    camera: &Camera,
    img_size: glam::UVec2,
    transforms: FloatTensor<MainBackendBase>,
    raw_opacities: FloatTensor<MainBackendBase>,
    scale_modifier: f32,
    v_output: FloatTensor<MainBackendBase>,
) -> XRaySplatGrads<MainBackendBase> {
    use brush_xray::XRayPass;
    let out = MainBackendBase::render_xray(
        camera,
        img_size,
        transforms.clone(),
        raw_opacities.clone(),
        scale_modifier,
        XRayPass::Backward,
    )
    .await;

    let raster_grads = MainBackendBase::rasterize_xray_bwd(
        out.projected_splats,
        out.compact_gid_from_isect,
        out.aux.tile_offsets,
        out.aux.n_contrib,
        img_size,
        v_output,
    );

    MainBackendBase::project_xray_bwd(
        transforms,
        raw_opacities,
        out.global_from_compact_gid,
        out.uniforms,
        raster_grads.v_combined,
    )
}
