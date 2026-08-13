//! Autodiff + fusion wiring for the cone-beam X-ray rasterizer,
//! mirroring `brush-render-bwd/src/burn_glue.rs`.
//!
//! [`XRaySplatBwdOps`] is implemented for the fusion backend
//! `Fusion<MainBackendBase>` (resolves fusion tensors, runs the raw cube
//! kernels on `MainBackendBase`, binds the results back into the fusion
//! stream). [`XRayRenderBackwards`] is the burn `Backward` op that saves
//! the forward intermediates and runs both backward stages, and
//! [`render_xray`] is the differentiable high-level entry point.

use brush_cube::{MainBackend, MainBackendBase};
use brush_render::burn_glue::{
    AutodiffMain, unwrap_ad_wgpu_float, wrap_ad_wgpu_float, wrap_wgpu_float,
};
use brush_render::camera::Camera;
use brush_xray::{XRayOps, XRayPass, XRayProjectUniformsHost, XRaySplats};
use burn::{
    backend::{
        Backend, TensorMetadata,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Shape, Tensor},
};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;

use crate::render_bwd::{XRayRasterizeGrads, XRaySplatBwdOps, XRaySplatGrads};

/// State saved during the forward pass for the backward computation.
#[derive(Debug, Clone)]
struct XRayBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    raw_opacity: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    uniforms: XRayProjectUniformsHost,
    global_from_compact_gid: IntTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    tile_offsets: IntTensor<B>,
    n_contrib: IntTensor<B>,

    img_size: glam::UVec2,
}

/// Differentiable X-ray render output: the density image plus the aux needed
/// by the density controller (visibility, max radius, refine weight).
#[derive(Debug, Clone)]
pub struct XRayRenderDiffOutput {
    /// Single-channel density projection `[H, W]` (autodiff).
    pub img: Tensor<2>,
    /// Number of visible splats (forward pass).
    pub num_visible: u32,
    /// Per-splat visibility aux on the **inner** backend (no gradients).
    pub visible: Tensor<1>,
    /// Per-splat max screen-space radius in pixels (inner backend).
    pub max_radius: Tensor<1>,
    /// Refine-weight holder — receives the per-splat viewspace gradient norm
    /// as its gradient (density control).
    pub refine_weight_holder: Tensor<1>,
}

#[derive(Debug)]
struct XRayRenderBackwards;

const NUM_BWD_ARGS: usize = 3;

/// Gradient registration for the X-ray render: backprop through
/// `render_xray`'s output (the single-channel density image) to the
/// `transforms`, `raw_opacities` and `refine_weight` parents.
impl<B: Backend + XRaySplatBwdOps> Backward<B, NUM_BWD_ARGS> for XRayRenderBackwards {
    type State = XRayBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_BWD_ARGS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("render_xray backwards").entered();

        let state = ops.state;
        let v_output = grads.consume::<B>(&ops.node);

        let [transforms_parent, raw_opacity_parent, refine_weight_parent] = ops.parents;

        let rasterize_grads = B::rasterize_xray_bwd(
            state.projected_splats,
            state.compact_gid_from_isect,
            state.tile_offsets,
            state.n_contrib,
            state.img_size,
            v_output,
        );

        let splat_grads = B::project_xray_bwd(
            state.transforms,
            state.raw_opacity,
            state.global_from_compact_gid,
            state.uniforms,
            rasterize_grads.v_combined,
        );

        if let Some(node) = transforms_parent {
            grads.register::<B>(node.id, splat_grads.v_transforms);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, splat_grads.v_raw_opac);
        }

        if let Some(node) = refine_weight_parent {
            grads.register::<B>(node.id, splat_grads.v_refine_weight);
        }
    }
}

/// Differentiable high-level X-ray projection: renders `splats` to a
/// single-channel density image `[H, W]` on an autodiff-enabled device.
/// The result is differentiable w.r.t. `transforms` and `raw_opacities`;
/// the output also carries the density-control aux.
pub async fn render_xray(
    splats: XRaySplats,
    camera: &Camera,
    img_size: glam::UVec2,
    scale_modifier: f32,
) -> XRayRenderDiffOutput {
    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_xray_bwd::render_xray requires an autodiff-enabled device"
    );

    let refine_weight_holder = Tensor::<1>::zeros([1], &device).require_grad();

    let transforms_ad = unwrap_ad_wgpu_float(splats.transforms.val());
    let raw_opac_ad = unwrap_ad_wgpu_float(splats.raw_opacities.val());
    let refine_weight_ad = unwrap_ad_wgpu_float(refine_weight_holder.clone());

    let prep_nodes = XRayRenderBackwards
        .prepare::<NoCheckpointing>([
            transforms_ad.node.clone(),
            raw_opac_ad.node.clone(),
            refine_weight_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();

    let transforms_inner: FloatTensor<MainBackend> = transforms_ad.primitive.clone();
    let raw_opac_inner: FloatTensor<MainBackend> = raw_opac_ad.primitive.clone();

    let output = <MainBackend as XRayOps>::render_xray(
        camera,
        img_size,
        transforms_inner.clone(),
        raw_opac_inner.clone(),
        scale_modifier,
        XRayPass::Backward,
    )
    .await;

    let img_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = XRayBackwardState {
                transforms: transforms_inner,
                raw_opacity: raw_opac_inner,
                projected_splats: output.projected_splats,
                uniforms: output.uniforms,
                global_from_compact_gid: output.global_from_compact_gid,
                compact_gid_from_isect: output.compact_gid_from_isect,
                tile_offsets: output.aux.tile_offsets,
                n_contrib: output.aux.n_contrib,
                img_size,
            };
            prep.finish(state, output.out_img)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_img),
    };

    XRayRenderDiffOutput {
        img: wrap_ad_wgpu_float(img_ad),
        num_visible: output.aux.num_visible,
        visible: wrap_wgpu_float(output.aux.visible),
        max_radius: wrap_wgpu_float(output.aux.max_radius),
        refine_weight_holder,
    }
}

impl XRaySplatBwdOps for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    fn rasterize_xray_bwd(
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        tile_offsets: IntTensor<Self>,
        n_contrib: IntTensor<Self>,
        img_size: glam::UVec2,
        v_output: FloatTensor<Self>,
    ) -> XRayRasterizeGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            img_size: glam::UVec2,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [v_output, projected_splats, compact_gid_from_isect, tile_offsets, n_contrib] =
                    inputs;

                let [v_combined] = outputs;

                let grads = <MainBackendBase as XRaySplatBwdOps>::rasterize_xray_bwd(
                    h.get_float_tensor::<MainBackendBase>(projected_splats),
                    h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                    h.get_int_tensor::<MainBackendBase>(tile_offsets),
                    h.get_int_tensor::<MainBackendBase>(n_contrib),
                    self.img_size,
                    h.get_float_tensor::<MainBackendBase>(v_output),
                );

                h.register_float_tensor::<MainBackendBase>(&v_combined.id, grads.v_combined);
            }
        }

        let client = v_output.client.clone();
        let num_visible = projected_splats.shape()[0].max(1);

        let input_tensors = [
            v_output,
            projected_splats,
            compact_gid_from_isect,
            tile_offsets,
            n_contrib,
        ];

        let outputs = {
            let v_combined_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_visible, 8]),
                DType::F32,
            );
            let stream = StreamId::current();
            let desc = CustomOpIr::new(
                "rasterize_xray_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[v_combined_out],
            );
            let op = CustomOp {
                desc: desc.clone(),
                img_size,
            };
            client
                .register(stream, OperationIr::Custom(desc), op)
                .outputs()
        };

        let [v_combined] = outputs;

        XRayRasterizeGrads { v_combined }
    }

    #[allow(clippy::too_many_arguments)]
    fn project_xray_bwd(
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        uniforms: XRayProjectUniformsHost,
        v_combined: FloatTensor<Self>,
    ) -> XRaySplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            uniforms: XRayProjectUniformsHost,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [transforms, raw_opacities, global_from_compact_gid, v_combined_in] = inputs;

                let [v_transforms, v_raw_opac, v_refine_weight] = outputs;

                let grads = <MainBackendBase as XRaySplatBwdOps>::project_xray_bwd(
                    h.get_float_tensor::<MainBackendBase>(transforms),
                    h.get_float_tensor::<MainBackendBase>(raw_opacities),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.uniforms,
                    h.get_float_tensor::<MainBackendBase>(v_combined_in),
                );

                h.register_float_tensor::<MainBackendBase>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
                h.register_float_tensor::<MainBackendBase>(
                    &v_refine_weight.id,
                    grads.v_refine_weight,
                );
            }
        }

        let client = transforms.client.clone();
        let num_points = transforms.shape()[0];

        let input_tensors = [
            transforms,
            raw_opacities,
            global_from_compact_gid,
            v_combined,
        ];

        let outputs = {
            let v_transforms_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points, 10]),
                DType::F32,
            );
            let v_raw_opac_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points]),
                DType::F32,
            );
            let v_refine_weight_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_points]),
                DType::F32,
            );

            let stream = StreamId::current();
            let desc = CustomOpIr::new(
                "project_xray_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[v_transforms_out, v_raw_opac_out, v_refine_weight_out],
            );

            client
                .register(
                    stream,
                    OperationIr::Custom(desc.clone()),
                    CustomOp {
                        desc,
                        uniforms,
                    },
                )
                .outputs()
        };

        let [v_transforms, v_raw_opac, v_refine_weight] = outputs;

        XRaySplatGrads {
            v_transforms,
            v_raw_opac,
            v_refine_weight,
        }
    }
}

