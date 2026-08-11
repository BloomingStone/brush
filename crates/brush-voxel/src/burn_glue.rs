//! `Fusion<MainBackendBase>` implementation of [`VoxelOps`] and
//! [`VoxelBwdOps`]: resolves fusion inputs, runs the raw cube pipeline on
//! `MainBackendBase`, and binds the results back into the fusion stream.
//! Also wires the autodiff `Backward` op and the differentiable
//! [`voxelize`] entry point.

use brush_cube::{MainBackend, MainBackendBase};
use brush_render::burn_glue::{AutodiffMain, unwrap_ad_wgpu_float, wrap_ad_wgpu_float};
use brush_xray::XRaySplats;
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
use burn_fusion::{Fusion, FusionHandle, stream::{Operation, StreamId}};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;

use crate::{
    VoxelBwdOps, VoxelOps, VoxelPass,
    aux::{VoxelAuxInner, VoxelOutput},
    backward::{VoxelRasterizeGrads, VoxelSplatGrads},
    settings::VoxelSettings,
};

impl VoxelOps for Fusion<MainBackendBase> {
    async fn voxelize(
        settings: &crate::settings::VoxelSettings,
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        pass: VoxelPass,
    ) -> VoxelOutput<Self> {
        let client = transforms.client.clone();

        // Resolve fusion inputs to MainBackendBase tensors.
        let base_transforms = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(transforms);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(raw_opacities);

        // Run the full pipeline on MainBackendBase.
        let out = MainBackendBase::voxelize(settings, base_transforms, base_raw_opac, pass).await;

        // Bind precomputed outputs back into the fusion stream.
        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            out_volume: FloatTensor<MainBackendBase>,
            projected_splats: FloatTensor<MainBackendBase>,
            cube_offsets: IntTensor<MainBackendBase>,
            compact_gid_from_isect: IntTensor<MainBackendBase>,
            global_from_compact_gid: IntTensor<MainBackendBase>,
            n_contrib: IntTensor<MainBackendBase>,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (_, outputs) = self.desc.as_fixed::<0, 6>();
                let [
                    out_volume,
                    projected_splats,
                    cube_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                    n_contrib,
                ] = outputs;

                h.register_float_tensor::<MainBackendBase>(&out_volume.id, self.out_volume.clone());
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    self.projected_splats.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(&cube_offsets.id, self.cube_offsets.clone());
                h.register_int_tensor::<MainBackendBase>(
                    &compact_gid_from_isect.id,
                    self.compact_gid_from_isect.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &global_from_compact_gid.id,
                    self.global_from_compact_gid.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(&n_contrib.id, self.n_contrib.clone());
            }
        }

        let out_volume_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.out_volume.shape(),
            DType::F32,
        );
        let projected_splats_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.projected_splats.shape(),
            DType::F32,
        );
        let cube_offsets_ir =
            TensorIr::uninit(client.create_empty_handle(), out.aux.cube_offsets.shape(), DType::U32);
        let compact_gid_from_isect_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.compact_gid_from_isect.shape(),
            DType::U32,
        );
        let global_from_compact_gid_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.global_from_compact_gid.shape(),
            DType::U32,
        );
        let n_contrib_ir =
            TensorIr::uninit(client.create_empty_handle(), out.aux.n_contrib.shape(), DType::U32);

        let stream = StreamId::current();
        let desc = CustomOpIr::new(
            "voxelize_bind",
            &[],
            &[
                out_volume_ir,
                projected_splats_ir,
                cube_offsets_ir,
                compact_gid_from_isect_ir,
                global_from_compact_gid_ir,
                n_contrib_ir,
            ],
        );
        let op = BindOp {
            desc: desc.clone(),
            out_volume: out.out_volume,
            projected_splats: out.projected_splats,
            cube_offsets: out.aux.cube_offsets,
            compact_gid_from_isect: out.compact_gid_from_isect,
            global_from_compact_gid: out.global_from_compact_gid,
            n_contrib: out.aux.n_contrib,
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [
            out_volume,
            projected_splats,
            cube_offsets,
            compact_gid_from_isect,
            global_from_compact_gid,
            n_contrib,
        ] = outputs;

        VoxelOutput {
            out_volume,
            aux: VoxelAuxInner {
                num_visible: out.aux.num_visible,
                num_intersections: out.aux.num_intersections,
                cube_offsets,
                n_contrib,
                n_voxel: out.aux.n_voxel,
            },
            projected_splats,
            compact_gid_from_isect,
            uniforms: out.uniforms,
            global_from_compact_gid,
        }
    }
}

/// State saved during the forward pass for the backward computation.
#[derive(Debug, Clone)]
struct VoxelBackwardState<B: Backend> {
    transforms: FloatTensor<B>,
    raw_opacity: FloatTensor<B>,

    projected_splats: FloatTensor<B>,
    uniforms: crate::VoxelUniformsHost,
    global_from_compact_gid: IntTensor<B>,
    compact_gid_from_isect: IntTensor<B>,
    cube_offsets: IntTensor<B>,
    n_contrib: IntTensor<B>,
}

#[derive(Debug)]
struct VoxelizeBackwards;

const NUM_BWD_ARGS: usize = 2;

/// Gradient registration for the voxelizer: backprop through the density
/// volume to the `transforms` and `raw_opacities` parents.
impl<B: Backend + VoxelBwdOps> Backward<B, NUM_BWD_ARGS> for VoxelizeBackwards {
    type State = VoxelBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_BWD_ARGS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let _span = tracing::trace_span!("voxelize backwards").entered();

        let state = ops.state;
        let v_volume = grads.consume::<B>(&ops.node);

        let [transforms_parent, raw_opacity_parent] = ops.parents;

        let raster_grads = B::rasterize_voxel_bwd(
            state.projected_splats,
            state.compact_gid_from_isect,
            state.cube_offsets,
            state.n_contrib,
            v_volume,
            state.uniforms,
        );

        let splat_grads = B::preprocess_voxel_bwd(
            state.transforms,
            state.raw_opacity,
            state.global_from_compact_gid,
            state.uniforms,
            raster_grads.v_combined,
        );

        if let Some(node) = transforms_parent {
            grads.register::<B>(node.id, splat_grads.v_transforms);
        }

        if let Some(node) = raw_opacity_parent {
            grads.register::<B>(node.id, splat_grads.v_raw_opac);
        }
    }
}

/// Differentiable high-level voxelization: renders `splats` into a density
/// volume `[nVoxel_x, nVoxel_y, nVoxel_z]` on an autodiff-enabled device.
/// The result is differentiable w.r.t. `transforms` and `raw_opacities`.
pub async fn voxelize(splats: XRaySplats, settings: &VoxelSettings) -> Tensor<3> {
    let device = splats.device();
    assert!(
        device.is_autodiff(),
        "brush_voxel::voxelize requires an autodiff-enabled device"
    );

    let transforms_ad = unwrap_ad_wgpu_float(splats.transforms.val());
    let raw_opac_ad = unwrap_ad_wgpu_float(splats.raw_opacities.val());

    let prep_nodes = VoxelizeBackwards
        .prepare::<NoCheckpointing>([
            transforms_ad.node.clone(),
            raw_opac_ad.node.clone(),
        ])
        .compute_bound()
        .stateful();

    let transforms_inner: FloatTensor<MainBackend> = transforms_ad.primitive.clone();
    let raw_opac_inner: FloatTensor<MainBackend> = raw_opac_ad.primitive.clone();

    let output = <MainBackend as VoxelOps>::voxelize(
        settings,
        transforms_inner.clone(),
        raw_opac_inner.clone(),
        VoxelPass::Backward,
    )
    .await;

    let vol_ad: FloatTensor<AutodiffMain> = match prep_nodes {
        OpsKind::Tracked(prep) => {
            let state = VoxelBackwardState {
                transforms: transforms_inner,
                raw_opacity: raw_opac_inner,
                projected_splats: output.projected_splats,
                uniforms: output.uniforms,
                global_from_compact_gid: output.global_from_compact_gid,
                compact_gid_from_isect: output.compact_gid_from_isect,
                cube_offsets: output.aux.cube_offsets,
                n_contrib: output.aux.n_contrib,
            };
            prep.finish(state, output.out_volume)
        }
        OpsKind::UnTracked(prep) => prep.finish(output.out_volume),
    };

    wrap_ad_wgpu_float(vol_ad)
}

impl VoxelBwdOps for Fusion<MainBackendBase> {
    fn rasterize_voxel_bwd(
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        cube_offsets: IntTensor<Self>,
        n_contrib: IntTensor<Self>,
        v_volume: FloatTensor<Self>,
        uniforms: crate::VoxelUniformsHost,
    ) -> VoxelRasterizeGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            uniforms: crate::VoxelUniformsHost,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [
                    v_volume,
                    projected_splats,
                    compact_gid_from_isect,
                    cube_offsets,
                    n_contrib,
                ] = inputs;

                let [v_combined] = outputs;

                let grads = <MainBackendBase as VoxelBwdOps>::rasterize_voxel_bwd(
                    h.get_float_tensor::<MainBackendBase>(projected_splats),
                    h.get_int_tensor::<MainBackendBase>(compact_gid_from_isect),
                    h.get_int_tensor::<MainBackendBase>(cube_offsets),
                    h.get_int_tensor::<MainBackendBase>(n_contrib),
                    h.get_float_tensor::<MainBackendBase>(v_volume),
                    self.uniforms,
                );

                h.register_float_tensor::<MainBackendBase>(&v_combined.id, grads.v_combined);
            }
        }

        let client = v_volume.client.clone();
        let num_visible = projected_splats.shape()[0].max(1);

        let input_tensors = [
            v_volume,
            projected_splats,
            compact_gid_from_isect,
            cube_offsets,
            n_contrib,
        ];

        let outputs = {
            let v_combined_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([num_visible, crate::kernels::render_bwd::BWD_LANES as usize]),
                DType::F32,
            );
            let stream = StreamId::current();
            let desc = CustomOpIr::new(
                "rasterize_voxel_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[v_combined_out],
            );
            let op = CustomOp {
                desc: desc.clone(),
                uniforms,
            };
            client
                .register(stream, OperationIr::Custom(desc), op)
                .outputs()
        };

        let [v_combined] = outputs;

        VoxelRasterizeGrads { v_combined }
    }

    fn preprocess_voxel_bwd(
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        uniforms: crate::VoxelUniformsHost,
        v_combined: FloatTensor<Self>,
    ) -> VoxelSplatGrads<Self> {
        #[derive(Debug)]
        struct CustomOp {
            desc: CustomOpIr,
            uniforms: crate::VoxelUniformsHost,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for CustomOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (inputs, outputs) = self.desc.as_fixed();

                let [transforms, raw_opacities, global_from_compact_gid, v_combined_in] = inputs;

                let [v_transforms, v_raw_opac] = outputs;

                let grads = <MainBackendBase as VoxelBwdOps>::preprocess_voxel_bwd(
                    h.get_float_tensor::<MainBackendBase>(transforms),
                    h.get_float_tensor::<MainBackendBase>(raw_opacities),
                    h.get_int_tensor::<MainBackendBase>(global_from_compact_gid),
                    self.uniforms,
                    h.get_float_tensor::<MainBackendBase>(v_combined_in),
                );

                h.register_float_tensor::<MainBackendBase>(&v_transforms.id, grads.v_transforms);
                h.register_float_tensor::<MainBackendBase>(&v_raw_opac.id, grads.v_raw_opac);
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

            let stream = StreamId::current();
            let desc = CustomOpIr::new(
                "preprocess_voxel_bwd",
                &input_tensors.map(|t| t.into_ir()),
                &[v_transforms_out, v_raw_opac_out],
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

        let [v_transforms, v_raw_opac] = outputs;

        VoxelSplatGrads {
            v_transforms,
            v_raw_opac,
        }
    }
}

