//! `Fusion<MainBackendBase>` implementation of [`XRayOps`]: resolves
//! fusion inputs, runs the raw cube pipeline on `MainBackendBase`, and
//! binds the results back into the fusion stream so downstream tensor
//! ops see concrete buffers.

use brush_cube::MainBackendBase;
use brush_render::camera::Camera;
use burn::backend::{
    TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::DType;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;

use crate::{
    XRayOps, XRayPass,
    aux::{XRayRenderAuxInner, XRayRenderOutput},
};

impl XRayOps for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    async fn render_xray(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        scale_modifier: f32,
        signed_opac: bool,
        screen_area_penalty: f32,
        pass: XRayPass,
    ) -> XRayRenderOutput<Self> {
        let client = transforms.client.clone();

        // Resolve fusion inputs to MainBackendBase tensors. This drains
        // any pending fusion operations into a concrete buffer.
        let base_transforms = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(transforms);
        let base_raw_opac = client
            .clone()
            .resolve_tensor_float::<MainBackendBase>(raw_opacities);

        // Run the full pipeline on MainBackendBase.
        let out = MainBackendBase::render_xray(
            camera,
            img_size,
            base_transforms,
            base_raw_opac,
            scale_modifier,
            signed_opac,
            screen_area_penalty,
            pass,
        )
        .await;

        // Bind precomputed outputs back into the fusion stream.
        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            out_img: FloatTensor<MainBackendBase>,
            visible: FloatTensor<MainBackendBase>,
            max_radius: FloatTensor<MainBackendBase>,
            projected_splats: FloatTensor<MainBackendBase>,
            tile_offsets: IntTensor<MainBackendBase>,
            compact_gid_from_isect: IntTensor<MainBackendBase>,
            global_from_compact_gid: IntTensor<MainBackendBase>,
            n_contrib: IntTensor<MainBackendBase>,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (_, outputs) = self.desc.as_fixed::<0, 8>();
                let [
                    out_img,
                    visible,
                    max_radius,
                    projected_splats,
                    tile_offsets,
                    compact_gid_from_isect,
                    global_from_compact_gid,
                    n_contrib,
                ] = outputs;

                h.register_float_tensor::<MainBackendBase>(&out_img.id, self.out_img.clone());
                h.register_float_tensor::<MainBackendBase>(&visible.id, self.visible.clone());
                h.register_float_tensor::<MainBackendBase>(&max_radius.id, self.max_radius.clone());
                h.register_float_tensor::<MainBackendBase>(
                    &projected_splats.id,
                    self.projected_splats.clone(),
                );
                h.register_int_tensor::<MainBackendBase>(
                    &tile_offsets.id,
                    self.tile_offsets.clone(),
                );
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

        let out_img_ir = TensorIr::uninit(client.create_empty_handle(), out.out_img.shape(), DType::F32);
        let visible_ir =
            TensorIr::uninit(client.create_empty_handle(), out.aux.visible.shape(), DType::F32);
        let max_radius_ir =
            TensorIr::uninit(client.create_empty_handle(), out.aux.max_radius.shape(), DType::F32);
        let projected_splats_ir = TensorIr::uninit(
            client.create_empty_handle(),
            out.projected_splats.shape(),
            DType::F32,
        );
        let tile_offsets_ir =
            TensorIr::uninit(client.create_empty_handle(), out.aux.tile_offsets.shape(), DType::U32);
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
            "xray_render_bind",
            &[],
            &[
                out_img_ir,
                visible_ir,
                max_radius_ir,
                projected_splats_ir,
                tile_offsets_ir,
                compact_gid_from_isect_ir,
                global_from_compact_gid_ir,
                n_contrib_ir,
            ],
        );
        let op = BindOp {
            desc: desc.clone(),
            out_img: out.out_img,
            visible: out.aux.visible,
            max_radius: out.aux.max_radius,
            projected_splats: out.projected_splats,
            tile_offsets: out.aux.tile_offsets,
            compact_gid_from_isect: out.compact_gid_from_isect,
            global_from_compact_gid: out.global_from_compact_gid,
            n_contrib: out.aux.n_contrib,
        };

        let outputs = client
            .register(stream, OperationIr::Custom(desc), op)
            .outputs();

        let [
            out_img,
            visible,
            max_radius,
            projected_splats,
            tile_offsets,
            compact_gid_from_isect,
            global_from_compact_gid,
            n_contrib,
        ] = outputs;

        XRayRenderOutput {
            out_img,
            aux: XRayRenderAuxInner {
                num_visible: out.aux.num_visible,
                num_intersections: out.aux.num_intersections,
                visible,
                max_radius,
                tile_offsets,
                n_contrib,
                img_size: out.aux.img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            uniforms: out.uniforms,
            global_from_compact_gid,
        }
    }
}
