//! Fusion glue for the DRR kernels: resolve the fusion inputs to
//! `MainBackendBase`, run the raw cube pipeline, and bind the results back
//! into the fusion stream. Mirrors `brush-voxel`'s `burn_glue.rs` — keeps
//! the volume + gradients on GPU (no CPU round-trips).

use brush_cube::{MainBackendBase, MainBackend as FusionBackend};
use burn::backend::TensorMetadata;
use burn::backend::tensor::FloatTensor;
use burn::tensor::DType;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;

use crate::{DrrOps, host::DrrSettings};

/// Bind a single `MainBackendBase` float tensor back into the fusion stream
/// under a pre-allocated handle.
#[derive(Debug)]
struct BindOne {
    desc: CustomOpIr,
    out: FloatTensor<MainBackendBase>,
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOne {
    fn execute(
        &self,
        h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
    ) {
        let (_, outputs) = self.desc.as_fixed::<0, 1>();
        let [out] = outputs;
        h.register_float_tensor::<MainBackendBase>(&out.id, self.out.clone());
    }
}

pub async fn drr_forward_fused(
    settings: &DrrSettings,
    volume: FloatTensor<Fusion<MainBackendBase>>,
) -> FloatTensor<Fusion<MainBackendBase>> {
    let client = volume.client.clone();
    let base_volume = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(volume);
    let out_base = <MainBackendBase as DrrOps>::drr_forward(settings, base_volume).await;

    let out_ir = TensorIr::uninit(client.create_empty_handle(), out_base.shape(), DType::F32);
    let stream = StreamId::current();
    let desc = CustomOpIr::new("drr_forward_bind", &[], &[out_ir]);
    let op = BindOne {
        desc: desc.clone(),
        out: out_base,
    };
    let outputs = client.register(stream, OperationIr::Custom(desc), op).outputs::<1>();
    let [out] = outputs;
    out
}

pub async fn drr_backward_fused(
    settings: &DrrSettings,
    v_proj: FloatTensor<Fusion<MainBackendBase>>,
) -> FloatTensor<Fusion<MainBackendBase>> {
    let client = v_proj.client.clone();
    let base_v_proj = client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(v_proj);
    let out_base = <MainBackendBase as DrrOps>::drr_backward(settings, base_v_proj).await;

    let out_ir = TensorIr::uninit(client.create_empty_handle(), out_base.shape(), DType::F32);
    let stream = StreamId::current();
    let desc = CustomOpIr::new("drr_backward_bind", &[], &[out_ir]);
    let op = BindOne {
        desc: desc.clone(),
        out: out_base,
    };
    let outputs = client.register(stream, OperationIr::Custom(desc), op).outputs::<1>();
    let [out] = outputs;
    out
}
