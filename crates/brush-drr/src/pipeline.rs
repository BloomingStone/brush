//! DRR kernel orchestration on `MainBackendBase`: allocate the output,
//! launch the cube kernels.

use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor};
use burn::backend::tensor::FloatTensor;
use burn::tensor::DType;
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::cubecl::features::AtomicUsage;
use burn_cubecl::cubecl::ir::{ElemType, FloatKind, Type};
use burn_wgpu::WgpuRuntime;
use tracing::trace_span;

use crate::{
    DrrOps,
    host::DrrSettings,
    kernels::{
        atomic::{CasAtomicAdd, HfAtomicAdd},
        drr_backward::drr_backward_kernel,
        drr_forward::{WG_SIZE, drr_forward_kernel},
    },
};

impl DrrOps for MainBackendBase {
    async fn drr_forward(
    settings: &DrrSettings,
    volume: FloatTensor<MainBackendBase>,
) -> FloatTensor<MainBackendBase> {
    let client = volume.client.clone();
    let device = volume.device.clone();
    let n_pix = (settings.img_w * settings.img_h) as usize;
    let u = settings.to_launch_object();
    let out_proj = create_tensor(
        [settings.img_h as usize, settings.img_w as usize],
        &device,
        DType::F32,
    );
    let _ = settings;
    trace_span!("DrrForward").in_scope(|| {
        drr_forward_kernel::launch::<WgpuRuntime>(
            &client,
            calc_cube_count_1d(n_pix as u32, WG_SIZE),
            CubeDim::new_1d(WG_SIZE),
            volume.into_tensor_arg(),
            out_proj.clone().into_tensor_arg(),
            u,
        );
    });
    out_proj
}

    async fn drr_backward(
    settings: &DrrSettings,
    v_proj: FloatTensor<MainBackendBase>,
) -> FloatTensor<MainBackendBase> {
    let client = v_proj.client.clone();
    let device = v_proj.device.clone();
    let n_pix = (settings.img_w * settings.img_h) as usize;
    let u = settings.to_launch_object();

    let hard_floats = client
        .properties()
        .atomic_type_usage(Type::atomic(Type::scalar(ElemType::Float(FloatKind::F32))))
        .contains(AtomicUsage::Add);

    // 原子累加: 必须先清零 (create_tensor 是未初始化内存)。
    let v_volume = <Self as burn::backend::ops::FloatTensorOps<Self>>::float_zeros(
        [
            settings.vol_x as usize,
            settings.vol_y as usize,
            settings.vol_z as usize,
        ]
        .into(),
        &device,
        burn::tensor::FloatDType::F32,
    );
    trace_span!("DrrBackward").in_scope(|| {
        if hard_floats {
            drr_backward_kernel::launch::<HfAtomicAdd, WgpuRuntime>(
                &client,
                calc_cube_count_1d(n_pix as u32, WG_SIZE),
                CubeDim::new_1d(WG_SIZE),
                v_proj.into_tensor_arg(),
                v_volume.clone().into_tensor_arg(),
                u,
            );
        } else {
            drr_backward_kernel::launch::<CasAtomicAdd, WgpuRuntime>(
                &client,
                calc_cube_count_1d(n_pix as u32, WG_SIZE),
                CubeDim::new_1d(WG_SIZE),
                v_proj.into_tensor_arg(),
                v_volume.clone().into_tensor_arg(),
                u,
            );
        }
    });
    v_volume
}
}
