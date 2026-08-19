//! Backend trait for the fused HexPlane query: forward (one kernel) and
//! backward (one kernel with atomic scatter). Implemented for the raw
//! `MainBackendBase` (direct cubecl launch) and for the fusion backend
//! `Fusion<MainBackendBase>` (custom op registered into the fusion stream —
//! synchronous, resolved lazily at execution).

use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor};
use burn::backend::{
    Backend, TensorMetadata,
    ops::FloatTensorOps,
    tensor::FloatTensor,
};
use burn::tensor::{DType, FloatDType, Shape};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_fusion::{
    Fusion, FusionHandle,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use burn_wgpu::WgpuRuntime;

use crate::fused::kernels::{
    HexPlaneKernelUniforms, HexPlaneKernelUniformsLaunch, hex_plane_query_bwd_kernel,
    hex_plane_query_fwd_kernel, WG_SIZE,
};

/// Backend trait for the fused HexPlane query kernels.
pub trait HexPlaneFusedOps: Backend {
    /// Six-plane bilinear query + sum → `[N, C]` feature tensor.
    #[allow(clippy::too_many_arguments)]
    fn hex_plane_query_fwd(
        xyz: FloatTensor<Self>,
        phase: FloatTensor<Self>,
        planes: [FloatTensor<Self>; 6],
        u: HexPlaneKernelUniforms,
    ) -> FloatTensor<Self>;

    /// Backward: atomic scatter to the six plane grids + xyz VJP.
    fn hex_plane_query_bwd(
        xyz: FloatTensor<Self>,
        phase: FloatTensor<Self>,
        planes: [FloatTensor<Self>; 6],
        v_feat: FloatTensor<Self>,
        u: HexPlaneKernelUniforms,
    ) -> HexPlaneFusedGrads<Self>;
}

/// Gradients produced by the fused backward.
#[derive(Debug, Clone)]
pub struct HexPlaneFusedGrads<B: Backend> {
    /// `[N, 3]` gradient w.r.t. the canonical positions.
    pub v_xyz: FloatTensor<B>,
    /// Per-plane grid gradients (same shapes as the planes).
    pub v_planes: [FloatTensor<B>; 6],
}

impl HexPlaneFusedOps for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    fn hex_plane_query_fwd(
        xyz: FloatTensor<Self>,
        phase: FloatTensor<Self>,
        planes: [FloatTensor<Self>; 6],
        u: HexPlaneKernelUniforms,
    ) -> FloatTensor<Self> {
        let client = xyz.client.clone();
        let device = xyz.device.clone();
        let n = xyz.shape()[0];
        let [xy, xz, yz, xt, yt, zt] = planes;

        // The xyz slice of `transforms` may be a non-contiguous view; the
        // kernels index flat buffers.
        let xyz = into_contiguous(xyz);
        // 2D `[N, C]` shape so the handle metadata ranks match the IR shape
        // (the kernel still indexes the flat buffer).
        let out = create_tensor([n, u.c as usize], &device, DType::F32);

        let cube_count = calc_cube_count_1d((n as u32) * u.c, WG_SIZE);
        let cube_dim = CubeDim::new_1d(WG_SIZE);
        let launch_obj = HexPlaneKernelUniformsLaunch::new(
            u.n,
            u.rs,
            u.rt,
            u.c,
            u.coord_scale,
            u.phase_min,
            u.phase_max,
        );
        hex_plane_query_fwd_kernel::launch::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            xyz.into_tensor_arg(),
            phase.into_tensor_arg(),
            xy.into_tensor_arg(),
            xz.into_tensor_arg(),
            yz.into_tensor_arg(),
            xt.into_tensor_arg(),
            yt.into_tensor_arg(),
            zt.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            launch_obj,
        );
        out
    }

    fn hex_plane_query_bwd(
        xyz: FloatTensor<Self>,
        phase: FloatTensor<Self>,
        planes: [FloatTensor<Self>; 6],
        v_feat: FloatTensor<Self>,
        u: HexPlaneKernelUniforms,
    ) -> HexPlaneFusedGrads<Self> {
        let client = xyz.client.clone();
        let device = xyz.device.clone();
        let n = xyz.shape()[0];
        let [xy, xz, yz, xt, yt, zt] = planes;
        // The xyz slice of `transforms` may be a non-contiguous view.
        let xyz = into_contiguous(xyz);

        // Pre-zeroed grad buffers (the kernel accumulates via atomics). The
        // shapes match the plane params (3D) so the Adam optimizer can apply
        // them; the kernels index the flat buffers.
        let s_shape = [u.rs as usize, u.rs as usize, u.c as usize];
        let t_shape = [u.rs as usize, u.rt as usize, u.c as usize];
        let v_xyz = Self::float_zeros([n, 3].into(), &device, FloatDType::F32);
        let v_xy = Self::float_zeros(s_shape.into(), &device, FloatDType::F32);
        let v_xz = Self::float_zeros(s_shape.into(), &device, FloatDType::F32);
        let v_yz = Self::float_zeros(s_shape.into(), &device, FloatDType::F32);
        let v_xt = Self::float_zeros(t_shape.into(), &device, FloatDType::F32);
        let v_yt = Self::float_zeros(t_shape.into(), &device, FloatDType::F32);
        let v_zt = Self::float_zeros(t_shape.into(), &device, FloatDType::F32);

        let hard_floats = client
            .properties()
            .atomic_type_usage(burn_cubecl::cubecl::ir::Type::atomic(
                burn_cubecl::cubecl::ir::Type::scalar(burn_cubecl::cubecl::ir::ElemType::Float(
                    burn_cubecl::cubecl::ir::FloatKind::F32,
                )),
            ))
            .contains(burn_cubecl::cubecl::features::AtomicUsage::Add);

        let cube_count = calc_cube_count_1d((n as u32) * u.c, WG_SIZE);
        let cube_dim = CubeDim::new_1d(WG_SIZE);
        let launch_obj = HexPlaneKernelUniformsLaunch::new(
            u.n,
            u.rs,
            u.rt,
            u.c,
            u.coord_scale,
            u.phase_min,
            u.phase_max,
        );
        if hard_floats {
            use crate::fused::atomic::HfAtomicAdd;
            hex_plane_query_bwd_kernel::launch::<HfAtomicAdd, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                xyz.into_tensor_arg(),
                phase.into_tensor_arg(),
                v_feat.into_tensor_arg(),
                xy.into_tensor_arg(),
                xz.into_tensor_arg(),
                yz.into_tensor_arg(),
                xt.into_tensor_arg(),
                yt.into_tensor_arg(),
                zt.into_tensor_arg(),
                v_xyz.clone().into_tensor_arg(),
                v_xy.clone().into_tensor_arg(),
                v_xz.clone().into_tensor_arg(),
                v_yz.clone().into_tensor_arg(),
                v_xt.clone().into_tensor_arg(),
                v_yt.clone().into_tensor_arg(),
                v_zt.clone().into_tensor_arg(),
                launch_obj,
            );
        } else {
            use crate::fused::atomic::CasAtomicAdd;
            hex_plane_query_bwd_kernel::launch::<CasAtomicAdd, WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                xyz.into_tensor_arg(),
                phase.into_tensor_arg(),
                v_feat.into_tensor_arg(),
                xy.into_tensor_arg(),
                xz.into_tensor_arg(),
                yz.into_tensor_arg(),
                xt.into_tensor_arg(),
                yt.into_tensor_arg(),
                zt.into_tensor_arg(),
                v_xyz.clone().into_tensor_arg(),
                v_xy.clone().into_tensor_arg(),
                v_xz.clone().into_tensor_arg(),
                v_yz.clone().into_tensor_arg(),
                v_xt.clone().into_tensor_arg(),
                v_yt.clone().into_tensor_arg(),
                v_zt.clone().into_tensor_arg(),
                launch_obj,
            );
        }

        HexPlaneFusedGrads {
            v_xyz,
            v_planes: [v_xy, v_xz, v_yz, v_xt, v_yt, v_zt],
        }
    }
}

/// Fusion-backend impl: synchronously resolve the fusion inputs to base
/// tensors, run the raw kernel on `MainBackendBase`, then bind the results
/// back into the fusion stream with an outputs-only custom op (the same
/// `BindOp` pattern as `brush-xray`'s forward).
impl HexPlaneFusedOps for Fusion<MainBackendBase> {
    #[allow(clippy::too_many_arguments)]
    fn hex_plane_query_fwd(
        xyz: FloatTensor<Self>,
        phase: FloatTensor<Self>,
        planes: [FloatTensor<Self>; 6],
        u: HexPlaneKernelUniforms,
    ) -> FloatTensor<Self> {
        let client = xyz.client.clone();
        let n = xyz.shape()[0];

        // Sync resolve: drains the stream and gives concrete base tensors.
        let xyz_base = client.resolve_tensor_float::<MainBackendBase>(xyz);
        let phase_base = client.resolve_tensor_float::<MainBackendBase>(phase);
        let planes_base = planes.map(|p| client.resolve_tensor_float::<MainBackendBase>(p));

        let feat = <MainBackendBase as HexPlaneFusedOps>::hex_plane_query_fwd(
            xyz_base,
            phase_base,
            planes_base,
            u,
        );

        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            out: FloatTensor<MainBackendBase>,
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (_, outputs) = self.desc.as_fixed::<0, 1>();
                let [out] = outputs;
                h.register_float_tensor::<MainBackendBase>(&out.id, self.out.clone());
            }
        }

        let out_ir = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([n, u.c as usize]),
            DType::F32,
        );
        let stream = StreamId::current();
        let desc = CustomOpIr::new("hex_plane_query_fwd", &[], &[out_ir]);
        let [out] = client
            .register(
                stream,
                OperationIr::Custom(desc.clone()),
                BindOp {
                    desc,
                    out: feat,
                },
            )
            .outputs::<1>();
        out
    }

    fn hex_plane_query_bwd(
        xyz: FloatTensor<Self>,
        phase: FloatTensor<Self>,
        planes: [FloatTensor<Self>; 6],
        v_feat: FloatTensor<Self>,
        u: HexPlaneKernelUniforms,
    ) -> HexPlaneFusedGrads<Self> {
        let client = v_feat.client.clone();
        let n = xyz.shape()[0];

        let xyz_base = client.resolve_tensor_float::<MainBackendBase>(xyz);
        let phase_base = client.resolve_tensor_float::<MainBackendBase>(phase);
        let planes_base = planes.map(|p| client.resolve_tensor_float::<MainBackendBase>(p));
        let v_feat_base = client.resolve_tensor_float::<MainBackendBase>(v_feat);

        let grads = <MainBackendBase as HexPlaneFusedOps>::hex_plane_query_bwd(
            xyz_base,
            phase_base,
            planes_base,
            v_feat_base,
            u,
        );

        #[derive(Debug)]
        struct BindOp {
            desc: CustomOpIr,
            v_xyz: FloatTensor<MainBackendBase>,
            v_planes: [FloatTensor<MainBackendBase>; 6],
        }

        impl Operation<FusionCubeRuntime<WgpuRuntime>> for BindOp {
            fn execute(
                &self,
                h: &mut HandleContainer<FusionHandle<FusionCubeRuntime<WgpuRuntime>>>,
            ) {
                let (_, outputs) = self.desc.as_fixed::<0, 7>();
                let [v_xyz, v_xy, v_xz, v_yz, v_xt, v_yt, v_zt] = outputs;
                h.register_float_tensor::<MainBackendBase>(&v_xyz.id, self.v_xyz.clone());
                let [g_xy, g_xz, g_yz, g_xt, g_yt, g_zt] = &self.v_planes;
                h.register_float_tensor::<MainBackendBase>(&v_xy.id, g_xy.clone());
                h.register_float_tensor::<MainBackendBase>(&v_xz.id, g_xz.clone());
                h.register_float_tensor::<MainBackendBase>(&v_yz.id, g_yz.clone());
                h.register_float_tensor::<MainBackendBase>(&v_xt.id, g_xt.clone());
                h.register_float_tensor::<MainBackendBase>(&v_yt.id, g_yt.clone());
                h.register_float_tensor::<MainBackendBase>(&v_zt.id, g_zt.clone());
            }
        }

        let s_shape = [u.rs as usize, u.rs as usize, u.c as usize];
        let t_shape = [u.rs as usize, u.rt as usize, u.c as usize];
        let outputs = {
            let v_xyz_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([n, 3]),
                DType::F32,
            );
            let v_xy_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new(s_shape),
                DType::F32,
            );
            let v_xz_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new(s_shape),
                DType::F32,
            );
            let v_yz_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new(s_shape),
                DType::F32,
            );
            let v_xt_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new(t_shape),
                DType::F32,
            );
            let v_yt_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new(t_shape),
                DType::F32,
            );
            let v_zt_out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new(t_shape),
                DType::F32,
            );

            let stream = StreamId::current();
            let desc = CustomOpIr::new(
                "hex_plane_query_bwd",
                &[],
                &[
                    v_xyz_out, v_xy_out, v_xz_out, v_yz_out, v_xt_out, v_yt_out, v_zt_out,
                ],
            );
            client
                .register(
                    stream,
                    OperationIr::Custom(desc.clone()),
                    BindOp {
                        desc,
                        v_xyz: grads.v_xyz,
                        v_planes: grads.v_planes,
                    },
                )
                .outputs::<7>()
        };

        let [v_xyz, v_xy, v_xz, v_yz, v_xt, v_yt, v_zt] = outputs;
        HexPlaneFusedGrads {
            v_xyz,
            v_planes: [v_xy, v_xz, v_yz, v_xt, v_yt, v_zt],
        }
    }
}
