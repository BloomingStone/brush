//! Backward pass for the voxelizer: trait definition + `MainBackendBase`
//! implementation + the `voxelize_bwd_pipeline` convenience function.

use brush_cube::{MainBackendBase, calc_cube_count_1d};
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
use tracing::trace_span;

use crate::VoxelOps;
use crate::kernels::helpers::BLOCK3D_SIZE;
use crate::kernels::render_bwd::BWD_LANES;
use crate::settings::VoxelSettings;
use crate::VoxelPass;

/// Intermediate gradients from the render backward.
/// Sparse `[num_visible, BWD_LANES]`, compact-indexed: `point_vol`(3) /
/// `conic`(6) / `opacity`.
#[derive(Debug, Clone)]
pub struct VoxelRasterizeGrads<B: burn::backend::Backend> {
    pub v_combined: FloatTensor<B>,
}

/// Final gradients w.r.t. the splat inputs.
#[derive(Debug, Clone)]
pub struct VoxelSplatGrads<B: burn::backend::Backend> {
    pub v_transforms: FloatTensor<B>,
    pub v_raw_opac: FloatTensor<B>,
}

/// Backward pass trait mirroring [`VoxelOps`].
pub trait VoxelBwdOps: VoxelOps {
    /// 3D render backward. Returns sparse `v_combined`
    /// `[num_visible, BWD_LANES]` indexed by `compact_gid`.
    fn rasterize_voxel_bwd(
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        cube_offsets: IntTensor<Self>,
        n_contrib: IntTensor<Self>,
        v_volume: FloatTensor<Self>,
        uniforms: crate::VoxelUniformsHost,
    ) -> VoxelRasterizeGrads<Self>;

    /// Preprocess + covariance backward. Reads sparse `v_combined`,
    /// writes dense outputs (scatter in kernel).
    fn preprocess_voxel_bwd(
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        uniforms: crate::VoxelUniformsHost,
        v_combined: FloatTensor<Self>,
    ) -> VoxelSplatGrads<Self>;
}

impl VoxelBwdOps for MainBackendBase {
    fn rasterize_voxel_bwd(
        projected_splats: FloatTensor<Self>,
        compact_gid_from_isect: IntTensor<Self>,
        cube_offsets: IntTensor<Self>,
        n_contrib: IntTensor<Self>,
        v_volume: FloatTensor<Self>,
        uniforms: crate::VoxelUniformsHost,
    ) -> VoxelRasterizeGrads<Self> {
        let _span = trace_span!("rasterize_voxel_bwd").entered();

        let device = projected_splats.device.clone();
        let num_visible = projected_splats.shape()[0].max(1);
        let client = projected_splats.client.clone();

        let v_combined =
            Self::float_zeros([num_visible, BWD_LANES as usize].into(), &device, FloatDType::F32);

        let grid = uniforms.grid;
        let cube_count = grid.x * grid.y * grid.z;

        let hard_floats = client
            .properties()
            .atomic_type_usage(Type::atomic(Type::scalar(ElemType::Float(FloatKind::F32))))
            .contains(AtomicUsage::Add);

        let cube_dim = CubeDim::new_1d(BLOCK3D_SIZE);
        let launch_uniforms = uniforms.to_launch_object();

        trace_span!("VoxelRenderBackwards").in_scope(|| {
            use crate::kernels::atomic::{CasAtomicAdd, HfAtomicAdd};
            use crate::kernels::render_bwd::render_voxel_bwd_kernel;
            // 3D grid dispatch (see VoxelRender: 1D form exceeded 65535).
            let (gx, gy, gz) = (grid.x, grid.y, grid.z);
            if hard_floats {
                render_voxel_bwd_kernel::launch::<HfAtomicAdd, WgpuRuntime>(
                    &client,
                    CubeCount::Static(gx, gy, gz),
                    cube_dim,
                    compact_gid_from_isect.into_tensor_arg(),
                    cube_offsets.into_tensor_arg(),
                    projected_splats.into_tensor_arg(),
                    n_contrib.into_tensor_arg(),
                    v_volume.into_tensor_arg(),
                    v_combined.clone().into_tensor_arg(),
                    launch_uniforms,
                );
            } else {
                render_voxel_bwd_kernel::launch::<CasAtomicAdd, WgpuRuntime>(
                    &client,
                    CubeCount::Static(gx, gy, gz),
                    cube_dim,
                    compact_gid_from_isect.into_tensor_arg(),
                    cube_offsets.into_tensor_arg(),
                    projected_splats.into_tensor_arg(),
                    n_contrib.into_tensor_arg(),
                    v_volume.into_tensor_arg(),
                    v_combined.clone().into_tensor_arg(),
                    launch_uniforms,
                );
            }
        });

        VoxelRasterizeGrads { v_combined }
    }

    fn preprocess_voxel_bwd(
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        global_from_compact_gid: IntTensor<Self>,
        uniforms: crate::VoxelUniformsHost,
        v_combined: FloatTensor<Self>,
    ) -> VoxelSplatGrads<Self> {
        let _span = trace_span!("preprocess_voxel_bwd").entered();

        let transforms = into_contiguous(transforms);
        let raw_opacities = into_contiguous(raw_opacities);

        let device = transforms.device.clone();
        let num_points = transforms.shape()[0];
        let client = transforms.client.clone();

        let v_transforms =
            Self::float_zeros([num_points, 10].into(), &device, FloatDType::F32);
        let v_raw_opac = Self::float_zeros([num_points].into(), &device, FloatDType::F32);

        let num_visible = uniforms.num_visible;
        let launch_uniforms = uniforms.to_launch_object();

        trace_span!("VoxelPreprocessBackwards").in_scope(|| {
            crate::kernels::preprocess_bwd::preprocess_voxel_bwd_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, crate::kernels::preprocess_bwd::WG_SIZE),
                CubeDim::new_1d(crate::kernels::preprocess_bwd::WG_SIZE),
                transforms.into_tensor_arg(),
                raw_opacities.into_tensor_arg(),
                global_from_compact_gid.into_tensor_arg(),
                v_combined.into_tensor_arg(),
                v_transforms.clone().into_tensor_arg(),
                v_raw_opac.clone().into_tensor_arg(),
                launch_uniforms,
            );
        });

        VoxelSplatGrads {
            v_transforms,
            v_raw_opac,
        }
    }
}

/// Convenience: run forward (Backward pass) then both backward stages for
/// a given upstream gradient, returning the dense splat gradients.
pub async fn voxelize_bwd_pipeline(
    settings: &VoxelSettings,
    transforms: FloatTensor<MainBackendBase>,
    raw_opacities: FloatTensor<MainBackendBase>,
    v_volume: FloatTensor<MainBackendBase>,
) -> VoxelSplatGrads<MainBackendBase> {
    let out = MainBackendBase::voxelize(
        settings,
        transforms.clone(),
        raw_opacities.clone(),
        VoxelPass::Backward,
    )
    .await;

    let raster_grads = MainBackendBase::rasterize_voxel_bwd(
        out.projected_splats,
        out.compact_gid_from_isect,
        out.aux.cube_offsets,
        out.aux.n_contrib,
        v_volume,
        out.uniforms,
    );

    MainBackendBase::preprocess_voxel_bwd(
        transforms,
        raw_opacities,
        out.global_from_compact_gid,
        out.uniforms,
        raster_grads.v_combined,
    )
}
