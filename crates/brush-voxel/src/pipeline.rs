//! Forward orchestration for the voxelizer: preprocess → readback →
//! cube bin → 3D additive render. Mirrors `brush-xray`'s pipeline but
//! without any camera (voxel-space geometry only).

use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor};
use brush_prefix_sum::prefix_sum;
use brush_render::get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets};
use brush_sort::radix_argsort;
use burn::backend::TensorMetadata;
use burn::backend::ops::{IntTensorOps, TransactionOps, TransactionPrimitive};
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, IntDType};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use tracing::trace_span;

use crate::{
    VoxelOps, VoxelPass,
    aux::{VoxelAuxInner, VoxelOutput},
    host::VoxelUniformsHost,
    kernels,
};
use kernels::helpers::{BLOCK3D_SIZE, VOXEL_LANES_USIZE};

impl VoxelOps for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    async fn voxelize(
        settings: &crate::settings::VoxelSettings,
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        pass: VoxelPass,
    ) -> VoxelOutput<Self> {
        assert!(
            settings.n_voxel.x > 0 && settings.n_voxel.y > 0 && settings.n_voxel.z > 0,
            "Can't voxelize into a 0-size grid."
        );
        let bwd_info = pass.bwd_info();

        let transforms = into_contiguous(transforms);
        let raw_opacities = into_contiguous(raw_opacities);

        let total_splats = transforms.shape()[0] as u32;
        let mut uniforms = VoxelUniformsHost::from_settings(settings, total_splats);

        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let (global_from_presort_gid, intersect_counts, num_visible_buf, num_intersections_buf) = {
            let u = uniforms.to_launch_object();
            let num_visible_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let num_intersections_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let intersect_counts =
                Self::int_zeros([total_splats as usize].into(), &device, IntDType::U32);
            let global_from_presort_gid = create_tensor([total_splats as usize], &device, DType::U32);

            trace_span!("VoxelPreprocess").in_scope(|| {
                kernels::preprocess::preprocess_voxel_kernel::launch::<WgpuRuntime>(
                    &client,
                    calc_cube_count_1d(total_splats, kernels::preprocess::WG_SIZE),
                    CubeDim::new_1d(kernels::preprocess::WG_SIZE),
                    transforms.clone().into_tensor_arg(),
                    global_from_presort_gid.clone().into_tensor_arg(),
                    num_visible_buf.clone().into_tensor_arg(),
                    intersect_counts.clone().into_tensor_arg(),
                    num_intersections_buf.clone().into_tensor_arg(),
                    u,
                );
            });
            (
                global_from_presort_gid,
                intersect_counts,
                num_visible_buf,
                num_intersections_buf,
            )
        };

        let (num_visible, num_intersections) = if total_splats == 0 {
            (0, 0)
        } else {
            let tp = TransactionPrimitive::<Self>::new(
                vec![],
                vec![],
                vec![num_visible_buf, num_intersections_buf],
                vec![],
            );
            let data = <Self as TransactionOps<Self>>::tr_execute(tp)
                .await
                .expect("Failed to read counts");
            let num_visible = data.read_ints[0]
                .clone()
                .into_vec::<u32>()
                .expect("num_visible")[0];
            let num_intersections = data.read_ints[1]
                .clone()
                .into_vec::<u32>()
                .expect("num_intersections")[0];
            (num_visible, num_intersections)
        };

        uniforms.num_visible = num_visible;
        uniforms.num_intersections = num_intersections;
        let num_visible_sz = (num_visible as usize).max(1);

        // No depth sort for the voxelizer: compact order = original order.
        let global_from_compact_gid =
            Self::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);

        let compact_counts =
            Self::int_gather(0, intersect_counts, global_from_compact_gid.clone());
        let cum_tiles_hit =
            trace_span!("VoxelPrefixSum").in_scope(|| prefix_sum(compact_counts));

        let projected_splats = create_tensor(
            [num_visible_sz, VOXEL_LANES_USIZE],
            &device,
            DType::F32,
        );

        trace_span!("VoxelProjectVisible").in_scope(|| {
            let u = uniforms.to_launch_object();
            kernels::project_visible::project_visible_voxel_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::project_visible::WG_SIZE),
                CubeDim::new_1d(kernels::project_visible::WG_SIZE),
                transforms.into_tensor_arg(),
                raw_opacities.into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                u,
            );
        });

        let num_cubes = uniforms.num_cubes;
        let buffer_size = (num_intersections as usize).max(1);
        let cube_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);

        trace_span!("VoxelMapCubes").in_scope(|| {
            let u = uniforms.to_launch_object();
            kernels::map_cubes::map_cubes_voxel_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::map_cubes::WG_SIZE),
                CubeDim::new_1d(kernels::map_cubes::WG_SIZE),
                projected_splats.clone().into_tensor_arg(),
                cum_tiles_hit.clone().into_tensor_arg(),
                cube_id_from_isect.clone().into_tensor_arg(),
                compact_gid_from_isect.clone().into_tensor_arg(),
                u,
            );
        });

        let bits = u32::BITS - num_cubes.leading_zeros();
        let (cube_id_from_isect, compact_gid_from_isect) =
            trace_span!("VoxelCubeSort").in_scope(|| {
                radix_argsort(cube_id_from_isect, compact_gid_from_isect, bits)
            });

        let cube_offsets = Self::int_zeros(
            [num_cubes as usize, 2].into(),
            &device,
            IntDType::U32,
        );
        let cube_dim = CubeDim::new_1d(256);
        trace_span!("VoxelGetCubeOffsets").in_scope(|| {
            get_tile_offsets::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_intersections, cube_dim.x * CHECKS_PER_ITER),
                cube_dim,
                num_intersections,
                num_cubes,
                cube_id_from_isect.into_tensor_arg(),
                cube_offsets.clone().into_tensor_arg(),
            );
        });

        let n_voxel = settings.n_voxel;
        let out_volume = create_tensor(
            [
                n_voxel.x as usize,
                n_voxel.y as usize,
                n_voxel.z as usize,
            ],
            &device,
            DType::F32,
        );
        let n_contrib = if bwd_info {
            Self::int_zeros(
                [(n_voxel.x * n_voxel.y * n_voxel.z) as usize].into(),
                &device,
                IntDType::U32,
            )
        } else {
            Self::int_zeros([1].into(), &device, IntDType::U32)
        };

        trace_span!("VoxelRender").in_scope(|| {
            let u = uniforms.to_launch_object();
            kernels::render::render_voxel_kernel::launch::<WgpuRuntime>(
                &client,
                burn_cubecl::cubecl::CubeCount::Static(num_cubes, 1, 1),
                CubeDim::new_1d(BLOCK3D_SIZE),
                compact_gid_from_isect.clone().into_tensor_arg(),
                cube_offsets.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                n_contrib.clone().into_tensor_arg(),
                out_volume.clone().into_tensor_arg(),
                u,
                bwd_info,
            );
        });

        VoxelOutput {
            out_volume,
            aux: VoxelAuxInner {
                num_visible,
                num_intersections,
                cube_offsets,
                n_contrib,
                n_voxel,
            },
            projected_splats,
            compact_gid_from_isect,
            uniforms,
            global_from_compact_gid,
        }
    }
}
