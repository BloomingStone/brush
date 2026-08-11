//! Forward orchestration for the cone-beam X-ray rasterizer: project →
//! readback → depth sort → tile bin → additive rasterize.

use brush_cube::{MainBackendBase, calc_cube_count_1d, create_tensor};
use brush_prefix_sum::prefix_sum;
use brush_render::camera::Camera;
use brush_render::get_tile_offset::{CHECKS_PER_ITER, get_tile_offsets};
use brush_sort::radix_argsort;
use burn::backend::TensorMetadata;
use burn::backend::ops::{FloatTensorOps, IntTensorOps, TransactionOps, TransactionPrimitive};
use burn::backend::tensor::FloatTensor;
use burn::tensor::{DType, FloatDType, IntDType};
use burn_cubecl::cubecl::CubeDim;
use burn_cubecl::kernel::into_contiguous;
use burn_wgpu::WgpuRuntime;
use tracing::trace_span;

use crate::{
    XRayOps, XRayPass,
    aux::{XRayRenderAuxInner, XRayRenderOutput},
    host::XRayProjectUniforms as XRayProjectUniformsHost,
    kernels,
};
use kernels::helpers::{TILE_SIZE, XRAY_LANES_USIZE};

impl XRayOps for MainBackendBase {
    #[allow(clippy::too_many_arguments)]
    async fn render_xray(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        scale_modifier: f32,
        pass: XRayPass,
    ) -> XRayRenderOutput<Self> {
        assert!(
            img_size[0] > 0 && img_size[1] > 0,
            "Can't render images with 0 size."
        );
        let bwd_info = pass.bwd_info();

        let transforms = into_contiguous(transforms);
        let raw_opacities = into_contiguous(raw_opacities);

        let total_splats = transforms.shape()[0] as u32;
        let mut project_uniforms =
            XRayProjectUniformsHost::from_camera(camera, img_size, total_splats, scale_modifier);

        let device = transforms.device.clone();
        let client = transforms.client.clone();

        let (
            global_from_presort_gid,
            depths,
            intersect_counts,
            max_radius,
            num_visible_buf,
            num_intersections_buf,
        ) = {
            let u = project_uniforms.to_launch_object();
            let num_visible_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let num_intersections_buf = Self::int_zeros([1].into(), &device, IntDType::U32);
            let intersect_counts =
                Self::int_zeros([total_splats as usize].into(), &device, IntDType::U32);
            let max_radius =
                Self::float_zeros([total_splats as usize].into(), &device, FloatDType::F32);
            let global_from_presort_gid = create_tensor([total_splats as usize], &device, DType::U32);
            let depths = create_tensor([total_splats as usize], &device, DType::F32);

            trace_span!("XRayProjectForward").in_scope(|| {
                kernels::project_forward::project_forward_xray_kernel::launch::<WgpuRuntime>(
                    &client,
                    calc_cube_count_1d(total_splats, kernels::project_forward::WG_SIZE),
                    CubeDim::new_1d(kernels::project_forward::WG_SIZE),
                    transforms.clone().into_tensor_arg(),
                    raw_opacities.clone().into_tensor_arg(),
                    global_from_presort_gid.clone().into_tensor_arg(),
                    depths.clone().into_tensor_arg(),
                    num_visible_buf.clone().into_tensor_arg(),
                    intersect_counts.clone().into_tensor_arg(),
                    num_intersections_buf.clone().into_tensor_arg(),
                    max_radius.clone().into_tensor_arg(),
                    u,
                );
            });
            (
                global_from_presort_gid,
                depths,
                intersect_counts,
                max_radius,
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

        project_uniforms.num_visible = num_visible;
        let tile_bounds = project_uniforms.tile_bounds;
        let num_visible_sz = (num_visible as usize).max(1);

        let global_from_compact_gid = {
            let depths = Self::float_slice(depths, &[(0..num_visible_sz).into()]);
            let global_from_presort_gid =
                Self::int_slice(global_from_presort_gid, &[(0..num_visible_sz).into()]);
            trace_span!("XRayDepthSort").in_scope(|| {
                let (_, global_from_compact_gid) =
                    radix_argsort(depths, global_from_presort_gid, 32);
                global_from_compact_gid
            })
        };
        let compact_counts =
            Self::int_gather(0, intersect_counts, global_from_compact_gid.clone());
        let cum_tiles_hit =
            trace_span!("XRayPrefixSum").in_scope(|| prefix_sum(compact_counts));
        let projected_splats = create_tensor(
            [num_visible_sz, XRAY_LANES_USIZE],
            &device,
            DType::F32,
        );

        trace_span!("XRayProjectVisible").in_scope(|| {
            let u = project_uniforms.to_launch_object();
            kernels::project_visible::project_visible_xray_kernel::launch::<WgpuRuntime>(
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

        let num_tiles = tile_bounds.x * tile_bounds.y;
        let buffer_size = (num_intersections as usize).max(1);
        let tile_id_from_isect = create_tensor([buffer_size], &device, DType::U32);
        let compact_gid_from_isect = create_tensor([buffer_size], &device, DType::U32);

        trace_span!("XRayMapGaussians").in_scope(|| {
            let u = project_uniforms.to_launch_object();
            kernels::map_gaussians::map_gaussians_xray_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_visible, kernels::map_gaussians::WG_SIZE),
                CubeDim::new_1d(kernels::map_gaussians::WG_SIZE),
                projected_splats.clone().into_tensor_arg(),
                cum_tiles_hit.clone().into_tensor_arg(),
                tile_id_from_isect.clone().into_tensor_arg(),
                compact_gid_from_isect.clone().into_tensor_arg(),
                u,
            );
        });

        let bits = u32::BITS - num_tiles.leading_zeros();
        let (tile_id_from_isect, compact_gid_from_isect) =
            trace_span!("XRayTileSort").in_scope(|| {
                radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits)
            });

        let tile_offsets = Self::int_zeros(
            [tile_bounds.y as usize, tile_bounds.x as usize, 2].into(),
            &device,
            IntDType::U32,
        );
        let cube_dim = CubeDim::new_1d(256);
        trace_span!("XRayGetTileOffsets").in_scope(|| {
            get_tile_offsets::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_intersections, cube_dim.x * CHECKS_PER_ITER),
                cube_dim,
                num_intersections,
                num_tiles,
                tile_id_from_isect.into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
            );
        });

        let out_img =
            create_tensor([img_size.y as usize, img_size.x as usize], &device, DType::F32);
        let total_splats = project_uniforms.total_splats as usize;
        let n_contrib = if bwd_info {
            Self::int_zeros(
                [img_size.y as usize * img_size.x as usize].into(),
                &device,
                IntDType::U32,
            )
        } else {
            Self::int_zeros([1].into(), &device, IntDType::U32)
        };
        let visible = if bwd_info {
            Self::float_zeros([total_splats].into(), &device, FloatDType::F32)
        } else {
            Self::float_zeros([1].into(), &device, FloatDType::F32)
        };

        trace_span!("XRayRasterize").in_scope(|| {
            let uniforms = kernels::types::XRayRasterizeUniformsLaunch::new(
                tile_bounds.x,
                img_size.x,
                img_size.y,
            );
            kernels::rasterize::rasterize_xray_kernel::launch::<WgpuRuntime>(
                &client,
                calc_cube_count_1d(num_tiles * TILE_SIZE, TILE_SIZE),
                CubeDim::new_1d(TILE_SIZE),
                compact_gid_from_isect.clone().into_tensor_arg(),
                tile_offsets.clone().into_tensor_arg(),
                projected_splats.clone().into_tensor_arg(),
                out_img.clone().into_tensor_arg(),
                global_from_compact_gid.clone().into_tensor_arg(),
                visible.clone().into_tensor_arg(),
                n_contrib.clone().into_tensor_arg(),
                uniforms,
                bwd_info,
            );
        });

        XRayRenderOutput {
            out_img,
            aux: XRayRenderAuxInner {
                num_visible,
                num_intersections,
                visible,
                max_radius,
                tile_offsets,
                n_contrib,
                img_size,
            },
            projected_splats,
            compact_gid_from_isect,
            uniforms: project_uniforms,
            global_from_compact_gid,
        }
    }
}
