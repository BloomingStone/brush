//! Forward smoke tests for the cone-beam X-ray rasterizer.

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_xray::{XRaySplats, render_xray_forward};
use burn::module::{Param, ParamId};
use burn::tensor::{Distribution, Tensor, s};

fn std_cam() -> Camera {
    Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

fn build_splats(n: usize, device: &burn::tensor::Device) -> XRaySplats {
    let means = Tensor::<2>::random([n, 3], Distribution::Uniform(-1.0, 1.0), device);
    let log_scales = Tensor::<2>::ones([n, 3], device) * -1.0;
    let quats: Tensor<2> = Tensor::<1>::from_floats(glam::Quat::IDENTITY.to_array(), device)
        .unsqueeze_dim(0)
        .repeat_dim(0, n);
    let raw_opac = Tensor::<1>::ones([n], device) * 2.0;
    XRaySplats::from_parts(means, quats, log_scales, raw_opac)
}

#[tokio::test]
async fn renders_density_in_front_of_camera() {
    let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);

    let splats = build_splats(32, &device);
    let proj = render_xray_forward(&splats, &cam, img_size, 1.0).await;

    let proj = proj.to_data_async().await.expect("readback");
    let vals = proj.as_slice::<f32>().expect("f32");
    assert_eq!(vals.len(), 64 * 64);

    // All finite & non-negative, and some density landed on screen.
    let mut sum = 0.0f32;
    for &v in vals {
        assert!(v.is_finite(), "non-finite projection value {v}");
        assert!(v >= 0.0, "negative density {v}");
        sum += v;
    }
    assert!(sum > 0.0, "no density rendered from visible splats");
}

#[tokio::test]
async fn renders_zero_with_splats_behind_camera() {
    let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
    let cam = Camera::new(
        glam::vec3(0.0, 0.0, -5.0),
        glam::Quat::IDENTITY,
        0.6,
        0.6,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    );
    let img_size = glam::uvec2(64, 64);

    let mut splats = build_splats(8, &device);
    // Move all splats behind the camera: cam_z = world_z - (-5) < 0.01.
    let means = splats.means();
    let z = means.clone().slice(s![.., 2..3]).add_scalar(-12.0);
    let means = means.slice_assign(s![.., 2..3], z);
    splats.transforms = Param::initialized(ParamId::new(), means.detach());
    let proj = render_xray_forward(&splats, &cam, img_size, 1.0).await;
    let proj = proj.to_data_async().await.expect("readback");
    let vals = proj.as_slice::<f32>().expect("f32");
    assert!(vals.iter().all(|&v| v == 0.0), "expected all-zero projection");
}

#[tokio::test]
async fn renders_zero_with_zero_opacity() {
    let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);

    let mut splats = build_splats(8, &device);
    // Zero out opacity (logit -20 → sigmoid≈0).
    let raw_opac = Tensor::<1>::full([8], -20.0, &device);
    splats.raw_opacities = Param::initialized(ParamId::new(), raw_opac.detach());
    let proj = render_xray_forward(&splats, &cam, img_size, 1.0).await;
    let proj = proj.to_data_async().await.expect("readback");
    let vals = proj.as_slice::<f32>().expect("f32");
    assert!(vals.iter().all(|&v| v == 0.0), "expected all-zero projection");
}
