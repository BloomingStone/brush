//! 用真实 xray render 模块复现 brush 训练里的陈旧读 bug。
//!
//! 复现 brush 训练循环 (静态重建, 无 deform):
//!   1. `lift_xray_splats_to_autodiff` 把 canonical (inner) 抬到 autodiff;
//!   2. `render_xray` (autodiff, XRayPass::Backward) 渲染密度图;
//!   3. L1 loss vs 固定随机 GT -> backward;
//!   4. AdamScaled (per-component scaling [1,10]) 步进 -> `.valid()` 写回 canonical。
//!
//! 关键: 多线程 tokio (work-stealing) 会让 `step()` 与 `render` 落在不同
//! worker 线程 (= 不同 `StreamId`), 触发 `FusionTensor::clone/into_ir` 的
//! 跨 stream `shared_view`。bug 表现为 autodiff (lift) 渲染的密度比 raw
//! forward 渲染少 (~4%)。
//!
//! 检查: 每 N 步在另一个 OS 线程上对比
//!   - `render_xray` (autodiff, lift 路径, 疑似陈旧) — 先做, 是 step 后
//!     第一个 drain canonical 的路径
//!   - `render_xray_forward` (raw, 无 lift, 真值) — 后做, canonical 已落地
//! 两者密度和之比。

use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_xray::{XRaySplats, render_xray_forward};
use brush_xray_bwd::{lift_xray_splats_to_autodiff, render_xray};
use burn::optim::GradientsParams;
use burn::tensor::{Tensor, TensorData};
use burn_fusion::stream::StreamId;

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
    let mut transforms = Vec::with_capacity(n * 10);
    let mut raw = Vec::with_capacity(n);
    for i in 0..n {
        let x = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
        let y = ((i * 104729) % 1000) as f32 / 500.0 - 1.0;
        let z = ((i * 1299721) % 1000) as f32 / 500.0 - 1.0;
        transforms.extend_from_slice(&[x, y, z, 1.0, 0.0, 0.0, 0.0, -0.3, -0.5, -0.7]);
        raw.push(1.0 + 0.001 * (i % 500) as f32);
    }
    let t = Tensor::from_data(TensorData::new(transforms, [n, 10]), device);
    let r = Tensor::from_data(TensorData::new(raw, [n]), device);
    XRaySplats::from_tensor_data(t, r)
}

fn img_sum(img: &Tensor<2>) -> f32 {
    img.clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .sum()
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let wgpu_device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup_async::<burn_wgpu::graphics::AutoGraphicsApi>(
        &wgpu_device,
        Default::default(),
    )
    .await;
    let device: burn::tensor::Device = wgpu_device.into();

    let cam = std_cam();
    let img_size = glam::uvec2(64, 64);
    let n = 2048usize;

    // 固定随机 GT (驱动梯度, 让 transforms 每步真实变化)。
    let mut seed = 42u64;
    let gt_vec: Vec<f32> = (0..(64 * 64))
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32)) * 0.5
        })
        .collect();
    let device_ad = device.clone().autodiff();
    let gt = Tensor::<2>::from_data(TensorData::new(gt_vec, [64, 64]), &device_ad);

    let mut canonical = build_splats(n, &device);

    // ---- AdamScaled 复刻 (与 brush xray_train.rs 一致) ----
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-15f32;
    let lr_mean = 1e-3f32;
    let lr_opac = 1e-3f32;
    let lr_values: [f32; 10] = [
        lr_mean, lr_mean, lr_mean, lr_mean, lr_mean, lr_mean, lr_mean, lr_mean, lr_mean, lr_mean,
    ];
    // 跨步持久的 momentum 状态 (moment_1, moment_2)。
    let mut mom_t: Option<(Tensor<2>, Tensor<2>)> = None;
    let mut mom_o: Option<(Tensor<1>, Tensor<1>)> = None;
    let mut time = 0usize;

    let steps = 3000u32;

    // 关键: 把训练循环放进 tokio::spawn (worker 线程池), 这样 task 会在 .await 时
    // 被 work-stealing 迁到不同 worker 线程 —— 复刻 brush 里同一 task 跨线程迁移。
    // (放在 #[tokio::main] 的 block_on 里则永远停在主线程, StreamId 恒为 0。)
    let handle = tokio::spawn(async move {
    for step in 0..steps {
        // ---- step (tokio 决定在哪个 worker 线程 poll 这个 future) ----
        let step_cur = StreamId::current().value;

        let canonical_ad = lift_xray_splats_to_autodiff(canonical.clone());
        let out = render_xray(canonical_ad.clone(), &cam, img_size, 1.0, false).await;

        let loss = (out.img.clone() - gt.clone()).abs().mean();
        let mut grads = loss.backward();
        // 复刻 brush step() 每步的 refine_weight readback (grad_remove → into_data,
        // 会 drain stream 并 mark_read 该 grad 张量)。
        let _refine_weight = out
            .refine_weight_holder
            .grad_remove(&mut grads)
            .expect("refine weight grad");
        let transforms_id = canonical_ad.transforms.id;
        let opacities_id = canonical_ad.raw_opacities.id;
        let mut gp = GradientsParams::from_module(&mut grads, &canonical_ad);
        let grad_t = gp.remove::<2>(transforms_id).expect("transforms grad");
        let grad_o = gp.remove::<1>(opacities_id).expect("opacity grad");

        time += 1;

        // transforms: scaling [1,10] + reduce_moment_2=false。
        let g2t = grad_t.clone().powi_scalar(2);
        let (m1t, m2t) = match mom_t.take() {
            None => (
                grad_t.clone().mul_scalar(1.0 - beta1),
                g2t.mul_scalar(1.0 - beta2),
            ),
            Some((m1, m2)) => (
                m1.mul_scalar(beta1).add(grad_t.clone().mul_scalar(1.0 - beta1)),
                m2.mul_scalar(beta2).add(g2t.mul_scalar(1.0 - beta2)),
            ),
        };
        mom_t = Some((m1t.clone(), m2t.clone()));
        let m1tc = m1t.div_scalar(1.0 - beta1.powi(time as i32));
        let m2tc = m2t.div_scalar(1.0 - beta2.powi(time as i32));
        let grad_hat_t = m1tc.div(m2tc.sqrt().add_scalar(eps));
        let scaling = Tensor::<1>::from_floats(lr_values.as_slice(), &device).reshape([1, 10]);
        let delta_t = grad_hat_t.mul(scaling.mul_scalar(lr_mean));

        // opacity: 无 scaling, reduce_moment_2=true (但 D=1 时退化为普通 Adam)。
        let g2o = grad_o.clone().powi_scalar(2);
        let (m1o, m2o) = match mom_o.take() {
            None => (
                grad_o.clone().mul_scalar(1.0 - beta1),
                g2o.mul_scalar(1.0 - beta2),
            ),
            Some((m1, m2)) => (
                m1.mul_scalar(beta1).add(grad_o.clone().mul_scalar(1.0 - beta1)),
                m2.mul_scalar(beta2).add(g2o.mul_scalar(1.0 - beta2)),
            ),
        };
        mom_o = Some((m1o.clone(), m2o.clone()));
        let m1oc = m1o.div_scalar(1.0 - beta1.powi(time as i32));
        let m2oc = m2o.div_scalar(1.0 - beta2.powi(time as i32));
        let grad_hat_o = m1oc.div(m2oc.sqrt().add_scalar(eps));
        let delta_o = grad_hat_o.mul_scalar(lr_opac);

        // 优化器在 inner 后端步进 (grad 是 inner), 结果直接作为 inner canonical。
        let t_inner = canonical_ad.transforms.val().inner();
        let o_inner = canonical_ad.raw_opacities.val().inner();
        let new_transforms = t_inner.sub(delta_t);
        let new_opac = o_inner.sub(delta_o);
        canonical = XRaySplats {
            transforms: burn::module::Param::initialized(transforms_id, new_transforms),
            raw_opacities: burn::module::Param::initialized(opacities_id, new_opac),
        };

        // 用 sleep 而不是 yield_now: sleep 会把 task 放回全局队列, 允许被其他
        // worker steal (真实读回 suspend 的等价行为), yield_now 通常回到同线程。
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        if step % 100 == 0 {
            // ---- 在另一个 OS 线程上做检查 (必然不同 StreamId) ----
            let c = canonical.clone();
            let h = std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async move {
                    let check_cur = StreamId::current().value;
                    // autodiff (lift) 渲染先做 —— 它是 step 后第一个 drain canonical
                    // 的路径 (等价 brush eval_view 的 render), 若读陈旧则密度偏少。
                    let ad = lift_xray_splats_to_autodiff(c.clone());
                    let out = render_xray(ad, &cam, img_size, 1.0, false).await;
                    let ad_sum = img_sum(&out.img);
                    // forward 渲染后做 (此时 canonical 已被 materialize, 读真值)。
                    let fwd = render_xray_forward(&c, &cam, img_size, 1.0).await;
                    let fwd_sum = img_sum(&fwd);
                    (check_cur, fwd_sum, ad_sum)
                })
            })
            .join()
            .unwrap();

            let (check_cur, fwd_sum, ad_sum) = h;
            let ratio = ad_sum / fwd_sum;
            println!(
                "step {step:5}: step_cur={step_cur} check_cur={check_cur} fwd_sum={fwd_sum:.4} ad_sum={ad_sum:.4} ratio={ratio:.5}",
            );
        }
    }
    });
    handle.await.unwrap();
}
