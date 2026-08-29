//! 应用无关的最小复现: burn-fusion 后端上, 带 pending op 的 fusion 张量经
//! lift (AutodiffBackend::from_inner) 后读到陈旧 buffer。
//!
//! 复现 brush 训练循环里 `lift_xray_splats_to_autodiff` 的行为:
//!   1. 一个 `Param<Tensor>` (inner device), 每步用 Adam 式融合 op 更新
//!      (momentum / variance 跨步持久, 等价 optimizer step)。
//!   2. 定期对比:
//!        - 直接读 `p.val().into_data()` (应 = 真值)
//!        - lift 后读 `lift_to_autodiff(p.val()).into_data()` (bug: 可能读到陈旧值)
//!
//! 若 lift 读到的值与真值不同, 即复现了训练中 eval/loss 渲染偏少的 bug。

use burn::backend::{
    Autodiff, AutodiffBackend, BackendTensor, CheckpointingStrategy, DispatchTensor,
    DispatchTensorKind,
};
use burn::module::{AutodiffModule, Module, Param, ParamId};
use burn::tensor::{Tensor, TensorData};
use burn_wgpu::graphics::AutoGraphicsApi;
use burn_wgpu::{Wgpu, WgpuDevice};

type B = Wgpu;
type AD = Autodiff<B>;

/// 复刻 `brush_render::burn_glue::lift_to_autodiff` (去掉了 brush 依赖)。
fn lift_to_autodiff<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        DispatchTensorKind::Wgpu(BackendTensor::Float(inner)) => {
            let ad = <AD as AutodiffBackend>::from_inner(inner);
            Tensor::from_dispatch(DispatchTensor {
                kind: DispatchTensorKind::Autodiff(Box::new(DispatchTensorKind::Wgpu(
                    BackendTensor::Autodiff(ad),
                ))),
                checkpointing: Some(CheckpointingStrategy::None),
            })
        }
        other => panic!("expected Wgpu float tensor, got: {:?}", other),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let device = WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
    let device: burn::tensor::Device = device.into();

    let n = 16384usize;
    let init: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 1e-4).collect();

    // 初始 Param (inner device), 同 brush 的 canonical.transforms。
    let mut p: Param<Tensor<1>> = Param::initialized(
        ParamId::new(),
        Tensor::from_data(TensorData::new(init.clone(), [n]), &device),
    );

    // Adam 状态 (跨步持久)。
    let mut m: Tensor<1> = Tensor::zeros([n], &device);
    let mut v: Tensor<1> = Tensor::zeros([n], &device);

    let lr = 1e-3f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let steps = 5000u32;

    for step in 1..=steps {
        // 假梯度 g = p (inner, 仅制造融合依赖链)。
        let g = p.val();
        // momentum: m = beta1*m + (1-beta1)*g
        m = m.clone().mul_scalar(beta1).add(g.clone().mul_scalar(1.0 - beta1));
        // variance: v = beta2*v + (1-beta2)*g^2
        v = v
            .clone()
            .mul_scalar(beta2)
            .add(g.clone().powi_scalar(2).mul_scalar(1.0 - beta2));
        // param: p = p - lr * m / (sqrt(v) + eps)
        let p_new = g.sub(m.clone().div(v.clone().sqrt().add_scalar(eps)).mul_scalar(lr));
        p = Param::initialized(ParamId::new(), p_new);

        // 每步都 lift 一次 (等价 brush 的 loss render 每步 lift canonical)。
        let _p_ad = lift_to_autodiff(p.val()).require_grad();

        // 模拟 render 的融合张量压力。
        let _noise = Tensor::<1>::zeros([n * 4], &device).add_scalar(1.0);

        if step % 500 == 0 {
            // 直接读 (真值)。
            let direct: Vec<f32> = p.val().into_data().to_vec::<f32>().expect("f32");
            // lift 后读 (bug 路径)。
            let lifted = lift_to_autodiff(p.val()).require_grad();
            let lifted_val: Vec<f32> = lifted.into_data().to_vec::<f32>().expect("f32");

            let mut ndiff = 0usize;
            let mut maxdiff = 0.0f32;
            for (a, b) in direct.iter().zip(lifted_val.iter()) {
                let d = (a - b).abs();
                if d > 1e-6 {
                    ndiff += 1;
                }
                if d > maxdiff {
                    maxdiff = d;
                }
            }
            println!(
                "step {step:5}: direct[0]={:.6} lifted[0]={:.6} ndiff={ndiff} maxdiff={maxdiff:.6e}",
                direct[0], lifted_val[0]
            );
        }
    }
}
