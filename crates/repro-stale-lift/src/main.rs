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
use burn::backend::tensor::FloatTensor;
use burn::module::{Module, Param, ParamId};
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

/// 复刻 `unwrap_wgpu_float`: 提取 inner FusionTensor。
fn unwrap_wgpu_float<const D: usize>(t: Tensor<D>) -> FloatTensor<B> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        DispatchTensorKind::Wgpu(BackendTensor::Float(inner)) => inner,
        other => panic!("expected Wgpu float, got: {:?}", other),
    }
}

/// 复刻 `unwrap_ad_wgpu_float`: 从 lifted tensor 提取 AutodiffTensor。
fn unwrap_ad<const D: usize>(t: Tensor<D>) -> FloatTensor<AD> {
    let dispatch: DispatchTensor = t.into_dispatch();
    match dispatch.kind {
        DispatchTensorKind::Autodiff(inner) => match *inner {
            DispatchTensorKind::Wgpu(BackendTensor::Autodiff(ad)) => ad,
            other => panic!("expected Wgpu autodiff, got: {:?}", other),
        },
        other => panic!("expected autodiff, got: {:?}", other),
    }
}

/// 复刻 `wrap_wgpu_float`: 把 FusionTensor 包回 `Tensor<D>`。
fn wrap_wgpu_float<const D: usize>(t: FloatTensor<B>) -> Tensor<D> {
    Tensor::from_dispatch(DispatchTensor {
        kind: DispatchTensorKind::Wgpu(BackendTensor::Float(t)),
        checkpointing: None,
    })
}

#[tokio::main]
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

    let steps = 5000u32;

    for step in 1..=steps {
        // 每步累积更新 (值持续变化, 便于检测陈旧读): p <- p + 0.001。
        let p_new = p.val().add_scalar(0.001);
        p = Param::initialized(ParamId::new(), p_new);

        // 每步都 lift 一次 (等价 brush 的 loss render 每步 lift canonical)。
        let _p_ad = lift_to_autodiff(p.val()).require_grad();

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

            // 跨线程测试: 在另一线程做 val() 克隆 + 读 (触发跨 stream shared_view)。
            {
                let p_clone = p.clone();
                let handle = std::thread::spawn(move || {
                    let ft = unwrap_wgpu_float(p_clone.val());
                    let v: Vec<f32> = wrap_wgpu_float::<1>(ft.clone()).into_data().to_vec::<f32>().expect("f32");
                    (ft.id, ft.stream.value, burn_fusion::stream::StreamId::current().value, v[0])
                });
                let (tid, tstream, tcur, v0) = handle.join().unwrap();
                println!(
                    "       cross-thread: id={:?} stream={} cur={} v[0]={:.6} (direct[0]={:.6})",
                    tid, tstream, tcur, v0, direct[0]
                );
            }
        }
    }
}
