//! X-ray deform-GS trainer: canonical [`XRaySplats`] + (optionally) a
//! cardiac-phase-guided [`DeformModel`], rendered with the differentiable
//! cone-beam rasterizer and optimized jointly (canonical params + deform
//! network).
//!
//! Per step:
//! 1. lift the canonical splats to autodiff,
//! 2. predict deformations from the (AST-noised) phase — skipped in static
//!    mode ([`XRayTrainConfig::enable_deform`] = `false`),
//! 3. deform the canonical splats (autodiff graph kept),
//! 4. `render_xray` → `exp(-clamp(proj))` intensity → [`gray_loss`] vs the
//!    normalized DICOM GT,
//! 5. backward; step the canonical params (per-component LR) and the deform
//!    network; accumulate the viewspace gradient stats for the density
//!    controller ([`XRayRefiner`]).

use brush_dataset::scene::SceneBatch;
use brush_deform::{
    DeformModel, DeformModelConfig, Deforms, HexPlaneDeformConfig, HexPlaneDeformModel,
    TimeEncodingConfig, deform_splats,
};
use brush_loss::gray::{GrayLossConfig, GrayLossType, gray_loss};
use brush_render::burn_glue::detach_autodiff;
use brush_xray::XRaySplats;
use brush_xray_bwd::{lift_xray_splats_to_autodiff, render_xray};
use burn::{
    lr_scheduler::{
        LrScheduler,
        composed::{ComposedLrScheduler, ComposedLrSchedulerConfig},
        cosine::CosineAnnealingLrSchedulerConfig,
        exponential::ExponentialLrSchedulerConfig,
    },
    module::{AutodiffModule, Module},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, IndexingUpdateOp, Int, Tensor, TensorData, s},
};

use crate::adam_scaled::{AdamScaled, AdamScaledConfig};
use crate::fdk_prior::FdkPrior;
use crate::xray_refine::{XRayRefineConfig, XRayRefineStats, XRayRefiner};

/// Which deform-network backend to train.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeformBackend {
    /// HexPlane encoder + small MLP (default: fast on wgpu, low-frequency
    /// cardiac motion).
    #[default]
    HexPlane,
    /// Multi-resolution hash grid + skip-MLP (original implementation).
    HashGrid,
}

/// The active deform network: a [`Module`] union over the two backends, so
/// the trainer optimizes whichever one is configured.
#[derive(Module, Debug)]
pub enum DeformNetwork {
    HexPlane(HexPlaneDeformModel),
    HashGrid(DeformModel),
}

impl DeformNetwork {
    /// Predict deformations for canonical positions `xyz` (`[N, 3]`) at a
    /// scalar `phase` (`[N, 1]`) and real `time` (`[N, 1]`, seconds; used by
    /// the learned time conditioning when enabled).
    pub fn forward(&self, xyz: Tensor<2>, phase: Tensor<2>, time: Tensor<2>) -> Deforms {
        match self {
            Self::HexPlane(model) => model.forward(xyz, phase, time),
            Self::HashGrid(model) => model.forward(xyz, phase, time),
        }
    }

    /// Spatial TV of the deform field's feature planes (HexPlane backend);
    /// `None` for HashGrid.
    pub fn plane_tv(&self) -> Option<Tensor<1>> {
        match self {
            Self::HexPlane(model) => Some(model.plane_tv()),
            Self::HashGrid(_) => None,
        }
    }
}

/// Hyperparameters for the X-ray deform-GS trainer.
#[derive(Debug, Clone)]
pub struct XRayTrainConfig {
    pub total_iters: u32,
    /// Start / end LR for the splat means.
    pub lr_mean: f64,
    pub lr_mean_end: f64,
    /// LR for log-scales / rotations / opacity logits.
    pub lr_scale: f64,
    pub lr_rotation: f64,
    pub lr_opac: f64,
    /// Start / end LR for the deform network.
    pub lr_deform: f64,
    pub lr_deform_end: f64,
    /// Add AST (asynchronous time) noise to the phase input during training.
    pub enable_ast: bool,
    /// Warm-up steps during which no deformation is applied (dummy gradients).
    pub warm_up: u32,
    /// When `false`, train a **static** reconstruction: no deform field and no
    /// phase conditioning — splats stay fixed in canonical space (the deform
    /// network and its optimizer are omitted entirely).
    pub enable_deform: bool,
    /// Deform-network backend (`HexPlane` by default).
    pub deform_backend: DeformBackend,
    /// `HexPlane` configuration (used when `deform_backend == HexPlane`).
    pub hex_plane: HexPlaneDeformConfig,
    /// Predict a per-splat scaling offset? Default `false` for both backends:
    /// the deform field is mass-conserving (displacement + rotation only) —
    /// a splat's integrated absorption is `∝ scale`, so scaling would alter
    /// the scene's total absorption.
    pub predict_scaling: bool,
    /// Condition the deform field on real time via a learnable Fourier bank:
    /// the network fits the respiratory (and other) temporal frequencies
    /// during training — no breathing-frequency prior needed.
    pub enable_time: bool,
    /// Time jitter (seconds, Gaussian std) added to the time conditioning
    /// each step — analogous to AST phase noise. Enforces that the deform
    /// field is locally smooth in time (nearby frames give nearby motion).
    /// 0 disables. 2026-08-21: pig data is a continuous-video-like sequence,
    /// adjacent frames move little; jitter teaches time-invariant smoothness.
    pub time_jitter: f32,
    /// Weight of a temporal smoothness (TV) regularizer on the deform field:
    /// `mean(|d_xyz(phase+dp, time+dt) - d_xyz(phase, time)|^2)` on a random
    /// subset of splats. Encodes "adjacent frames deform little", helping
    /// held-out time generalization. 0 disables.
    pub time_tv_weight: f32,
    /// Phase delta for the TV regularizer (per-frame cardiac advance).
    pub time_tv_dp: f32,
    /// Time delta (s) for the TV regularizer (one frame step). Default
    /// 0.0125 = 1 frame at 80 fps.
    pub time_tv_dt: f32,
    /// Splat count sampled per step for the TV term (keeps it cheap).
    pub time_tv_sample: usize,
    /// Staged two-field training: after this step, the (phase-conditioned)
    /// cardiac deform field is **frozen** (detached) and a second,
    /// time-conditioned respiratory field starts training. 0 disables staged
    /// training (single field, current behaviour). When > 0 the cardiac field
    /// is built phase-only (`enable_time = false`) so the two fields don't
    /// couple, letting the respiratory field learn motion the cardiac phase
    /// can't explain.
    pub respi_after: u32,
    /// Freeze the cardiac field once the respiratory field activates
    /// (`respi_after`)? `false` trains both fields jointly (architecturally
    /// decoupled — separate networks, phase-only vs time-only — but both keep
    /// updating). Default `true` (the staged decoupling the field is for).
    pub respi_freeze: bool,
    /// Learnable temporal encoding configuration (used when `enable_time`).
    pub time_enc: TimeEncodingConfig,
    /// Density-control configuration.
    pub refine: XRayRefineConfig,
    /// Initial activated density (mm⁻¹) for the random splats. Defaults to
    /// μ_water (0.002). Raise it when the normalized / gamma-corrected GT
    /// target sits at higher intensity so the init ball starts at the right
    /// gray level (Beer-Lambert `proj` scales linearly with density).
    pub init_density: f32,
    /// FDK-residual mode: render with **signed opacity** (`opac = MU_WATER ·
    /// raw`, raw used directly so residual splats can subtract absorption),
    /// init raw small (`±init_density/MU_WATER`, sign-randomized), and the
    /// total projection = splat residual + FDK prior DRR.
    pub fdk_residual: bool,
    /// Small signed density magnitude the residual splats are initialized at
    /// (`|density| ≈ 1e-5` by default) in FDK-residual mode.
    pub fdk_residual_init_density: f32,
    /// L1 residual-sparsity prior (FDK-residual mode): `λ · mean(|density|)`
    /// over all splats, where density = `MU_WATER·raw` (signed). Promotes a
    /// sparse residual — splats only grow where they are needed to correct
    /// the static FDK prior. 0 disables.
    pub resid_sparse_weight: f32,
    /// L1 / SSIM weights for the gray loss.
    pub l1_weight: f32,
    pub ssim_weight: f32,
    /// Pixel-wise penalty for the gray (and proj) loss — L1 / Charbonnier /
    /// Huber / L2. Robust penalties tolerate X-ray noise better than L1.
    pub loss_type: GrayLossType,
    /// Charbonnier `ε` (smoothing floor), default 1e-3.
    pub loss_eps: f32,
    /// Huber threshold `δ`, default 0.1.
    pub loss_delta: f32,
    /// Weight of the optional **projection-domain** L1 loss: compares
    /// `proj = -ln(intensity)` (the raw attenuation path integral) instead of
    /// the Beer-Lambert-compressed intensity. 0 disables it. Default 1.0
    /// (2026-08-17: +1.07dB on RXA_chest).
    pub proj_weight: f32,
    /// Weight of the projection-domain SSIM term (`1 - SSIM(proj_pred, proj_gt)`)
    /// added to the proj loss. 0 keeps proj loss as pure L1.
    pub proj_ssim_weight: f32,    /// Weight of an optional multi-window (WW/WL) loss in the proj domain:
    /// several window transforms highlight different structures (soft tissue /
    /// bone / fine detail). 0 disables.
    pub window_weight: f32,
    /// Weight of an optional gradient (Sobel-style finite-difference) loss
    /// that sharpens edges. 0 disables.
    pub grad_weight: f32,
    /// Edge-weighted gradient loss ramp start step. Before this step the
    /// gradient term is 0 (L1/SSIM dominate the early/coarse phase), then its
    /// weight ramps (smoothstep) to `grad_weight` over
    /// `[grad_ramp_from, grad_ramp_to]`.
    pub grad_ramp_from: u32,
    /// Ramp end step (full `grad_weight`). 0 = `total_iters`.
    pub grad_ramp_to: u32,
    /// Per-pixel weight `clamp(|∇gt| / scale, 0, 1)` so the gradient loss
    /// concentrates on strong GT edges. 0 = plain (unweighted) gradient loss.
    pub grad_edge_scale: f32,    /// Use cosine-annealing for the mean LR (reference-project style) instead
    /// of exponential decay.
    pub cosine_lr: bool,
    /// Weight of an optional multi-scale (pyramid) loss: the gray L1+SSIM loss
    /// is also applied at 1/2 and 1/4 resolution. 0 disables.
    pub multiscale_weight: f32,
}

impl Default for XRayTrainConfig {
    fn default() -> Self {
        Self {
            total_iters: 30_000,
            lr_mean: 2e-5,
            // 末期不冻结: 2e-7 → 2e-6 (cosine/指数末段仍可微调位置)。
            lr_mean_end: 2e-6,
            lr_scale: 5e-3,
            lr_rotation: 2e-3,
            lr_opac: 0.012,
            lr_deform: 1e-3,
            lr_deform_end: 1e-4,
            enable_ast: true,
            warm_up: 300,
            enable_deform: true,
            deform_backend: DeformBackend::HexPlane,
            hex_plane: HexPlaneDeformConfig::default(),
            predict_scaling: false,
            enable_time: false,
            time_jitter: 0.0,   // 时间抖动 (秒, 高斯std), 0 = 关
            time_tv_weight: 0.0, // 时间 TV 正则权重, 0 = 关
            time_tv_dp: 0.0,    // TV 相位步长 (每帧心搏推进)
            time_tv_dt: 0.0125, // TV 时间步长 = 1 帧 @80fps
            time_tv_sample: 1024,
            respi_after: 0, // 0 = 单场训练 (当前行为); >0 = 分阶段双场
            respi_freeze: true,
            time_enc: TimeEncodingConfig::default(),
            refine: XRayRefineConfig::default(),
            init_density: brush_cube::MU_WATER,
            fdk_residual: false,
            fdk_residual_init_density: 1e-5,
            resid_sparse_weight: 0.0,
            l1_weight: 1.0,
            ssim_weight: 1.0,
            loss_type: GrayLossType::L1,
            loss_eps: 1e-3,
            loss_delta: 0.1,
            // Proj 域损失为默认开启 (w=1.0): 2026-08-17 实验 RXA_chest
            // 34.82dB vs 33.75 (+1.07), LPIPS 0.574 vs 0.594。
            proj_weight: 1.0,
            proj_ssim_weight: 0.0,
            // 多窗宽窗位损失默认开启 (w=0.5): LPIPS 0.5455 (vs 0.5570), 结构感知增强。
            window_weight: 0.5,
            grad_weight: 0.0,
            grad_ramp_from: 3_000,
            grad_ramp_to: 0, // 0 = total_iters
            grad_edge_scale: 0.03,
            cosine_lr: false,
            // 多尺度金字塔损失默认开启 (w=0.5): 2026-08-17 最强项 34.13dB/LPIPS 0.529。
            multiscale_weight: 0.5,
        }
    }
}

/// Standard-normal sample via Box-Muller from two `[0, 1)` uniforms
/// (rand has no built-in normal distribution; `rand::random` is uniform).
fn randn_f32() -> f32 {
    let u1 = rand::random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rand::random::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    z as f32
}

/// One training step's stats.
#[derive(Debug, Clone)]
pub struct XRayTrainStats {
    pub loss: f32,
    pub num_visible: u32,
    pub num_splats: u32,
    pub lr_mean: f64,
    /// Predicted intensity image `[H, W]` f32 (normalized `[0, 1]`), when
    /// [`XRayTrainer::set_collect_pred`] is enabled (visualization paths).
    pub pred_img: Option<TensorData>,
    /// Per-parameter gradient magnitudes (mean |∂L/∂·| per splat), when
    /// [`XRayTrainer::set_collect_grads`] is enabled. Use to check whether a
    /// parameter's LR actually moves it (position step ≈ `lr_mean · mean_grad`).
    pub grad_norms: Option<XRayGradStats>,
}

/// Gradient magnitude diagnostics (mean |∂L/∂·| per splat).
#[derive(Debug, Clone, Copy, Default)]
pub struct XRayGradStats {
    /// Mean |∂L/∂x| over the 3 position coords of every splat (mm⁻¹).
    pub mean_grad: f32,
    /// Mean |∂L/∂q| over the 4 rotation coords.
    pub rot_grad: f32,
    /// Mean |∂L/∂s| over the 3 scale-logits.
    pub scale_grad: f32,
    /// Mean |∂L/∂raw| over the density logits.
    pub density_grad: f32,
}

type OptimType = OptimizerAdaptor<AdamScaled, XRaySplats>;
type DeformOptimType = OptimizerAdaptor<burn::optim::Adam, DeformNetwork>;

/// X-ray (deform-)GS trainer. With `enable_deform=false` this is a plain
/// static reconstruction: canonical splats optimized directly against the
/// multi-angle DICOM projections.
pub struct XRayTrainer {
    config: XRayTrainConfig,
    canonical: XRaySplats,
    deform: Option<DeformNetwork>,
    /// Stage-2 respiratory field (time-conditioned, hash-grid). `None` unless
    /// `config.respi_after > 0`.
    respi: Option<DeformNetwork>,
    refiner: XRayRefiner,
    optim_splats: Option<OptimType>,
    optim_deform: Option<DeformOptimType>,
    optim_respi: Option<DeformOptimType>,
    sched_mean: ComposedLrScheduler,
    step_count: u32,
    /// Read back the predicted intensity image every step (for visualization).
    collect_pred: bool,
    /// Collect per-parameter gradient norms every step (diagnostics).
    collect_grads: bool,
    /// Read back the loss scalar every step. Disable when the loss is only
    /// needed at eval steps — every readback is a GPU→CPU sync that
    /// serializes the pipeline (CPU submission waits for GPU execution).
    collect_loss: bool,
    /// Last read-back loss value (used when `collect_loss` is off).
    last_loss: f32,
    /// FDK static prior (constant DRR volume). `None` = plain splat render.
    fdk: Option<FdkPrior>,
    /// VGG-LPIPS model for perceptual eval (loaded once; `None` keeps the
    /// eval free of the extra GPU memory).
    lpips: Option<lpips::LpipsModel>,
}

impl XRayTrainer {
    pub fn new(
        config: XRayTrainConfig,
        canonical: XRaySplats,
        deform: Option<DeformNetwork>,
        respi: Option<DeformNetwork>,
        fdk: Option<FdkPrior>,
        device: &Device,
    ) -> Self {
        let mut refine_cfg = config.refine.clone();
        refine_cfg.total_iters = config.total_iters;
        // 软重置的密度 cap 与训练初始化密度保持一致。
        refine_cfg.init_density = if config.fdk_residual {
            config.fdk_residual_init_density
        } else {
            config.init_density
        };
        refine_cfg.signed_opac = config.fdk_residual;
        let num_points = canonical.num_splats();
        let refiner = XRayRefiner::new(refine_cfg, num_points, device);

        // Mean LR schedule: exponential decay (default) or cosine annealing
        // (reference-project style, no warm restarts). Both end at lr_mean_end.
        let mut sched_mean = if config.cosine_lr {
            ComposedLrSchedulerConfig::new()
                .cosine(
                    CosineAnnealingLrSchedulerConfig::new(
                        config.lr_mean,
                        config.total_iters.max(1) as usize,
                    )
                    .with_min_lr(config.lr_mean_end),
                )
                .init()
                .expect("valid cosine lr scheduler")
        } else {
            let decay = (config.lr_mean_end / config.lr_mean)
                .powf(1.0 / config.total_iters.max(1) as f64);
            ComposedLrSchedulerConfig::new()
                .exponential(ExponentialLrSchedulerConfig::new(config.lr_mean, decay))
                .init()
                .expect("valid exponential lr scheduler")
        };

        // First LR step aligns the current step count.
        sched_mean.step();

        Self {
            canonical,
            deform,
            respi,
            refiner,
            optim_splats: None,
            optim_deform: None,
            optim_respi: None,
            sched_mean,
            config,
            step_count: 0,
            collect_pred: false,
            collect_grads: false,
            collect_loss: true,
            last_loss: f32::NAN,
            fdk,
            lpips: Some(lpips::load_vgg_lpips(device)),
        }
    }

    /// Enable / disable per-step readback of the predicted intensity image.
    /// Only turn this on when a visualization sink consumes it — it adds a
    /// GPU→CPU sync to every step.
    pub fn set_collect_pred(&mut self, collect: bool) {
        self.collect_pred = collect;
    }

    /// Enable / disable per-step gradient-magnitude diagnostics (adds a
    /// GPU→CPU readback per step).
    pub fn set_collect_grads(&mut self, collect: bool) {
        self.collect_grads = collect;
    }

    /// Enable / disable the per-step loss readback. Turn it off when the loss
    /// scalar is only needed at eval steps — the readback is a GPU→CPU sync
    /// that prevents CPU submission from overlapping GPU execution.
    pub fn set_collect_loss(&mut self, collect: bool) {
        self.collect_loss = collect;
    }

    pub fn config(&self) -> &XRayTrainConfig {
        &self.config
    }

    pub fn num_splats(&self) -> u32 {
        self.canonical.num_splats()
    }

    pub fn canonical(&self) -> &XRaySplats {
        &self.canonical
    }

    /// The trained deform network (for checkpoint / deform-field export).
    pub fn deform(&self) -> Option<&DeformNetwork> {
        self.deform.as_ref()
    }

    /// Read back the learned temporal frequencies (Hz) of the deform network,
    /// when time conditioning is enabled. `None` otherwise / on read failure.
    pub async fn learned_time_freqs(&self) -> Option<Vec<f32>> {
        let enc = match &self.deform {
            Some(DeformNetwork::HexPlane(m)) => m.time_encoding()?,
            Some(DeformNetwork::HashGrid(m)) => m.time_encoding()?,
            None => return None,
        };
        let f = enc.frequencies();
        Some(f.into_data_async().await.ok()?.to_vec::<f32>().ok()?)
    }

    /// Read back the learned temporal frequencies (Hz) of the stage-2
    /// respiratory field (hash-grid + time encoding), when staged training is
    /// active. `None` otherwise.
    pub async fn respi_learned_time_freqs(&self) -> Option<Vec<f32>> {
        let enc = match &self.respi {
            Some(DeformNetwork::HexPlane(m)) => m.time_encoding()?,
            Some(DeformNetwork::HashGrid(m)) => m.time_encoding()?,
            None => return None,
        };
        let f = enc.frequencies();
        Some(f.into_data_async().await.ok()?.to_vec::<f32>().ok()?)
    }

    /// Render the current model (canonical splats, deformed at `phase` when in
    /// deform mode) against a GT frame and compute grayscale PSNR / SSIM.
    /// Forward-only — no gradients are accumulated.
    pub async fn eval_view(
        &self,
        camera: &brush_render::camera::Camera,
        gt: &TensorData,
        phase: f32,
        time: f32,
    ) -> crate::xray_eval::XRayEvalSample {
        use crate::xray_eval::XRayEvalSample;
        use brush_loss::gray::{gray_psnr, gray_ssim};

        let device = self.canonical.device();
        let device_ad = device.clone().autodiff();
        let canonical_ad = lift_xray_splats_to_autodiff(self.canonical.clone());

        let deformed = if let Some(deform) = &self.deform {
            let n = canonical_ad.num_splats() as usize;
            let phase_t =
                Tensor::<2>::from_data(TensorData::new(vec![phase; n], [n, 1]), &device_ad);
            let time_t =
                Tensor::<2>::from_data(TensorData::new(vec![time; n], [n, 1]), &device_ad);
            let xyz = canonical_ad.means();
            let deforms = deform.forward(xyz, phase_t, time_t);
            deform_splats(&canonical_ad, &deforms)
        } else {
            canonical_ad
        };

        let img_size = glam::uvec2(gt.shape[1] as u32, gt.shape[0] as u32);
        // 计时探针: 设置环境变量 BRUSH_PROFILE_EVAL=1 打印各段耗时(排查瓶颈)。
        let profile = std::env::var("BRUSH_PROFILE_EVAL").is_ok();
        let t0 = std::time::Instant::now();
        let out = render_xray(deformed, camera, img_size, 1.0, self.config.fdk_residual).await;
        let t_render = t0.elapsed();
        let mut proj = out.img;
        if let Some(fdk) = &self.fdk {
            let fdk_drr = fdk.drr_for(camera, img_size).await;
            proj = proj.add(fdk_drr);
        }
        let intensity = (-proj.clamp(1e-3, 14.0)).exp();
        let gt_t = Tensor::<2>::from_data(gt.clone(), &device_ad);

        let psnr = gray_psnr(intensity.clone(), gt_t.clone())
            .into_scalar_async::<f32>()
            .await
            .expect("psnr readback");
        let t_psnr = t0.elapsed();
        let ssim = gray_ssim(intensity.clone(), gt_t.clone())
            .into_scalar_async::<f32>()
            .await
            .expect("ssim readback");

        // LPIPS: grayscale → 3-channel (VGG expects RGB), on the inner
        // (non-autodiff) device. Lower is more perceptually similar.
        let lpips = match &self.lpips {
            Some(model) => {
                let pred_inner = intensity.clone().inner();
                let gt_inner = gt_t.clone().inner();
                let [h, w] = [pred_inner.dims()[0], pred_inner.dims()[1]];
                // `[H,W]` → `[1,H,W,1]` → expand to `[1,H,W,3]` (broadcast
                // aligns from the trailing dim, so the channel axis must be
                // explicit).
                let pred3 = pred_inner.reshape([1, h, w, 1]).expand([1, h, w, 3]);
                let gt3 = gt_inner.reshape([1, h, w, 1]).expand([1, h, w, 3]);
                // 标准 LPIPS 输入 256×256。全分辨率(862×634)下 VGG 极慢:
                // 实测 eval_view 3.35s 里 ~99% 花在 LPIPS 上。缩放后预期
                // <0.5s。PSNR/SSIM 仍在全分辨率算; LPIPS 是感知指标,
                // 低分辨率输入是论文标准做法(分数会有数值偏移, 趋势一致)。
                use burn::tensor::module::interpolate;
                use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
                let pred3 = interpolate(
                    pred3.permute([0, 3, 1, 2]),
                    [256, 256],
                    InterpolateOptions::new(InterpolateMode::Bilinear)
                        .with_align_corners(false),
                )
                .permute([0, 2, 3, 1]);
                let gt3 = interpolate(
                    gt3.permute([0, 3, 1, 2]),
                    [256, 256],
                    InterpolateOptions::new(InterpolateMode::Bilinear)
                        .with_align_corners(false),
                )
                .permute([0, 2, 3, 1]);
                model
                    .lpips(pred3, gt3)
                    .into_scalar_async::<f32>()
                    .await
                    .expect("lpips readback")
            }
            None => f32::NAN,
        };
        let t_lpips = t0.elapsed();

        let pred = intensity
            .into_data_async()
            .await
            .expect("pred readback");
        let t_readback = t0.elapsed();

        if profile {
            let splats = self.canonical.num_splats();
            println!(
                "[profile] eval_view {} splats: total={:.3}s render={:.3}s psnr_ssim={:.3}s lpips={:.3}s readback={:.3}s",
                splats,
                t_readback.as_secs_f32(),
                t_render.as_secs_f32(),
                t_psnr.as_secs_f32(),
                t_lpips.as_secs_f32(),
                t_readback.as_secs_f32()
            );
        }

        XRayEvalSample {
            pred,
            gt: gt.clone(),
            psnr,
            ssim,
            lpips,
        }
    }

    /// One training step against a DICOM X-ray batch.
    pub async fn step(&mut self, batch: &SceneBatch) -> XRayTrainStats {
        let device = self.canonical.device();
        let device_ad = device.clone().autodiff();

        // ---- Lift canonical splats to autodiff ---------------------------
        let canonical_ad = lift_xray_splats_to_autodiff(self.canonical.clone());

        // ---- Phase + deform (static mode skips both) --------------------
        let (deformed, tv_term, anchor_term) = if let Some(deform) = &self.deform {
            let mut phase = batch.phase;
            if self.config.enable_ast && self.step_count >= self.config.warm_up {
                // Python `get_linear_noise_func`: noise = randn · 1/(step+1) ·
                // smooth_term where smooth_term ramps from 0 to 1 after warm-up.
                let interval =
                    1.0 / ((self.step_count % self.config.total_iters.max(1)) + 1) as f32;
                let progress =
                    ((self.step_count as f32 - self.config.warm_up as f32) / 100.0)
                        .clamp(0.0, 1.0);
                // CPU-side standard normal (Box-Muller): the old code sampled
                // a `[1,1]` GPU tensor and read it back **every step**, which
                // forces a GPU→CPU sync that serializes the pipeline.
                let delta = randn_f32() * interval * progress;
                phase += delta;
            }

            let n = canonical_ad.num_splats() as usize;
            let phase_t =
                Tensor::<2>::from_data(TensorData::new(vec![phase; n], [n, 1]), &device_ad);
            // D: 时间抖动 — 对 time 条件加高斯噪声, 强制形变场时间局部平滑
            // (类似 AST 相位噪声)。time 单调 = 帧号/fps (连续视频)。
            let mut time = batch.time;
            if self.config.time_jitter > 0.0 {
                time += randn_f32() * self.config.time_jitter;
            }
            let time_t =
                Tensor::<2>::from_data(TensorData::new(vec![time; n], [n, 1]), &device_ad);
            let xyz = canonical_ad.means();
            // 分阶段双场形变:
            // - 阶段1 (step < respi_after): 仅心电场 (相位条件), 呼吸场不用不训。
            // - 阶段2 (step >= respi_after): 心电场冻结 (detach), 呼吸场开训。
            let respi_active = self.config.respi_after > 0
                && self.step_count >= self.config.respi_after
                && self.respi.is_some();
            let mut deforms = deform.forward(xyz.clone(), phase_t, time_t.clone());
            if respi_active {
                let respi = self.respi.as_ref().expect("respi active requires field");
                // 呼吸场纯时间条件: phase 输入置 0 (不与其耦合)。零头初始化 →
                // identity 起步, 不注入随机位移。
                let phase0 =
                    Tensor::<2>::from_data(TensorData::new(vec![0.0f32; n], [n, 1]), &device_ad);
                let r = respi.forward(xyz.clone(), phase0, time_t.clone());
                deforms = if self.config.respi_freeze {
                    deforms.detach().compose(&r)
                } else {
                    deforms.compose(&r)
                };
            }
            let deformed = deform_splats(&canonical_ad, &deforms);
            // 刚性锚定: |mean(d_xyz)|² 惩罚形变场全场平均位移 (规范自由度 —
            // 整体平移应留在 canonical, 静态区域位移应为 0)。批次即全部
            // splats (means()), 均值 = 全场均值。
            let anchor_term = if self.config.hex_plane.rigid_anchor_weight > 0.0 {
                Some(deforms.d_xyz.clone().mean_dim(0).powf_scalar(2.0).sum())
            } else {
                None
            };
            // A: 时间 TV 正则 — 惩罚 deform 场在 (phase+dp, time+dt) 与
            // (phase, time) 的位移差 (随机子集), 编码"相邻帧形变小"。
            // 位置 detach: 只正则化 deform 网络参数, 不扰动 splat 放置。
            let tv_term = if self.config.time_tv_weight > 0.0 {
                use rand::seq::IteratorRandom;
                let sample = self.config.time_tv_sample.min(n);
                let idx: Vec<u32> = (0..n as u32)
                    .sample(&mut rand::rng(), sample);
                let idx_t = Tensor::<1, Int>::from_data(TensorData::new(idx, [sample]), &device_ad);
                let xyz_sub = xyz.select(0, idx_t).detach();
                let phase_next = (phase + self.config.time_tv_dp).clamp(0.0, 1.0);
                let time_next = time + self.config.time_tv_dt;
                let p0 = Tensor::<2>::from_data(TensorData::new(vec![phase; sample], [sample, 1]), &device_ad);
                let t0 = Tensor::<2>::from_data(TensorData::new(vec![time; sample], [sample, 1]), &device_ad);
                let p1 = Tensor::<2>::from_data(TensorData::new(vec![phase_next; sample], [sample, 1]), &device_ad);
                let t1 = Tensor::<2>::from_data(TensorData::new(vec![time_next; sample], [sample, 1]), &device_ad);
                let d0 = deform.forward(xyz_sub.clone(), p0, t0).d_xyz;
                let d1 = deform.forward(xyz_sub, p1, t1).d_xyz;
                Some((d1.sub(d0)).powi_scalar(2).mean())
            } else {
                None
            };
            (deformed, tv_term, anchor_term)
        } else {
            // Static reconstruction: no deform field, render the canonical
            // splats directly (shallow clone — tensors are Arc-backed).
            (canonical_ad.clone(), None, None)
        };

        // ---- Render + loss ----------------------------------------------
        let img_size = glam::uvec2(
            batch
                .img_gray
                .as_ref()
                .map_or(0, |g| g.shape[1]) as u32,
            batch
                .img_gray
                .as_ref()
                .map_or(0, |g| g.shape[0]) as u32,
        );
        assert!(img_size[0] > 0 && img_size[1] > 0, "X-ray batch needs a gray GT image");
        let out = render_xray(deformed, &batch.camera, img_size, 1.0, self.config.fdk_residual).await;

        // FDK-residual: total projection = splat residual + static prior DRR
        // (both in the `-ln(gray)` proj domain, so they sum additively).
        let mut proj = out.img;
        if let Some(fdk) = &self.fdk {
            let fdk_drr = fdk.drr_for(&batch.camera, img_size).await;
            proj = proj.add(fdk_drr);
        }

        let intensity = (-proj.clone().clamp(1e-3, 14.0)).exp();
        let gt = Tensor::<2>::from_data(batch.img_gray.clone().expect("gray GT"), &device_ad);
        let loss_cfg = GrayLossConfig {
            l1_weight: self.config.l1_weight,
            ssim_weight: self.config.ssim_weight,
            loss_type: self.config.loss_type,
            charbonnier_eps: self.config.loss_eps,
            huber_delta: self.config.loss_delta,
        };
        let mut loss = gray_loss(intensity.clone(), gt.clone(), &loss_cfg);
        // L1 残差稀疏先验 (FDK-residual): λ·mean(|density|), density=MU_WATER·raw
        // (signed)。只对残差 splat 生效 (fdk_residual 模式) — 促进残差稀疏。
        if self.config.resid_sparse_weight > 0.0 {
            let raw = canonical_ad
                .raw_opacities
                .val()
                .clamp(-20.0, 20.0)
                .abs()
                .mul_scalar(brush_cube::MU_WATER);
            loss = loss.add(raw.mean().mul_scalar(self.config.resid_sparse_weight));
        }
        // 时间 TV 正则项 (deform 块算好, 这里加入总损失)。
        if let Some(tv) = tv_term {
            loss = loss.add(tv.mul_scalar(self.config.time_tv_weight));
        }
        // 刚性锚定正则: |mean(d_xyz)|² → 消除规范自由度 (形变场里的整体平移)。
        if let Some(anchor) = anchor_term {
            loss = loss.add(anchor.mul_scalar(self.config.hex_plane.rigid_anchor_weight));
        }
        // 空间 TV 正则: HexPlane 特征平面 TV → 强制形变场低频/平滑
        // (否则形变场退化为带限周期模式拟合投影噪声)。
        if self.config.hex_plane.plane_tv_weight > 0.0
            && let Some(deform) = &self.deform
            && let Some(tv) = deform.plane_tv()
        {
            loss = loss.add(tv.mul_scalar(self.config.hex_plane.plane_tv_weight));
        }
        // Proj 域损失: 在 `proj = -ln(intensity)`（Beer-Lambert 衰减积分）域比较,
        // 避开 exp 压缩导致暗部/高 proj 区梯度衰减的问题。
        if self.config.proj_weight > 0.0 || self.config.proj_ssim_weight > 0.0 {
            let proj_pred = proj.clone().clamp(1e-3, 14.0); // = -ln(intensity)
            let proj_gt = gt.clone().clamp(1e-4, 1.0).log().neg(); // = -ln(gt)
            if self.config.proj_weight > 0.0 {
                // 同样使用配置的 robust 惩罚 (L1/Charbonnier/Huber/L2)。
                let proj_loss = if self.config.loss_type == GrayLossType::L1 {
                    (proj_pred.clone() - proj_gt.clone()).abs().mean()
                } else {
                    let mut proj_cfg = loss_cfg;
                    proj_cfg.ssim_weight = 0.0;
                    gray_loss(proj_pred.clone(), proj_gt.clone(), &proj_cfg)
                };
                loss = loss.add(proj_loss.mul_scalar(self.config.proj_weight));
            }
            if self.config.proj_ssim_weight > 0.0 {
                // Proj 域 SSIM: 结构感知项 (L1 只敏感绝对差)。值域非 [0,1],
                // c1/c2 常数相对偏小 → 更接近纯结构比, 仍有效。
                use brush_loss::gray::gray_ssim;
                let ssim = gray_ssim(proj_pred, proj_gt);
                loss = loss
                    .add(ssim.ones_like().sub(ssim).mul_scalar(self.config.proj_ssim_weight));
            }
        }
        // 多尺度(金字塔)损失: 在 1/2、1/4 分辨率上叠加 gray loss, 强制多尺度一致。
        if self.config.multiscale_weight > 0.0 {
            use burn::tensor::module::adaptive_avg_pool2d;
            let i4 = intensity.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(1);
            let g4 = gt.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(1);
            for scale in [2usize, 4] {
                let (h, w) = (img_size.y as usize / scale, img_size.x as usize / scale);
                let p = adaptive_avg_pool2d(i4.clone(), [h, w])
                    .squeeze_dim::<3>(0)
                    .squeeze_dim::<2>(0);
                let gg = adaptive_avg_pool2d(g4.clone(), [h, w])
                    .squeeze_dim::<3>(0)
                    .squeeze_dim::<2>(0);
                loss = loss
                    .add(gray_loss(p, gg, &loss_cfg).mul_scalar(self.config.multiscale_weight));
            }
        }
        // 多窗宽窗位损失: 在 proj(衰减)域做多个窗变换, 各窗下增强不同结构
        // (软组织 / 骨 / 细细节)。窗变换: clamp((x-(wl-ww/2))/ww, 0, 1)。
        if self.config.window_weight > 0.0 {
            let proj_pred = proj.clone().clamp(1e-3, 14.0);
            let proj_gt = gt.clone().clamp(1e-4, 1.0).log().neg();
            for (wl, ww) in [(0.6, 0.4), (1.2, 0.6), (0.4, 0.25)] {
                let wp = (proj_pred.clone() - (wl - ww / 2.0))
                    .div_scalar(ww)
                    .clamp(0.0, 1.0);
                let wg = (proj_gt.clone() - (wl - ww / 2.0))
                    .div_scalar(ww)
                    .clamp(0.0, 1.0);
                loss = loss
                    .add((wp - wg).abs().mean().mul_scalar(self.config.window_weight));
            }
        }
        // 梯度(Sobel 差分)损失: 增强边缘结构。
        // 后期 ramp: 权重从 0 平滑升到 grad_weight (3000 步前 L1/SSIM 主导),
        // 且按 GT 边缘幅度加权 → 集中推强边缘 (高频细节), 直击后期边缘模糊。
        if self.config.grad_weight > 0.0 {
            let ramp_to = if self.config.grad_ramp_to == 0 {
                self.config.total_iters
            } else {
                self.config.grad_ramp_to
            };
            let ramp = if self.step_count <= self.config.grad_ramp_from {
                0.0f32
            } else {
                let span = ramp_to.saturating_sub(self.config.grad_ramp_from).max(1) as f32;
                let u = ((self.step_count - self.config.grad_ramp_from) as f32 / span).clamp(0.0, 1.0);
                u * u * (3.0 - 2.0 * u) // smoothstep: 平滑进入后期强化
            };
            let pgx = intensity.clone().slice(s![.., 1..]) - intensity.clone().slice(s![.., ..-1]);
            let pgy = intensity.clone().slice(s![1.., ..]) - intensity.clone().slice(s![..-1, ..]);
            let ggx = gt.clone().slice(s![.., 1..]) - gt.clone().slice(s![.., ..-1]);
            let ggy = gt.clone().slice(s![1.., ..]) - gt.clone().slice(s![..-1, ..]);
            // gx 形状 [h, w-1], gy 形状 [h-1, w] → 对齐到公共内部 [h-1, w-1]。
            let pgx = pgx.slice(s![..-1, ..]);
            let pgy = pgy.slice(s![.., ..-1]);
            let ggx = ggx.slice(s![..-1, ..]);
            let ggy = ggy.slice(s![.., ..-1]);
            let grad_diff = (pgx.clone() - ggx.clone()).abs().add((pgy.clone() - ggy.clone()).abs());
            let gl = if self.config.grad_edge_scale > 0.0 {
                // GT 边缘幅度加权: 强边缘像素权重更高, 弱/平坦区权重低。
                let emag = ggx.powf_scalar(2.0).add(ggy.powf_scalar(2.0)).sqrt();
                let w = emag.div_scalar(self.config.grad_edge_scale).clamp(0.0, 1.0);
                grad_diff.mul(w).mean()
            } else {
                grad_diff.mean()
            };
            loss = loss.add(gl.mul_scalar(self.config.grad_weight * ramp));
        }
        let loss_inner = loss.clone().inner();
        let mut grads = loss.backward();

        // Optional readback of the predicted intensity for visualization.
        let pred_img = if self.collect_pred {
            Some(
                intensity
                    .into_data_async()
                    .await
                    .expect("pred image readback"),
            )
        } else {
            None
        };

        // ---- Density-control stats --------------------------------------
        let refine_weight = out
            .refine_weight_holder
            .grad_remove(&mut grads)
            .expect("viewspace gradients must be computed");
        let visible = out.visible;
        let refine_weight = detach_autodiff(refine_weight);
        // 屏幕半径(px) = 归一化半径 × 图像宽; 用于 screen-size prune。
        let max_radius_px = out.max_radius.clone().mul_scalar(img_size.x as f32);
        self.refiner.gather_stats(refine_weight, visible, Some(max_radius_px));

        // ---- Optimizer: canonical splats --------------------------------
        // Mean LR decays exponentially via `sched_mean`. The actual
        // per-component LR is encoded in the transforms scaling record, which
        // is **rebuilt every step** from the current `lr_mean` (mirrors the
        // RGB path in `train.rs`). Previously the scaling record was created
        // once with the first-step LR, so `lr_mean_end` never reached the
        // optimizer (the decay was only used for logging). Transforms are
        // stepped at base lr=1.0 (LR lives in the scaling) and the opacity
        // logits are stepped separately at `lr_opac` — the old code stepped
        // both together at lr=1.0, silently giving opacity an LR of 1.0
        // instead of the configured `lr_opac` (~80× too large → density
        // oscillation / saturation).
        let lr_mean = self.sched_mean.step();
        let opt_device = device.clone();
        let optimizer = self.optim_splats.get_or_insert_with(|| {
            AdamScaledConfig::new().with_epsilon(1e-15).init::<XRaySplats>()
        });
        {
            use burn::optim::record::AdaptorRecord;
            // transforms layout: means(3) + rotations(4) + log_scales(3).
            let lr_values: [f32; 10] = [
                lr_mean as f32, lr_mean as f32, lr_mean as f32,
                self.config.lr_rotation as f32, self.config.lr_rotation as f32,
                self.config.lr_rotation as f32, self.config.lr_rotation as f32,
                self.config.lr_scale as f32, self.config.lr_scale as f32,
                self.config.lr_scale as f32,
            ];
            let transform_scaling =
                Tensor::<1>::from_floats(lr_values.as_slice(), &opt_device).reshape([1, 10]);
            let mut record = optimizer.to_record();
            let existing = record.remove(&canonical_ad.transforms.id);
            let momentum = existing.and_then(|r| r.into_state::<2>().momentum);
            record.insert(
                canonical_ad.transforms.id,
                AdaptorRecord::from_state(crate::adam_scaled::AdamState::<2> {
                    momentum,
                    scaling: Some(transform_scaling),
                    reduce_moment_2: false,
                }),
            );
            // Keep the opacity state in the record (created on the first step
            // with `reduce_moment_2`); it is stepped separately at `lr_opac`
            // below. Because the scaling tensor is rebuilt every step from the
            // current schedule, newly densified splats automatically inherit
            // the correct LR (no per-splat scaling rows to keep in sync).
            if !record.contains_key(&canonical_ad.raw_opacities.id) {
                record.insert(
                    canonical_ad.raw_opacities.id,
                    AdaptorRecord::from_state(crate::adam_scaled::AdamState::<1> {
                        momentum: None,
                        scaling: None,
                        reduce_moment_2: true,
                    }),
                );
            }
            *optimizer = AdamScaledConfig::new()
                .with_epsilon(1e-15)
                .init::<XRaySplats>()
                .load_record(record);
        }

        // Step transforms at base lr=1.0 (real LR is in the scaling record),
        // then the opacity logits separately at `lr_opac`.
        let transforms_id = canonical_ad.transforms.id;
        let opacities_id = canonical_ad.raw_opacities.id;
        let grad_transforms =
            GradientsParams::from_params(&mut grads, &canonical_ad, &[transforms_id]);
        let grad_opac =
            GradientsParams::from_params(&mut grads, &canonical_ad, &[opacities_id]);

        // ---- Gradient diagnostics (optional) ----------------------------
        // Mean |∂L/∂·| per splat — verify that each parameter's LR actually
        // moves it (position step ≈ lr_mean · mean_grad mm/step). Non-consuming
        // reads via `GradientsParams::get` (before the grads are consumed by
        // the optimizer steps below).
        let grad_norms = if self.collect_grads {
            let n = canonical_ad.num_splats().max(1) as f32;
            let transforms_grad = grad_transforms.get::<2>(transforms_id);
            let density_grad = grad_opac.get::<1>(opacities_id);
            let (mean_grad, rot_grad, scale_grad, density_grad) =
                if let Some(t) = transforms_grad {
                    let means = t.clone().slice(s![.., 0..3]);
                    let rots = t.clone().slice(s![.., 3..7]);
                    let scales = t.slice(s![.., 7..10]);
                    let mean_grad = means
                        .abs()
                        .sum()
                        .into_scalar_async::<f32>()
                        .await
                        .unwrap_or(0.0)
                        / n;
                    let rot_grad = rots
                        .abs()
                        .sum()
                        .into_scalar_async::<f32>()
                        .await
                        .unwrap_or(0.0)
                        / n;
                    let scale_grad = scales
                        .abs()
                        .sum()
                        .into_scalar_async::<f32>()
                        .await
                        .unwrap_or(0.0)
                        / n;
                    let density_grad = if let Some(d) = density_grad {
                        d.abs()
                            .sum()
                            .into_scalar_async::<f32>()
                            .await
                            .unwrap_or(0.0)
                            / n
                    } else {
                        0.0
                    };
                    (mean_grad, rot_grad, scale_grad, density_grad)
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
            Some(XRayGradStats {
                mean_grad,
                rot_grad,
                scale_grad,
                density_grad,
            })
        } else {
            None
        };

        let canonical_updated = optimizer.step(1.0, canonical_ad, grad_transforms);
        let canonical_updated =
            optimizer.step(self.config.lr_opac, canonical_updated, grad_opac);

        // ---- Optimizer: deform network (static mode: skipped) -----------
        if let Some(deform) = &self.deform {
            let deform_optim = self.optim_deform.get_or_insert_with(|| {
                burn::optim::AdamConfig::new().init()
            });
            let deform_progress =
                (self.step_count as f64 / self.config.total_iters.max(1) as f64).clamp(0.0, 1.0);
            let lr_deform = self.config.lr_deform * (1.0 - deform_progress)
                + self.config.lr_deform_end * deform_progress;
            let deform_grads = GradientsParams::from_module(&mut grads, deform);
            let deform_updated = deform_optim.step(lr_deform, deform.clone(), deform_grads);
            self.deform = Some(deform_updated);
        }
        // 阶段2: 呼吸场 optimizer (阶段1 其无梯度, 不步进)。
        if let Some(respi) = &self.respi {
            if self.config.respi_after > 0 && self.step_count >= self.config.respi_after {
                let respi_optim = self.optim_respi.get_or_insert_with(|| {
                    burn::optim::AdamConfig::new().init()
                });
                let deform_progress = (self.step_count as f64
                    / self.config.total_iters.max(1) as f64)
                    .clamp(0.0, 1.0);
                let lr_deform = self.config.lr_deform * (1.0 - deform_progress)
                    + self.config.lr_deform_end * deform_progress;
                let respi_grads = GradientsParams::from_module(&mut grads, respi);
                let respi_updated = respi_optim.step(lr_deform, respi.clone(), respi_grads);
                self.respi = Some(respi_updated);
            }
        }

        // ---- Strip canonical back to inner backend ----------------------
        self.canonical = canonical_updated.valid();

        self.step_count += 1;

        // Loss readback only when requested: it is a GPU→CPU sync, and with
        // it disabled the CPU can keep submitting the next step while the GPU
        // is still executing the current one.
        let loss_val = if self.collect_loss {
            let v = loss_inner
                .into_scalar_async::<f32>()
                .await
                .expect("loss readback");
            self.last_loss = v;
            v
        } else {
            self.last_loss
        };

        XRayTrainStats {
            loss: loss_val,
            num_visible: out.num_visible,
            num_splats: self.canonical.num_splats(),
            lr_mean,
            pred_img,
            grad_norms,
        }
    }

    /// Run the density controller (densify + prune) if due.
    /// Diagnostic: count canonical splats whose distance from the isocenter
    /// exceeds `r0` (mm) and their mean activated density (mm^-1). Used to
    /// verify the mostly-air region beyond the FOV cylinder (r0 = W/2) is
    /// pruned after densification starts — if not, the prune threshold is
    /// wrong.
    pub async fn splats_beyond_radius(&self, r0: f32) -> (u32, u32, f32) {
        let means: Vec<f32> = self
            .canonical
            .means()
            .into_data_async()
            .await
            .expect("means readback")
            .into_vec::<f32>()
            .unwrap();
        let raw: Vec<f32> = self
            .canonical
            .raw_opacities
            .val()
            .into_data_async()
            .await
            .expect("raw readback")
            .into_vec::<f32>()
            .unwrap();
        let n = raw.len();
        let mut count = 0u32;
        let mut den = 0.0f64;
        for i in 0..n {
            let r = (means[i * 3] * means[i * 3]
                + means[i * 3 + 1] * means[i * 3 + 1]
                + means[i * 3 + 2] * means[i * 3 + 2])
            .sqrt();
            if r > r0 {
                count += 1;
                den += (brush_cube::MU_WATER * brush_cube::silu(raw[i])) as f64;
            }
        }
        let mean_den = if count > 0 { den / count as f64 } else { 0.0 };
        (count, n as u32, mean_den as f32)
    }

    pub async fn maybe_refine(&mut self, iter: u32) -> Option<XRayRefineStats> {
        if iter.is_multiple_of(self.config.refine.refine_every) {
            let (canonical, update, stats) = self.refiner.refine(iter, self.canonical.clone()).await;

            // Sync optimizer state: keep rows matching `keep_mask`, append
            // zero state for the densified clones.
            let keep_inds = update
                .keep_mask
                .argwhere_async()
                .await
                .squeeze_dim::<1>(1);
            let add_count = update.densify_inds.dims()[0] as u32;
            let split_count = update.split_inds.dims()[0] as u32;

            if let Some(optim) = &mut self.optim_splats {
                use burn::optim::record::AdaptorRecord;
                let transforms_id = self.canonical.transforms.id;
                let opacities_id = self.canonical.raw_opacities.id;
                let mut record = optim.to_record();
                // The per-component LR scaling is rebuilt from the current
                // schedule on every `step()` (shape `[1,10]`, broadcastable to
                // any splat count), so only the momentum tensors need to be
                // reindexed here. Newly densified clones start from zero
                // momentum (standard Adam behavior) and pick up the current LR
                // automatically on the next step — no per-splat scaling rows
                // to keep in sync.
                #[allow(clippy::explicit_iter_loop)]
                for (id, state) in record.iter_mut() {
                    let s = state.to_owned();
                    if *id == transforms_id {
                        // Rank-2 state: `[N,10]` momentum tensors.
                        let mut st: crate::adam_scaled::AdamState<2> = s.into_state();
                        if let Some(moment) = &mut st.momentum {
                            moment.moment_1 = moment.moment_1.clone().select(0, keep_inds.clone());
                            moment.moment_2 = moment.moment_2.clone().select(0, keep_inds.clone());
                            let [_, d] = moment.moment_1.dims();
                            // Split parents reset their momentum — both
                            // halves of a split start from zero (their
                            // scale/position changed discontinuously).
                            if split_count > 0 {
                                let split_global =
                                    keep_inds.clone().select(0, update.split_inds.clone());
                                let inds = split_global.clone().unsqueeze_dim(1).repeat_dim(1, d);
                                let neg1 = -moment.moment_1.clone().select(0, split_global.clone());
                                moment.moment_1 = moment
                                    .moment_1
                                    .clone()
                                    .scatter(0, inds, neg1, IndexingUpdateOp::Add);
                                let inds2 = split_global.clone().unsqueeze_dim(1).repeat_dim(1, d);
                                let neg2 = -moment.moment_2.clone().select(0, split_global.clone());
                                moment.moment_2 = moment
                                    .moment_2
                                    .clone()
                                    .scatter(0, inds2, neg2, IndexingUpdateOp::Add);
                            }
                            let zeros = Tensor::<2>::zeros([add_count as usize, d], &moment.moment_1.device());
                            moment.moment_1 = Tensor::cat(vec![moment.moment_1.clone(), zeros.clone()], 0);
                            let zeros2 = Tensor::<2>::zeros(
                                [add_count as usize, moment.moment_2.dims()[1]],
                                &moment.moment_2.device(),
                            );
                            moment.moment_2 = Tensor::cat(vec![moment.moment_2.clone(), zeros2], 0);
                        }
                        *state = AdaptorRecord::from_state(st);
                    } else if *id == opacities_id {
                        // Rank-1 state: `[N]` momentum tensors.
                        let mut st: crate::adam_scaled::AdamState<1> = s.into_state();
                        if let Some(moment) = &mut st.momentum {
                            moment.moment_1 = moment.moment_1.clone().select(0, keep_inds.clone());
                            moment.moment_2 = moment.moment_2.clone().select(0, keep_inds.clone());
                            let [_n] = moment.moment_1.dims();
                            // Split parents reset their momentum.
                            if split_count > 0 {
                                let split_global =
                                    keep_inds.clone().select(0, update.split_inds.clone());
                                let neg1 = -moment.moment_1.clone().select(0, split_global.clone());
                                moment.moment_1 = moment
                                    .moment_1
                                    .clone()
                                    .scatter(0, split_global.clone(), neg1, IndexingUpdateOp::Add);
                                let neg2 = -moment.moment_2.clone().select(0, split_global.clone());
                                moment.moment_2 = moment
                                    .moment_2
                                    .clone()
                                    .scatter(0, split_global, neg2, IndexingUpdateOp::Add);
                            }
                            let zeros = Tensor::<1>::zeros([add_count as usize], &moment.moment_1.device());
                            moment.moment_1 = Tensor::cat(vec![moment.moment_1.clone(), zeros.clone()], 0);
                            let zeros2 = Tensor::<1>::zeros([add_count as usize], &moment.moment_2.device());
                            moment.moment_2 = Tensor::cat(vec![moment.moment_2.clone(), zeros2], 0);
                        }
                        *state = AdaptorRecord::from_state(st);
                    }
                }
                self.optim_splats = Some(optim.clone().load_record(record));
            }

            self.canonical = canonical;
            return Some(stats);
        }
        None
    }
}

/// Initial canonical splat sampling region.
#[derive(Debug, Clone, Copy)]
pub enum InitRegion {
    /// Uniform in a ball of `radius` around the isocenter (historical).
    Ball { radius: f32 },
    /// Uniform in a cylinder aligned with the rotation axis (Z): radius
    /// `radius` in XY, half-height `half_height` along Z. Matches the
    /// cone-beam FOV geometry better than a ball.
    Cylinder { radius: f32, half_height: f32 },
}

/// Convenience: create an X-ray trainer with random canonical splats inside a
/// ball of radius `scene_extent` around the isocenter (used by the CLI path).
///
/// `fov = Some((cameras, img_size))` filters the random points to keep only
/// those that project inside at least one camera's FOV — points outside every
/// view frustum receive no image gradient, so their density stays at the init
/// value forever and they become high-opacity outliers. Pass `None` to keep
/// all sampled points (e.g. for cylinder init where points re-enter the FOV
/// during rotation).
pub fn create_xray_trainer(
    config: XRayTrainConfig,
    num_points: u32,
    scene_extent: f32,
    init: InitRegion,
    device: &Device,
    fov: Option<(&[brush_render::camera::Camera], glam::UVec2)>,
    fdk: Option<FdkPrior>,
) -> XRayTrainer {
    let mut config = config;
    // 分阶段双场模式: 心电场必须纯相位 (关 time), 避免两场耦合。
    if config.respi_after > 0 {
        config.enable_time = false;
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.refine.seed);
    use rand::{RngExt, SeedableRng};

    // 可选 FOV 过滤: 预计算各相机的 world→cam 变换 + 针孔参数 (fx, fy, cx, cy)。
    let fov_views: Option<(Vec<(glam::Affine3A, (f32, f32, f32, f32))>, glam::UVec2)> =
        fov.map(|(cams, img_size)| {
            (
                cams.iter()
                    .map(|c| {
                        let pin = c.build_pinhole_params(img_size);
                        (c.world_to_local(), (pin.fx, pin.fy, pin.cx, pin.cy))
                    })
                    .collect(),
                img_size,
            )
        });

    // Random positions, uniform in the sampling region (ball or cylinder,
    // optionally filtered to keep only points visible in ≥1 view).
    let mut means = Vec::with_capacity(num_points as usize * 3);
    let mut rots = Vec::with_capacity(num_points as usize * 4);
    // Beer-Lambert alpha = `opac · mu · exp(power)` with `mu ≈ scale·√(2π)`
    // (the along-ray integration factor). With KNN-sized scales (~5-13 mm) a
    // Beer-Lambert alpha = `opac · mu · exp(power)` with `mu ≈ scale·√(2π)`
    // (the along-ray integration factor). Start the density at the **physical
    // linear attenuation coefficient of water**, `μ_water ≈ 0.002 mm⁻¹` — the
    // DICOM GT is `raw = exp(-∫μ dl)` (uint16-quantized), so the splats'
    // density must live in the same μ units to land in the visible intensity
    // band. With KNN scales (~5-13 mm) one splat contributes an optical depth
    // of `μ·scale·√(2π) ≈ 0.03`, and a ray crossing the scene accumulates
    // `proj ≈ 0.4-1.0` → `intensity = exp(-proj) ∈ (0.35, 0.7)`: a visible
    // mid-gray with real gradients (vs. `sigmoid(0)=0.5` → `proj` saturating
    // `clamp(proj, 14)` → all-black). The optimizer + density controller then
    // grow/lower density where the anatomy (iodine, bone) demands it.
    const MU_WATER: f32 = 0.002; // mm^-1 (density activation scale)
    // Density activation is `MU_WATER · silu(raw)` (exp6). Start at the
    // configured init density: raw = inverse_silu(init_density / MU_WATER).
    // FDK-residual mode: signed (opac = MU_WATER·raw) + tiny |density| with a
    // random sign, so the residual starts near zero and can go either way.
    let (init_raw_opac, fdk_residual): (f32, bool) = if config.fdk_residual {
        let mag = config.fdk_residual_init_density / MU_WATER;
        (mag, true)
    } else {
        (brush_cube::inverse_silu(config.init_density / MU_WATER), false)
    };
    let mut raw_opac = Vec::with_capacity(num_points as usize);
    let mut attempts = 0u32;
    // 防死循环上限 (FOV 过滤可能拒绝大量随机点)。
    let max_attempts = (num_points as usize * 30).max(10_000) as u32;
    while means.len() < num_points as usize * 3 && attempts < max_attempts {
        attempts += 1;
        // 采样区域: 球 (均匀体积) 或绕 Z 的圆柱 (均匀体积)。
        let p = match init {
            InitRegion::Ball { radius } => {
                let u: f32 = rng.random_range(0.0..1.0);
                let r = radius * u.cbrt();
                let theta = rng.random_range(0.0..std::f32::consts::PI);
                let phi = rng.random_range(0.0..2.0 * std::f32::consts::PI);
                let (s, c) = theta.sin_cos();
                [r * s * phi.cos(), r * s * phi.sin(), r * c]
            }
            InitRegion::Cylinder { radius, half_height } => {
                // 均匀圆柱体积: r = R·√u1, θ = 2π·u2, z = half_h·(2u3−1)。
                let u1: f32 = rng.random_range(0.0..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                let u3: f32 = rng.random_range(0.0..1.0);
                let r = radius * u1.sqrt();
                let phi = 2.0 * std::f32::consts::PI * u2;
                let z = half_height * (2.0 * u3 - 1.0);
                [r * phi.cos(), r * phi.sin(), z]
            }
        };
        if let Some((views, img_size)) = &fov_views {
            let mut visible = false;
            for (view, (fx, fy, cx, cy)) in views {
                let cp = view.transform_point3(glam::Vec3::new(p[0], p[1], p[2]));
                if cp.z <= 0.0 {
                    continue;
                }
                let ux = fx * cp.x / cp.z + cx;
                let uy = fy * cp.y / cp.z + cy;
                const MARGIN: f32 = 10.0; // px, 留边避免贴边
                if ux > -MARGIN
                    && ux < img_size.x as f32 + MARGIN
                    && uy > -MARGIN
                    && uy < img_size.y as f32 + MARGIN
                {
                    visible = true;
                    break;
                }
            }
            if !visible {
                continue;
            }
        }
        means.extend_from_slice(&p);
        rots.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        let raw = if fdk_residual {
            if rng.random_range(0.0..1.0) < 0.5 {
                init_raw_opac
            } else {
                -init_raw_opac
            }
        } else {
            init_raw_opac
        };
        raw_opac.push(raw);
    }
    if means.len() < num_points as usize * 3 {
        log::warn!(
            "FOV filter kept only {} of requested {} points (max_attempts {})",
            means.len() / 3,
            num_points,
            max_attempts
        );
    }
    if fov_views.is_some() {
        let kept = means.len() / 3;
        println!(
            "FOV filter: kept {kept}/{num_points} random points (rejected {} outside every view)",
            attempts.saturating_sub(kept as u32)
        );
    }

    // Scale from local point density via KNN (same routine as the RGB
    // point-cloud init / the Python reference): `(1st + 2nd NN dist) / 4`,
    // clamped to `[1e-3, median_size·0.1]`, in log space. A too-tiny fixed
    // scale (e.g. 0.135 mm) would project to well under one pixel at C-arm
    // distances → `proj ≈ 0` → an all-white Beer-Lambert image with ~zero
    // gradients; KNN sizing guarantees continuous early coverage.
    //
    // Scale activation is `exp(raw)` (exp5 experiment), so the stored raw is
    // just the log-scale itself: `raw = ln(σ)`.
    let log_scales = crate::splat_init::compute_knn_scales(&means);
    debug_assert_eq!(log_scales.len(), means.len(), "one log-scale per axis");
    let raw_scales: Vec<f32> = log_scales;
    let canonical = XRaySplats::from_raw(means, rots, raw_scales, raw_opac, device);

    // Deform network only in deform mode; static mode omits it entirely.
    // The backend is selected by `config.deform_backend` (HexPlane default).
    let deform = if config.enable_deform {
        let device_ad = device.clone().autodiff();
        match config.deform_backend {
            DeformBackend::HashGrid => {
                let deform_cfg = DeformModelConfig {
                    coord_scale: scene_extent.max(1.0),
                    predict_scaling: config.predict_scaling,
                    enable_time: config.enable_time,
                    time_enc: config.time_enc.clone(),
                    ..DeformModelConfig::default()
                };
                Some(DeformNetwork::HashGrid(DeformModel::new(deform_cfg, &device_ad)))
            }
            DeformBackend::HexPlane => {
                let mut hex_cfg = config.hex_plane.clone();
                hex_cfg.hex_plane.coord_scale = scene_extent.max(1.0);
                hex_cfg.predict_scaling = config.predict_scaling;
                hex_cfg.enable_time = config.enable_time;
                hex_cfg.time_enc = config.time_enc.clone();
                Some(DeformNetwork::HexPlane(HexPlaneDeformModel::new(
                    hex_cfg,
                    &device_ad,
                )))
            }
        }
    } else {
        None
    };

    // Push the scene extent into the density controller (scale cap).
    config.refine.scene_extent = scene_extent;

    // 分阶段双场模式: 另建时间条件呼吸场 (融合 HexPlane + 时间 Fourier 基,
    // phase 输入恒定 0 → 时间条件化走 time_enc)。输出头零初始化 →
    // identity 起步, 切换时不注入随机位移噪声。
    let respi = if config.respi_after > 0 && config.enable_deform {
        let device_ad = device.clone().autodiff();
        let mut rcfg = config.hex_plane.clone();
        rcfg.hex_plane.coord_scale = scene_extent.max(1.0);
        rcfg.predict_scaling = false;
        rcfg.enable_time = true;
        rcfg.time_enc = config.time_enc.clone();
        Some(DeformNetwork::HexPlane(
            HexPlaneDeformModel::new(rcfg, &device_ad).zero_warp_heads(),
        ))
    } else {
        None
    };

    XRayTrainer::new(config, canonical, deform, respi, fdk, device)
}
