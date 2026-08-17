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
use brush_deform::{DeformModel, DeformModelConfig, deform_splats};
use brush_loss::gray::{GrayLossConfig, gray_loss};
use brush_render::burn_glue::detach_autodiff;
use brush_xray::XRaySplats;
use brush_xray_bwd::{lift_xray_splats_to_autodiff, render_xray};
use burn::{
    lr_scheduler::{
        LrScheduler,
        exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    },
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Device, IndexingUpdateOp, Tensor, TensorData, Distribution, s},
};

use crate::adam_scaled::{AdamScaled, AdamScaledConfig};
use crate::xray_refine::{XRayRefineConfig, XRayRefineStats, XRayRefiner};

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
    /// Density-control configuration.
    pub refine: XRayRefineConfig,
    /// Initial activated density (mm⁻¹) for the random splats. Defaults to
    /// μ_water (0.002). Raise it when the normalized / gamma-corrected GT
    /// target sits at higher intensity so the init ball starts at the right
    /// gray level (Beer-Lambert `proj` scales linearly with density).
    pub init_density: f32,
    /// L1 / SSIM weights for the gray loss.
    pub l1_weight: f32,
    pub ssim_weight: f32,
    /// Weight of the optional **projection-domain** L1 loss: compares
    /// `proj = -ln(intensity)` (the raw attenuation path integral) instead of
    /// the Beer-Lambert-compressed intensity. 0 disables it.
    pub proj_weight: f32,
}

impl Default for XRayTrainConfig {
    fn default() -> Self {
        Self {
            total_iters: 30_000,
            lr_mean: 2e-5,
            lr_mean_end: 2e-7,
            lr_scale: 5e-3,
            lr_rotation: 2e-3,
            lr_opac: 0.012,
            lr_deform: 1e-3,
            lr_deform_end: 1e-4,
            enable_ast: true,
            warm_up: 300,
            enable_deform: true,
            refine: XRayRefineConfig::default(),
            init_density: brush_cube::MU_WATER,
            l1_weight: 1.0,
            ssim_weight: 1.0,
            proj_weight: 0.0,
        }
    }
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
type DeformOptimType = OptimizerAdaptor<burn::optim::Adam, DeformModel>;

/// X-ray (deform-)GS trainer. With `enable_deform=false` this is a plain
/// static reconstruction: canonical splats optimized directly against the
/// multi-angle DICOM projections.
pub struct XRayTrainer {
    config: XRayTrainConfig,
    canonical: XRaySplats,
    deform: Option<DeformModel>,
    refiner: XRayRefiner,
    optim_splats: Option<OptimType>,
    optim_deform: Option<DeformOptimType>,
    sched_mean: ExponentialLrScheduler,
    step_count: u32,
    /// Read back the predicted intensity image every step (for visualization).
    collect_pred: bool,
    /// Collect per-parameter gradient norms every step (diagnostics).
    collect_grads: bool,
    /// VGG-LPIPS model for perceptual eval (loaded once; `None` keeps the
    /// eval free of the extra GPU memory).
    lpips: Option<lpips::LpipsModel>,
}

impl XRayTrainer {
    pub fn new(
        config: XRayTrainConfig,
        canonical: XRaySplats,
        deform: Option<DeformModel>,
        device: &Device,
    ) -> Self {
        let mut refine_cfg = config.refine.clone();
        refine_cfg.total_iters = config.total_iters;
        // 软重置的密度 cap 与训练初始化密度保持一致。
        refine_cfg.init_density = config.init_density;
        let num_points = canonical.num_splats();
        let refiner = XRayRefiner::new(refine_cfg, num_points, device);

        // Exponential mean LR decay: lr_end = lr_start · gamma^total_iters.
        let decay = (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_iters.max(1) as f64);
        let mut sched_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay)
            .init()
            .expect("valid lr scheduler");

        // First LR step aligns the current step count.
        sched_mean.step();

        Self {
            canonical,
            deform,
            refiner,
            optim_splats: None,
            optim_deform: None,
            sched_mean,
            config,
            step_count: 0,
            collect_pred: false,
            collect_grads: false,
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

    pub fn config(&self) -> &XRayTrainConfig {
        &self.config
    }

    pub fn num_splats(&self) -> u32 {
        self.canonical.num_splats()
    }

    pub fn canonical(&self) -> &XRaySplats {
        &self.canonical
    }

    /// Render the current model (canonical splats, deformed at `phase` when in
    /// deform mode) against a GT frame and compute grayscale PSNR / SSIM.
    /// Forward-only — no gradients are accumulated.
    pub async fn eval_view(
        &self,
        camera: &brush_render::camera::Camera,
        gt: &TensorData,
        phase: f32,
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
            let xyz = canonical_ad.means();
            let deforms = deform.forward(xyz, phase_t);
            deform_splats(&canonical_ad, &deforms)
        } else {
            canonical_ad
        };

        let img_size = glam::uvec2(gt.shape[1] as u32, gt.shape[0] as u32);
        let out = render_xray(deformed, camera, img_size, 1.0).await;
        let intensity = (-out.img.clamp(1e-3, 14.0)).exp();
        let gt_t = Tensor::<2>::from_data(gt.clone(), &device_ad);

        let psnr = gray_psnr(intensity.clone(), gt_t.clone())
            .into_scalar_async::<f32>()
            .await
            .expect("psnr readback");
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
                model
                    .lpips(pred3, gt3)
                    .into_scalar_async::<f32>()
                    .await
                    .expect("lpips readback")
            }
            None => f32::NAN,
        };

        let pred = intensity
            .into_data_async()
            .await
            .expect("pred readback");

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
        let deformed = if let Some(deform) = &self.deform {
            let mut phase = batch.phase;
            if self.config.enable_ast && self.step_count >= self.config.warm_up {
                // Python `get_linear_noise_func`: noise = randn · 1/(step+1) ·
                // smooth_term where smooth_term ramps from 0 to 1 after warm-up.
                let interval =
                    1.0 / ((self.step_count % self.config.total_iters.max(1)) + 1) as f32;
                let progress =
                    ((self.step_count as f32 - self.config.warm_up as f32) / 100.0)
                        .clamp(0.0, 1.0);
                let noise =
                    Tensor::<2>::random([1, 1], Distribution::Normal(0.0, 1.0), &device_ad);
                let n = noise.into_data_async().await.unwrap().to_vec::<f32>().unwrap()[0];
                let delta = n * interval * progress;
                phase += delta;
            }

            let n = canonical_ad.num_splats() as usize;
            let phase_t =
                Tensor::<2>::from_data(TensorData::new(vec![phase; n], [n, 1]), &device_ad);
            let xyz = canonical_ad.means();
            let deforms = deform.forward(xyz, phase_t);
            deform_splats(&canonical_ad, &deforms)
        } else {
            // Static reconstruction: no deform field, render the canonical
            // splats directly (shallow clone — tensors are Arc-backed).
            canonical_ad.clone()
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
        let out = render_xray(deformed, &batch.camera, img_size, 1.0).await;

        let intensity = (-out.img.clone().clamp(1e-3, 14.0)).exp();
        let gt = Tensor::<2>::from_data(batch.img_gray.clone().expect("gray GT"), &device_ad);
        let loss_cfg = GrayLossConfig {
            l1_weight: self.config.l1_weight,
            ssim_weight: self.config.ssim_weight,
        };
        let mut loss = gray_loss(intensity.clone(), gt.clone(), &loss_cfg);
        // Proj 域损失: 在 `proj = -ln(intensity)`（Beer-Lambert 衰减积分）域比较,
        // 避开 exp 压缩导致暗部/高 proj 区梯度衰减的问题。
        if self.config.proj_weight > 0.0 {
            let proj_pred = out.img.clone().clamp(1e-3, 14.0); // = -ln(intensity)
            let proj_gt = gt.clone().clamp(1e-4, 1.0).log().neg(); // = -ln(gt)
            loss = loss
                .add((proj_pred - proj_gt).abs().mean().mul_scalar(self.config.proj_weight));
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
        self.refiner.gather_stats(refine_weight, visible);

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
            let deform_grads = GradientsParams::from_grads(grads, deform);
            let deform_updated = deform_optim.step(lr_deform, deform.clone(), deform_grads);
            self.deform = Some(deform_updated);
        }

        // ---- Strip canonical back to inner backend ----------------------
        self.canonical = canonical_updated.valid();

        self.step_count += 1;

        let loss_val = loss_inner
            .into_scalar_async::<f32>()
            .await
            .expect("loss readback");

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

/// Convenience: create an X-ray trainer with random canonical splats inside a
/// ball of radius `scene_extent` around the isocenter (used by the CLI path).
pub fn create_xray_trainer(
    config: XRayTrainConfig,
    num_points: u32,
    scene_extent: f32,
    device: &Device,
) -> XRayTrainer {
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.refine.seed);
    use rand::{RngExt, SeedableRng};

    // Random positions, uniform in a ball of radius scene_extent.
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
    let init_raw_opac = brush_cube::inverse_silu(config.init_density / MU_WATER);
    let mut raw_opac = Vec::with_capacity(num_points as usize);
    for _ in 0..num_points {
        let u: f32 = rng.random_range(0.0..1.0);
        let r = scene_extent * u.cbrt();
        let theta = rng.random_range(0.0..std::f32::consts::PI);
        let phi = rng.random_range(0.0..2.0 * std::f32::consts::PI);
        let (s, c) = theta.sin_cos();
        means.push(r * s * phi.cos());
        means.push(r * s * phi.sin());
        means.push(r * c);
        rots.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        raw_opac.push(init_raw_opac);
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
    let deform = if config.enable_deform {
        let deform_cfg = DeformModelConfig {
            coord_scale: scene_extent.max(1.0),
            ..DeformModelConfig::default()
        };
        Some(DeformModel::new(deform_cfg, &device.clone().autodiff()))
    } else {
        None
    };

    // Push the scene extent into the density controller (scale cap).
    let mut config = config;
    config.refine.scene_extent = scene_extent;
    XRayTrainer::new(config, canonical, deform, device)
}
