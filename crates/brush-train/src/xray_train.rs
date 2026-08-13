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
    tensor::{Device, Tensor, TensorData, Distribution},
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
    /// L1 / SSIM weights for the gray loss.
    pub l1_weight: f32,
    pub ssim_weight: f32,
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
            l1_weight: 1.0,
            ssim_weight: 1.0,
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
        }
    }

    /// Enable / disable per-step readback of the predicted intensity image.
    /// Only turn this on when a visualization sink consumes it — it adds a
    /// GPU→CPU sync to every step.
    pub fn set_collect_pred(&mut self, collect: bool) {
        self.collect_pred = collect;
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
        let pred = intensity
            .into_data_async()
            .await
            .expect("pred readback");

        XRayEvalSample {
            pred,
            gt: gt.clone(),
            psnr,
            ssim,
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
        let loss = gray_loss(intensity.clone(), gt, &loss_cfg);
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
        let lr_mean = self.sched_mean.step();
        let opt_device = device.clone();
        let optimizer = self.optim_splats.get_or_insert_with(|| {
            // transforms layout: means(3) + rotations(4) + log_scales(3).
            let lr_values: [f32; 10] = [
                lr_mean as f32, lr_mean as f32, lr_mean as f32,
                self.config.lr_rotation as f32, self.config.lr_rotation as f32,
                self.config.lr_rotation as f32, self.config.lr_rotation as f32,
                self.config.lr_scale as f32, self.config.lr_scale as f32,
                self.config.lr_scale as f32,
            ];
            let scaling = Tensor::<1>::from_floats(lr_values.as_slice(), &opt_device)
                .reshape([1, 10]);
            let mut optim = AdamScaledConfig::new().with_epsilon(1e-15).init::<XRaySplats>();
            use burn::optim::record::AdaptorRecord;
            let record = optim.to_record();
            let mut record = record;
            record.insert(
                canonical_ad.transforms.id,
                AdaptorRecord::from_state(crate::adam_scaled::AdamState::<2> {
                    momentum: None,
                    scaling: Some(scaling),
                    reduce_moment_2: false,
                }),
            );
            record.insert(
                canonical_ad.raw_opacities.id,
                AdaptorRecord::from_state(crate::adam_scaled::AdamState::<1> {
                    momentum: None,
                    scaling: None,
                    reduce_moment_2: true,
                }),
            );
            optim = optim.load_record(record);
            optim
        });

        let splat_grads = GradientsParams::from_params(
            &mut grads,
            &canonical_ad,
            &[canonical_ad.transforms.id, canonical_ad.raw_opacities.id],
        );
        let canonical_updated = optimizer.step(1.0, canonical_ad, splat_grads);

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

            if let Some(optim) = &mut self.optim_splats {
                use burn::optim::record::AdaptorRecord;
                let transforms_id = self.canonical.transforms.id;
                let opacities_id = self.canonical.raw_opacities.id;
                let mut record = optim.to_record();
                // Need the ParamIds to dispatch on rank; `&mut record` would
                // yield `&mut (ParamId, _)` which is less ergonomic here.
                #[allow(clippy::explicit_iter_loop)]
                for (id, state) in record.iter_mut() {
                    let s = state.to_owned();
                    if *id == transforms_id {
                        // Rank-2 state: `[N,10]` momentum / scaling tensors.
                        let mut st: crate::adam_scaled::AdamState<2> = s.into_state();
                        if let Some(moment) = &mut st.momentum {
                            moment.moment_1 = moment.moment_1.clone().select(0, keep_inds.clone());
                            moment.moment_2 = moment.moment_2.clone().select(0, keep_inds.clone());
                        }
                        if let Some(scaling) = &mut st.scaling {
                            *scaling = scaling.clone().select(0, keep_inds.clone());
                            // Append zero state for the new clones: the
                            // per-component LR scaling rows must match the
                            // splat count, or the next optimizer step trips a
                            // broadcast mismatch (`[N+add,10]` vs `[N,10]`).
                            let [_, d] = scaling.dims();
                            let zeros = Tensor::<2>::zeros(
                                [add_count as usize, d],
                                &scaling.device(),
                            );
                            *scaling = Tensor::cat(vec![scaling.clone(), zeros], 0);
                        }
                        // Append zero state for the new clones.
                        if let Some(moment) = &mut st.momentum {
                            let [_, d] = moment.moment_1.dims();
                            let zeros = Tensor::<2>::zeros([add_count as usize, d], &moment.moment_1.device());
                            moment.moment_1 = Tensor::cat(vec![moment.moment_1.clone(), zeros.clone()], 0);
                            let zeros2 = Tensor::<2>::zeros([add_count as usize, moment.moment_2.dims()[1]], &moment.moment_2.device());
                            moment.moment_2 = Tensor::cat(vec![moment.moment_2.clone(), zeros2], 0);
                        }
                        *state = AdaptorRecord::from_state(st);
                    } else if *id == opacities_id {
                        // Rank-1 state: `[N]` momentum tensors, no scaling.
                        let mut st: crate::adam_scaled::AdamState<1> = s.into_state();
                        if let Some(moment) = &mut st.momentum {
                            moment.moment_1 = moment.moment_1.clone().select(0, keep_inds.clone());
                            moment.moment_2 = moment.moment_2.clone().select(0, keep_inds.clone());
                        }
                        if let Some(moment) = &mut st.momentum {
                            let [_n] = moment.moment_1.dims();
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
    const MU_WATER: f32 = 0.002; // mm^-1
    let init_density = MU_WATER;
    let init_raw_opac = (init_density / (1.0 - init_density)).ln();
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
    let log_scales = crate::splat_init::compute_knn_scales(&means);
    debug_assert_eq!(log_scales.len(), means.len(), "one log-scale per axis");
    let canonical = XRaySplats::from_raw(means, rots, log_scales, raw_opac, device);

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
