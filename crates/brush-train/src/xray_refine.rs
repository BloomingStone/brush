//! X-ray density control (densify + prune) for [`XRaySplats`], mirroring the
//! Python project's `RotateXrayDensityController` and reusing the brush RGB
//! refinement patterns.
//!
//! The controller accumulates per-splat viewspace (mean2D) gradient norms
//! between refine steps (the `v_refine_weight` exposed by
//! `brush-xray-bwd`), then at each refine step:
//!
//! - **prunes** splats whose density (`sigmoid(raw_opacity)`) is below
//!   `cull_density_threshold`, plus NaN / out-of-bounds splats;
//! - **densifies** splats whose mean viewspace gradient exceeds a
//!   (fixed or dynamically percentile-based) threshold, capping child scale
//!   at `scene_extent * percent_dense`;
//! - **resets density** every `density_reset_interval` steps.
//!
//! The returned [`XRayRefineUpdate`] lets the owning trainer sync its
//! optimizer state (same keep / append indexing as the splat tensors).

use std::collections::VecDeque;

use brush_xray::XRaySplats;
use burn::tensor::activation::sigmoid;
use burn::tensor::{Bool, Device, Distribution, Int, Tensor, TensorData, s};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::quat_vec::quaternion_vec_multiply;

/// Floor used when converting raw opacity → density and back.
/// TODO NOT USED FOR NOW
pub const MIN_DENSITY: f32 = 1.0 / 255.0;

/// Hyperparameters for the X-ray density controller (defaults follow the
/// Python `RotateXrayDensityController`).
#[derive(Debug, Clone)]
pub struct XRayRefineConfig {
    /// How often (in steps) `refine` should be invoked.
    pub refine_every: u32,
    /// Total training steps (used for `densify_until_frac`).
    pub total_iters: u32,
    /// Do not densify before this step.
    pub densify_from_iter: u32,
    /// Stop densifying after `densify_until_frac * total_iters`.
    pub densify_until_frac: f32,
    /// Percentile of the viewspace gradient norm used as the dynamic
    /// densification threshold (max over the recent 5 refine steps).
    pub densify_grad_percentile: f32,
    /// If `Some`, use a fixed threshold instead of the dynamic percentile.
    pub fixed_grad_threshold: Option<f32>,
    /// Prune splats whose density is below this.
    pub cull_density_threshold: f32,
    /// Reset opacity every this many steps.
    pub density_reset_interval: u32,
    /// Max child scale = `scene_extent * percent_dense` (mm).
    pub percent_dense: f32,
    /// Upper bound on splat count.
    pub max_splats: u32,
    /// Scene extent in mm (C-arm isocenter distance; used for the scale cap).
    pub scene_extent: f32,
    /// Fraction of above-threshold splats actually densified per refine.
    pub growth_select_fraction: f32,
    /// RNG seed (position jitter for cloned splats).
    pub seed: u64,
}

impl Default for XRayRefineConfig {
    fn default() -> Self {
        Self {
            refine_every: 300,
            total_iters: 30_000,
            densify_from_iter: 500,
            densify_until_frac: 0.8,
            densify_grad_percentile: 0.98,
            fixed_grad_threshold: None,
            // Physical floor: μ_water ≈ 0.002 mm⁻¹ is a meaningful Beer-Lambert
            // contribution, so only prune splats that are numerically dead (a
            // density below MIN_ALPHA-level). A higher threshold (e.g. 5e-4 =
            // a quarter of water) culls nearly all low-density splats during
            // early training and starves the reconstruction.
            cull_density_threshold: 1e-5,
            // Disabled by default: an opacity/density reset is an RGB 3DGS
            // habit that does not survive the Beer-Lambert mapping — resetting
            // density mid-training destroys the learned attenuation field (the
            // path integral is a direct function of density) and PSNR dives
            // until it re-converges. Keep it off unless you know what you are
            // doing.
            density_reset_interval: 0,
            percent_dense: 0.0005,
            max_splats: 1_000_000,
            scene_extent: 760.0,
            growth_select_fraction: 0.25,
            seed: 0,
        }
    }
}

/// Stats for one refine step.
#[derive(Debug, Clone, Default)]
pub struct XRayRefineStats {
    pub num_added: u32,
    pub num_pruned: u32,
    pub grad_threshold: Option<f32>,
    pub density_reset: bool,
    pub total_splats: u32,
}

/// Index bookkeeping for the trainer to sync optimizer state after refine.
#[derive(Debug, Clone)]
pub struct XRayRefineUpdate {
    /// Keep-mask (bool `[N]`) over the pre-refine splats (pruned → false).
    pub keep_mask: Tensor<1, Bool>,
    /// Indices (into the kept splats) that were cloned/densified.
    pub densify_inds: Tensor<1, Int>,
}

/// Per-step accumulator for the viewspace gradient statistics.
pub struct XRayRefiner {
    config: XRayRefineConfig,
    xyz_gradient_accum: Tensor<1>,
    denom: Tensor<1>,
    recent_grad_percentile: VecDeque<f32>,
    grad_threshold: Option<f32>,
    rng: StdRng,
}

impl XRayRefiner {
    pub fn new(config: XRayRefineConfig, num_points: u32, device: &Device) -> Self {
        Self {
            xyz_gradient_accum: Tensor::<1>::zeros([num_points as usize], device),
            denom: Tensor::<1>::zeros([num_points as usize], device),
            recent_grad_percentile: VecDeque::with_capacity(5),
            grad_threshold: config.fixed_grad_threshold,
            rng: StdRng::seed_from_u64(config.seed),
            config,
        }
    }

    pub fn config(&self) -> &XRayRefineConfig {
        &self.config
    }

    /// Accumulate one step's viewspace gradient norm and visibility.
    pub fn gather_stats(&mut self, refine_weight: Tensor<1>, visible: Tensor<1>) {
        self.xyz_gradient_accum = self.xyz_gradient_accum.clone() + refine_weight;
        self.denom = self.denom.clone() + visible;
    }

    /// Mean viewspace gradient norm since the last refine.
    fn mean_grads(&self) -> Tensor<1> {
        let denom = self.denom.clone().clamp_min(1.0);
        let grads = self.xyz_gradient_accum.clone() / denom;
        // NaN / inf → 0 so the percentile stays well-defined.
        let bad = grads.clone().is_finite().bool_not();
        let zero = Tensor::<1>::zeros_like(&grads);
        grads.mask_where(bad, zero)
    }

    /// Fixed or dynamic (recent-5 max percentile) grad threshold.
    async fn compute_threshold(&mut self, grads: &Tensor<1>) -> Option<f32> {
        if let Some(t) = self.config.fixed_grad_threshold {
            return Some(t);
        }
        let values = grads
            .clone()
            .into_data_async()
            .await
            .expect("read grads")
            .into_vec::<f32>()
            .expect("grads f32");
        let mut finite: Vec<f32> = values.into_iter().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return self.grad_threshold;
        }
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = finite.len();
        let idx = ((self.config.densify_grad_percentile * (n - 1) as f32) as usize).min(n - 1);
        let pct = finite[idx];
        self.recent_grad_percentile.push_back(pct);
        while self.recent_grad_percentile.len() > 5 {
            self.recent_grad_percentile.pop_front();
        }
        let threshold = self
            .recent_grad_percentile
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        self.grad_threshold = Some(threshold);
        Some(threshold)
    }

    /// Densify + prune one step. `iter` is the current training step.
    pub async fn refine(
        &mut self,
        iter: u32,
        mut splats: XRaySplats,
    ) -> (XRaySplats, XRayRefineUpdate, XRayRefineStats) {
        let device = splats.device();

        let progress = iter as f32 / self.config.total_iters.max(1) as f32;
        let densifying = iter >= self.config.densify_from_iter
            && progress < self.config.densify_until_frac
            && splats.num_splats() < self.config.max_splats;

        let grads = self.mean_grads();
        let threshold = if densifying {
            self.compute_threshold(&grads).await
        } else {
            None
        };

        // ---- Prune -------------------------------------------------------
        let density = sigmoid(splats.raw_opacities.val());
        let prune_density = density.lower_elem(self.config.cull_density_threshold);

        let transforms_bad = row_non_finite(&splats.transforms.val());
        let opac_bad = row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1));
        let max_allowed_bounds = self.config.scene_extent * 100.0;
        let bound_mask = (splats.means().abs())
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        let prune_mask = prune_density
            .bool_or(transforms_bad)
            .bool_or(opac_bad)
            .bool_or(bound_mask);
        let num_pruned = prune_mask
            .clone()
            .int()
            .sum()
            .into_scalar_async::<i32>()
            .await
            .expect("count pruned") as u32;

        let keep_mask = prune_mask.bool_not();
        let keep_inds = keep_mask
            .clone()
            .argwhere_async()
            .await
            .squeeze_dim::<1>(1);

        // ---- Densify -----------------------------------------------------
        let mut densify_inds = Vec::new();

        if densifying {
            let threshold = threshold.expect("threshold set when densifying");

            // Above-threshold + visible splats, in **kept** index space (the
            // prune above reindexes; densify indices must target the kept set).
            let grads_kept = grads.select(0, keep_inds.clone());
            let denom_kept = self.denom.clone().select(0, keep_inds.clone());
            let above = grads_kept
                .clone()
                .greater_elem(threshold)
                .bool_and(denom_kept.greater_elem(0.0));

            let candidates: Vec<i32> = above
                .argwhere_async()
                .await
                .squeeze_dim::<1>(1)
                .into_data_async()
                .await
                .expect("above inds")
                .into_vec::<i32>()
                .expect("above inds vec");

            // Cap growth by max_splats.
            let cur = splats.num_splats() as usize;
            let headroom = self.config.max_splats.saturating_sub(cur as u32) as usize;
            let mut grow = (candidates.len() as f32 * self.config.growth_select_fraction).round()
                as usize;
            grow = grow.min(headroom).min(candidates.len());

            // Deterministic (seeded) selection of `grow` candidates.
            use rand::seq::SliceRandom;
            let mut idxs: Vec<usize> = (0..candidates.len()).collect();
            idxs.shuffle(&mut self.rng);
            densify_inds = idxs[..grow]
                .iter()
                .map(|&i| candidates[i])
                .collect();
        }

        // ---- Apply prune -------------------------------------------------
        // Splats keep `keep_inds`; clone `densify_inds` (indices into the kept
        // set) appended at the end.
        let transforms_kept = splats.transforms.val().select(0, keep_inds.clone());
        let raw_opac_kept = splats.raw_opacities.val().select(0, keep_inds.clone());

        let mut new_transforms = transforms_kept;
        let mut new_raw_opac = raw_opac_kept;

        let densify_count = densify_inds.len();
        if densify_count > 0 {
            let inds = Tensor::from_data(
                TensorData::new(densify_inds.clone(), [densify_count]),
                &device,
            );

            let parent_t = new_transforms.clone().select(0, inds.clone());
            let parent_o = new_raw_opac.clone().select(0, inds);

            let parent_means = parent_t.clone().slice(s![.., 0..3]);
            let parent_rots = parent_t.clone().slice(s![.., 3..7]);
            let parent_log_scale = parent_t.slice(s![.., 7..10]);
            let parent_scales = parent_log_scale.exp();

            // Child scale: half of parent, capped at scene_extent * percent_dense.
            let max_scale = self.config.scene_extent * self.config.percent_dense;
            let child_scales = (parent_scales.clone() * 0.5).clamp_max(max_scale);
            let child_log_scale = child_scales.log();

            // Position jitter: rotate a small offset by the splat orientation.
            let samples = Tensor::random(
                [densify_count, 3],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let offset = (parent_scales * 0.5) * samples;
            let child_means = parent_means + quaternion_vec_multiply(parent_rots.clone(), offset);

            let child_transforms = Tensor::cat(
                vec![child_means, parent_rots, child_log_scale],
                1,
            );

            new_transforms = Tensor::cat(vec![new_transforms, child_transforms], 0);
            new_raw_opac = Tensor::cat(vec![new_raw_opac, parent_o], 0);
        }

        // ---- Density reset -----------------------------------------------
        let mut density_reset = false;
        if self.config.density_reset_interval > 0
            && iter.is_multiple_of(self.config.density_reset_interval)
        {
            // Reset density to the *physical water background* (μ_water ≈
            // 0.002 mm⁻¹). Resetting raw_opac to 0 would mean `sigmoid(0) =
            // 0.5` — 250× water — which instantly overexposes every ray
            // (Beer-Lambert `proj` saturates) and makes training oscillate
            // between all-black and washed-out.
            let reset_logit = (0.002f32 / (1.0 - 0.002f32)).ln();
            new_raw_opac = new_raw_opac.mul_scalar(0.0).add_scalar(reset_logit);
            density_reset = true;
        }

        splats.transforms = splats
            .transforms
            .map(|_| new_transforms.detach().require_grad());
        splats.raw_opacities = splats
            .raw_opacities
            .map(|_| new_raw_opac.detach().require_grad());

        // ---- Reset accumulators ------------------------------------------
        // Must match the *new* splat count: `zeros_like` would keep the
        // pre-refine length and the next `gather_stats` would broadcast-mismatch.
        let new_n = splats.num_splats() as usize;
        self.xyz_gradient_accum = Tensor::<1>::zeros([new_n], &device);
        self.denom = Tensor::<1>::zeros([new_n], &device);

        let densify_inds_tensor = if densify_count > 0 {
            Tensor::from_data(
                TensorData::new(densify_inds.clone(), [densify_count]),
                &device,
            )
        } else {
            Tensor::<1, Int>::from_data(TensorData::new(vec![0i32; 0], [0usize]), &device)
        };

        let stats = XRayRefineStats {
            num_added: densify_count as u32,
            num_pruned,
            grad_threshold: threshold,
            density_reset,
            total_splats: splats.num_splats(),
        };

        (
            splats,
            XRayRefineUpdate {
                keep_mask,
                densify_inds: densify_inds_tensor,
            },
            stats,
        )
    }
}

/// Row-wise non-finite mask for a `[N, ...]` tensor → `[N]` bool.
fn row_non_finite(t: &Tensor<2>) -> Tensor<1, Bool> {
    t.clone().is_finite().bool_not().any_dim(1).squeeze_dim(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_splats(n: usize, device: &Device) -> XRaySplats {
        let means: Vec<f32> = (0..n * 3)
            .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let rots: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 0 { 1.0 } else { 0.0 })
            .collect();
        let log_scales: Vec<f32> = (0..n * 3).map(|_| -1.0).collect();
        // First quarter have near-zero density (logit -20 → sigmoid ≈ 2e-9,
        // below the cull_density_threshold default of 1e-5); the rest are
        // dense (logit 2 → ≈ 0.88).
        let raw_opac: Vec<f32> = (0..n)
            .map(|i| if i < n / 4 { -20.0 } else { 2.0 })
            .collect();
        XRaySplats::from_raw(means, rots, log_scales, raw_opac, device)
    }

    #[tokio::test]
    async fn refine_prunes_low_density_and_densifies_high_grad() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let n = 256;
        let splats = test_splats(n, &device);

        let config = XRayRefineConfig {
            total_iters: 1000,
            densify_from_iter: 0,
            densify_until_frac: 1.0,
            fixed_grad_threshold: Some(5.0),
            growth_select_fraction: 0.25,
            max_splats: 100_000,
            ..Default::default()
        };

        let mut refiner = XRayRefiner::new(config, n as u32, &device);
        // All splats visible with high viewspace gradient → the 192 dense ones
        // get densified (25% of them), the 64 low-density ones get pruned.
        refiner.gather_stats(
            Tensor::<1>::ones([n], &device).mul_scalar(10.0),
            Tensor::<1>::ones([n], &device),
        );

        let (new_splats, update, stats) = refiner.refine(100, splats).await;

        // 64 pruned, 192 kept, 25% of 192 = 48 densified → 192 + 48 = 240.
        assert_eq!(stats.num_pruned, 64, "pruned");
        assert_eq!(stats.num_added, 48, "densified");
        assert_eq!(stats.total_splats, 240, "total");
        assert_eq!(new_splats.num_splats(), 240);
        assert_eq!(update.densify_inds.dims()[0], 48);

        // No NaN in the refined splats.
        let t = new_splats.transforms.val().into_data_async().await.unwrap();
        let t_vals = t.into_vec::<f32>().unwrap();
        assert!(t_vals.iter().all(|v| v.is_finite()));
    }

    #[tokio::test]
    async fn refine_keeps_count_when_not_densifying() {
        let device: Device = brush_cube::test_helpers::test_device().await.into();
        let n = 64;
        let splats = test_splats(n, &device);

        let config = XRayRefineConfig {
            total_iters: 1000,
            densify_from_iter: 10_000, // never reached
            densify_until_frac: 1.0,
            ..Default::default()
        };

        let mut refiner = XRayRefiner::new(config, n as u32, &device);
        refiner.gather_stats(
            Tensor::<1>::ones([n], &device).mul_scalar(10.0),
            Tensor::<1>::ones([n], &device),
        );

        let (new_splats, _update, stats) = refiner.refine(100, splats).await;
        // Only the 16 low-density splats are pruned; no densification.
        assert_eq!(stats.num_pruned, 16);
        assert_eq!(stats.num_added, 0);
        assert_eq!(new_splats.num_splats(), 48);
    }
}
