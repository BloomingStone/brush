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
//!   (fixed or dynamically percentile-based) threshold. Small splats are
//!   cloned (child scale capped at `scene_extent * percent_dense`), oversized
//!   ones (`max_scale > scene_extent * percent_dense`) are **split** in two —
//!   the parent shrinks by `split_scale_factor` and is offset one way while a
//!   same-density child is appended offset the other way (centroid-
//!   preserving), mirroring the RGB `SplatTrainer::refine_splats`;
//! - **resets density** every `density_reset_interval` steps.
//!
//! The returned [`XRayRefineUpdate`] lets the owning trainer sync its
//! optimizer state (same keep / append indexing as the splat tensors).

use std::collections::VecDeque;

use brush_cube::MU_WATER;
use brush_xray::XRaySplats;
use burn::tensor::{Bool, Device, Distribution, IndexingUpdateOp, Int, Tensor, TensorData, s};
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
    /// Soft-reset cap (mm⁻¹): every `density_reset_interval` steps, activated
    /// densities above this are clamped back down to it
    /// (`min(density, init_density)` — only ever lowers). Mirrors the Python
    /// project's `_reset_density`. Kept in sync with the trainer's
    /// `init_density`.
    pub init_density: f32,
    /// Max child scale = `scene_extent * percent_dense` (mm).
    pub percent_dense: f32,
    /// Upper bound on splat count.
    pub max_splats: u32,
    /// Scene extent in mm (C-arm isocenter distance; used for the scale cap).
    pub scene_extent: f32,
    /// Splats whose center lies farther than `max_bound_factor × scene_extent`
    /// from the isocenter are pruned every refine. The anatomy (human) always
    /// sits in a fixed region well inside the isocenter FOV, so this can be
    /// aggressive (default 3×) to kill drifted outliers that never receive
    /// gradients (they keep high opacity forever).
    pub max_bound_factor: f32,
    /// Prune splats whose accumulated max screen radius (px, larger axis)
    /// exceeds this every refine. 0 disables. Mirrors the Python project's
    /// `max_radii2D > max_screen_size` prune — kills oversized-on-screen blobs
    /// (a blur source).
    pub max_screen_size: f32,
    /// Fraction of above-threshold splats actually densified per refine.
    pub growth_select_fraction: f32,
    /// Split oversized high-gradient splats (`max_scale > scene_extent *
    /// percent_dense`) in two instead of only cloning them. Disabled keeps
    /// the historical clone-only behavior.
    pub enable_split: bool,
    /// Per-axis scale shrink applied by a split. `1/√2` conserves the total
    /// projected coverage (matches the RGB `SplatTrainer`).
    pub split_scale_factor: f32,
    /// RNG seed (position jitter for cloned / split splats).
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
            // Activated density = MU_WATER·softplus(raw) ≈ 0.002 mm⁻¹ at
            // water level. 5e-5 ≈ 2.5% of water — prune splats that have
            // essentially decayed to zero (matches the Python project's
            // `cull_density_threshold: 5e-5`).
            cull_density_threshold: 5e-5,
            // Disabled by default: an opacity/density reset is an RGB 3DGS
            // habit that does not survive the Beer-Lambert mapping — resetting
            // density mid-training destroys the learned attenuation field (the
            // path integral is a direct function of density) and PSNR dives
            // until it re-converges. Keep it off unless you know what you are
            // doing.
            density_reset_interval: 0,
            init_density: MU_WATER,
            percent_dense: 0.0005,
            max_splats: 1_000_000,
            scene_extent: 760.0,
            max_bound_factor: 3.0,
            max_screen_size: 0.0,
            growth_select_fraction: 0.25,
            enable_split: false,
            split_scale_factor: std::f32::consts::FRAC_1_SQRT_2,
            seed: 0,
        }
    }
}

/// Stats for one refine step.
#[derive(Debug, Clone, Default)]
pub struct XRayRefineStats {
    pub num_added: u32,
    /// Number of split parents (each also appends one child, counted in
    /// `num_added`).
    pub num_split: u32,
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
    /// Indices (into the kept splats) that were cloned/densified (each
    /// appends exactly one new splat).
    pub densify_inds: Tensor<1, Int>,
    /// Indices (into the kept splats) of split parents — their optimizer
    /// state must be reset (scale/position changed discontinuously).
    pub split_inds: Tensor<1, Int>,
}

/// Per-step accumulator for the viewspace gradient statistics.
pub struct XRayRefiner {
    config: XRayRefineConfig,
    xyz_gradient_accum: Tensor<1>,
    denom: Tensor<1>,
    /// Accumulated per-splat max screen radius in pixels (screen-size prune).
    #[allow(non_snake_case)]
    max_radii2D: Tensor<1>,
    recent_grad_percentile: VecDeque<f32>,
    grad_threshold: Option<f32>,
    rng: StdRng,
}

impl XRayRefiner {
    pub fn new(config: XRayRefineConfig, num_points: u32, device: &Device) -> Self {
        Self {
            xyz_gradient_accum: Tensor::<1>::zeros([num_points as usize], device),
            denom: Tensor::<1>::zeros([num_points as usize], device),
            max_radii2D: Tensor::<1>::zeros([num_points as usize], device),
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
    pub fn gather_stats(
        &mut self,
        refine_weight: Tensor<1>,
        visible: Tensor<1>,
        max_radius_px: Option<Tensor<1>>,
    ) {
        self.xyz_gradient_accum = self.xyz_gradient_accum.clone() + refine_weight;
        self.denom = self.denom.clone() + visible;
        if let Some(r) = max_radius_px {
            // 累积每 splat 见过的最大屏幕半径 (px): max_radii2D = max(·, r)。
            let grow = r.clone().greater(self.max_radii2D.clone());
            self.max_radii2D = self.max_radii2D.clone().mask_where(grow, r);
        }
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
        // Activated density = MU_WATER · silu(raw) (exp6); silu(raw)<0 for
        // raw<0 → decaying air splats fall under the cull threshold.
        let raw = splats.raw_opacities.val().clamp(-20.0, 20.0);
        let sig = raw.clone().neg().exp().add_scalar(1.0).recip();
        let density = raw.mul(sig).mul_scalar(MU_WATER);
        let prune_density = density.lower_elem(self.config.cull_density_threshold);

        let transforms_bad = row_non_finite(&splats.transforms.val());
        let opac_bad = row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1));
        let max_allowed_bounds = self.config.scene_extent * self.config.max_bound_factor;
        let bound_mask = (splats.means().abs())
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        let mut prune_mask = prune_density
            .bool_or(transforms_bad)
            .bool_or(opac_bad)
            .bool_or(bound_mask);
        if self.config.max_screen_size > 0.0 {
            // 屏幕上过大的点 (参考项目 max_radii2D > max_screen_size)。
            let screen_big = self
                .max_radii2D
                .clone()
                .greater_elem(self.config.max_screen_size);
            prune_mask = prune_mask.bool_or(screen_big);
        }
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

        // ---- Densify (clone small + split oversized) ---------------------
        let mut clone_inds = Vec::new();
        let mut split_inds = Vec::new();

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

            let (small_candidates, big_candidates): (Vec<i32>, Vec<i32>) =
                if self.config.enable_split && !candidates.is_empty() {
                    // Partition by max rendered scale `softplus(raw_log_scale)`
                    // (kept space): small → clone, oversized → split.
                    let kept_transforms = splats.transforms.val().select(0, keep_inds.clone());
                    let kept_raw_scale = kept_transforms.slice(s![.., 7..10]);
                    let kept_scales = kept_raw_scale.clamp(-20.0, 6.9078).exp();
                    let max_scale: Tensor<1> = kept_scales.max_dim(1).squeeze_dim(1); // [N_kept]
                    let oversized_thr = self.config.scene_extent * self.config.percent_dense;
                    let oversized = max_scale.greater_elem(oversized_thr);

                    let cand_t = Tensor::<1, Int>::from_data(
                        TensorData::new(candidates.clone(), [candidates.len()]),
                        &device,
                    );
                    let cand_oversized = oversized.select(0, cand_t.clone());
                    let big_idx: Vec<usize> = cand_oversized
                        .clone()
                        .argwhere_async()
                        .await
                        .squeeze_dim::<1>(1)
                        .into_data_async()
                        .await
                        .expect("big idx")
                        .into_vec::<i32>()
                        .expect("big idx vec")
                        .into_iter()
                        .map(|v| v as usize)
                        .collect();
                    let small_idx: Vec<usize> = cand_oversized
                        .bool_not()
                        .argwhere_async()
                        .await
                        .squeeze_dim::<1>(1)
                        .into_data_async()
                        .await
                        .expect("small idx")
                        .into_vec::<i32>()
                        .expect("small idx vec")
                        .into_iter()
                        .map(|v| v as usize)
                        .collect();

                    (
                        small_idx.into_iter().map(|i| candidates[i]).collect(),
                        big_idx.into_iter().map(|i| candidates[i]).collect(),
                    )
                } else {
                    (candidates, Vec::new())
                };

            // Cap growth by max_splats — both a clone and a split append
            // exactly one new splat (the split keeps its parent).
            let cur = splats.num_splats() as usize;
            let headroom = self.config.max_splats.saturating_sub(cur as u32) as usize;

            // Clones keep the conservative growth fraction; splits take the
            // full oversized set (they are the blur fix), bounded by whatever
            // headroom the clones leave.
            let mut grow = (small_candidates.len() as f32 * self.config.growth_select_fraction)
                .round() as usize;
            grow = grow.min(headroom).min(small_candidates.len());

            // Deterministic (seeded) selection of `grow` clones.
            use rand::seq::SliceRandom;
            let mut idxs: Vec<usize> = (0..small_candidates.len()).collect();
            idxs.shuffle(&mut self.rng);
            clone_inds = idxs[..grow]
                .iter()
                .map(|&i| small_candidates[i])
                .collect();

            let split_budget = headroom.saturating_sub(clone_inds.len());
            let split_grow = big_candidates.len().min(split_budget);
            let mut sidxs: Vec<usize> = (0..big_candidates.len()).collect();
            sidxs.shuffle(&mut self.rng);
            split_inds = sidxs[..split_grow]
                .iter()
                .map(|&i| big_candidates[i])
                .collect();
        }

        // ---- Apply prune -------------------------------------------------
        // Splats keep `keep_inds`; clone `densify_inds` (indices into the kept
        // set) appended at the end.
        let transforms_kept = splats.transforms.val().select(0, keep_inds.clone());
        let raw_opac_kept = splats.raw_opacities.val().select(0, keep_inds.clone());

        let mut new_transforms = transforms_kept;
        let mut new_raw_opac = raw_opac_kept;

        // ---- Clone (small high-gradient splats) --------------------------
        let clone_count = clone_inds.len();
        if clone_count > 0 {
            let inds = Tensor::from_data(
                TensorData::new(clone_inds.clone(), [clone_count]),
                &device,
            );

            let parent_t = new_transforms.clone().select(0, inds.clone());
            let parent_o = new_raw_opac.clone().select(0, inds);

            let parent_means = parent_t.clone().slice(s![.., 0..3]);
            let parent_rots = parent_t.clone().slice(s![.., 3..7]);
            let parent_log_scale = parent_t.slice(s![.., 7..10]);
            // Reverse activation: stored raw → rendered scale `exp(raw)` (exp5).
            let raw_clamped = parent_log_scale.clamp(-20.0, 6.9078);
            let parent_scales = raw_clamped.exp();

            // Child scale: half of parent, capped at scene_extent * percent_dense.
            let max_scale = self.config.scene_extent * self.config.percent_dense;
            let child_scales = (parent_scales.clone() * 0.5).clamp_max(max_scale);
            let child_log_scale = child_scales
                .into_data_async()
                .await
                .expect("child scales readback")
                .to_vec::<f32>()
                .expect("f32")
                .into_iter()
                .map(|s| s.max(1e-6).ln()) // inverse of exp activation
                .collect::<Vec<f32>>();
            let child_log_scale = Tensor::<2>::from_data(
                TensorData::new(child_log_scale, [parent_scales.dims()[0], 3]),
                &splats.raw_opacities.device(),
            );

            // Position jitter: rotate a small offset by the splat orientation.
            let samples = Tensor::random(
                [clone_count, 3],
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

        // ---- Split (oversized high-gradient splats) ----------------------
        // Mirrors the RGB `SplatTrainer::refine_splats`: the parent shrinks by
        // `split_scale_factor` (per-axis; the dominant axis gets the full
        // shrink) and is offset one way, while a same-density child is
        // appended offset the other way — centroid-preserving, so the
        // Beer-Lambert path integral is roughly conserved.
        let split_count = split_inds.len();
        if split_count > 0 {
            let inds = Tensor::from_data(
                TensorData::new(split_inds.clone(), [split_count]),
                &device,
            );

            let parent_t = new_transforms.clone().select(0, inds.clone());
            let parent_means = parent_t.clone().slice(s![.., 0..3]);
            let parent_rots = parent_t.clone().slice(s![.., 3..7]);
            let parent_log_scale = parent_t.slice(s![.., 7..10]);
            let parent_scales = parent_log_scale
                .clone()
                .clamp(-20.0, 6.9078)
                .exp();

            // Smooth covariance-aware shrink: k=1 on minor axes, k=
            // split_scale_factor on the dominant axis.
            let scales_sq = parent_scales.clone().powi_scalar(2);
            let max_sq = scales_sq.clone().max_dim(1).clamp_min(1e-30);
            let ratio = scales_sq / max_sq;
            let k = -ratio * (1.0_f32 - self.config.split_scale_factor) + 1.0;
            let offset_factor = (-k.clone().powi_scalar(2) + 1.0).clamp_min(0.0).sqrt();
            let offset_local = offset_factor * parent_scales;
            let samples = quaternion_vec_multiply(parent_rots.clone(), offset_local);
            let new_log_scale = parent_log_scale.clone() + k.log();

            // Parent: means -= samples, log_scale += log(k) (scatter-add).
            let scale_diff = new_log_scale.clone() - parent_log_scale;
            let inds_10 = inds.clone().unsqueeze_dim(1).repeat_dim(1, 10);
            let mut update = Tensor::zeros([split_count, 10], &device);
            update = update.slice_assign(s![.., 0..3], -samples.clone());
            update = update.slice_assign(s![.., 7..10], scale_diff);
            new_transforms = new_transforms.scatter(0, inds_10, update, IndexingUpdateOp::Add);

            // Child: means + samples, same rotation, same shrunk scale,
            // same density.
            let child_transforms = Tensor::cat(
                vec![parent_means + samples, parent_rots, new_log_scale],
                1,
            );
            new_transforms = Tensor::cat(vec![new_transforms, child_transforms], 0);

            let parent_raw = new_raw_opac.clone().select(0, inds);
            new_raw_opac = Tensor::cat(vec![new_raw_opac, parent_raw], 0);
        }

        // ---- Density soft-reset (min(density, init_density), 只降不升) ---
        let mut density_reset = false;
        if self.config.density_reset_interval > 0
            && iter.is_multiple_of(self.config.density_reset_interval)
        {
            // 软重置: new_density = min(current_density, init_density)。
            // 只把密度高于 init 的点压回 init（参考项目 `_reset_density`），
            // 空气点（已低于 cull）与正常组织保持不变 —— 避免硬重置毁掉
            // Beer-Lambert 衰减场。
            let reset_raw = brush_cube::inverse_silu(self.config.init_density / MU_WATER);
            let n = new_raw_opac.dims()[0];
            let reset_t = Tensor::<1>::from_data(TensorData::new(vec![reset_raw; n], [n]), &device);
            let raw = new_raw_opac.clone().clamp(-20.0, 20.0);
            let sig = raw.clone().neg().exp().add_scalar(1.0).recip();
            let density = raw.clone().mul(sig).mul_scalar(MU_WATER); // MU_WATER·silu(raw)
            let over = density.greater_elem(self.config.init_density);
            new_raw_opac = new_raw_opac.mask_where(over, reset_t);
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
        self.max_radii2D = Tensor::<1>::zeros([new_n], &device);

        // Appended index list = clones then splits (each appends exactly one).
        let mut appended = clone_inds.clone();
        appended.extend(split_inds.iter().copied());
        let append_count = appended.len();
        let densify_inds_tensor = if append_count > 0 {
            Tensor::from_data(TensorData::new(appended, [append_count]), &device)
        } else {
            Tensor::<1, Int>::from_data(TensorData::new(vec![0i32; 0], [0usize]), &device)
        };
        let split_inds_tensor = if split_count > 0 {
            Tensor::from_data(TensorData::new(split_inds.clone(), [split_count]), &device)
        } else {
            Tensor::<1, Int>::from_data(TensorData::new(vec![0i32; 0], [0usize]), &device)
        };

        let stats = XRayRefineStats {
            num_added: append_count as u32,
            num_split: split_count as u32,
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
                split_inds: split_inds_tensor,
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
        // First quarter have near-zero density (logit -20 → softplus ≈ 2e-9,
        // density ≈ 4e-12 ≪ cull_density_threshold 5e-5); the rest are
        // dense (logit 2 → density ≈ 0.002·2.13 ≈ 0.0043).
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
