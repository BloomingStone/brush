//! Single-channel (grayscale) image loss for X-ray training.
//!
//! The predicted X-ray projection is mapped to `[0, 1]` intensity via
//! `exp(-clamp(proj))` (Beer–Lambert → intensity) and compared against the
//! normalized DICOM ground truth (also `[0, 1]`). The loss is
//! `l1_weight * mean(|pred - gt|) + ssim_weight * (1 - mean(SSIM))` computed
//! with differentiable tensor ops (safe to call inside an autodiff tape).

use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};

/// Pixel-wise loss type for the gray intensity term (replaces plain L1).
/// X-ray projections are noisy, so a robust penalty is more stable than L1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrayLossType {
    /// Plain L1: `|r|`.
    #[default]
    L1,
    /// Charbonnier: `sqrt(r² + ε²)` — smooth, close to L1, finite at 0.
    Charbonnier,
    /// Huber: `0.5 r²` for `|r| ≤ δ`, else `δ(|r| - 0.5δ)`.
    Huber,
    /// Squared L2: `r²` (emphasizes large residuals / noise).
    L2,
}

/// Weights for the gray image loss (defaults match the Python project's
/// `RotateXrayMetrics`: `w_gray_loss = 1.0`, `w_ssim_loss = 1.0`).
#[derive(Debug, Clone, Copy)]
pub struct GrayLossConfig {
    /// Weight of the L1 / robust term.
    pub l1_weight: f32,
    /// Weight of the SSIM term (computed as `1 - SSIM`).
    pub ssim_weight: f32,
    /// Pixel-wise penalty type (L1 / Charbonnier / Huber / L2).
    pub loss_type: GrayLossType,
    /// Charbonnier / robust `ε` (smoothing floor), default `1e-3`.
    pub charbonnier_eps: f32,
    /// Huber threshold `δ`, default `0.1`.
    pub huber_delta: f32,
}

impl Default for GrayLossConfig {
    fn default() -> Self {
        Self {
            l1_weight: 1.0,
            ssim_weight: 1.0,
            loss_type: GrayLossType::L1,
            charbonnier_eps: 1e-3,
            huber_delta: 0.1,
        }
    }
}

/// Per-pixel residual penalty `φ(pred - gt)` (`[H, W]`) for the configured
/// [`GrayLossType`]. Differentiable.
fn pixel_loss(pred: &Tensor<2>, gt: &Tensor<2>, cfg: &GrayLossConfig) -> Tensor<2> {
    let r = pred.clone() - gt.clone();
    match cfg.loss_type {
        GrayLossType::L1 => r.abs(),
        GrayLossType::Charbonnier => {
            // sqrt(r² + ε²) — smooth L1.
            (r.clone().powi_scalar(2) + cfg.charbonnier_eps * cfg.charbonnier_eps).sqrt()
        }
        GrayLossType::L2 => r.powi_scalar(2),
        GrayLossType::Huber => {
            let d = cfg.huber_delta;
            let r2 = r.clone().powi_scalar(2) * 0.5;
            let abs = r.clone().abs();
            let lin = abs.clone().sub_scalar(0.5 * d).mul_scalar(d);
            let outside = abs.greater_elem(d); // BoolTensor
            r2.mask_where(outside, lin)
        }
    }
}

/// Gray (single-channel) image loss between a predicted X-ray projection and
/// the normalized ground truth, both `[H, W]` f32 in `[0, 1]`.
///
/// `loss = l1_weight * mean(φ(pred - gt)) + ssim_weight * (1 - mean(SSIM))`
/// where `φ` is the configured robust penalty (L1 / Charbonnier / Huber / L2).
pub fn gray_loss(pred: Tensor<2>, gt: Tensor<2>, cfg: &GrayLossConfig) -> Tensor<1> {
    let pixel = pixel_loss(&pred, &gt, cfg).mean().mul_scalar(cfg.l1_weight);
    if cfg.ssim_weight <= 0.0 {
        return pixel;
    }

    let ssim = ssim_2d(pred, gt, /*radius=*/ 5, /*sigma=*/ 1.5);
    pixel + (ssim.ones_like() - ssim).mul_scalar(cfg.ssim_weight)
}

/// Mean SSIM for `[0, 1]` grayscale images (same 11×11 Gaussian window as
/// [`gray_loss`]). Scalar tensor.
pub fn gray_ssim(pred: Tensor<2>, gt: Tensor<2>) -> Tensor<1> {
    ssim_2d(pred, gt, /*radius=*/ 5, /*sigma=*/ 1.5)
}

/// PSNR (dB) for `[0, 1]` grayscale images: `10·log10(1 / mean((p-g)²))`.
/// Scalar tensor.
pub fn gray_psnr(pred: Tensor<2>, gt: Tensor<2>) -> Tensor<1> {
    let mse = (pred - gt).powi_scalar(2).mean().clamp_min(1e-10);
    mse.recip().log() * (10.0 / std::f32::consts::LN_10)
}

/// Per-pixel SSIM (11×11 Gaussian window, `c1 = 0.01²`, `c2 = 0.03²`),
/// averaged to a scalar.
fn ssim_2d(pred: Tensor<2>, gt: Tensor<2>, radius: usize, sigma: f32) -> Tensor<1> {    let kernel = gaussian_kernel_2d(radius, sigma, &pred.device());
    let options = ConvOptions::new([1, 1], [radius, radius], [1, 1], 1);

    let x = pred.unsqueeze::<3>().unsqueeze::<4>(); // [H, W] -> [1, 1, H, W]
    let y = gt.unsqueeze::<3>().unsqueeze::<4>();

    let mu_x = conv2d(x.clone(), kernel.clone(), None, options.clone());
    let mu_y = conv2d(y.clone(), kernel.clone(), None, options.clone());

    let mu_x2 = mu_x.clone() * mu_x.clone();
    let mu_y2 = mu_y.clone() * mu_y.clone();
    let mu_xy = mu_x * mu_y;

    let sigma_x2 = conv2d(x.clone() * x.clone(), kernel.clone(), None, options.clone()) - mu_x2.clone();
    let sigma_y2 = conv2d(y.clone() * y.clone(), kernel.clone(), None, options.clone()) - mu_y2.clone();
    let sigma_xy = conv2d(x * y, kernel, None, options) - mu_xy.clone();

    let c1 = 0.01_f32 * 0.01;
    let c2 = 0.03_f32 * 0.03;

    let numerator = (mu_xy * 2.0 + c1) * (sigma_xy * 2.0 + c2);
    let denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2);
    let ssim_map = numerator / (denominator + 1e-8);

    ssim_map.mean()
}

/// Normalized 2D Gaussian kernel of size `(2*radius + 1)²` as `[1, 1, k, k]`.
fn gaussian_kernel_2d(radius: usize, sigma: f32, device: &Device) -> Tensor<4> {
    let size = 2 * radius + 1;
    let mut weights1d = vec![0.0_f32; size];
    let mut sum = 0.0_f32;
    for (i, w) in weights1d.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        *w = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += *w;
    }
    for w in &mut weights1d {
        *w /= sum;
    }

    let mut kernel = vec![0.0_f32; size * size];
    for i in 0..size {
        for j in 0..size {
            kernel[i * size + j] = weights1d[i] * weights1d[j];
        }
    }

    Tensor::from_data(TensorData::new(kernel, [1, 1, size, size]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_kernel_is_normalized() {
        let device = Device::default();
        let k = gaussian_kernel_2d(5, 1.5, &device);
        let data = k.into_data().to_vec::<f32>().unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "kernel sum = {sum}");
    }

    fn img(device: &Device, seed: f32) -> Tensor<2> {
        let data: Vec<f32> = (0..64).map(|i| ((i % 8) as f32 / 8.0 + seed) % 1.0).collect();
        Tensor::<2>::from_data(TensorData::new(data, [8, 8]), device)
    }

    #[test]
    fn gray_loss_is_zero_for_identical_images() {
        let device = Device::default();
        let pred = img(&device, 0.0);
        let gt = img(&device, 0.0);
        let loss = gray_loss(pred, gt, &GrayLossConfig::default());
        let v = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(v.abs() < 1e-4, "loss = {v}");
    }

    #[test]
    fn gray_loss_is_positive_for_different_images() {
        let device = Device::default();
        let pred = img(&device, 0.0);
        let gt = img(&device, 0.5);
        let loss = gray_loss(pred, gt, &GrayLossConfig::default());
        let v = loss.into_data().to_vec::<f32>().unwrap()[0];
        assert!(v > 0.0, "loss = {v}");
    }

    /// Robust penalties: Charbonnier ≈ L1 (finite at 0, no NaN at r=0),
    /// and L2 penalizes a large residual more than L1.
    #[test]
    fn robust_penalties_are_finite_and_ordered() {
        let device = Device::default();
        let pred = img(&device, 0.0);
        let gt = img(&device, 0.4);
        let m = |cfg: GrayLossConfig| {
            let c = GrayLossConfig {
                ssim_weight: 0.0,
                ..cfg
            };
            gray_loss(pred.clone(), gt.clone(), &c).into_data().to_vec::<f32>().unwrap()[0]
        };
        let l1 = m(GrayLossConfig::default());
        let char = m(GrayLossConfig { loss_type: GrayLossType::Charbonnier, ..Default::default() });
        let huber = m(GrayLossConfig { loss_type: GrayLossType::Huber, ..Default::default() });
        let l2 = m(GrayLossConfig { loss_type: GrayLossType::L2, ..Default::default() });

        for (name, v) in [("l1", l1), ("char", char), ("huber", huber), ("l2", l2)] {
            assert!(v.is_finite(), "{name} loss not finite: {v}");
            assert!(v > 0.0, "{name} loss should be > 0");
        }
        // Charbonnier ≈ L1 (with ε=1e-3); for residuals in [0,1], the sub-L1
        // penalties (Huber, L2) are strictly below L1.
        assert!((char - l1).abs() < 0.05, "charbonnier({char}) should ≈ L1({l1})");
        assert!(l2 < l1, "L2({l2}) should be < L1({l1}) for residuals in [0,1]");
        assert!(huber < l1, "Huber({huber}) should be < L1({l1})");
    }
}
