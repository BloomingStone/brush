//! Learnable temporal encoding for the deform network.
//!
//! Conditioning the deform field on real time `t` (seconds) lets the network
//! itself fit whatever temporal motion the data contains — the quasi-periodic
//! respiratory displacement (whose frequency is *not* known a priori), plus
//! any other non-periodic drift — during training. No breathing-frequency
//! prior / band-pass extraction is needed.
//!
//! [`TimeEncoding`] is a Fourier bank `emb(t) = [sin(2π f_k t), cos(2π f_k t)]`
//! whose frequencies `f_k` are **trainable parameters**, log-spaced over a
//! wide physiological range at init. Backprop moves `f_k` toward the dominant
//! temporal frequencies actually present in the data (e.g. ~0.8 Hz breathing),
//! so the network "discovers" the respiratory frequency distribution.
//!
//! The heart phase keeps its own circular HexPlane axis (periodic, exact from
//! the DICOM tag); `TimeEncoding` covers the *residual* (breathing + drift).

use burn::module::{Module, Param, ParamId};
use burn::tensor::{Tensor, TensorData};

/// Configuration for [`TimeEncoding`].
#[derive(Debug, Clone)]
pub struct TimeEncodingConfig {
    /// Number of Fourier pairs (sin + cos) in the bank.
    pub n_freqs: usize,
    /// Log-spaced frequency range (Hz) at init — wide enough to be
    /// acquisition-agnostic (human ~0.2 Hz .. small-animal ~1 Hz breathing).
    pub min_freq: f32,
    pub max_freq: f32,
    /// Keep the frequencies frozen at init (amplitudes still learned)?
    /// Default `false` → frequencies are trainable.
    pub learn_freqs: bool,
}

impl Default for TimeEncodingConfig {
    fn default() -> Self {
        Self {
            n_freqs: 10,
            min_freq: 0.15,
            max_freq: 3.0,
            learn_freqs: true,
        }
    }
}

/// Learnable Fourier temporal bank. See the module docs.
#[derive(Module, Debug)]
pub struct TimeEncoding {
    freqs: Param<Tensor<1>>,
    #[module(skip)]
    cfg: TimeEncodingConfig,
}

impl TimeEncoding {
    pub fn new(cfg: TimeEncodingConfig, device: &burn::tensor::Device) -> Self {
        let k = cfg.n_freqs.max(1);
        let freqs: Vec<f32> = (0..k)
            .map(|i| {
                let r = if k == 1 { 0.5 } else { i as f32 / (k - 1) as f32 };
                cfg.min_freq * (cfg.max_freq / cfg.min_freq.max(1e-6)).powf(r)
            })
            .collect();
        let t = Tensor::<1>::from_data(TensorData::new(freqs, [k]), device)
            .detach()
            .require_grad();
        Self {
            freqs: Param::initialized(ParamId::new(), t),
            cfg,
        }
    }

    pub fn config(&self) -> &TimeEncodingConfig {
        &self.cfg
    }

    /// Current (learned) frequencies, `[K]` Hz. Read back for diagnostics.
    pub fn frequencies(&self) -> Tensor<1> {
        self.freqs.val()
    }

    /// Output feature dimension per sample (`2 * n_freqs`).
    pub fn output_channels(&self) -> usize {
        self.cfg.n_freqs.max(1) * 2
    }

    /// Encode real time `t` (`[N, 1]`, seconds) → `[N, 2K]`
    /// `[sin(2π f_1 t), cos(2π f_1 t), ...]` with learnable `f_k`.
    pub fn forward(&self, time: Tensor<2>) -> Tensor<2> {
        let k = self.cfg.n_freqs.max(1);
        let n = time.dims()[0];
        // Frozen mode: detach so no gradient flows into the frequencies.
        let freqs = if self.cfg.learn_freqs {
            self.freqs.val()
        } else {
            self.freqs.val().detach()
        };
        // Broadcast [N,1] x [1,K] -> [N,K].
        let f = freqs.reshape([1, k]).expand([n, k]);
        let t = time.clone().expand([n, k]);
        let ang = t * f * std::f32::consts::TAU; // [N, K] radians
        let s = ang.clone().sin();
        let c = ang.cos();
        Tensor::cat(vec![s, c], 1) // [N, 2K]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_trainable_freqs() {
        let cfg = TimeEncodingConfig::default();
        assert!(cfg.learn_freqs);
        assert_eq!(cfg.n_freqs, 10);
    }

    #[tokio::test]
    async fn forward_shape_and_freq_learning() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let cfg = TimeEncodingConfig {
            n_freqs: 6,
            ..TimeEncodingConfig::default()
        };
        let enc = TimeEncoding::new(cfg, &device);
        assert_eq!(enc.output_channels(), 12);

        let n = 32;
        let t = Tensor::<2>::from_data(TensorData::new(vec![1.0f32; n], [n, 1]), &device);
        let emb = enc.forward(t);
        assert_eq!(emb.dims(), [n, 12]);
        let data = emb
            .clone()
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "time encoding has NaN");

        // Gradient flows back into the learnable frequencies.
        let loss = emb.sum();
        let grads = loss.backward();
        let g = enc
            .freqs
            .val()
            .grad(&grads)
            .expect("freqs gradient");
        let gd = g.into_data_async().await.unwrap().to_vec::<f32>().unwrap();
        assert!(
            gd.iter().any(|v| v.abs() > 1e-8),
            "frequencies should receive gradient (learnable)"
        );
    }

    #[tokio::test]
    async fn frozen_freqs_get_no_gradient() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let cfg = TimeEncodingConfig {
            learn_freqs: false,
            ..TimeEncodingConfig::default()
        };
        let enc = TimeEncoding::new(cfg, &device);
        let t = Tensor::<2>::from_data(TensorData::new(vec![0.5f32; 8], [8, 1]), &device);
        let loss = enc.forward(t).sum();
        let grads = loss.backward();
        let g = enc.freqs.val().grad(&grads);
        assert!(g.is_none(), "frozen freqs should not get gradient");
    }
}
