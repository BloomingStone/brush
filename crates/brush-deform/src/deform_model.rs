//! Cardiac-phase-guided deformation network (port of the Python
//! `HashGridDefromModel`).
//!
//! Given the canonical splat positions and a scalar cardiac phase, predicts a
//! per-splat deformation `(d_xyz, d_scaling, d_rotation)`; applying it yields
//! the deformed splats that are then X-ray rendered.
//!
//! All parameters are `burn::nn` `Param`s, so the whole network trains via
//! autodiff.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, s};
use brush_xray::XRaySplats;

use crate::hash_grid::{HashGrid, HashGridConfig};
use crate::mlp::SkipMlp;
use crate::positional::PositionalEncoding;
use crate::time_encoding::{TimeEncoding, TimeEncodingConfig};

/// Per-splat deformation outputs (matching the Python `Deforms`).
#[derive(Debug, Clone)]
pub struct Deforms {
    /// Position offset `[N, 3]`.
    pub d_xyz: Tensor<2>,
    /// Relative scale change `[N, 3]` in `[-1, 1]`
    /// (`new_scale = scale * (1 + d_scaling)`).
    pub d_scaling: Tensor<2>,
    /// Unit quaternion `[N, 4]` `(w, x, y, z)` rotation offset
    /// (`new_rot = rot * d_rotation`).
    pub d_rotation: Tensor<2>,
}

impl Deforms {
    /// Compose two deformation fields: `self` applied first, then `other`
    /// (`d_xyz = a + b`, `q = a·b`, `(1+s) = (1+sa)(1+sb)`). Used by staged
    /// training to sum a frozen cardiac field and a learned respiratory field.
    pub fn compose(&self, other: &Deforms) -> Deforms {
        let sa = self.d_scaling.clone();
        let sb = other.d_scaling.clone();
        Deforms {
            d_xyz: self.d_xyz.clone() + other.d_xyz.clone(),
            d_scaling: sa.clone() * sb.clone() + sa + sb,
            d_rotation: quat_multiply(self.d_rotation.clone(), other.d_rotation.clone()),
        }
    }

    /// Detach from autodiff — gradients no longer flow to this field's
    /// parameters (used to freeze the cardiac field in stage 2).
    pub fn detach(&self) -> Deforms {
        Deforms {
            d_xyz: self.d_xyz.clone().detach(),
            d_scaling: self.d_scaling.clone().detach(),
            d_rotation: self.d_rotation.clone().detach(),
        }
    }
}

/// Configuration for [`DeformModel`].
#[derive(Debug, Clone)]
pub struct DeformModelConfig {
    /// Hash-grid resolution levels (`x_multires` in the Python config).
    pub x_multires: u32,
    /// Hash-grid features per level.
    pub n_features_per_level: u32,
    /// Log2 hash table size.
    pub log2_hashmap_size: u32,
    /// Hash-grid base resolution.
    pub base_resolution: u32,
    /// Hash-grid max resolution.
    pub max_resolution: u32,
    /// Positional-encoding frequencies for the phase input (`t_multires`).
    pub t_multires: u32,
    /// Number of skip-connected MLP blocks (`combine_layers`).
    pub combine_layers: u32,
    /// MLP hidden width (`combine_W`).
    pub combine_width: u32,
    /// Inner block depth per skip block (`D`).
    pub block_depth: u32,
    /// Normalization scale for world coordinates: `xyz / coord_scale` is
    /// mapped to `[0, 1]` before the hash grid (use the scene extent, e.g.
    /// SOD for a C-arm scan).
    pub coord_scale: f32,
    /// Predict a per-splat scaling offset? Default `false` (mass-conserving
    /// deform field — displacement + rotation only; a scale change would
    /// alter a splat's integrated absorption).
    pub predict_scaling: bool,
    /// Condition the deform field on real time via a learnable Fourier bank
    /// (lets the network fit the respiratory frequency during training).
    pub enable_time: bool,
    /// Learnable temporal encoding configuration (used when `enable_time`).
    pub time_enc: TimeEncodingConfig,
}

impl Default for DeformModelConfig {
    fn default() -> Self {
        Self {
            x_multires: 7,
            n_features_per_level: 4,
            log2_hashmap_size: 7,
            base_resolution: 16,
            max_resolution: 128,
            t_multires: 6,
            combine_layers: 4,
            combine_width: 128,
            block_depth: 2,
            coord_scale: 760.0,
            predict_scaling: false,
            enable_time: false,
            time_enc: TimeEncodingConfig::default(),
        }
    }
}

/// Cardiac-phase (+ optional learned-time) deformation network.
#[derive(Module, Debug)]
pub struct DeformModel {
    hash_grid: HashGrid,
    #[module(skip)]
    phase_enc: PositionalEncoding,
    combine_mlp: SkipMlp,
    xyz_warp: Linear,
    scaling_warp: Option<Linear>,
    axial_warp: Linear,
    /// Learnable temporal Fourier bank (`None` when `enable_time = false`).
    #[module(skip)]
    time_enc: Option<TimeEncoding>,
    #[module(skip)]
    cfg: DeformModelConfig,
}

impl DeformModel {
    pub fn new(cfg: DeformModelConfig, device: &burn::tensor::Device) -> Self {
        let hash_grid_cfg = HashGridConfig {
            n_levels: cfg.x_multires,
            n_features_per_level: cfg.n_features_per_level,
            log2_hashmap_size: cfg.log2_hashmap_size,
            base_resolution: cfg.base_resolution,
            max_resolution: cfg.max_resolution,
            init_scale: 0.1,
            seed: 0,
        };
        let hash_grid = HashGrid::new(hash_grid_cfg, device);
        let phase_enc = PositionalEncoding::new(1, cfg.t_multires);

        let emb_x = hash_grid.output_channels() as usize;
        let emb_t = phase_enc.output_channels() as usize;
        let time_dim = if cfg.enable_time {
            TimeEncoding::new(cfg.time_enc.clone(), device).output_channels()
        } else {
            0
        };
        let width = cfg.combine_width as usize;

        let combine_mlp = SkipMlp::new(
            emb_x + emb_t + time_dim,
            width,
            cfg.combine_layers as usize,
            cfg.block_depth as usize,
            width,
            device,
        );

        let time_enc = cfg
            .enable_time
            .then(|| TimeEncoding::new(cfg.time_enc.clone(), device));

        Self {
            hash_grid,
            phase_enc,
            combine_mlp,
            xyz_warp: LinearConfig::new(width, 3).init(device),
            scaling_warp: cfg
                .predict_scaling
                .then(|| LinearConfig::new(width, 3).init(device)),
            axial_warp: LinearConfig::new(width, 3).init(device),
            time_enc,
            cfg,
        }
    }

    pub fn config(&self) -> &DeformModelConfig {
        &self.cfg
    }

    /// The learned temporal encoding (when `enable_time`), for diagnostics.
    pub fn time_encoding(&self) -> Option<&TimeEncoding> {
        self.time_enc.as_ref()
    }

    /// Predict deformations for canonical positions `xyz` (`[N, 3]`, world
    /// mm) at a scalar `phase` (`[N, 1]`) and (optionally) real `time`
    /// (`[N, 1]`, seconds).
    pub fn forward(&self, xyz: Tensor<2>, phase: Tensor<2>, time: Tensor<2>) -> Deforms {
        // Normalize world mm coords to [0, 1] for the hash grid.
        let xyz_norm = (xyz.clone() / self.cfg.coord_scale + 1.0) * 0.5;

        let x_emb = self.hash_grid.forward(xyz_norm); // [N, emb_x]
        let t_emb = self.phase_enc.forward(phase); // [N, emb_t]

        let mut inputs = vec![x_emb, t_emb];
        if let Some(enc) = &self.time_enc {
            inputs.push(enc.forward(time));
        }
        let h = self.combine_mlp.forward(Tensor::cat(inputs, 1)); // [N, W]

        let d_xyz = self.xyz_warp.forward(h.clone());
        let d_scaling = match &self.scaling_warp {
            Some(warp) => warp.forward(h.clone()).tanh(),
            None => Tensor::<2>::zeros(d_xyz.dims(), &xyz.device()),
        };
        let axial = self.axial_warp.forward(h).tanh();
        let d_rotation = axial_angle_to_quat(axial);

        Deforms {
            d_xyz,
            d_scaling,
            d_rotation,
        }
    }
}

/// Convert an axial-angle vector `v` (`[N, 3]`) to a unit quaternion
/// `(w, x, y, z)`: `q = (cos(ω/2), v/ω · sin(ω/2))` with `ω = |v|`.
pub(crate) fn axial_angle_to_quat(axial: Tensor<2>) -> Tensor<2> {
    let omega = axial.clone().powi_scalar(2).sum_dim(1).add_scalar(1e-10).sqrt(); // [N, 1]
    let q_w = (omega.clone() * 0.5).cos();
    // v/ω · sin(ω/2) = v · sin(ω/2)/ω
    let sinc = (omega.clone() * 0.5).sin() / omega.clamp_min(1e-10);
    let q_v = axial * sinc;
    let q = Tensor::cat(vec![q_w, q_v], 1); // [N, 4]

    let norm = q.clone().powi_scalar(2).sum_dim(1).add_scalar(1e-12).sqrt();
    q / norm
}

/// Apply a deformation to the canonical splats (keeps the autodiff graph:
/// the resulting splats are differentiable w.r.t. both the canonical params
/// and the deform network).
pub fn deform_splats(splats: &XRaySplats, deforms: &Deforms) -> XRaySplats {
    let transforms = splats.transforms.val();
    let means = transforms.clone().slice(s![.., 0..3]) + deforms.d_xyz.clone();
    let rots = quat_multiply(transforms.clone().slice(s![.., 3..7]), deforms.d_rotation.clone());
    // new_scale = scale * (1 + d_scaling)  →  log-space: + log(1 + d)
    let log_scales = transforms.slice(s![.., 7..10]) + (deforms.d_scaling.clone() + 1.0).log();

    let deformed = Tensor::cat(vec![means, rots, log_scales], 1);
    XRaySplats::from_tensor_data_autodiff(deformed, splats.raw_opacities.val())
}

/// Hamilton product of two quaternions `[N, 4]` in `(w, x, y, z)` order:
/// `result = a * b`.
fn quat_multiply(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let aw = a.clone().slice(s![.., 0..1]);
    let ax = a.clone().slice(s![.., 1..2]);
    let ay = a.clone().slice(s![.., 2..3]);
    let az = a.slice(s![.., 3..4]);
    let bw = b.clone().slice(s![.., 0..1]);
    let bx = b.clone().slice(s![.., 1..2]);
    let by = b.clone().slice(s![.., 2..3]);
    let bz = b.slice(s![.., 3..4]);

    let w = aw.clone() * bw.clone() - ax.clone() * bx.clone() - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x = aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
        - az.clone() * by.clone();
    let y = aw.clone() * by.clone() + ax.clone() * bz.clone() + ay.clone() * bw.clone()
        + az.clone() * bx.clone();
    let z = aw.clone() * bz.clone() - ax.clone() * by.clone() + ay.clone() * bx.clone()
        + az.clone() * bw.clone();

    Tensor::cat(vec![w, x, y, z], 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    #[test]
    fn axial_angle_to_quat_is_unit() {
        let device = burn::tensor::Device::default();
        let axial = Tensor::<2>::from_data(
            TensorData::new(vec![0.1f32, -0.2, 0.3], [1, 3]),
            &device,
        );
        let q = axial_angle_to_quat(axial);
        let data = q.into_data().to_vec::<f32>().unwrap();
        let n = (data[0].powi(2) + data[1].powi(2) + data[2].powi(2) + data[3].powi(2)).sqrt();
        assert!((n - 1.0).abs() < 1e-5, "norm = {n}");
    }

    #[tokio::test]
    async fn forward_produces_valid_deforms() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let model = DeformModel::new(
            DeformModelConfig {
                predict_scaling: true,
                ..DeformModelConfig::default()
            },
            &device,
        );

        let n = 32;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; n * 3], [n, 3]), &device);
        let phase =
            Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let time =
            Tensor::<2>::from_data(TensorData::new(vec![1.1f32; n], [n, 1]), &device);
        let deforms = model.forward(xyz, phase, time);

        assert_eq!(deforms.d_xyz.dims(), [n, 3]);
        assert_eq!(deforms.d_scaling.dims(), [n, 3]);
        assert_eq!(deforms.d_rotation.dims(), [n, 4]);

        // d_scaling ∈ [-1, 1] (tanh head).
        let ds = deforms
            .d_scaling
            .clone()
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(ds.iter().all(|v| v.abs() <= 1.0 + 1e-6), "d_scaling out of range");

        // d_rotation is a unit quaternion per splat.
        let q = deforms
            .d_rotation
            .clone()
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for row in q.chunks_exact(4) {
            let nrm = (row[0].powi(2) + row[1].powi(2) + row[2].powi(2) + row[3].powi(2)).sqrt();
            assert!((nrm - 1.0).abs() < 1e-4, "rotation norm = {nrm}");
        }

        // Deformed splats stay valid (scales > 0, finite).
        let splats = XRaySplats::from_raw(
            vec![0.0f32; n * 3],
            vec![1.0f32, 0.0, 0.0, 0.0].repeat(n),
            vec![-1.0f32; n * 3],
            vec![2.0f32; n],
            &device,
        );
        let deformed = deform_splats(&splats, &deforms);
        let t = deformed
            .transforms
            .val()
            .clone()
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(t.iter().all(|v| v.is_finite()), "deformed has NaN/inf");
        // log-scales (cols 7..10) after deformation must be finite (1 + d > 0).
    }

    #[tokio::test]
    async fn scaling_off_yields_zero_d_scaling() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let model = DeformModel::new(DeformModelConfig::default(), &device);

        let n = 16;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; n * 3], [n, 3]), &device);
        let phase =
            Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let time =
            Tensor::<2>::from_data(TensorData::new(vec![1.1f32; n], [n, 1]), &device);
        let deforms = model.forward(xyz, phase, time);
        let ds = deforms
            .d_scaling
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            ds.iter().all(|v| *v == 0.0),
            "d_scaling should be zero when predict_scaling is off"
        );
    }

    #[tokio::test]
    async fn gradients_flow_through_model() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let model = DeformModel::new(DeformModelConfig::default(), &device);

        let n = 16;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; n * 3], [n, 3]), &device)
            .require_grad();
        let phase =
            Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let time =
            Tensor::<2>::from_data(TensorData::new(vec![3.0f32; n], [n, 1]), &device);
        let deforms = model.forward(xyz.clone(), phase, time);

        let loss = deforms.d_xyz.sum() + deforms.d_scaling.sum() + deforms.d_rotation.sum();
        let grads = loss.backward();

        // Gradient flows all the way back to the input positions.
        let g = xyz.grad(&grads).expect("xyz gradient");
        let data = g.into_data_async().await.unwrap().to_vec::<f32>().unwrap();
        assert!(data.iter().any(|v| v.abs() > 1e-6), "no gradient on xyz");
    }
}
