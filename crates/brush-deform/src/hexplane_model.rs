//! HexPlane-based cardiac-phase deformation network (Stage 1: pure burn ops).
//!
//! Same interface and output semantics as [`crate::deform_model::DeformModel`]
//! (predicts per-splat `(d_xyz, d_scaling, d_rotation)` from canonical
//! positions + cardiac phase), but the spatial/time conditioning is a
//! [`HexPlane`] encoder instead of a hash grid + skip-MLP: six 2D feature
//! planes (XY/XZ/YZ + XT/YT/ZT) queried bilinearly and summed, then decoded
//! by a small MLP. Cardiac phase is periodic: the HexPlane time axis wraps,
//! and the MLP also receives `[sin(2πt), cos(2πt)]`.
//!
//! [`HexPlaneDeformConfig::predict_scaling`] defaults to `false`: the scaling
//! head is dropped entirely, so the deform field is **mass-conserving** — a
//! splat's integrated absorption is `∝ scale`, so a scale change would alter
//! the scene's total absorption. Local density changes come from splat
//! displacement / rotation only (elastic cardiac motion).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;

use crate::deform_model::{Deforms, axial_angle_to_quat};
use crate::hex_plane::{HexPlane, HexPlaneConfig};
use crate::mlp::Mlp;

/// Configuration for [`HexPlaneDeformModel`].
#[derive(Debug, Clone)]
pub struct HexPlaneDeformConfig {
    pub hex_plane: HexPlaneConfig,
    /// Decoder MLP hidden width.
    pub mlp_hidden: usize,
    /// Decoder MLP depth (number of Linear layers).
    pub mlp_layers: usize,
    /// Predict a per-splat scaling offset? Default `false` (mass-conserving
    /// deform field — displacement + rotation only).
    pub predict_scaling: bool,
}

impl Default for HexPlaneDeformConfig {
    fn default() -> Self {
        Self {
            hex_plane: HexPlaneConfig::default(),
            mlp_hidden: 128,
            mlp_layers: 2,
            predict_scaling: false,
        }
    }
}

/// Cardiac-phase-guided HexPlane deformation network.
#[derive(Module, Debug)]
pub struct HexPlaneDeformModel {
    hex_plane: HexPlane,
    decoder: Mlp,
    xyz_warp: Linear,
    scaling_warp: Option<Linear>,
    axial_warp: Linear,
    #[module(skip)]
    cfg: HexPlaneDeformConfig,
}

impl HexPlaneDeformModel {
    pub fn new(cfg: HexPlaneDeformConfig, device: &burn::tensor::Device) -> Self {
        let hex_plane = HexPlane::new(cfg.hex_plane.clone(), device);
        // Plane features + periodic phase pair [sin(2πt), cos(2πt)].
        let decoder = Mlp::new(hex_plane.output_channels() + 2, cfg.mlp_hidden, cfg.mlp_layers, device);
        let width = cfg.mlp_hidden;
        Self {
            hex_plane,
            decoder,
            xyz_warp: LinearConfig::new(width, 3).init(device),
            scaling_warp: cfg
                .predict_scaling
                .then(|| LinearConfig::new(width, 3).init(device)),
            axial_warp: LinearConfig::new(width, 3).init(device),
            cfg,
        }
    }

    pub fn config(&self) -> &HexPlaneDeformConfig {
        &self.cfg
    }

    /// Predict deformations for canonical positions `xyz` (`[N, 3]`, world
    /// mm) at a scalar `phase` (`[N, 1]`).
    pub fn forward(&self, xyz: Tensor<2>, phase: Tensor<2>) -> Deforms {
        let cfg = &self.cfg;
        let feat = self.hex_plane.forward(xyz.clone(), phase.clone()); // [N, C]

        // Periodic phase encoding: t=0 and t=1 give identical codes.
        let t = (phase - cfg.hex_plane.phase_min) / (cfg.hex_plane.phase_max - cfg.hex_plane.phase_min);
        let ang = t * std::f32::consts::TAU;
        let phase_enc = Tensor::cat(vec![ang.clone().sin(), ang.cos()], 1); // [N, 2]

        let h = self.decoder.forward(Tensor::cat(vec![feat, phase_enc], 1)); // [N, hidden]

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deform_splats;
    use brush_xray::XRaySplats;
    use burn::tensor::TensorData;

    #[tokio::test]
    async fn forward_produces_valid_deforms() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let model = HexPlaneDeformModel::new(HexPlaneDeformConfig::default(), &device);

        let n = 32;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; n * 3], [n, 3]), &device);
        let phase = Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let deforms = model.forward(xyz, phase);

        assert_eq!(deforms.d_xyz.dims(), [n, 3]);
        assert_eq!(deforms.d_scaling.dims(), [n, 3]);
        assert_eq!(deforms.d_rotation.dims(), [n, 4]);

        // predict_scaling=false (default): d_scaling must be exactly zero
        // (mass-conserving deform field).
        let ds = deforms
            .d_scaling
            .clone()
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            ds.iter().all(|v| *v == 0.0),
            "d_scaling should be zero when predict_scaling is off"
        );

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
    }

    #[tokio::test]
    async fn scaling_head_is_active_when_enabled() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let cfg = HexPlaneDeformConfig {
            predict_scaling: true,
            ..HexPlaneDeformConfig::default()
        };
        let model = HexPlaneDeformModel::new(cfg, &device);

        let n = 16;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.2f32; n * 3], [n, 3]), &device);
        let phase = Tensor::<2>::from_data(TensorData::new(vec![0.3f32; n], [n, 1]), &device);
        let deforms = model.forward(xyz, phase);
        let ds = deforms
            .d_scaling
            .clone()
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            ds.iter().any(|v| v.abs() > 1e-6),
            "d_scaling should be non-trivial with predict_scaling on"
        );
        assert!(
            ds.iter().all(|v| v.abs() <= 1.0 + 1e-6),
            "d_scaling out of tanh range"
        );
    }

    #[tokio::test]
    async fn gradients_flow_through_model() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let model = HexPlaneDeformModel::new(HexPlaneDeformConfig::default(), &device);

        let n = 16;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; n * 3], [n, 3]), &device)
            .require_grad();
        let phase = Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let deforms = model.forward(xyz.clone(), phase);

        let loss = deforms.d_xyz.sum() + deforms.d_scaling.sum() + deforms.d_rotation.sum();
        let grads = loss.backward();

        // Gradient flows all the way back to the canonical positions.
        let g = xyz.grad(&grads).expect("xyz gradient");
        let data = g.into_data_async().await.unwrap().to_vec::<f32>().unwrap();
        assert!(
            data.iter().any(|v| v.abs() > 1e-6),
            "no gradient on xyz"
        );
    }
}
