//! HexPlane encoding (Cao & Johnson, "HexPlane: A Fast Representation for
//! Dynamic Scenes", CVPR 2023), implemented with differentiable burn tensor
//! ops — Stage 1 (no custom kernels; a fused cubecl version may follow).
//!
//! The 4D point `(x, y, z, phase)` is projected onto six 2D feature planes:
//!   spatial-only:     XY, XZ, YZ — grids `[rs, rs, C]`
//!   spatiotemporal:   XT, YT, ZT — grids `[rs, rt, C]`
//! Each plane is queried with bilinear interpolation and the six feature
//! vectors are combined element-wise (**sum**).
//!
//! The **time axis is circular** (toroidal interpolation): cardiac phase is
//! periodic — phase 0 and phase 1 are the same point of the cycle, so the
//! last time cell wraps into the first. A linear time axis would force the
//! field to learn the seam itself (wasteful and never exact).

use burn::module::{Module, Param, ParamId};
use burn::tensor::{Device, Int, Tensor, TensorData, s};
use rand::{RngExt, SeedableRng};

/// Configuration for the HexPlane encoding.
#[derive(Debug, Clone)]
pub struct HexPlaneConfig {
    /// Feature channels per plane cell (`C`).
    pub n_feature_dim: usize,
    /// Resolution of the spatial plane axes (`rs`; XY/XZ/YZ are `rs×rs`).
    pub spatial_resolution: u32,
    /// Resolution of the time axis (`rt`; XT/YT/ZT are `rs×rt`, circular).
    pub time_resolution: u32,
    /// World-coordinate normalization scale: `xyz / coord_scale + 1` maps to
    /// `[0, 2]` (use the scene extent, e.g. SOD for a C-arm scan).
    pub coord_scale: f32,
    /// Cardiac phase normalization range: `(phase - min) / (max - min)`.
    pub phase_min: f32,
    pub phase_max: f32,
    /// Init range for the plane values (uniform `[-init_scale, init_scale]`).
    pub init_scale: f32,
    /// RNG seed for the plane initialization.
    pub seed: u64,
    /// Use the fused cubecl kernels (Stage 2: one forward + one backward
    /// kernel) instead of the pure-burn reference implementation. Disable
    /// for the reference path / non-wgpu backends.
    pub fused: bool,
}

impl Default for HexPlaneConfig {
    fn default() -> Self {
        Self {
            n_feature_dim: 16,
            spatial_resolution: 64,
            time_resolution: 32,
            coord_scale: 760.0,
            phase_min: 0.0,
            phase_max: 1.0,
            init_scale: 0.1,
            seed: 0,
            fused: true,
        }
    }
}

/// HexPlane encoding module: six trainable 2D feature planes.
#[derive(Module, Debug)]
pub struct HexPlane {
    xy: Param<Tensor<3>>,
    xz: Param<Tensor<3>>,
    yz: Param<Tensor<3>>,
    xt: Param<Tensor<3>>,
    yt: Param<Tensor<3>>,
    zt: Param<Tensor<3>>,
    #[module(skip)]
    cfg: HexPlaneConfig,
}

impl HexPlane {
    pub fn new(cfg: HexPlaneConfig, device: &Device) -> Self {
        let rs = cfg.spatial_resolution as usize;
        let rt = cfg.time_resolution as usize;
        let c = cfg.n_feature_dim;
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let mut make_plane = |h: usize, w: usize| {
            let values: Vec<f32> = (0..h * w * c)
                .map(|_| rng.random_range(-cfg.init_scale..cfg.init_scale))
                .collect();
            let t = Tensor::<3>::from_data(TensorData::new(values, [h, w, c]), device)
                .detach()
                .require_grad();
            Param::initialized(ParamId::new(), t)
        };
        Self {
            xy: make_plane(rs, rs),
            xz: make_plane(rs, rs),
            yz: make_plane(rs, rs),
            xt: make_plane(rs, rt),
            yt: make_plane(rs, rt),
            zt: make_plane(rs, rt),
            cfg,
        }
    }

    pub fn config(&self) -> &HexPlaneConfig {
        &self.cfg
    }

    /// Output feature dimension per splat.
    pub fn output_channels(&self) -> usize {
        self.cfg.n_feature_dim
    }

    /// Spatial total-variation of the feature planes: mean |adjacent-cell
    /// feature diff| along each plane axis. Penalizing it forces the plane
    /// features (and hence the interpolated deform field) to be **smooth /
    /// low-frequency** — the learned field otherwise degenerates to a
    /// band-limited periodic pattern that fits projection noise.
    pub fn tv(&self) -> Tensor<1> {
        let plane_tv = |t: &Param<Tensor<3>>| {
            let v = t.val();
            let dx = v.clone().slice(s![1.., .., ..]) - v.clone().slice(s![..-1, .., ..]);
            let dy = v.clone().slice(s![.., 1.., ..]) - v.clone().slice(s![.., ..-1, ..]);
            dx.abs().mean().add(dy.abs().mean())
        };
        plane_tv(&self.xy)
            .add(plane_tv(&self.xz))
            .add(plane_tv(&self.yz))
            .add(plane_tv(&self.xt))
            .add(plane_tv(&self.yt))
            .add(plane_tv(&self.zt))
    }

    /// Diagnostic: the six feature planes `[(name, tensor), ...]`
    /// (`xy/xz/yz` are `[rs, rs, C]`, `xt/yt/zt` are `[rs, rt, C]`).
    pub fn planes(&self) -> Vec<(&'static str, Tensor<3>)> {
        vec![
            ("xy", self.xy.val()),
            ("xz", self.xz.val()),
            ("yz", self.yz.val()),
            ("xt", self.xt.val()),
            ("yt", self.yt.val()),
            ("zt", self.zt.val()),
        ]
    }

    /// Encode canonical positions `xyz` (`[N, 3]`, world mm) + cardiac phase
    /// (`[N, 1]`) into `[N, C]` plane-summed features. Uses the fused cubecl
    /// kernels by default; `HexPlaneConfig::fused = false` falls back to the
    /// pure-burn reference.
    pub fn forward(&self, xyz: Tensor<2>, phase: Tensor<2>) -> Tensor<2> {
        if self.cfg.fused {
            self.forward_fused(xyz, phase)
        } else {
            self.forward_pure(xyz, phase)
        }
    }

    /// Fused path: one forward kernel + one backward kernel (see
    /// [`crate::fused`]).
    fn forward_fused(&self, xyz: Tensor<2>, phase: Tensor<2>) -> Tensor<2> {
        let planes = [
            self.xy.val(),
            self.xz.val(),
            self.yz.val(),
            self.xt.val(),
            self.yt.val(),
            self.zt.val(),
        ];
        crate::fused::burn_glue::hex_plane_query_ad(xyz, phase, planes, &self.cfg)
    }

    /// Pure-burn reference implementation.
    fn forward_pure(&self, xyz: Tensor<2>, phase: Tensor<2>) -> Tensor<2> {
        let cfg = &self.cfg;
        let rs = cfg.spatial_resolution;
        let rt = cfg.time_resolution;

        // Normalize world mm -> [0, 1] (same map as the hash-grid model).
        let xyz_norm = ((xyz / cfg.coord_scale + 1.0) * 0.5).clamp(0.0, 1.0);
        let t = ((phase - cfg.phase_min) / (cfg.phase_max - cfg.phase_min)).clamp(0.0, 1.0);

        let x = xyz_norm.clone().slice(s![.., 0..1]);
        let y = xyz_norm.clone().slice(s![.., 1..2]);
        let z = xyz_norm.slice(s![.., 2..3]);

        let f_xy = self.query(&self.xy.val(), x.clone(), y.clone(), rs, rs, (false, false));
        let f_xz = self.query(&self.xz.val(), x.clone(), z.clone(), rs, rs, (false, false));
        let f_yz = self.query(&self.yz.val(), y.clone(), z.clone(), rs, rs, (false, false));
        let f_xt = self.query(&self.xt.val(), x.clone(), t.clone(), rs, rt, (false, true));
        let f_yt = self.query(&self.yt.val(), y.clone(), t.clone(), rs, rt, (false, true));
        let f_zt = self.query(&self.zt.val(), z, t, rs, rt, (false, true));

        f_xy + f_xz + f_yz + f_xt + f_yt + f_zt
    }

    /// Bilinear query of one plane. `plane` is `[h_axis, w_axis, C]`;
    /// `u`/`v` are normalized `[0, 1]` coordinates along each axis
    /// (`[N, 1]`). A wrapped axis interpolates circularly (`lo -> hi` wraps
    /// around the grid edge), so phase 0 and phase 1 land on the same cell.
    fn query(
        &self,
        plane: &Tensor<3>,
        u: Tensor<2>,
        v: Tensor<2>,
        h_axis: u32,
        w_axis: u32,
        wrap: (bool, bool),
    ) -> Tensor<2> {
        let (wrap_u, wrap_v) = wrap;
        let [h, w, c] = plane.dims();
        let table = plane.clone().reshape([h * w, c]);
        let (u0, u1, fu) = axis_indices(u, h_axis, wrap_u);
        let (v0, v1, fv) = axis_indices(v, w_axis, wrap_v);

        // Row-major flattening of `[h, w, c]`: row = u_idx * w + v_idx.
        let w_scale = w as i64;
        let row00 = u0.clone() * w_scale + v0.clone();
        let row01 = u0 * w_scale + v1.clone();
        let row10 = u1.clone() * w_scale + v0.clone();
        let row11 = u1 * w_scale + v1;

        let f00 = table.clone().select(0, row00);
        let f01 = table.clone().select(0, row01);
        let f10 = table.clone().select(0, row10);
        let f11 = table.select(0, row11);

        let one = Tensor::<1>::ones_like(&fu);
        let w00 = (one.clone() - fu.clone()) * (one.clone() - fv.clone());
        let w01 = (one.clone() - fu.clone()) * fv.clone();
        let w10 = fu.clone() * (one.clone() - fv.clone());
        let w11 = fu * fv;

        f00 * w00.unsqueeze_dim(1)
            + f01 * w01.unsqueeze_dim(1)
            + f10 * w10.unsqueeze_dim(1)
            + f11 * w11.unsqueeze_dim(1)
    }
}

/// Per-axis interpolation indices `(lo, hi, frac)` for a clamped (spatial)
/// or circular (time) axis of resolution `res`. `coord` is `[N, 1]` in
/// `[0, 1]`.
fn axis_indices(
    coord: Tensor<2>,
    res: u32,
    wrap: bool,
) -> (Tensor<1, Int>, Tensor<1, Int>, Tensor<1>) {
    let coord = coord.squeeze_dim::<1>(1); // [N]
    let scaled = if wrap {
        coord * res as f32
    } else {
        coord * (res as f32 - 1.0)
    };
    let lo_f = scaled.clone().floor();
    let frac = scaled - lo_f.clone();
    let lo = if wrap {
        // `lo` is non-negative here, so `%` is a true wrap; phase=1.0 lands
        // on cell 0 exactly (t·rt = rt → rt % rt = 0), same as phase=0.0.
        lo_f.clone().int() % res as i64
    } else {
        lo_f.clone().int()
    };
    let hi = if wrap {
        (lo.clone() + 1) % res as i64
    } else {
        (lo_f + 1.0).clamp_max(res as f32 - 1.0).int()
    };
    (lo, hi, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    #[test]
    fn default_config_has_16_features() {
        assert_eq!(HexPlaneConfig::default().n_feature_dim, 16);
    }

    #[tokio::test]
    async fn forward_produces_finite_features() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let cfg = HexPlaneConfig::default();
        let model = HexPlane::new(cfg.clone(), &device);

        let n = 64;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.1f32; n * 3], [n, 3]), &device);
        let phase = Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let feat = model.forward(xyz, phase);
        assert_eq!(feat.dims(), [n, cfg.n_feature_dim]);
        let data = feat
            .into_data_async()
            .await
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "features have NaN/inf");
    }

    /// The cardiac cycle is periodic: phase 0 and phase 1 are the same point,
    /// so the wrapped time axis must make them query the exact same cells
    /// (identical features — even with random plane init).
    #[tokio::test]
    async fn phase_zero_and_one_are_identical() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();
        let cfg = HexPlaneConfig {
            n_feature_dim: 4,
            spatial_resolution: 8,
            time_resolution: 16,
            ..HexPlaneConfig::default()
        };
        let model = HexPlane::new(cfg, &device);

        let n = 32;
        let xyz = Tensor::<2>::from_data(
            TensorData::new(vec![0.37f32; n * 3], [n, 3]),
            &device,
        );
        let p0 = Tensor::<2>::from_data(TensorData::new(vec![0.0f32; n], [n, 1]), &device);
        let p1 = Tensor::<2>::from_data(TensorData::new(vec![1.0f32; n], [n, 1]), &device);
        let f0 = model.forward(xyz.clone(), p0);
        let f1 = model.forward(xyz.clone(), p1);
        let diff = (f0.clone() - f1)
            .abs()
            .sum()
            .into_scalar_async::<f32>()
            .await
            .unwrap();
        assert_eq!(diff, 0.0, "phase 0 and phase 1 must query identical cells");

        // Sanity: a mid-cycle phase does differ (different time cells).
        let p_mid =
            Tensor::<2>::from_data(TensorData::new(vec![0.5f32; n], [n, 1]), &device);
        let f_mid = model.forward(xyz, p_mid);
        let diff = (f0 - f_mid)
            .abs()
            .sum()
            .into_scalar_async::<f32>()
            .await
            .unwrap();
        assert!(diff > 1e-6, "phase 0.5 should query different cells");
    }

    /// Pure-burn reference path: outputs finite, wrap semantics hold.
    #[tokio::test]
    async fn pure_path_forward_is_valid() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let cfg = HexPlaneConfig {
            fused: false,
            ..HexPlaneConfig::default()
        };
        let model = HexPlane::new(cfg.clone(), &device);

        let n = 32;
        let xyz = Tensor::<2>::from_data(TensorData::new(vec![0.2f32; n * 3], [n, 3]), &device);
        let p0 = Tensor::<2>::from_data(TensorData::new(vec![0.0f32; n], [n, 1]), &device);
        let p1 = Tensor::<2>::from_data(TensorData::new(vec![1.0f32; n], [n, 1]), &device);
        let f0 = model.forward(xyz.clone(), p0);
        let f1 = model.forward(xyz, p1);
        let diff = (f0 - f1)
            .abs()
            .sum()
            .into_scalar_async::<f32>()
            .await
            .unwrap();
        assert_eq!(diff, 0.0, "pure path: phase 0 and 1 must match");
    }

    /// The fused kernel must reproduce the pure-burn reference both in the
    /// forward output and in the gradients w.r.t. `xyz` and the planes.
    #[tokio::test]
    async fn fused_matches_pure_reference() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let device = device.autodiff();

        let cfg = |fused| HexPlaneConfig {
            n_feature_dim: 4,
            spatial_resolution: 8,
            time_resolution: 16,
            fused,
            seed: 7,
            ..HexPlaneConfig::default()
        };
        let fused_model = HexPlane::new(cfg(true), &device);
        let pure_model = HexPlane::new(cfg(false), &device);

        let n = 48;
        let xyz_data = TensorData::new(
            (0..n * 3)
                .map(|k| {
                    let v = (k as f32 * 0.137) % 1.0; // spread across [0, 1)
                    -1.0 + v * 2.0 // world space around the isocenter
                })
                .collect(),
            [n, 3],
        );
        let xyz = Tensor::<2>::from_data(xyz_data, &device).require_grad();
        let phase = Tensor::<2>::from_data(TensorData::new(vec![0.61f32; n], [n, 1]), &device);
        // Non-uniform upstream weights: a flat `sum()` would give every
        // (splat, feature) thread the same g=1 and mask xyz-VJP scaling bugs.
        let w_data: Vec<f32> = (0..n * 4).map(|k| 0.5 + ((k as f32 * 0.17) % 1.0)).collect();
        let weights = Tensor::<2>::from_data(TensorData::new(w_data, [n, 4]), &device);

        let f_fused = fused_model.forward(xyz.clone(), phase.clone());
        let f_pure = pure_model.forward(xyz.clone(), phase);

        let loss_fused = (f_fused.clone() * weights.clone()).sum();
        let loss_pure = (f_pure.clone() * weights).sum();
        let g_fused = loss_fused.backward();
        let g_pure = loss_pure.backward();

        // Forward outputs match.
        let d_fwd = (f_fused - f_pure)
            .abs()
            .sum()
            .into_scalar_async::<f32>()
            .await
            .unwrap();
        assert!(
            d_fwd < 1e-4,
            "fused vs pure forward mismatch: {d_fwd}"
        );

        // xyz gradients match.
        let xyz_g_fused = xyz.grad(&g_fused).expect("fused xyz grad");
        let xyz_g_pure = xyz.grad(&g_pure).expect("pure xyz grad");
        let d_g = (xyz_g_fused - xyz_g_pure)
            .abs()
            .sum()
            .into_scalar_async::<f32>()
            .await
            .unwrap();
        assert!(
            d_g < 1e-4,
            "fused vs pure xyz-gradient mismatch: {d_g}"
        );

        // Plane gradients match. Note: the fused path's grads come back as
        // inner (non-AD) tensors while the pure path's are AD-wrapped, so
        // read each sum back to a scalar separately instead of mixing them
        // in one arithmetic graph.
        let plane_grads = |model: &HexPlane, g: &burn::tensor::Gradients| {
            [
                model.xy.val().grad(g),
                model.xz.val().grad(g),
                model.yz.val().grad(g),
                model.xt.val().grad(g),
                model.yt.val().grad(g),
                model.zt.val().grad(g),
            ]
            .into_iter()
            .map(|p| p.expect("plane grad"))
            .collect::<Vec<_>>()
        };
        let mut fused_sum = 0.0f32;
        for t in plane_grads(&fused_model, &g_fused) {
            fused_sum += t.sum().into_scalar_async::<f32>().await.unwrap();
        }
        let mut pure_sum = 0.0f32;
        for t in plane_grads(&pure_model, &g_pure) {
            pure_sum += t.sum().into_scalar_async::<f32>().await.unwrap();
        }
        assert!(
            (fused_sum - pure_sum).abs() < 1e-4,
            "fused vs pure plane-gradient mismatch: {fused_sum} vs {pure_sum}"
        );
    }
}
