//! Minimal splat parameter container for the X-ray (density) path.
//!
//! The cone-beam X-ray renderer only needs `transforms` (means + quats +
//! log-scales, `[N,10]`) and `raw_opacities` (logits, `[N]`). Unlike
//! [`brush_render::gaussian_splats::Splats`] it carries **no SH / color /
//! mip / min-scale surface**, which the additive density projection never
//! uses. This keeps the X-ray API type-honest: callers cannot accidentally
//! pass (or be forced to fabricate) SH data, and the differentiable path
//! still gets real `Param`s to register gradients on.
//!
//! Convert from a full [`Splats`] with [`From`] — SH, `render_mip` and
//! `min_scale` are dropped (same behavior as passing a `Splats` directly
//! to the renderer before this type existed).

use brush_render::gaussian_splats::Splats;
use burn::module::{Module, Param, ParamId};
use burn::tensor::{Device, Tensor, TensorData, s};

/// X-ray splat parameters: `transforms` `[N,10]` (means + quats +
/// log-scales) and `raw_opacities` `[N]` logits.
#[derive(Module, Debug)]
pub struct XRaySplats {
    pub transforms: Param<Tensor<2>>,
    pub raw_opacities: Param<Tensor<1>>,
}
impl XRaySplats {
    /// Build from raw CPU data (means / rotations / log-scales / opacity
    /// logits). No SH is required — the X-ray renderer is SH-free.
    pub fn from_raw(
        means: Vec<f32>,
        rots: Vec<f32>,
        log_scales: Vec<f32>,
        raw_opac: Vec<f32>,
        device: &Device,
    ) -> Self {
        let n = means.len() / 3;
        let means = Tensor::from_data(TensorData::new(means, [n, 3]), device);
        let rots = Tensor::from_data(TensorData::new(rots, [n, 4]), device);
        let log_scales = Tensor::from_data(TensorData::new(log_scales, [n, 3]), device);
        let raw_opacities = Tensor::from_data(TensorData::new(raw_opac, [n]), device);
        Self::from_parts(means, rots, log_scales, raw_opacities)
    }

    /// Build from separate device tensors (means / rotations / log-scales /
    /// opacity logits), packed into the `[N,10]` `transforms` layout.
    pub fn from_parts(
        means: Tensor<2>,
        rotations: Tensor<2>,
        log_scales: Tensor<2>,
        raw_opacities: Tensor<1>,
    ) -> Self {
        assert_eq!(means.dims()[1], 3, "Means must be 3D");
        assert_eq!(rotations.dims()[1], 4, "Rotations must be 4D");
        assert_eq!(log_scales.dims()[1], 3, "Log-scales must be 3D");
        let transforms = Tensor::cat(vec![means, rotations, log_scales], 1);
        Self::from_tensor_data(transforms, raw_opacities)
    }

    /// Build from an already-packed `[N,10]` `transforms` tensor and `[N]`
    /// `raw_opacities`.
    pub fn from_tensor_data(transforms: Tensor<2>, raw_opacities: Tensor<1>) -> Self {
        assert_eq!(transforms.dims()[1], 10, "transforms must be [N,10]");
        Self {
            transforms: Param::initialized(ParamId::new(), transforms.detach().require_grad()),
            raw_opacities: Param::initialized(
                ParamId::new(),
                raw_opacities.detach().require_grad(),
            ),
        }
    }

    /// Like [`from_tensor_data`](Self::from_tensor_data) but **without**
    /// detaching or re-requiring gradients: the packed tensors keep their
    /// autodiff graph, so gradients flow back through the tensors that
    /// produced them (e.g. a deformation network applied to the canonical
    /// splats). The tensors must already be tracked (autodiff) — this is the
    /// training path.
    pub fn from_tensor_data_autodiff(transforms: Tensor<2>, raw_opacities: Tensor<1>) -> Self {
        assert_eq!(transforms.dims()[1], 10, "transforms must be [N,10]");
        Self {
            transforms: Param::initialized(ParamId::new(), transforms),
            raw_opacities: Param::initialized(ParamId::new(), raw_opacities),
        }
    }

    pub fn num_splats(&self) -> u32 {
        self.transforms.dims()[0] as u32
    }

    pub fn device(&self) -> Device {
        self.transforms.device()
    }

    /// Means (positions) — slice of transforms columns 0..3.
    pub fn means(&self) -> Tensor<2> {
        self.transforms.val().slice(s![.., 0..3])
    }

    /// Rotation quaternions — slice of transforms columns 3..7.
    pub fn rotations(&self) -> Tensor<2> {
        self.transforms.val().slice(s![.., 3..7])
    }

    /// Log-space scales — slice of transforms columns 7..10.
    pub fn log_scales(&self) -> Tensor<2> {
        self.transforms.val().slice(s![.., 7..10])
    }
}

impl From<&Splats> for XRaySplats {
    fn from(s: &Splats) -> Self {
        Self::from_tensor_data(s.transforms.val(), s.raw_opacities.val())
    }
}

impl From<Splats> for XRaySplats {
    fn from(s: Splats) -> Self {
        (&s).into()
    }
}
