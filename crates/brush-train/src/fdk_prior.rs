//! FDK static-prior integration: a calibrated FDK volume whose per-camera
//! DRR (`proj = scale · integral + bias`, the `-ln(gray)` domain) is added
//! to the splat renderer's projection so the Gaussian splats only have to
//! fit the *residual* (dynamic) part.
//!
//! The DRR is computed lazily per step on the GPU as a **constant in the
//! autodiff graph** (a burn `Backward` op with the volume as a parent whose
//! gradient is discarded) — no per-step GPU→CPU readback, and no raw-wgpu /
//! fusion-allocator interleaving (the pattern fit_volume uses in its loop).

use brush_drr::{DrrOps, DrrSettings};
use brush_render::burn_glue::{unwrap_ad_wgpu_float, wrap_ad_wgpu_float};
use brush_render::camera::Camera;
use burn::{
    backend::{
        Backend,
        autodiff::{
            checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::FloatTensor,
    },
    tensor::{Tensor, TensorData},
};

/// Custom AD op: forward projects the (constant) FDK volume for one camera,
/// producing a projection-domain image `[H, W]` in the autodiff graph. The
/// backward pass discards `v_output` — no gradients flow into the prior.
#[derive(Debug)]
struct FdkDrrBackwards;

impl<B: Backend + DrrOps> Backward<B, 1> for FdkDrrBackwards {
    type State = ();

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        // The FDK volume is a constant prior: consume the output gradient and
        // do not register anything for the volume parent.
        let _ = grads.consume::<B>(&ops.node);
    }
}

/// FDK static prior: holds the (constant, non-optimized) volume + geometry.
#[derive(Debug)]
pub struct FdkPrior {
    volume: Tensor<3>,
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
    rx: f32,
    ry: f32,
    rz: f32,
    steps: u32,
    scale: f32,
    bias: f32,
}

impl FdkPrior {
    /// Build from a raw float32 volume `[vol_x*vol_y*vol_z]` (kernel layout
    /// `[y,z,x]`) + world half extents + projection calibration.
    pub fn new(
        volume: Vec<f32>,
        vol_x: usize,
        vol_y: usize,
        vol_z: usize,
        rx: f32,
        ry: f32,
        rz: f32,
        steps: u32,
        scale: f32,
        bias: f32,
        device_ad: &burn::tensor::Device,
    ) -> Self {
        assert_eq!(
            volume.len(),
            vol_x * vol_y * vol_z,
            "FDK volume size mismatch"
        );
        let volume = Tensor::<3>::from_data(
            TensorData::new::<f32, _>(volume, [vol_x, vol_y, vol_z]),
            device_ad,
        )
        .require_grad();
        Self {
            volume,
            vol_x: vol_x as u32,
            vol_y: vol_y as u32,
            vol_z: vol_z as u32,
            rx,
            ry,
            rz,
            steps,
            scale,
            bias,
        }
    }

    /// Projection-domain DRR `[H, W]` for `camera` as an autodiff constant
    /// tensor (safe to add to the splat render's projection).
    pub async fn drr_for(&self, camera: &Camera, img_size: glam::UVec2) -> Tensor<2> {
        let settings = DrrSettings::new(
            camera,
            img_size.x,
            img_size.y,
            self.vol_x,
            self.vol_y,
            self.vol_z,
            self.steps,
            self.rx,
            self.ry,
            self.rz,
            self.scale,
            self.bias,
        );
        let volume_ad = unwrap_ad_wgpu_float(self.volume.clone());
        let prep = FdkDrrBackwards
            .prepare::<NoCheckpointing>([volume_ad.node.clone()])
            .compute_bound()
            .stateful();

        let volume_inner: FloatTensor<brush_cube::MainBackend> = volume_ad.primitive.clone();
        let out = <brush_cube::MainBackend as DrrOps>::drr_forward(&settings, volume_inner).await;

        let img_ad: FloatTensor<brush_render::burn_glue::AutodiffMain> = match prep {
            OpsKind::Tracked(prep) => prep.finish((), out),
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        wrap_ad_wgpu_float(img_ad)
    }
}
