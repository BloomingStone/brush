//! Autodiff wiring for the fused HexPlane query: a hand-rolled burn
//! `Backward` op (mirroring `brush-xray-bwd::burn_glue`) that saves the
//! forward intermediates and runs the fused backward kernel, plus the
//! high-level `hex_plane_query_ad` entry used by [`crate::hex_plane::HexPlane`].

use brush_cube::MainBackend;
use brush_render::burn_glue::{unwrap_ad_wgpu_float, wrap_ad_wgpu_float};
use burn::backend::{
    Backend,
    autodiff::{
        checkpoint::{base::Checkpointer, strategy::NoCheckpointing},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::FloatTensor,
};
use burn::tensor::Tensor;

use super::kernels::HexPlaneKernelUniforms;
use super::ops::{HexPlaneFusedGrads, HexPlaneFusedOps};
use crate::hex_plane::HexPlaneConfig;

/// State saved during the forward pass for the backward computation.
#[derive(Debug, Clone)]
struct HexPlaneQueryBackwardState<B: Backend> {
    /// Contiguous canonical positions `[N, 3]` (indices/weights recomputed
    /// deterministically from `xyz` + `phase` — no per-corner state needed).
    xyz: FloatTensor<B>,
    /// Phase tensor `[N, 1]` (constant across splats; no gradient).
    phase: FloatTensor<B>,
    /// Plane grids (needed for the xyz VJP through the interpolation).
    planes: [FloatTensor<B>; 6],
    u: HexPlaneKernelUniforms,
}

#[derive(Debug)]
struct HexPlaneQueryBackwards;

const NUM_PARENTS: usize = 7; // xyz + 6 planes

impl<B: Backend + HexPlaneFusedOps> Backward<B, NUM_PARENTS> for HexPlaneQueryBackwards {
    type State = HexPlaneQueryBackwardState<B>;

    fn backward(
        self,
        ops: Ops<Self::State, NUM_PARENTS>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;
        let v_feat = grads.consume::<B>(&ops.node);

        let HexPlaneFusedGrads { v_xyz, v_planes } = B::hex_plane_query_bwd(
            state.xyz,
            state.phase,
            state.planes,
            v_feat,
            state.u,
        );

        let [xyz_parent, p_xy, p_xz, p_yz, p_xt, p_yt, p_zt] = ops.parents;
        if let Some(node) = xyz_parent {
            grads.register::<B>(node.id, v_xyz);
        }
        for (parent, grad) in [p_xy, p_xz, p_yz, p_xt, p_yt, p_zt]
            .into_iter()
            .zip(v_planes)
        {
            if let Some(node) = parent {
                grads.register::<B>(node.id, grad);
            }
        }
    }
}

/// Differentiable fused HexPlane query: `xyz` (`[N, 3]`, world mm) + `phase`
/// (`[N, 1]`) → `[N, C]` plane-summed features. All inputs must live on an
/// autodiff-enabled wgpu device. Gradient flows to `xyz` and the six plane
/// tensors.
pub fn hex_plane_query_ad(
    xyz: Tensor<2>,
    phase: Tensor<2>,
    planes: [Tensor<3>; 6],
    cfg: &HexPlaneConfig,
) -> Tensor<2> {
    let n = xyz.dims()[0];
    let u = HexPlaneKernelUniforms::from_config(cfg, n as u32);

    let xyz_ad = unwrap_ad_wgpu_float(xyz);
    let phase_ad = unwrap_ad_wgpu_float(phase);
    let planes_ad = planes.map(unwrap_ad_wgpu_float);

    let prep = HexPlaneQueryBackwards
        .prepare::<NoCheckpointing>([
            xyz_ad.node.clone(),
            planes_ad[0].node.clone(),
            planes_ad[1].node.clone(),
            planes_ad[2].node.clone(),
            planes_ad[3].node.clone(),
            planes_ad[4].node.clone(),
            planes_ad[5].node.clone(),
        ])
        .compute_bound()
        .stateful();

    // The xyz slice of `transforms` may be a non-contiguous view; the base
    // impls make it contiguous before launching.
    let xyz_inner = xyz_ad.primitive;
    let phase_inner = phase_ad.primitive;
    let planes_inner = planes_ad.map(|p| p.primitive);

    let output = <MainBackend as HexPlaneFusedOps>::hex_plane_query_fwd(
        xyz_inner.clone(),
        phase_inner.clone(),
        planes_inner.clone(),
        u,
    );

    let out_ad = match prep {
        OpsKind::Tracked(prep) => {
            let state = HexPlaneQueryBackwardState {
                xyz: xyz_inner,
                phase: phase_inner,
                planes: planes_inner,
                u,
            };
            prep.finish(state, output)
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    };

    wrap_ad_wgpu_float(out_ad)
}
