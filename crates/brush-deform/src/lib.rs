//! Cardiac-phase-guided deformation network for dynamic X-ray Gaussian splats.
//!
//! Port of the Python `HashGridDefromModel` (GS-dev-contrast-flow
//! `internal/deform_models/hashgrid_deform.py`) to burn:
//!
//! - a multi-resolution **hash grid** encoding of the (normalized) splat
//!   position,
//! - a **sin/cos positional encoding** of the cardiac phase,
//! - a skip-connected **MLP** fusing the two,
//! - heads predicting `d_xyz`, `d_scaling` and `d_rotation` (axial angle →
//!   quaternion), plus the `deform()` application to [`XRaySplats`].
//!
//! A faster [`hex_plane`] alternative ([`hexplane_model::HexPlaneDeformModel`])
//! replaces the hash grid + skip-MLP with a HexPlane encoder (six bilinear
//! feature planes, circular time axis for the periodic cardiac phase) + a
//! small MLP — better suited to low-frequency cardiac motion and the wgpu
//! backend (far fewer gather kernels and matmuls).
//!
//! The model is a `burn::nn` module, so all parameters are optimizable via
//! autodiff.

pub mod deform_model;
#[doc(hidden)]
pub mod fused;
pub mod hash_grid;
pub mod hex_plane;
pub mod hexplane_model;
pub mod mlp;
pub mod positional;

pub use deform_model::{DeformModel, DeformModelConfig, Deforms, deform_splats};
pub use hex_plane::{HexPlane, HexPlaneConfig};
pub use hexplane_model::{HexPlaneDeformConfig, HexPlaneDeformModel};
