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
//! The model is a `burn::nn` module, so all parameters are optimizable via
//! autodiff.

pub mod deform_model;
pub mod hash_grid;
pub mod mlp;
pub mod positional;

pub use deform_model::{DeformModel, DeformModelConfig, Deforms, deform_splats};
