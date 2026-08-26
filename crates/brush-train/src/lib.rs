#![recursion_limit = "256"]

pub mod config;
pub mod eval;
pub mod fdk_prior;
pub mod lod;
pub mod msg;
pub mod train;
pub mod xray_eval;
pub mod xray_refine;
pub mod xray_train;

mod adam_scaled;
mod multinomial;
mod quat_vec;
mod stats;

mod splat_init;

pub use splat_init::{RandomSplatsConfig, create_random_splats, to_init_splats};
