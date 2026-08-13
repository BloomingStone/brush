use clap::Args;
use serde::{Deserialize, Serialize};

#[cfg(not(target_family = "wasm"))]
pub mod burn_to_rerun;

// visualize_tools has a noop implementation for WASM.
pub mod visualize_tools;

#[derive(Clone, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct RerunConfig {
    /// Whether to enable rerun.io logging for this run.
    #[arg(long, help_heading = "Rerun options", default_value = "false")]
    pub rerun_enabled: bool,
    /// Headless rerun output: write the recording to this .rrd file instead
    /// of spawning a local viewer. Works without any GUI / X11 (ideal for
    /// remote SSH sessions) — open the file later with a local Rerun Viewer
    /// (`rerun out.rrd`). Implies rerun logging is on.
    #[arg(long, help_heading = "Rerun options")]
    pub rerun_rrd: Option<String>,
    /// How often to log basic training statistics.
    #[arg(
        long,
        help_heading = "Rerun options",
        default_value = "50",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub rerun_log_train_stats_every: u32,
    /// How often to log out the full splat point cloud to rerun (warning: heavy).
    #[arg(
        long,
        help_heading = "Rerun options",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub rerun_log_splats_every: Option<u32>,
    /// How often to log the splat scale/opacity/anisotropy distribution stats.
    #[arg(
        long,
        help_heading = "Rerun options",
        default_value = "1000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub rerun_log_distribution_every: u32,
    /// The maximum size of images from the dataset logged to rerun.
    #[arg(long, help_heading = "Rerun options", default_value = "512")]
    pub rerun_max_img_size: u32,
}
