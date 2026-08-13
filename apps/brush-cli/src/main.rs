#![recursion_limit = "256"]

// Headless trainer binary. The viewer lives in brush-app (the `brush` binary);
// this is a lean build of just the training path for quick CLI iteration.
#[cfg(not(target_family = "wasm"))]
fn main() -> anyhow::Result<()> {
    use brush_cli::{Cli, build_process, run_headless};
    use clap::Parser;
    use std::io::Write;

    let args = Cli::parse().validate()?;

    if args.with_viewer {
        anyhow::bail!(
            "brush-cli is headless and can't open a viewer. Pass a source to train, \
             or build the `brush` binary (brush-app) for the viewer."
        );
    }

    // `validate` guarantees a source is present when the viewer is off.
    let process = build_process(&args).expect("source must be present");

    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime")
        .block_on(run_headless(process, args.train_stream));

    // Workaround for the NVIDIA 550.76 driver bug: the driver's `[vkps]`
    // thread (libnvidia-eglcore.so) can segfault while releasing GPU
    // resources during normal process teardown, after training has already
    // finished and all artifacts are written. Skip the normal destructors on
    // success — the OS reclaims GPU memory on process exit, and this avoids
    // turning a successful training run into a nonzero exit code.
    if result.is_ok() {
        std::io::stdout().flush().ok();
        std::io::stderr().flush().ok();
        std::process::exit(0);
    }
    result
}

#[cfg(target_family = "wasm")]
fn main() {}
