//! X-ray training stream.
//!
//! Driven from the CLI with `--xray --source <path.dcm>`. Loads the DICOM
//! dataset (frames + phase + C-arm cameras via [`brush_dataset`]), builds an
//! [`XRayTrainer`] over random canonical splats, and iterates the render loop,
//! emitting the same [`TrainMessage`]s the CLI UI already understands.
//!
//! Two modes:
//! - **deform** (default): phase-conditioned [`DeformModel`] on top of the
//!   canonical splats (cardiac-phase-guided reconstruction);
//! - **static** (`--xray-static`): no deform field, no phase — the canonical
//!   splats are optimized directly against the multi-angle projections.

use crate::{
    Emitter,
    config::{TrainStreamConfig, XRayEvalFormat},
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
    wait_for_device,
};
use anyhow::Context;
use brush_dataset::{load_dataset, scene::SceneView, scene_loader::SceneLoader};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_rerun::visualize_tools::VisualizeTools;
use brush_train::xray_eval::{save_gray_nrrd_f32, save_gray_png16};
use brush_train::xray_train::{XRayTrainConfig, create_xray_trainer};
use brush_vfs::BrushVfs;
use brush_xray::XRaySplats;
use burn::tensor::TensorData;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{AutoCompiler, WgpuRuntime};
use std::sync::Arc;
use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

/// Convert canonical [`brush_xray::XRaySplats`] into a viewer-able
/// [`Splats`] (SH degree 0 → grayscale; the X-ray renderer is SH-free).
fn xray_to_splats(canonical: &XRaySplats, device: &burn::tensor::Device) -> Splats {
    let n = canonical.num_splats() as usize;
    let means = canonical.means();
    let rots = canonical.rotations();
    let log_scales = canonical.log_scales();
    let opac = canonical.raw_opacities.val();
    let sh = burn::tensor::Tensor::<3>::zeros([n, 1, 3], device);
    Splats::from_tensor_data(means, rots, log_scales, sh, opac, SplatRenderMode::Default)
}

/// Sample up to `count` views, spread evenly across the sequence so each eval
/// round covers a range of C-arm angles / cardiac phases. Returns all views
/// when `count` exceeds the sequence length.
fn sample_eval_views(views: &[SceneView], count: usize) -> Vec<&SceneView> {
    let n = views.len();
    if count >= n {
        views.iter().collect()
    } else {
        (0..count).map(|i| &views[i * n / count]).collect()
    }
}

#[allow(clippy::large_stack_frames)]
pub(crate) async fn xray_stream(
    vfs: Arc<BrushVfs>,
    config: TrainStreamConfig,
    emitter: &Emitter,
    slot: SlotSender<Splats>,
) -> anyhow::Result<()> {
    let process_config = &config.process_config;
    let static_mode = process_config.xray_static;
    log::info!(
        "X-ray {} training (seed {})",
        if static_mode { "static" } else { "deform-GS" },
        process_config.seed
    );

    let wgpu_device = wait_for_device().await;
    let device: burn::tensor::Device = wgpu_device.clone().into();
    device.seed(process_config.seed);

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
            config: Box::new(config.clone()),
        }))
        .await;

    log::info!("Loading X-ray dataset");
    let load_result = load_dataset(vfs.clone(), &config.load_config)
        .instrument(trace_span!("Load dataset"))
        .await?;
    for warning in load_result.warnings {
        emitter
            .emit(ProcessMessage::Warning {
                error: anyhow::anyhow!("{warning}"),
            })
            .await;
    }
    let dataset = load_result.dataset;
    let n_views = dataset.train.views.len();
    log::info!("Loaded X-ray dataset with {n_views} views");

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
            dataset: dataset.clone(),
        }))
        .await;

    // Sanity: the X-ray path needs a grayscale GT on each train view.
    if dataset
        .train
        .views
        .iter()
        .any(|v| v.gray_image.is_none())
    {
        anyhow::bail!(
            "X-ray training requires DICOM views (--xray), got an RGB dataset instead. \
             Make sure --source points at a .dcm file."
        );
    }

    let total_iters = config.train_config.total_iters();
    let xray_cfg = XRayTrainConfig {
        total_iters,
        // Static mode: no deform field, no phase conditioning, no AST noise.
        enable_deform: !static_mode,
        enable_ast: !static_mode && process_config.xray_enable_ast,
        warm_up: if static_mode { 0 } else { process_config.xray_warm_up },
        refine: brush_train::xray_refine::XRayRefineConfig {
            refine_every: process_config.xray_refine_every,
            cull_density_threshold: process_config.xray_cull_density as f32,
            density_reset_interval: process_config.xray_density_reset_interval,
            ..brush_train::xray_refine::XRayRefineConfig::default()
        },
        lr_mean: process_config.xray_lr_mean,
        lr_mean_end: process_config.xray_lr_mean_end,
        lr_scale: process_config.xray_lr_scale,
        lr_rotation: process_config.xray_lr_rotation,
        lr_opac: process_config.xray_lr_opac,
        lr_deform: process_config.xray_lr_deform,
        ..XRayTrainConfig::default()
    };
    let mut trainer = create_xray_trainer(
        xray_cfg,
        process_config.xray_num_points,
        process_config.xray_scene_extent,
        brush_train::xray_train::InitRegion::Ball {
            radius: process_config.xray_scene_extent,
        },
        &device,
        None,
        None,
    );
    if static_mode {
        log::info!(
            "Static reconstruction: {} canonical splats in a ball of radius {} mm",
            process_config.xray_num_points,
            process_config.xray_scene_extent,
        );
    }

    // Visualization: spawn a local Rerun Viewer, or (headless, recommended for
    // remote SSH sessions) write a .rrd file via `--rerun-rrd out.rrd`.
    let visualize = VisualizeTools::new(
        config.rerun_config.rerun_enabled,
        config.rerun_config.rerun_rrd.clone(),
    )
    .await;
    if visualize.is_enabled() {
        if let Err(error) = visualize.log_scene(
            &dataset.train,
            config.rerun_config.rerun_max_img_size,
        ) {
            emitter.emit(ProcessMessage::Warning { error }).await;
        }
        if let Err(error) = visualize.send_xray_blueprint() {
            emitter.emit(ProcessMessage::Warning { error }).await;
        }
        log::info!(
            "Rerun visualization on (rrd: {:?})",
            config.rerun_config.rerun_rrd
        );
    }
    // Read back the predicted image only when a visualization sink is active.
    trainer.set_collect_pred(visualize.is_enabled());

    // Publish the initial canonical splats to the viewer slot.
    let splats = xray_to_splats(trainer.canonical(), &device);
    slot.set(0, splats.clone());
    emitter
        .emit(ProcessMessage::SplatsUpdated {
            up_axis: Some(glam::Vec3::Y),
            frame: 0,
            total_frames: 1,
            num_splats: splats.num_splats(),
            sh_degree: 0,
        })
        .await;
    emitter.emit(ProcessMessage::DoneLoading).await;

    let mut dataloader = SceneLoader::new(&dataset.train, 42, &config.load_config);
    let client = WgpuRuntime::<AutoCompiler>::client(wgpu_device);
    client.memory_cleanup();

    // Sample eval views for PSNR/SSIM. When `--eval-split-every` is set the
    // DICOM loader holds out every `eval_split_every`-th frame in
    // `dataset.eval` — those are *never* seen during training, so eval on them
    // is honest generalization. Otherwise fall back to evenly sampling the
    // train views (that scores training views, so the reported PSNR/SSIM is
    // optimistic — log a warning so the numbers aren't misread).
    let eval_count = process_config.xray_eval_views as usize;
    let eval_views: Vec<&SceneView> = if let Some(eval_scene) = &dataset.eval
        && !eval_scene.views.is_empty()
    {
        // Held-out views (never seen during training) → honest generalization.
        sample_eval_views(&eval_scene.views, eval_count)
    } else {
        log::warn!(
            "No held-out eval split (pass --eval-split-every=N): evaluating on \
             training views, PSNR/SSIM will be optimistic"
        );
        sample_eval_views(&dataset.train.views, eval_count)
    };

    // Startup diagnostic: render the first eval view BEFORE training and log
    // the predicted intensity / path-integral (proj = -ln intensity) spread.
    // `proj ≈ 0` → all-white image (no visible Beer-Lambert contrast and ~no
    // learning signal); useful for sanity-checking the density init.
    {
        let view = &eval_views[0];
        let gray = view.gray_image.as_ref().expect("gray GT");
        let gt_data = TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width]);
        let sample = trainer.eval_view(&view.camera, &gt_data, view.phase, view.time).await;
        let pred = sample.pred.as_slice::<f32>().expect("f32");
        let mut vals: Vec<f32> = pred.iter().copied().filter(|v| v.is_finite()).collect();
        vals.sort_by(|a, b| a.total_cmp(b));
        let pct = |p: f32| {
            vals[((p * (vals.len().max(1) - 1) as f32).round() as usize).min(vals.len() - 1)]
        };
        log::info!(
            "X-ray init diag (view 0): intensity min={:.4} p50={:.4} mean={:.4} p95={:.4} max={:.4}; \
             proj=-ln: p5={:.3} p50={:.3} p95={:.3}",
            vals.first().copied().unwrap_or(0.0),
            pct(0.5),
            pred.iter().sum::<f32>() / pred.len().max(1) as f32,
            pct(0.95),
            vals.last().copied().unwrap_or(0.0),
            (-pct(0.05).ln()).max(0.0),
            (-pct(0.50).ln()).max(0.0),
            (-pct(0.95).ln()).max(0.0),
        );
    }

    // Resolve the export directory (shared by eval images and the final PLY).
    let dataset_name = vfs
        .base_path()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dataset".to_owned());
    let export_path_str = process_config
        .export_path
        .replace("{dataset}", &dataset_name);
    let base_path = vfs
        .base_path()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let export_path: std::path::PathBuf =
        base_path.join(&export_path_str).components().collect();

    let train_start = Instant::now();
    let mut last_emit = Instant::now();
    for iter in process_config.start_iter..total_iters {
        let step = iter + 1;

        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("X-ray next data batch"))
            .await;

        let stats = trainer.step(&batch).await;

        // Density control (skip the very first step so gradients accumulate).
        if step > 1
            && let Some(refine_stats) = trainer.maybe_refine(step).await
        {
            log::info!(
                "Refine iter {step}: {} splats (added {}, pruned {})",
                refine_stats.total_splats,
                refine_stats.num_added,
                refine_stats.num_pruned,
            );
            let splats = xray_to_splats(trainer.canonical(), &device);
            slot.set(0, splats.clone());
            if visualize.is_enabled()
                && let Err(error) = visualize.log_splats(step, splats).await
            {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                    cur_splat_count: refine_stats.total_splats,
                    iter: step,
                }))
                .await;
        }

        // Periodic visualization: scalars, pred/GT image pair, canonical
        // splat point cloud (when enabled).
        let log_every = config.rerun_config.rerun_log_train_stats_every.max(1);
        if visualize.is_enabled() && (step.is_multiple_of(log_every) || step == total_iters) {
            if let Err(error) = visualize.log_xray_train_stats(
                step,
                stats.loss,
                stats.num_splats,
                stats.num_visible,
                stats.lr_mean,
            ) {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
            if let (Some(pred), Some(gt)) = (&stats.pred_img, &batch.img_gray)
                && let Err(error) = visualize
                    .log_xray_render(
                        step,
                        0,
                        batch.phase,
                        pred,
                        gt,
                        config.rerun_config.rerun_max_img_size,
                    )
                    .await
            {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
            if let Some(interval) = config.rerun_config.rerun_log_splats_every
                && step.is_multiple_of(interval)
            {
                let splats = xray_to_splats(trainer.canonical(), &device);
                if let Err(error) = visualize.log_splats(step, splats).await {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }
        }

        // Throttled progress + loss reporting.
        if last_emit.elapsed() >= Duration::from_millis(250) || step == total_iters {
            log::info!(
                "Iter {step}/{total_iters}: loss {:.4} visible {} splats {} lr_mean {:.2e}",
                stats.loss,
                stats.num_visible,
                stats.num_splats,
                stats.lr_mean,
            );
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter: step,
                    total_elapsed: train_start.elapsed(),
                    lod_progress: None,
                }))
                .await;
            last_emit = Instant::now();
        }

        // Periodic evaluation: grayscale PSNR/SSIM over the sampled views,
        // logged to rerun and (optionally) saved losslessly to disk.
        let eval_every = process_config.eval_every.max(1);
        if step.is_multiple_of(eval_every) || step == total_iters {
            let mut avg_psnr = 0.0_f32;
            let mut avg_ssim = 0.0_f32;
            let count = eval_views.len() as f32;
            for (i, view) in eval_views.iter().enumerate() {
                brush_async::yield_now().await;
                let gray = view.gray_image.as_ref().expect("xray eval needs gray GT");
                let gt_data =
                    TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width]);
                let sample = trainer.eval_view(&view.camera, &gt_data, view.phase, view.time).await;
                avg_psnr += sample.psnr;
                avg_ssim += sample.ssim;

                if visualize.is_enabled()
                    && let Err(error) = visualize
                        .log_xray_render(
                            step,
                            i as u32,
                            view.phase,
                            &sample.pred,
                            &gt_data,
                            config.rerun_config.rerun_max_img_size,
                        )
                        .await
                {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }

                // Save the pred / GT intensity images losslessly.
                if process_config.eval_save_to_disk {
                    let img_name = format!("view_{i:03}_phase_{:.3}", view.phase);
                    let dir = export_path.join(format!("eval_{step}"));
                    let res = (|| -> anyhow::Result<()> {
                        match process_config.xray_eval_format {
                            XRayEvalFormat::Png16 => {
                                save_gray_png16(&dir.join(format!("{img_name}_pred.png")), &sample.pred)?;
                                save_gray_png16(&dir.join(format!("{img_name}_gt.png")), &sample.gt)?;
                            }
                            XRayEvalFormat::Nrrd => {
                                save_gray_nrrd_f32(&dir.join(format!("{img_name}_pred.nrrd")), &sample.pred)?;
                                save_gray_nrrd_f32(&dir.join(format!("{img_name}_gt.nrrd")), &sample.gt)?;
                            }
                        }
                        Ok(())
                    })();
                    if let Err(error) = res {
                        emitter.emit(ProcessMessage::Warning { error }).await;
                    }
                }
            }
            avg_psnr /= count;
            avg_ssim /= count;
            if visualize.is_enabled()
                && let Err(error) = visualize.log_eval_stats(step, avg_psnr, avg_ssim)
            {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
            log::info!("Eval iter {step}: PSNR {avg_psnr:.2} dB, SSIM {avg_ssim:.4}");
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                    iter: step,
                    avg_psnr,
                    avg_ssim,
                }))
                .await;
        }

        // Free GPU memory periodically (splat counts grow with densification).
        if step.is_multiple_of(100) {
            client.memory_cleanup();
        }
    }

    log::info!("X-ray training done ({total_iters} steps)");
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::DoneTraining))
        .await;

    // Export the canonical splats as a PLY checkpoint.
    let export_name = config
        .process_config
        .export_name
        .replace("{iter}", &format!("{total_iters}"));
    let splats = xray_to_splats(trainer.canonical(), &device);
    std::fs::create_dir_all(&export_path)
        .with_context(|| format!("Creating export directory {}", export_path.display()))?;
    let ply = brush_serde::splat_to_ply(splats, Some(glam::Vec3::Y))
        .await
        .context("Serializing canonical splats to PLY")?;
    std::fs::write(export_path.join(&export_name), ply)
        .with_context(|| format!("Failed to export ply {}", export_path.display()))?;
    log::info!("Exported canonical splats to {}", export_path.join(&export_name).display());

    // Ensure the rerun `.rrd` sink is flushed before the process may be torn
    // down with `std::process::exit` (which skips destructors — a truncated
    // recording would lose the final splats / eval logs).
    if visualize.is_enabled()
        && let Err(error) = visualize.flush()
    {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    Ok(())
}
