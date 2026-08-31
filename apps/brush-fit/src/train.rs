//! 训练入口: `run_static` / `run_deform` — 移植 fit_static.rs / fit_deform.rs
//! 的完整流程 (数据加载 → trainer 构建 → 训练循环 → eval/CSV → 导出)。
//! 差异仅: deform 配置块 / 导出 (FDK 全部剔除)。

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use brush_train::xray_refine::XRayRefineGradThreshold;
use brush_train::xray_train::{
    XRayTrainConfig, create_static_xray_trainer, create_xray_trainer,
};

use crate::config::{FitConfig, FitMode, Resolved};
use crate::data;
use crate::export;

pub type GradThr = XRayRefineGradThreshold;

/// 一次训练的全部输出产物。
#[derive(Debug, Clone)]
pub struct FitOutcome {
    pub out: PathBuf,
    /// 必须: phase=0 volume (nii.gz)。
    pub volume_phase0: PathBuf,
    /// 可选 (save_ply)。
    pub ply: Option<PathBuf>,
    /// 可选 (deform 模式 + save_deform): 网络权重。
    pub deform_ckpt: Option<PathBuf>,
    /// 可选 (deform 模式 + save_deform): 每相位网格场 nii.gz。
    pub deform_fields: Vec<PathBuf>,
    /// 可选 (save_bin): transforms/raw 前缀。
    pub bin_prefix: Option<PathBuf>,
    /// 最后 eval 指标 (psnr, ssim, lpips)。
    pub final_eval: (f32, f32, f32),
}

pub async fn run_static(cfg: FitConfig) -> anyhow::Result<FitOutcome> {
    run_mode(cfg, FitMode::Static, None).await
}

pub async fn run_deform(cfg: FitConfig) -> anyhow::Result<FitOutcome> {
    run_mode(cfg, FitMode::Deform, None).await
}

/// 带进度回调的运行入口 (FFI 使用)。
pub async fn run_static_with_progress(
    cfg: FitConfig,
    progress: ProgressFn,
) -> anyhow::Result<FitOutcome> {
    run_mode(cfg, FitMode::Static, Some(progress)).await
}

pub async fn run_deform_with_progress(
    cfg: FitConfig,
    progress: ProgressFn,
) -> anyhow::Result<FitOutcome> {
    run_mode(cfg, FitMode::Deform, Some(progress)).await
}

async fn run_mode(
    cfg: FitConfig,
    mode: FitMode,
    progress: Option<ProgressFn>,
) -> anyhow::Result<FitOutcome> {
    let mut cfg = cfg;
    cfg.mode = mode;
    run(cfg, progress).await
}

/// `[HH:MM:SS]` 北京时间 (UTC+8, 固定偏移; 轻量, 无 chrono 依赖)。
pub fn ts() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = (d.as_secs() + 8 * 3600) % 86_400;
    format!(
        "[{:02}:{:02}:{:02}]",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

#[allow(clippy::too_many_lines)]
async fn run(cfg: FitConfig, mut progress: Option<ProgressFn>) -> anyhow::Result<FitOutcome> {
    let res: Resolved = cfg.resolve()?;
    let deform = cfg.mode == FitMode::Deform;
    let out = cfg.out.clone();
    println!("{} brush-fit {} -> {}", ts(), if deform { "deform" } else { "static" }, out.display());

    // ---- 后端 -------------------------------------------------------------
    let wgpu = crate::burn_init_setup().await;
    let device: burn::tensor::Device = wgpu.clone().into();

    // ---- 数据 -------------------------------------------------------------
    let loaded = data::load(&cfg, &res).await?;
    let scene_extent = loaded.scene_extent;

    // ---- Trainer ----------------------------------------------------------
    let mut tcfg = XRayTrainConfig::default();
    tcfg.total_iters = cfg.iters;
    tcfg.init_density = res.init_density;
    tcfg.lr_mean = cfg.lr_mean;
    tcfg.lr_mean_end = cfg.lr_mean_end;
    tcfg.lr_scale = cfg.lr_scale;
    tcfg.lr_opac = cfg.lr_opac;
    tcfg.proj_weight = cfg.proj_weight;
    tcfg.proj_ssim_weight = cfg.proj_ssim_weight;
    tcfg.loss_type = res.loss_type;
    tcfg.loss_eps = cfg.loss_eps;
    tcfg.loss_delta = cfg.loss_delta;
    tcfg.cosine_lr = cfg.cosine_lr;
    tcfg.multiscale_weight = cfg.multiscale_weight;
    tcfg.window_weight = cfg.window_weight;
    tcfg.grad_weight = cfg.grad_weight;
    tcfg.grad_ramp_from = cfg.grad_ramp_from;
    tcfg.grad_ramp_to = cfg.grad_ramp_to;
    tcfg.grad_edge_scale = cfg.grad_edge_scale;
    tcfg.screen_area_penalty = cfg.screen_area_penalty;
    tcfg.scale_aniso_weight = cfg.scale_aniso_weight;
    tcfg.scale_cap_mm = cfg.scale_cap_mm;
    tcfg.scale_cap_weight = cfg.scale_cap_weight;
    tcfg.refine = brush_train::xray_refine::XRayRefineConfig {
        refine_every: cfg.refine_every,
        scene_extent,
        growth_select_fraction: cfg.growth_frac,
        grad_threshold: res.grad_threshold.clone(),
        enable_split: cfg.split,
        density_reset_interval: cfg.density_reset,
        percent_dense: res.percent_dense,
        split_scale_factor: res.split_scale,
        max_bound_factor: res.bound_factor,
        cull_density_threshold: res.cull_density,
        max_splats: cfg.max_splats,
        max_screen_size: res.max_screen_size,
        cull_contribution: cfg.cull_contribution,
        cull_contribution_percentile: cfg.cull_percentile,
        cull_contribution_floor: cfg.cull_floor,
        min_splats: cfg.min_splats,
        refine_until_frac: cfg.refine_until_frac,
        ..brush_train::xray_refine::XRayRefineConfig::default()
    };
    if deform {
        tcfg.enable_deform = true;
        tcfg.enable_ast = res.enable_ast;
        tcfg.warm_up = cfg.warm_up;
        tcfg.deform_backend = res.deform_backend;
        tcfg.predict_scaling = res.predict_scaling;
        // time 条件化已移除 (CLI/config 无 time 输入; 后端 time 输入固定 0,
        // enable_time 保持默认 false → 形变场仅由 phase 驱动)。
        tcfg.hex_plane = brush_deform::HexPlaneDeformConfig {
            hex_plane: brush_deform::HexPlaneConfig {
                n_feature_dim: cfg.hex_features,
                spatial_resolution: cfg.hex_res,
                time_resolution: cfg.hex_time_res,
                ..brush_deform::HexPlaneConfig::default()
            },
            mlp_hidden: cfg.hex_mlp_width,
            mlp_layers: cfg.hex_mlp_layers,
            predict_scaling: res.predict_scaling,
            plane_tv_weight: cfg.plane_tv_weight,
            rigid_anchor_weight: cfg.rigid_anchor_weight,
            ..brush_deform::HexPlaneDeformConfig::default()
        };
        tcfg.lr_deform = cfg.lr_deform;
        tcfg.lr_deform_end = cfg.lr_deform_end;
    }

    let init = data::init_region(&loaded, &cfg, &res);
    let train_cams: Vec<_> = loaded.dataset.train.views.iter().map(|v| v.camera).collect();
    let img = loaded.img_size;
    let fov = if cfg.fov_filter {
        Some((train_cams.as_slice(), img))
    } else {
        None
    };
    let mut trainer = if deform {
        create_xray_trainer(tcfg, cfg.points, scene_extent, init, &device, fov, None)
    } else {
        create_static_xray_trainer(tcfg, cfg.points, scene_extent, init, &device, fov, None)
    };
    println!(
        "{} init splats: {} (init region {:?}, r={}mm, init μ={} mm⁻¹, lr_mean={}->{})",
        ts(),
        trainer.num_splats(),
        init,
        scene_extent,
        res.init_density,
        cfg.lr_mean,
        cfg.lr_mean_end,
    );

    let mut dataloader = data::make_loader(&loaded);

    // ---- Eval 视图 (--no-eval 时完全跳过) ---------------------------------
    let eval_on = cfg.eval_enabled;
    let eval_views = if eval_on {
        data::eval_views(&loaded, cfg.eval_views)
    } else {
        Vec::new()
    };
    let eval_every = res.eval_every;
    if eval_on {
        println!("{} eval on {} views every {} steps", ts(), eval_views.len(), eval_every);
    }

    std::fs::create_dir_all(&out)?;
    let eval_nrrd = out.join("eval/nrrd");
    if eval_on {
        std::fs::create_dir_all(&eval_nrrd)?;
    }

    // 指标 CSV (no-eval 时不写)。
    let t0 = Instant::now();
    let mut csv_writer: Option<std::fs::File> = {
        let off = cfg.log_csv.as_deref() == Some(Path::new("off")) || !eval_on;
        if off {
            None
        } else {
            let path = cfg
                .log_csv
                .clone()
                .unwrap_or_else(|| out.join("metrics.csv"));
            let mut f = std::fs::File::create(&path)?;
            writeln!(
                f,
                "iter,time_bj,elapsed_s,loss,psnr,ssim,lpips,visible,splats,lr_mean,grad_mean,grad_rot,grad_scale,grad_density"
            )?;
            f.flush()?;
            println!("{} logging metrics -> {}", ts(), path.display());
            Some(f)
        }
    };

    // ---- 初始 eval (no-eval 跳过) -------------------------------------------
    let mut final_eval = (0.0f32, 0.0f32, 0.0f32);
    if eval_on {
        let (mut p, mut s, mut l) = (0.0f32, 0.0f32, 0.0f32);
        let mut pairs = Vec::with_capacity(eval_views.len());
        for view in eval_views.iter() {
            let gt = data::gt_tensor(view);
            let sample = trainer.eval_view(&view.camera, &gt, view.phase, 0.0).await;
            p += sample.psnr;
            s += sample.ssim;
            l += sample.lpips;
            pairs.push(merge_pair(&sample.pred, &sample.gt));
        }
        if cfg.save_eval {
            save_stack(&eval_nrrd, 0, &pairs);
        }
        p /= eval_views.len().max(1) as f32;
        s /= eval_views.len().max(1) as f32;
        l /= eval_views.len().max(1) as f32;
        log_metrics_row(&mut csv_writer, 0, &t0, f32::NAN, p, s, l, 0, trainer.num_splats(), 0.0, None)?;
        println!("{} iter {:4} psnr={:6.2} ssim={:5.3} lpips={:.4}", ts(), "init", p, s, l);
    }

    // ---- 训练循环 -----------------------------------------------------------
    for iter in 0..cfg.iters {
        let step = iter + 1;
        let is_eval_step = eval_on && (step % eval_every == 0 || step == cfg.iters);
        trainer.set_collect_grads(is_eval_step);
        trainer.set_collect_loss(is_eval_step);
        let batch = dataloader.next_batch().await;
        let stats = trainer.step(&batch).await;

        if let Some(cb) = progress.as_mut() {
            cb(FitProgress::Step {
                iter: step,
                total: cfg.iters,
                loss: stats.loss,
            });
        }

        if step > 1
            && let Some(refine_stats) = trainer.maybe_refine(step).await
        {
            let (beyond, total, md) = trainer.splats_beyond_radius(loaded.r0).await;
            println!(
                "{} refine iter {step}: {} splats (added {}, split {}, pruned {}) grad_thr={} | >R0={beyond}/{total} ({:.1}%), meanμ={md:.5}",
                ts(),
                refine_stats.total_splats,
                refine_stats.num_added,
                refine_stats.num_split,
                refine_stats.num_pruned,
                refine_stats.grad_threshold.map_or(-1.0f32, |t| t),
                beyond as f32 / total.max(1) as f32 * 100.0,
            );
        }

        if is_eval_step {
            let (mut p, mut s, mut l) = (0.0f32, 0.0f32, 0.0f32);
            let mut pairs = Vec::with_capacity(eval_views.len());
            for view in eval_views.iter() {
                let vgt = data::gt_tensor(view);
                let sample = trainer.eval_view(&view.camera, &vgt, view.phase, 0.0).await;
                p += sample.psnr;
                s += sample.ssim;
                l += sample.lpips;
                pairs.push(merge_pair(&sample.pred, &sample.gt));
            }
            if cfg.save_eval {
                save_stack(&eval_nrrd, step, &pairs);
            }
            p /= eval_views.len().max(1) as f32;
            s /= eval_views.len().max(1) as f32;
            l /= eval_views.len().max(1) as f32;
            final_eval = (p, s, l);
            log_metrics_row(
                &mut csv_writer,
                step,
                &t0,
                stats.loss,
                p,
                s,
                l,
                stats.num_visible,
                stats.num_splats,
                stats.lr_mean,
                stats.grad_norms.as_ref(),
            )?;
            if let Some(g) = &stats.grad_norms {
                println!(
                    "{} iter {:4} loss={:8.4} psnr={:6.2} ssim={:5.3} lpips={:.4} visible={} splats={} (eval {} views) | grads mean={:.1e} rot={:.1e} scale={:.1e} density={:.1e} | pos step≈{:.2e}mm",
                    ts(), step, stats.loss, p, s, l, stats.num_visible, stats.num_splats,
                    eval_views.len(), g.mean_grad, g.rot_grad, g.scale_grad, g.density_grad,
                    stats.lr_mean as f32 * g.mean_grad,
                );
            } else {
                println!(
                    "{} iter {:4} loss={:8.4} psnr={:6.2} ssim={:5.3} lpips={:.4} visible={} splats={} (eval {} views)",
                    ts(), step, stats.loss, p, s, l, stats.num_visible, stats.num_splats, eval_views.len(),
                );
            }
        }
    }

    // ---- 导出 ---------------------------------------------------------------
    // 训练后内存池不稳: deform 网格场交给独立进程 brush-fit-dump (干净设备),
    // 先小后大 (field 子进程 → volume voxelize)。不做 memory_cleanup: 会触发
    // cubecl 内存池断言 (训练后池状态不稳定, 见 fit_deform 注释)。
    let (mut deform_ckpt, mut deform_fields) = (None, Vec::new());
    if deform && cfg.save_deform {
        let (ck, fields) = export::export_deform(&cfg, &trainer, scene_extent, &out).await?;
        deform_ckpt = Some(ck);
        deform_fields = fields;
    }

    let volume_phase0 =
        export::export_volume_phase0(&cfg, &trainer, &device, loaded.half_w, loaded.half_h, &out).await?;

    let mut ply = None;
    if cfg.save_ply {
        ply = Some(export::export_ply(&trainer, &device, &out).await?);
    }

    let mut bin_prefix = None;
    if cfg.save_bin {
        bin_prefix = Some(export::export_bin(&trainer, &out).await?);
    }

    if let Some(cb) = progress.as_mut() {
        cb(FitProgress::Done);
    }
    println!("{} done -> {}", ts(), out.display());
    Ok(FitOutcome {
        out,
        volume_phase0,
        ply,
        deform_ckpt,
        deform_fields,
        bin_prefix,
        final_eval,
    })
}

/// 合并 GT | pred 为 `[H, 2W]` (左 GT, 右 pred)。
fn merge_pair(pred: &burn::tensor::TensorData, gt: &burn::tensor::TensorData) -> burn::tensor::TensorData {
    let pred_v: Vec<f32> = pred.as_slice::<f32>().expect("f32 pred").to_vec();
    let gt_v: Vec<f32> = gt.as_slice::<f32>().expect("f32 gt").to_vec();
    let h = pred.shape[0];
    let w = pred.shape[1];
    assert_eq!(gt.shape, pred.shape, "GT/pred must be same size");
    let mut merged = Vec::with_capacity(h * w * 2);
    for y in 0..h {
        merged.extend_from_slice(&gt_v[y * w..(y + 1) * w]);
        merged.extend_from_slice(&pred_v[y * w..(y + 1) * w]);
    }
    burn::tensor::TensorData::new(merged, [h, w * 2])
}

/// 把所有视图的 GT|pred 拼接图 stack 成单个 3D NRRD (`[N, H, 2W]`)。
fn save_stack(dir: &Path, iter: u32, pairs: &[burn::tensor::TensorData]) {
    if pairs.is_empty() {
        return;
    }
    let [h, w2] = [pairs[0].shape[0], pairs[0].shape[1]];
    let mut vol = Vec::with_capacity(pairs.len() * h * w2);
    for p in pairs {
        assert_eq!([p.shape[0], p.shape[1]], [h, w2], "view size mismatch");
        vol.extend_from_slice(p.as_slice::<f32>().expect("f32"));
    }
    let td = burn::tensor::TensorData::new(vol, [pairs.len(), h, w2]);
    let p = dir.join(format!("gt_pred_{iter:05}.nrrd"));
    brush_train::xray_eval::save_gray_nrrd_f32_stack(&p, &td).expect("save GT|pred NRRD stack");
    println!("{} saved {} ({} views stacked)", ts(), p.display(), pairs.len());
}

/// 追加一行指标到 CSV(若启用)。
#[allow(clippy::too_many_arguments)]
fn log_metrics_row(
    w: &mut Option<std::fs::File>,
    iter: u32,
    t0: &Instant,
    loss: f32,
    psnr: f32,
    ssim: f32,
    lpips: f32,
    visible: u32,
    splats: u32,
    lr_mean: f64,
    grad: Option<&brush_train::xray_train::XRayGradStats>,
) -> anyhow::Result<()> {
    if let Some(f) = w {
        let grad_csv = grad.map_or_else(String::new, |g| {
            format!(
                ",{:.3e},{:.3e},{:.3e},{:.3e}",
                g.mean_grad, g.rot_grad, g.scale_grad, g.density_grad
            )
        });
        writeln!(
            f,
            "{iter},{},{:.1},{loss:.6},{psnr:.4},{ssim:.4},{lpips:.4},{visible},{splats},{lr_mean:.3e}{grad_csv}",
            ts(),
            t0.elapsed().as_secs_f32()
        )?;
        f.flush()?;
    }
    Ok(())
}

/// 供外部 (FFI) 使用的进度回调类型: 每步 iter, 结束信号。
pub type ProgressFn = Box<dyn FnMut(FitProgress)>;

#[derive(Debug, Clone, Copy)]
pub enum FitProgress {
    Step { iter: u32, total: u32, loss: f32 },
    Done,
}
