//! 临时验证: 随机点云初始化 + 启用 density control(refine), 拟合
//! `images/RXA_brain.dcm`(静态旋转扫描)。
//!
//! 与 fit_centerline 的区别:
//!   1. 纯随机点云初始化(复用 `create_xray_trainer`: 球内随机点 + KNN scale +
//!      μ_water 密度), 没有中心线先验。
//!   2. **启用 refine**(densify/prune, `refine_every` 可配)——验证密度控制
//!      在静态脑部重建中的效果。
//! 每 `eval_every` 步对固定视图(frame 0)做一次 eval, 打印 PSNR/SSIM 并保存
//! GT|pred 拼接图(**float32 NRRD**, 无损, 可在 3D Slicer/ParaView/napari 中
//! 调窗宽窗位); 训练结束导出最终 canonical splats 的 PLY。
//!
//! 用法(默认参数对齐之前 rxa8 最好结果: 30000 点 / scene_extent=50mm /
//! refine_every=400 / 4000 步, PSNR ≈ 20.1dB):
//!   cargo run -p brush-process --bin fit_static -- \
//!     images/RXA_brain.dcm \
//!     --iters=4000 --points=30000 --scene-extent=50 \
//!     --refine-every=400 --eval-every=100 --out=target/fit_static

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene::SceneView;
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::xray_eval::save_gray_nrrd_f32_stack;
use brush_train::xray_refine::XRayRefineConfig;
use brush_train::xray_train::{XRayTrainConfig, create_xray_trainer};
use brush_vfs::BrushVfs;
use brush_xray::XRaySplats;
use burn::tensor::{Device, TensorData};

/// Convert canonical [`brush_xray::XRaySplats`] into a viewer-able
/// [`Splats`] (SH degree 0 → grayscale; the X-ray renderer is SH-free).
/// Same helper as `xray_stream::xray_to_splats` (private there).
fn xray_to_splats(canonical: &XRaySplats, device: &burn::tensor::Device) -> Splats {
    let n = canonical.num_splats() as usize;
    let means = canonical.means();
    let rots = canonical.rotations();
    let log_scales = canonical.log_scales();
    let opac = canonical.raw_opacities.val();
    let sh = burn::tensor::Tensor::<3>::zeros([n, 1, 3], device);
    Splats::from_tensor_data(means, rots, log_scales, sh, opac, SplatRenderMode::Default)
}

/// Sample up to `count` views, spread evenly across the sequence (for eval on
/// held-out sets, covering a range of C-arm angles).
fn sample_eval_views(views: &[SceneView], count: usize) -> Vec<&SceneView> {
    let n = views.len();
    if count >= n {
        views.iter().collect()
    } else {
        (0..count).map(|i| &views[i * n / count]).collect()
    }
}

/// 合并 GT | pred 为 `[H, 2W]` (左 GT, 右 pred, 均为 [0,1] f32)。
fn merge_pair(pred: &TensorData, gt: &TensorData) -> TensorData {
    let pred_v: Vec<f32> = pred.as_slice::<f32>().expect("f32 pred").to_vec();
    let gt_v: Vec<f32> = gt.as_slice::<f32>().expect("f32 gt").to_vec();
    let h = pred.shape[0];
    let w = pred.shape[1];
    assert_eq!(gt.shape, pred.shape, "GT/pred must be same size");
    let mut merged = Vec::with_capacity(h * w * 2);
    for y in 0..h {
        merged.extend_from_slice(&gt_v[y * w..(y + 1) * w]); // 左: GT
        merged.extend_from_slice(&pred_v[y * w..(y + 1) * w]); // 右: pred
    }
    TensorData::new(merged, [h, w * 2])
}

/// 把所有视图的 GT|pred 拼接图 (`[H, 2W]`) stack 成单个 3D NRRD
/// (`[N, H, 2W]`, z = 视图序号), 便于在 3D Slicer / ParaView / napari 中
/// 批量翻阅同一迭代的所有视图。
fn save_stack(dir: &Path, iter: u32, pairs: &[TensorData]) {
    if pairs.is_empty() {
        return;
    }
    let [h, w2] = [pairs[0].shape[0], pairs[0].shape[1]];
    let mut vol = Vec::with_capacity(pairs.len() * h * w2);
    for p in pairs {
        assert_eq!([p.shape[0], p.shape[1]], [h, w2], "view size mismatch");
        vol.extend_from_slice(p.as_slice::<f32>().expect("f32"));
    }
    let td = TensorData::new(vol, [pairs.len(), h, w2]);
    let p = dir.join(format!("gt_pred_{iter:05}.nrrd"));
    save_gray_nrrd_f32_stack(&p, &td).expect("save GT|pred NRRD stack");
    println!("{} saved {} ({} views stacked)", ts(), p.display(), pairs.len());
}

/// `[HH:MM:SS]` 北京时间 (UTC+8, 固定偏移; 轻量, 无 chrono 依赖)。
fn ts() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = (d.as_secs() + 8 * 3600) % 86_400; // +8h → 北京时间
    format!(
        "[{:02}:{:02}:{:02}]",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

/// 追加一行指标到 CSV(若启用)。列:
/// `iter,time_bj,elapsed_s,loss,psnr,ssim,lpips,visible,splats,lr_mean,grad_*`
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
    use std::io::Write;
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut dcm: Option<PathBuf> = None;
    let mut iters = 4_000u32;
    let mut points = 30_000u32;
    // None → 从相机几何自动计算(等中心 FOV 半径), 保证点云覆盖整个视野。
    let mut scene_extent: Option<f32> = None;
    // 自动 gamma: 让全局强度中位数映射到该目标灰度(0.5 = 中灰)。
    let mut gamma_target: Option<f32> = Some(0.5);
    let mut init_density = 0.02f32;
    let mut lr_mean = 2e-5f64;
    // 末期不冻结: 默认 2e-6 (cosine min / 指数末段)。
    let mut lr_mean_end = 2e-6f64;
    let mut lr_scale = 5e-3f64;
    let mut lr_opac = 0.012f64;
    let mut growth_frac = 0.25f32;
    let mut refine_every = 400u32;
    let mut eval_every = 100u32;
    // 验证集: `--eval-split-every=N` 每 N 帧扣一个 held-out 视图;
    // `--eval-views=M` 每次 eval 采 M 个验证视图(均匀)。
    let mut eval_split_every: Option<usize> = None;
    let mut eval_views_count = 8usize;
    // 固定 densify 梯度阈值(替代动态百分位); None = 用 densify_grad_percentile。
    let mut fixed_grad_thr: Option<f32> = None;
    // 启用 oversized 高梯度点拆分(clone-only → clone+split, 参考 RGB refine_splats)。
    let mut enable_split = false;
    // proj 域损失权重 (在 -ln(intensity) 域比较; 默认 1.0 已作为最优默认)。
    let mut proj_weight = 1.0f32;
    // proj 域 SSIM 权重 (0 = 关闭, proj 损失保持纯 L1)。
    let mut proj_ssim_weight = 0.0f32;
    // 使用 cosine LR (默认指数衰减)。
    let mut cosine_lr = false;
    // clone/split 分界阈值系数 (默认 0.0005)。
    let mut percent_dense: Option<f32> = None;
    // split 尺度收缩系数 (默认 1/√2)。
    let mut split_scale: Option<f32> = None;
    // 离群点位置剪枝系数 (默认 3× scene_extent, 人体固定区域)。
    let mut bound_factor: Option<f32> = None;
    // prune 密度阈值 (默认 5e-5)。
    let mut cull_density: Option<f32> = None;
    // screen-size prune 阈值 (px, 0 = 关闭)。
    let mut max_screen_size: Option<f32> = None;
    // 多尺度(金字塔)损失权重 (默认 0.5, 最强项)。
    let mut multiscale_weight = 0.5f32;
    // 多窗宽窗位损失权重 (默认 0.5, LPIPS 感知增强)。
    let mut window_weight = 0.5f32;
    // 梯度(Sobel 差分)损失权重 (0 = 关闭)。
    let mut grad_weight = 0.0f32;
    // 密度软重置间隔 (0 = 关闭; 参考项目用 2000)。
    let mut density_reset_interval = 0u32;
    let mut out = PathBuf::from("target/fit_static");
    // 指标 CSV 记录器: 默认 <out>/metrics.csv, `--log-csv=FILE` 覆盖,
    // `--log-csv=off` 关闭。每次 eval 追加一行(含时间戳 + 各指标)。
    let mut log_csv: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--iters=") {
            iters = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--points=") {
            points = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--scene-extent=") {
            scene_extent = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--gamma-target=") {
            gamma_target = Some(v.parse()?);
        } else if a == "--no-gamma" {
            gamma_target = None;
        } else if let Some(v) = a.strip_prefix("--init-density=") {
            init_density = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--lr-mean=") {
            lr_mean = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--lr-mean-end=") {
            lr_mean_end = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--lr-scale=") {
            lr_scale = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--lr-opac=") {
            lr_opac = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--growth-frac=") {
            growth_frac = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--refine-every=") {
            refine_every = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--eval-every=") {
            eval_every = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--eval-split-every=") {
            eval_split_every = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--eval-views=") {
            eval_views_count = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--fixed-grad-thr=") {
            fixed_grad_thr = Some(v.parse()?);
        } else if a == "--split" {
            enable_split = true;
        } else if let Some(v) = a.strip_prefix("--proj-weight=") {
            proj_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--proj-ssim-weight=") {
            proj_ssim_weight = v.parse()?;
        } else if a == "--cosine-lr" {
            cosine_lr = true;
        } else if let Some(v) = a.strip_prefix("--percent-dense=") {
            percent_dense = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--split-scale=") {
            split_scale = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--bound-factor=") {
            bound_factor = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--cull-density=") {
            cull_density = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--max-screen-size=") {
            max_screen_size = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--multiscale-weight=") {
            multiscale_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--window-weight=") {
            window_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--grad-weight=") {
            grad_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--density-reset=") {
            density_reset_interval = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--log-csv=") {
            log_csv = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if dcm.is_none() {
            dcm = Some(PathBuf::from(a));
        }
        i += 1;
    }
    let dcm = dcm.expect(
        "usage: fit_static <dcm> [--iters=N] [--points=N] [--scene-extent=MM] \
         [--gamma-target=G] [--init-density=MU] [--lr-mean=LR] \
         [--lr-mean-end=LR] [--lr-scale=LR] [--lr-opac=LR] [--growth-frac=F] \
         [--refine-every=N] [--eval-split-every=N] [--eval-views=M] \
         [--fixed-grad-thr=F] [--split] [--proj-weight=W] [--proj-ssim-weight=S]
         [--cosine-lr] [--percent-dense=F] [--split-scale=F] [--bound-factor=F]
         [--cull-density=MU] [--max-screen-size=PX] [--multiscale-weight=W]
         [--window-weight=W] [--grad-weight=W]
         [--density-reset=N] [--eval-every=N] [--log-csv=FILE] [--out=DIR]",
    );

    // ---- 后端 + 数据集 ---------------------------------------------------
    let wgpu = brush_process::burn_init_setup().await;
    let device: Device = wgpu.into();

    // 精确加载指定的 dcm 文件(不能 from_path(父目录): 会扫到 images/ 下其它
    // DICOM, load_dataset 会选中错误文件)。
    let file = tokio::fs::File::open(&dcm).await?;
    let name = dcm
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "input.dcm".to_owned());
    let vfs = Arc::new(
        BrushVfs::from_reader(tokio::io::BufReader::new(file), Some(name))
            .await
            .expect("construct vfs"),
    );
    // 归一化: **min-max**(保留全部动态范围, 不裁剪暗部——percentile 会把
    // 造影剂/暗部截到 0) + 自动 gamma(中位数→目标灰度, 增强暗部)。
    let load_config = LoadDatasetConfig {
        max_frames: None,
        max_resolution: 1920,
        eval_split_every,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        dicom_orientation: XRayOrientation::Ap,
        dicom_normalization: DicomNormalization::Minmax,
        dicom_gamma: None,
        dicom_gamma_target: gamma_target,
        roi: brush_dataset::config::RoiSpec::None,
        max_scene_batch_cache_size: 1 << 30,
    };
    let result = brush_dataset::load_dataset(vfs, &load_config).await?;
    let dataset = result.dataset;
    let g0 = dataset.train.views[0].gray_image.as_ref().expect("gray");
    println!(
        "{} loaded {} views, frame0 {}x{}",
        ts(),
        dataset.train.views.len(),
        g0.width,
        g0.height
    );

    // 球半径: 显式 `--scene-extent` 优先, 否则按相机几何自动计算等中心 FOV
    // 半径 (半 FOV = (W/2)·SOD/fx), 乘 1.05 留边距 —— 保证点云覆盖整个视野。
    let scene_extent = match scene_extent {
        Some(r) => r,
        None => {
            let r = dataset.train.isocenter_fov_radius() * 1.05;
            println!("{} auto scene_extent = {r:.1} mm (isocenter FOV radius x1.05)", ts());
            r
        }
    };
    if gamma_target.is_some() {
        // 打印实际用的 gamma(由 dicom.rs 在加载时计算)。
        if let Some(g) = result.gamma {
            println!("{} auto gamma = {g:.3} (median -> {:.2})", ts(), gamma_target.unwrap());
        }
    }

    // ---- Trainer: 随机初始化 + 启用 refine -------------------------------
    // `create_xray_trainer` 做随机球内点云(KNN scale + μ_water 密度初始化),
    // 并把 config.refine.scene_extent 设为 scene_extent。
    let mut cfg = XRayTrainConfig::default();
    cfg.total_iters = iters;
    cfg.enable_deform = false; // 静态重建
    cfg.enable_ast = false;
    cfg.warm_up = 0;
    cfg.init_density = init_density;
    cfg.lr_mean = lr_mean;
    cfg.lr_mean_end = lr_mean_end;
    cfg.lr_scale = lr_scale;
    cfg.lr_opac = lr_opac;
    cfg.proj_weight = proj_weight;
    cfg.proj_ssim_weight = proj_ssim_weight;
    cfg.cosine_lr = cosine_lr;
    cfg.multiscale_weight = multiscale_weight;
    cfg.window_weight = window_weight;
    cfg.grad_weight = grad_weight;
    cfg.refine = XRayRefineConfig {
        refine_every,
        scene_extent,
        growth_select_fraction: growth_frac,
        fixed_grad_threshold: fixed_grad_thr,
        enable_split,
        density_reset_interval,
        percent_dense: percent_dense.unwrap_or(0.0005),
        split_scale_factor: split_scale.unwrap_or(std::f32::consts::FRAC_1_SQRT_2),
        max_bound_factor: bound_factor.unwrap_or(3.0),
        cull_density_threshold: cull_density.unwrap_or(5e-5),
        max_screen_size: max_screen_size.unwrap_or(0.0),
        ..XRayRefineConfig::default()
    };
    // FOV 过滤初始化: 只保留至少在一个视角内投影的点, 消除 FOV 外的高
    // opacity 离群点 (无梯度 → 密度永不下降)。
    let train_cams: Vec<_> = dataset.train.views.iter().map(|v| v.camera).collect();
    let mut trainer = create_xray_trainer(
        cfg,
        points,
        scene_extent,
        &device,
        Some((&train_cams, glam::uvec2(g0.width, g0.height))),
    );
    // 梯度诊断只在 eval 步收集(打印 + CSV 用), 见训练循环。
    println!(
        "{} init splats: {} (random ball r={}mm, init μ={} mm⁻¹, lr_mean={}->{}), refine every {}",
        ts(),
        trainer.num_splats(),
        scene_extent,
        init_density,
        lr_mean,
        lr_mean_end,
        refine_every
    );

    let mut dataloader = SceneLoader::new(&dataset.train, 42, &load_config);

    // ---- Eval 视图: held-out 验证集(若有 split)否则 frame 0 ----------------
    let eval_views: Vec<&SceneView> = if let Some(eval_scene) = &dataset.eval
        && !eval_scene.views.is_empty()
    {
        println!(
            "{} held-out eval set: {} views (split every {})",
            ts(),
            eval_scene.views.len(),
            eval_split_every.unwrap_or(0)
        );
        sample_eval_views(&eval_scene.views, eval_views_count)
    } else {
        log::warn!(
            "no held-out split (--eval-split-every unset): evaluating on train view 0"
        );
        vec![&dataset.train.views[0]]
    };
    println!("{} eval on {} views every {} steps", ts(), eval_views.len(), eval_every);

    std::fs::create_dir_all(&out)?;

    // 指标 CSV 记录器: 每次 eval 追加一行(时间戳 + 各指标)。
    let t0 = Instant::now();
    let mut csv_writer: Option<std::fs::File> = {
        use std::io::Write;
        let off = log_csv.as_deref() == Some(Path::new("off"));
        if off {
            None
        } else {
            let path = log_csv
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

    // 初始(iter 0): 在验证集上平均 PSNR/SSIM。
    {
        let mut p = 0.0f32;
        let mut s = 0.0f32;
        let mut l = 0.0f32;
        let mut pairs = Vec::with_capacity(eval_views.len());
        for view in eval_views.iter() {
            let gray = view.gray_image.as_ref().expect("gray GT");
            let gt = TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width]);
            let sample = trainer.eval_view(&view.camera, &gt, view.phase, view.time).await;
            p += sample.psnr;
            s += sample.ssim;
            l += sample.lpips;
            pairs.push(merge_pair(&sample.pred, &sample.gt));
        }
        save_stack(&out, 0, &pairs);
        p /= eval_views.len().max(1) as f32;
        s /= eval_views.len().max(1) as f32;
        l /= eval_views.len().max(1) as f32;
        log_metrics_row(
            &mut csv_writer,
            0,
            &t0,
            f32::NAN,
            p,
            s,
            l,
            0,
            trainer.num_splats(),
            0.0,
            None,
        )?;
        println!(
            "{} iter {:4} psnr={:6.2} ssim={:5.3} lpips={:.4}",
            ts(),
            "init",
            p,
            s,
            l
        );
    }

    // ---- 训练循环(与标准流程一致, 含 density control) -------------------
    for iter in 0..iters {
        let step = iter + 1;
        // 梯度诊断只在 eval 步需要(打印 + CSV): 其余步关闭, 省掉每步 4 次
        // GPU→CPU readback(原先硬编码开启时的固定开销)。
        trainer.set_collect_grads(step % eval_every == 0 || step == iters);
        let batch = dataloader.next_batch().await;
        let stats = trainer.step(&batch).await;

        // Density control (skip the very first step so gradients accumulate)。
        if step > 1
            && let Some(refine_stats) = trainer.maybe_refine(step).await
        {
            println!(
                "{} refine iter {step}: {} splats (added {}, split {}, pruned {}) grad_thr={}",
                ts(),
                refine_stats.total_splats,
                refine_stats.num_added,
                refine_stats.num_split,
                refine_stats.num_pruned,
                refine_stats
                    .grad_threshold
                    .map_or(-1.0f32, |t| t),
            );
        }

        if step % eval_every == 0 || step == iters {
            let mut avg_psnr = 0.0f32;
            let mut avg_ssim = 0.0f32;
            let mut avg_lpips = 0.0f32;
            let mut pairs = Vec::with_capacity(eval_views.len());
            for view in eval_views.iter() {
                let gray = view.gray_image.as_ref().expect("gray GT");
                let vgt = TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width]);
                let sample = trainer.eval_view(&view.camera, &vgt, view.phase, view.time).await;
                avg_psnr += sample.psnr;
                avg_ssim += sample.ssim;
                avg_lpips += sample.lpips;
                pairs.push(merge_pair(&sample.pred, &sample.gt));
            }
            save_stack(&out, step, &pairs);
            avg_psnr /= eval_views.len().max(1) as f32;
            avg_ssim /= eval_views.len().max(1) as f32;
            avg_lpips /= eval_views.len().max(1) as f32;
            log_metrics_row(
                &mut csv_writer,
                step,
                &t0,
                stats.loss,
                avg_psnr,
                avg_ssim,
                avg_lpips,
                stats.num_visible,
                stats.num_splats,
                stats.lr_mean,
                stats.grad_norms.as_ref(),
            )?;
            // 梯度诊断: 位置每步移动 ≈ lr_mean × mean_grad。
            if let Some(g) = &stats.grad_norms {
                println!(
                    "{} iter {:4} loss={:8.4} psnr={:6.2} ssim={:5.3} lpips={:.4} visible={} splats={} (eval {} views) | \
                     grads mean={:.1e} rot={:.1e} scale={:.1e} density={:.1e} | pos step≈{:.2e}mm",
                    ts(),
                    step,
                    stats.loss,
                    avg_psnr,
                    avg_ssim,
                    avg_lpips,
                    stats.num_visible,
                    stats.num_splats,
                    eval_views.len(),
                    g.mean_grad,
                    g.rot_grad,
                    g.scale_grad,
                    g.density_grad,
                    stats.lr_mean as f32 * g.mean_grad,
                );
            } else {
                println!(
                    "{} iter {:4} loss={:8.4} psnr={:6.2} ssim={:5.3} lpips={:.4} visible={} splats={} (eval {} views)",
                    ts(),
                    step,
                    stats.loss,
                    avg_psnr,
                    avg_ssim,
                    avg_lpips,
                    stats.num_visible,
                    stats.num_splats,
                    eval_views.len(),
                );
            }
        }
    }

    // ---- 导出最终 canonical splats 为 PLY -------------------------------
    let splats = xray_to_splats(trainer.canonical(), &device);
    let ply = brush_serde::splat_to_ply(splats, Some(glam::Vec3::Y)).await?;
    let ply_path = out.join("canonical_final.ply");
    std::fs::write(&ply_path, ply)?;
    println!("{} exported {}", ts(), ply_path.display());

    println!("{} done -> {}", ts(), out.display());
    Ok(())
}
