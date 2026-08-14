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

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene_loader::SceneLoader;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::xray_eval::save_gray_nrrd_f32;
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

/// 保存 GT | pred 水平拼接图 (均为 [0,1] f32) 为 float32 NRRD
/// (无损, 可在 3D Slicer/ParaView/napari 中调窗宽窗位)。
fn save_pair(dir: &Path, iter: u32, pred: &TensorData, gt: &TensorData) {
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
    let td = TensorData::new(merged, [h, w * 2]);
    let p = dir.join(format!("gt_pred_{iter:05}.nrrd"));
    save_gray_nrrd_f32(&p, &td).expect("save GT|pred NRRD");
    println!("saved {}", p.display());
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
    let mut lr_mean_end = 2e-7f64;
    let mut refine_every = 400u32;
    let mut eval_every = 100u32;
    let mut out = PathBuf::from("target/fit_static");
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
        } else if let Some(v) = a.strip_prefix("--refine-every=") {
            refine_every = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--eval-every=") {
            eval_every = v.parse()?;
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
         [--lr-mean-end=LR] [--refine-every=N] [--eval-every=N] [--out=DIR]",
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
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        dicom_orientation: XRayOrientation::Ap,
        dicom_normalization: DicomNormalization::Minmax,
        dicom_gamma: None,
        dicom_gamma_target: gamma_target,
        max_scene_batch_cache_size: 1 << 30,
    };
    let result = brush_dataset::load_dataset(vfs, &load_config).await?;
    let dataset = result.dataset;
    let g0 = dataset.train.views[0].gray_image.as_ref().expect("gray");
    println!(
        "loaded {} views, frame0 {}x{}",
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
            println!("auto scene_extent = {r:.1} mm (isocenter FOV radius x1.05)");
            r
        }
    };
    if gamma_target.is_some() {
        // 打印实际用的 gamma(由 dicom.rs 在加载时计算)。
        if let Some(g) = result.gamma {
            println!("auto gamma = {g:.3} (median -> {:.2})", gamma_target.unwrap());
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
    cfg.refine = XRayRefineConfig {
        refine_every,
        scene_extent,
        ..XRayRefineConfig::default()
    };
    let mut trainer = create_xray_trainer(cfg, points, scene_extent, &device);
    // 打开梯度诊断, 检查各参数实际更新幅度。
    trainer.set_collect_grads(true);
    println!(
        "init splats: {} (random ball r={}mm, init μ={} mm⁻¹, lr_mean={}->{}), refine every {}",
        trainer.num_splats(),
        scene_extent,
        init_density,
        lr_mean,
        lr_mean_end,
        refine_every
    );

    let mut dataloader = SceneLoader::new(&dataset.train, 42, &load_config);

    // 固定 eval 视图: frame 0。
    let eval_view = &dataset.train.views[0];
    let gray = eval_view.gray_image.as_ref().expect("gray GT");
    let gt_data = TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width]);

    std::fs::create_dir_all(&out)?;

    // 初始(iter 0)。
    let s0 = trainer.eval_view(&eval_view.camera, &gt_data, eval_view.phase).await;
    println!(
        "iter {:4} psnr={:6.2} ssim={:5.3}",
        "init", s0.psnr, s0.ssim
    );
    save_pair(&out, 0, &s0.pred, &s0.gt);

    // ---- 训练循环(与标准流程一致, 含 density control) -------------------
    for iter in 0..iters {
        let step = iter + 1;
        let batch = dataloader.next_batch().await;
        let stats = trainer.step(&batch).await;

        // Density control (skip the very first step so gradients accumulate)。
        if step > 1
            && let Some(refine_stats) = trainer.maybe_refine(step).await
        {
            println!(
                "refine iter {step}: {} splats (added {}, pruned {}) grad_thr={}",
                refine_stats.total_splats,
                refine_stats.num_added,
                refine_stats.num_pruned,
                refine_stats
                    .grad_threshold
                    .map_or(-1.0f32, |t| t),
            );
        }

        if step % eval_every == 0 || step == iters {
            let sample = trainer
                .eval_view(&eval_view.camera, &gt_data, eval_view.phase)
                .await;
            save_pair(&out, step, &sample.pred, &sample.gt);
            // 梯度诊断: 位置每步移动 ≈ lr_mean × mean_grad。
            if let Some(g) = &stats.grad_norms {
                println!(
                    "iter {:4} loss={:8.4} psnr={:6.2} ssim={:5.3} visible={} splats={} | \
                     grads mean={:.1e} rot={:.1e} scale={:.1e} density={:.1e} | pos step≈{:.2e}mm",
                    step,
                    stats.loss,
                    sample.psnr,
                    sample.ssim,
                    stats.num_visible,
                    stats.num_splats,
                    g.mean_grad,
                    g.rot_grad,
                    g.scale_grad,
                    g.density_grad,
                    stats.lr_mean as f32 * g.mean_grad,
                );
            } else {
                println!(
                    "iter {:4} loss={:8.4} psnr={:6.2} ssim={:5.3} visible={} splats={}",
                    step,
                    stats.loss,
                    sample.psnr,
                    sample.ssim,
                    stats.num_visible,
                    stats.num_splats,
                );
            }
        }
    }

    // ---- 导出最终 canonical splats 为 PLY -------------------------------
    let splats = xray_to_splats(trainer.canonical(), &device);
    let ply = brush_serde::splat_to_ply(splats, Some(glam::Vec3::Y)).await?;
    let ply_path = out.join("canonical_final.ply");
    std::fs::write(&ply_path, ply)?;
    println!("exported {}", ply_path.display());

    println!("done -> {}", out.display());
    Ok(())
}
