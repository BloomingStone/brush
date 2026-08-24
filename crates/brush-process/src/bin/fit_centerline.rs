//! 临时验证: 中心线点云 + 随机点云背景 作为初始化, 拟合 `RXA_static_coronary.dcm`。
//!
//! 流程与正常 X-ray 训练一致(复用 `XRayTrainer::step` + `SceneLoader` +
//! `eval_view`), 仅两处不同:
//!   1. 初始化 splats = 中心线点云(血管, μ≈0.02, σ=0.7mm) + 随机点云背景
//!      (软组织, μ_water≈0.002, σ=5mm), 而不是纯随机球。
//!   2. 禁用 density control(refine_every = u32::MAX), 验证纯梯度更新能否
//!      优化 GS 点云重建质量。
//! 每 100 步对固定视图(frame 0)做一次 eval, 打印 PSNR/SSIM 并保存渲染
//! 结果(**float32 NRRD**, 无损, 可在 3D Slicer/ParaView/napari 中调窗宽窗位)。
//!
//! 用法:
//!   cargo run -p brush-process --bin fit_centerline -- \
//!     images/RXA_static_coronary.dcm \
//!     --centerline=crates/brush-train/tests/data/coronary/central_line_world.xyz \
//!     --iters=1000 --background=20000 --out=target/fit_centerline

use std::path::{Path, PathBuf};
use std::sync::Arc;

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene_loader::SceneLoader;
use brush_train::xray_eval::save_gray_nrrd_f32;
use brush_train::xray_refine::XRayRefineConfig;
use brush_train::xray_train::{XRayTrainConfig, XRayTrainer};
use brush_vfs::BrushVfs;
use brush_xray::XRaySplats;
use burn::tensor::{Device, TensorData};
use glam::Vec3;
use rand::{RngExt, SeedableRng};

/// 读取中心线 `.xyz`(每行 "x y z", world RAS mm)。
fn parse_centerline(path: &Path) -> Vec<Vec3> {
    let text = std::fs::read_to_string(path).expect("read centerline xyz");
    text.lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let x: f32 = it.next()?.parse().ok()?;
            let y: f32 = it.next()?.parse().ok()?;
            let z: f32 = it.next()?.parse().ok()?;
            Some(Vec3::new(x, y, z))
        })
        .collect()
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
    let mut centerline: Option<PathBuf> = None;
    let mut iters = 1000u32;
    let mut background = 2_000u32;
    let mut out = PathBuf::from("target/fit_centerline");
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--centerline=") {
            centerline = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--iters=") {
            iters = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--background=") {
            background = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if dcm.is_none() {
            dcm = Some(PathBuf::from(a));
        }
        i += 1;
    }
    let dcm = dcm.expect(
        "usage: fit_centerline <dcm> [--centerline=...] [--iters=N] [--background=N] [--out=DIR]",
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
    let load_config = LoadDatasetConfig {
        max_frames: None,
        max_resolution: 1920,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        dicom_orientation: XRayOrientation::Ap,
        dicom_normalization: DicomNormalization::Percentile,
        dicom_gamma: None,
        dicom_gamma_target: None,
        roi: brush_dataset::config::RoiSpec::None,
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

    // ---- 初始化 splats: 中心线 + 随机背景 ---------------------------------
    let centerline_pts = centerline.as_deref().map(parse_centerline).unwrap_or_default();
    println!("centerline points: {}", centerline_pts.len());

    const SCENE_EXTENT: f32 = 100.0; // mm, 覆盖 coronary FOV (±94mm)
    let mut means = Vec::new();
    let mut rots = Vec::new();
    let mut log_scales = Vec::new();
    let mut raw_opac = Vec::new();

    // 随机背景: 均匀球, 软组织密度 μ_water=0.002, σ=5mm。
    // 激活: density = MU_WATER·softplus(raw), 因此 raw = inverse_softplus(1) ≈ 0.541。
    // scale 激活 = softplus(raw), 要渲染 σ=5mm: raw = inverse_softplus(5)。
    let bg_logit = brush_cube::inverse_softplus(0.002 / brush_cube::MU_WATER);
    let bg_log_scale = brush_cube::inverse_softplus(5.0f32);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..background {
        let u: f32 = rng.random_range(0.0..1.0);
        let r = SCENE_EXTENT * u.cbrt();
        let theta = rng.random_range(0.0..std::f32::consts::PI);
        let phi = rng.random_range(0.0..2.0 * std::f32::consts::PI);
        let (s, c) = theta.sin_cos();
        means.extend([r * s * phi.cos(), r * s * phi.sin(), r * c]);
        rots.extend([1.0, 0.0, 0.0, 0.0]);
        log_scales.extend([bg_log_scale; 3]);
        raw_opac.push(bg_logit);
    }
    // 中心线: 血管密度 μ=0.02, σ=0.7mm(3σ≈2.1mm 匹配血管半径)。
    // raw = inverse_softplus(10) ≈ 10 → density = 0.002·softplus(10) ≈ 0.02。
    // scale 激活 = softplus(raw), 要渲染 σ=0.7mm: raw = inverse_softplus(0.7)。
    let cl_logit = brush_cube::inverse_softplus(0.02 / brush_cube::MU_WATER);
    let cl_log_scale = brush_cube::inverse_softplus(0.7f32);
    for p in &centerline_pts {
        means.extend([p.x, p.y, p.z]);
        rots.extend([1.0, 0.0, 0.0, 0.0]);
        log_scales.extend([cl_log_scale; 3]);
        raw_opac.push(cl_logit);
    }
    let canonical = XRaySplats::from_raw(means, rots, log_scales, raw_opac, &device);
    println!(
        "init splats: {} background + {} centerline = {}",
        background,
        centerline_pts.len(),
        canonical.num_splats()
    );

    // ---- Trainer: 静态模式, 禁用 refine -----------------------------------
    let mut cfg = XRayTrainConfig::default();
    cfg.total_iters = iters;
    cfg.enable_deform = false;
    cfg.enable_ast = false;
    cfg.warm_up = 0;
    cfg.refine = XRayRefineConfig {
        refine_every: u32::MAX, // 禁用 densify/prune, 验证纯梯度
        ..XRayRefineConfig::default()
    };
    let mut trainer = XRayTrainer::new(cfg, canonical, None, None, &device);

    let mut dataloader = SceneLoader::new(&dataset.train, 42, &load_config);

    // 固定 eval 视图: frame 0。
    let eval_view = &dataset.train.views[0];
    let gray = eval_view.gray_image.as_ref().expect("gray GT");
    let gt_data = TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width]);

    std::fs::create_dir_all(&out)?;

    // 初始(iter 0)。
    let s0 = trainer.eval_view(&eval_view.camera, &gt_data, eval_view.phase, eval_view.time).await;
    println!(
        "iter {:4} psnr={:6.2} ssim={:5.3}",
        "init", s0.psnr, s0.ssim
    );
    save_pair(&out, 0, &s0.pred, &s0.gt);

    // ---- 训练循环(与标准流程一致) -----------------------------------------
    for iter in 0..iters {
        let step = iter + 1;
        let batch = dataloader.next_batch().await;
        let stats = trainer.step(&batch).await;

        if step % 100 == 0 {
            let sample = trainer
                .eval_view(&eval_view.camera, &gt_data, eval_view.phase, eval_view.time)
                .await;
            save_pair(&out, step, &sample.pred, &sample.gt);
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

    println!("done -> {}", out.display());
    Ok(())
}
