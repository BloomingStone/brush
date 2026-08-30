//! DICOM 数据加载 + 相机几何 / 场景尺度计算 (移植 fit_static / fit_deform
//! 的 L449-660 段, 两者逐行一致)。

use std::sync::Arc;

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene_loader::SceneLoader;
use brush_dataset::Dataset;
use brush_vfs::BrushVfs;
use burn::tensor::TensorData;

use crate::config::{FitConfig, Resolved};

pub struct LoadedData {
    pub dataset: Dataset,
    pub load_config: LoadDatasetConfig,
    pub sdd: f32,
    /// 最终场景半径 (mm), 显式或相机几何自动。
    pub scene_extent: f32,
    pub img_size: glam::UVec2,
    pub half_w: f32,
    pub half_h: f32,
    pub sod: f32,
    pub r0: f32,
    pub gamma_used: Option<f32>,
}

/// 精确加载指定 dcm (不能 from_path 父目录: 会扫到其它 DICOM)。
pub async fn load(cfg: &FitConfig, res: &Resolved) -> anyhow::Result<LoadedData> {
    let file = tokio::fs::File::open(&cfg.dcm).await?;
    let name = cfg
        .dcm
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
        max_resolution: cfg.max_resolution,
        eval_split_every: cfg.eval_split_every,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        dicom_orientation: XRayOrientation::Ap,
        dicom_normalization: DicomNormalization::Minmax,
        dicom_gamma: None,
        dicom_gamma_target: res.gamma_target,
        roi: res.roi,
        max_scene_batch_cache_size: 1 << 30,
    };
    let result = brush_dataset::load_dataset(vfs, &load_config).await?;
    let dataset = result.dataset;
    let gamma_used = result.gamma;

    let bytes = std::fs::read(&cfg.dcm)?;
    let dcm_meta = brush_dicom::parse_dicom(&bytes)?;
    let sdd = dcm_meta.geometry.sdd as f32;

    let g0 = dataset.train.views[0].gray_image.as_ref().expect("gray");
    println!(
        "[loaded] {} views, frame0 {}x{}",
        dataset.train.views.len(),
        g0.width,
        g0.height
    );

    // 心动相位诊断。
    let phases: Vec<f32> = dataset.train.views.iter().map(|v| v.phase).collect();
    let (min_p, max_p) = phases.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(lo, hi), p| (lo.min(p), hi.max(p)),
    );
    let mut s: Vec<f32> = phases.clone();
    s.sort_by(|a, b| a.total_cmp(b));
    s.dedup();
    println!(
        "[cardiac phase] {} views, range [{:.3}, {:.3}], {} unique values",
        phases.len(),
        min_p,
        max_p,
        s.len()
    );
    if s.len() <= 1 {
        log::warn!(
            "phase appears constant ({} unique) — check the (0071,1010) private tag",
            s.len()
        );
    }

    // 物理时间诊断。
    let t0 = dataset.train.views[0].time;
    let t1 = dataset.train.views.last().map(|v| v.time).unwrap_or(t0);
    println!(
        "[real time] [{:.3}, {:.3}] s ({} frames, {:.3} s total)",
        t0,
        t1,
        dataset.train.views.len(),
        t1 - t0
    );
    if t1 - t0 <= 1e-6 {
        log::warn!("time is constant — check FrameTimeVector / fps fallback");
    }

    // 场景半径: 显式优先, 否则 min(SOD, SDD-SOD) × 0.6。
    let scene_extent = match cfg.scene_extent {
        Some(r) => r,
        None => {
            let sod = dataset.train.views[0].camera.position.length();
            let r = sod.min(sdd - sod) * 0.6;
            println!(
                "[auto scene_extent] {r:.1} mm (min(SOD {sod:.0}, SDD-SOD {:.0}) x 0.6)",
                sdd - sod
            );
            r
        }
    };
    if res.gamma_target.is_some() {
        if let Some(g) = gamma_used {
            println!(
                "[auto gamma] = {g:.3} (median -> {:.2})",
                res.gamma_target.unwrap()
            );
        }
    }

    // 相机几何: R0 = half_w (W/2 世界), half_h, sod。
    let img_size = glam::uvec2(g0.width, g0.height);
    let cam0 = &dataset.train.views[0].camera;
    let focal = cam0.focal(img_size);
    let sod = cam0.position.length();
    let half_w = (g0.width as f32 * 0.5) * sod / focal.x;
    let half_h = (g0.height as f32 * 0.5) * sod / focal.y;
    let r0 = half_w;

    Ok(LoadedData {
        dataset,
        load_config,
        sdd,
        scene_extent,
        img_size,
        half_w,
        half_h,
        sod,
        r0,
        gamma_used,
    })
}

/// 构造 SceneLoader (seed 42, 与 fit_*.rs 一致)。
pub fn make_loader(data: &LoadedData) -> SceneLoader {
    SceneLoader::new(&data.dataset.train, 42, &data.load_config)
}

/// 初始采样区域 (ball / cylinder), cylinder 时按体积比缩放点数。
pub fn init_region(
    data: &LoadedData,
    cfg: &FitConfig,
    res: &Resolved,
) -> brush_train::xray_train::InitRegion {
    let r0 = data.half_w;
    if res.init_shape == "cylinder" {
        let r = r0 * res.init_radius_scale;
        let half_h_cyl = if cfg.init_height_factor > 0.0 {
            data.half_h * cfg.init_height_factor
        } else {
            data.half_h * (1.0 + r / data.sod)
        };
        let h0 = 2.0 * data.half_h * (1.0 + r0 / data.sod);
        let vol_ratio = (r / r0).powi(2) * (2.0 * half_h_cyl) / h0;
        let points = (cfg.init_density_base as f32 * vol_ratio).round().max(1.0) as u32;
        println!(
            "[init] cylinder: R={r:.1} (R0 {r0:.1} x {:.1}), half_h={half_h_cyl:.1}, N={points} (density base {} @ R0)",
            res.init_radius_scale, cfg.init_density_base
        );
        brush_train::xray_train::InitRegion::Cylinder {
            radius: r,
            half_height: half_h_cyl,
        }
    } else {
        brush_train::xray_train::InitRegion::Ball {
            radius: data.scene_extent,
        }
    }
}

/// 采样 eval 视图: held-out 集 (均匀 spread) 或回退 train view 0。
pub fn eval_views(
    data: &LoadedData,
    count: usize,
) -> Vec<&brush_dataset::scene::SceneView> {
    let views = &data.dataset.train.views;
    if let Some(eval_scene) = &data.dataset.eval
        && !eval_scene.views.is_empty()
    {
        println!(
            "[eval] held-out set: {} views",
            eval_scene.views.len()
        );
        sample(&eval_scene.views, count)
    } else {
        log::warn!("no held-out split (eval_split_every unset): evaluating on train view 0");
        vec![&views[0]]
    }
}

fn sample(views: &[brush_dataset::scene::SceneView], count: usize) -> Vec<&brush_dataset::scene::SceneView> {
    let n = views.len();
    if count >= n {
        views.iter().collect()
    } else {
        (0..count).map(|i| &views[i * n / count]).collect()
    }
}

/// 从灰度 GT 视图构造 eval 输入 TensorData ([H, W] f32)。
pub fn gt_tensor(view: &brush_dataset::scene::SceneView) -> TensorData {
    let gray = view.gray_image.as_ref().expect("gray GT");
    TensorData::new(gray.data.as_ref().to_vec(), [gray.height, gray.width])
}
