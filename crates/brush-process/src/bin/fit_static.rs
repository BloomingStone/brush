//! 静态 X-ray 重建 (与 `fit_deform` 逻辑一致, 删除形变场部分):
//! 拟合 `images/RXA_chest.dcm` 等纯静态数据。
//!
//! 与 `fit_deform` 的差异: 只有形变相关部分被删除 —
//!   - 无 deform 网络 (`enable_deform=false`): canonical splats 直接渲染
//!   - 无 phase/time 条件化、无 AST/warm-up、无时间编码
//!   - 无 deform ckpt / deform_field 导出
//! 其余 (初始化/圆柱、loss、refine、eval、PLY 导出) 与 fit_deform 一致。
//!
//! 每 `eval_every` 步对 held-out 视图做 eval, 打印 PSNR/SSIM/LPIPS 并保存
//! GT|pred 拼接 NRRD stack; 训练结束导出 canonical splats 的 PLY。
//!
//! 用法:
//!   cargo run -p brush-process --bin fit_static -- \
//!     images/RXA_chest.dcm \
//!     --iters=20000 --points=5000 --refine-every=400 \
//!     --eval-split-every=5 --eval-views=8 --eval-every=1000 \
//!     --fixed-grad-thr=5e-6 --split --out=target/fit_static

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use brush_dataset::config::{parse_roi, DicomNormalization, LoadDatasetConfig, RoiSpec, XRayOrientation};
use brush_dataset::scene::SceneView;
use brush_dataset::scene_loader::SceneLoader;
use brush_loss::gray::GrayLossType;
use clap::Parser;

use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::xray_eval::save_gray_nrrd_f32_stack;
use brush_train::xray_refine::{XRayRefineConfig, XRayRefineGradThreshold};
use brush_train::xray_train::{XRayTrainConfig, create_xray_trainer};
use brush_vfs::BrushVfs;
use brush_xray::XRaySplats;
use burn::tensor::{Device, TensorData};

/// 解析 `--loss` (l1|charbonnier|huber|l2) → [`GrayLossType`]。
fn parse_loss(s: &str) -> Result<GrayLossType, String> {
    match s {
        "l1" => Ok(GrayLossType::L1),
        "charbonnier" => Ok(GrayLossType::Charbonnier),
        "huber" => Ok(GrayLossType::Huber),
        "l2" => Ok(GrayLossType::L2),
        _ => Err(format!("invalid --loss '{s}' (l1|charbonnier|huber|l2)")),
    }
}

/// 静态 X-ray 重建 CLI (clap)。参数按功能分组, 默认值与旧手写解析一致。
#[derive(Parser)]
#[command(name = "fit_static", about = "静态 X-ray 重建 (与 fit_deform 一致, 删除形变场部分)")]
struct FitStaticArgs {
    /// DICOM 序列文件路径 (如 images/RXA_chest.dcm)。
    #[arg(value_name = "DCM", help_heading = "输入数据")]
    dcm: PathBuf,

    // ---- 数据预处理 ------------------------------------------------------
    /// 自动 gamma: 让全局强度中位数映射到该目标灰度 (0.5 = 中灰)。
    #[arg(long, value_name = "G", default_value_t = 0.5, help_heading = "数据预处理")]
    gamma_target: f32,
    /// 关闭自动 gamma。
    #[arg(long, help_heading = "数据预处理")]
    no_gamma: bool,
    /// 场景半径 (mm); 缺省按相机几何自动计算等中心 FOV 半径。
    #[arg(long, value_name = "MM", help_heading = "数据预处理")]
    scene_extent: Option<f32>,
    /// ROI 截取: no | N | x0,y0,w,h (像素)。默认四边各裁 20。
    #[arg(long, value_name = "ROI", default_value = "20", value_parser = parse_roi, help_heading = "数据预处理")]
    roi: RoiSpec,

    // ---- 初始化 ----------------------------------------------------------
    /// 初始化 splat 点数 (densify 会按梯度阈值补足)。
    #[arg(long, value_name = "N", default_value_t = 5_000, help_heading = "初始化")]
    points: u32,
    /// 初始化采样区域: ball | cylinder (绕 Z 圆柱, 匹配锥束 FOV)。
    #[arg(long, default_value = "cylinder", value_parser = ["ball", "cylinder"], help_heading = "初始化")]
    init_shape: String,
    /// 圆柱半径 = R0(=half_w 世界) × scale; 高度自动 = 2·half_h·(1+R/SOD)。
    #[arg(long, default_value_t = 1.5, help_heading = "初始化")]
    init_radius_scale: f32,

    /// >0 覆盖自动高度 = F × (2·half_h)。
    #[arg(long, default_value_t = 0.0, help_heading = "初始化")]
    init_height_factor: f32,
    /// 初始密度 (mm⁻¹, 默认 0.01 = 5×水)。
    #[arg(long, value_name = "MU", default_value_t = 0.002, help_heading = "初始化")]
    init_density: f32,

    // ---- 优化 / 学习率 ----------------------------------------------------
    /// 总训练步数。
    #[arg(long, value_name = "N", default_value_t = 10_000, help_heading = "优化 / 学习率")]
    iters: u32,
    /// 位置学习率起点。
    #[arg(long, default_value_t = 2e-5, help_heading = "优化 / 学习率")]
    lr_mean: f64,
    /// 位置学习率终点 (cosine min / 指数末段)。
    #[arg(long, default_value_t = 2e-6, help_heading = "优化 / 学习率")]
    lr_mean_end: f64,
    /// 尺度 (log-scale) 学习率。
    #[arg(long, default_value_t = 5e-3, help_heading = "优化 / 学习率")]
    lr_scale: f64,
    /// 不透明度 (密度) 学习率。
    #[arg(long, default_value_t = 0.012, help_heading = "优化 / 学习率")]
    lr_opac: f64,
    /// 使用 cosine LR (默认指数衰减)。
    #[arg(long, help_heading = "优化 / 学习率")]
    cosine_lr: bool,

    // ---- 密度控制 / refine ------------------------------------------------
    /// 每次 refine 只 densify 25% 的过阈值 splat (平衡增长与速度)。
    #[arg(long, default_value_t = 0.25, help_heading = "密度控制 / refine")]
    growth_frac: f32,
    /// 密度控制 (densify/prune) 间隔 (步)。
    #[arg(long, value_name = "N", default_value_t = 400, help_heading = "密度控制 / refine")]
    refine_every: u32,
    /// 硬性 splat 数上限 (到顶后只 prune 不再增)。
    #[arg(long, value_name = "N", default_value_t = 300_000, help_heading = "密度控制 / refine")]
    max_splats: u32,
    /// densify 梯度百分位 (0~1, 缺省 0.98; None = 固定阈值)。
    #[arg(long, value_name = "PCT", help_heading = "密度控制 / refine")]
    dyn_grad_percentile: Option<f32>,
    /// 固定 densify 梯度阈值 (缺省 5e-6, 与 dyn_grad_percentile 互斥)。
    #[arg(long, value_name = "F", default_value_t = 5e-6, help_heading = "密度控制 / refine")]
    fixed_grad_thr: f32,
    /// 启用 oversized 高梯度点拆分 (默认开)。
    #[arg(long, default_value_t = true, help_heading = "密度控制 / refine")]
    split: bool,
    /// clone/split 分界阈值系数 (默认 0.0005)。
    #[arg(long, default_value_t = 0.0005, help_heading = "密度控制 / refine")]
    percent_dense: f32,
    /// split 尺度收缩系数 (默认 1/√2)。
    #[arg(long, default_value_t = std::f32::consts::FRAC_1_SQRT_2, help_heading = "密度控制 / refine")]
    split_scale: f32,
    /// 离群点位置剪枝系数 (默认 1.0 * scene_extent)。
    #[arg(long, default_value_t = 1.0, help_heading = "密度控制 / refine")]
    bound_factor: f32,
    /// prune 密度阈值 (mm⁻¹, 默认 5e-4)。
    #[arg(long, value_name = "MU", default_value_t = 5e-4, help_heading = "密度控制 / refine")]
    cull_density: f32,
    /// screen-size prune 阈值 (px, <0 = 关闭)。
    #[arg(long, value_name = "PX", default_value_t = -1.0, help_heading = "密度控制 / refine")]
    max_screen_size: f32,
    /// 贡献裁剪: 剪掉 density×屏幕面积×可见性 都低且处于最低百分位的 splat。
    #[arg(long, help_heading = "密度控制 / refine")]
    cull_contribution: bool,
    /// 关闭贡献裁剪。
    #[arg(long, help_heading = "密度控制 / refine")]
    no_cull_contribution: bool,
    /// 贡献裁剪百分位 (默认 0.05)。
    #[arg(long, default_value_t = 0.05, help_heading = "密度控制 / refine")]
    cull_percentile: f32,
    /// 贡献裁剪下限 (默认 1e-3)。
    #[arg(long, default_value_t = 1e-3, help_heading = "密度控制 / refine")]
    cull_floor: f32,
    /// 最小保留 splat 数。
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "密度控制 / refine")]
    min_splats: u32,
    /// 密度软重置间隔 (0 = 关闭; 参考项目用 2000)。
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "密度控制 / refine")]
    density_reset: u32,

    // ---- 损失 ------------------------------------------------------------
    /// 像素级损失类型: l1 | charbonnier | huber | l2。
    #[arg(long, default_value = "charbonnier", value_parser = parse_loss, help_heading = "损失")]
    loss: GrayLossType,
    /// Charbonnier ε (平滑底)。
    #[arg(long, default_value_t = 1e-3, help_heading = "损失")]
    loss_eps: f32,
    /// Huber δ。
    #[arg(long, default_value_t = 0.1, help_heading = "损失")]
    loss_delta: f32,
    /// proj 域损失权重 (在 -ln(intensity) 域比较)。
    #[arg(long, default_value_t = 1.0, help_heading = "损失")]
    proj_weight: f32,
    /// proj 域 SSIM 权重 (0 = 关闭, 纯 L1)。
    #[arg(long, default_value_t = 0.0, help_heading = "损失")]
    proj_ssim_weight: f32,
    /// 多尺度(金字塔)损失权重。
    #[arg(long, default_value_t = 0.5, help_heading = "损失")]
    multiscale_weight: f32,
    /// 多窗宽窗位损失权重 (LPIPS 感知增强)。
    #[arg(long, default_value_t = 0.5, help_heading = "损失")]
    window_weight: f32,
    /// 梯度(Sobel 差分)损失权重 (0 = 关闭)。
    #[arg(long, default_value_t = 0.0, help_heading = "损失")]
    grad_weight: f32,
    /// 边缘加权梯度损失 ramp 起点 (默认 3000 起步)。
    #[arg(long, value_name = "N", default_value_t = 3_000, help_heading = "损失")]
    grad_ramp_from: u32,
    /// 边缘加权梯度损失 ramp 终点 (0 = total_iters)。
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "损失")]
    grad_ramp_to: u32,
    /// GT 边缘幅度加权: clamp(|∇gt|/scale, 0, 1); 0 = 纯梯度损失。
    #[arg(long, default_value_t = 0.03, help_heading = "损失")]
    grad_edge_scale: f32,

    // ---- 评估与输出 -------------------------------------------------------
    /// 每 N 步做一次 eval (PSNR/SSIM/LPIPS + 保存 GT|pred stack)。
    #[arg(long, value_name = "N", default_value_t = 500, help_heading = "评估与输出")]
    eval_every: u32,
    /// 验证集: 每 N 帧扣一个 held-out 视图 (缺省用 train view 0)。
    #[arg(long, value_name = "N", help_heading = "评估与输出")]
    eval_split_every: Option<usize>,
    /// 每次 eval 采 M 个验证视图 (均匀)。
    #[arg(long, value_name = "M", default_value_t = 8, help_heading = "评估与输出")]
    eval_views: usize,
    /// 指标 CSV: 默认 <out>/metrics.csv, "off" 关闭。
    #[arg(long, value_name = "FILE", help_heading = "评估与输出")]
    log_csv: Option<PathBuf>,
    /// 输出目录。
    #[arg(long, value_name = "DIR", default_value = "target/fit_static", help_heading = "评估与输出")]
    out: PathBuf,
}

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

/// Sample up to `count` views, spread evenly across the sequence (for eval on
/// held-out sets, covering a range of C-arm angles + cardiac phases).
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
    let args = FitStaticArgs::parse();

    // ---- 输入 / 数据预处理 -------------------------------------------------
    let dcm = args.dcm.clone();
    let roi = args.roi;
    // None → 从相机几何自动计算(等中心 FOV 半径), 保证点云覆盖整个视野。
    let scene_extent = args.scene_extent;
    // 自动 gamma: 让全局强度中位数映射到该目标灰度 (0.5 = 中灰); --no-gamma 关闭。
    let gamma_target = if args.no_gamma { None } else {  Some(args.gamma_target) };
    // ---- 初始化 -----------------------------------------------------------
    // 初始化点数: 5000 (2026-08-21 扫描最优: 比 30000 少, PSNR 反而更高
    // 41.12 vs 40.63, 且快 ~1.5×)。densify 会按梯度阈值补足, 初始种子少
    // → 空气区噪声少、放置更高效。
    let points = args.points;
    let init_shape = args.init_shape.clone();
    let init_radius_scale = args.init_radius_scale;
    let init_height_factor = args.init_height_factor;
    // 初始密度 = 0.01 mm⁻¹ (5×水): 2026-08-21 扫描 points=5000 下甜点
    // (PSNR 41.32)。之前 0.02 (10×水) 让初始球成高密度雾, 空气 splat 密度
    // 饱和永不衰减 → pruned≈0; 降到 0.01 后空气点可衰减并被密度裁剪。
    let init_density = args.init_density;
    // ---- 优化 / 学习率 ----------------------------------------------------
    let iters = args.iters;
    let lr_mean = args.lr_mean;
    // 末期不冻结: 默认 2e-6 (cosine min / 指数末段)。
    let lr_mean_end = args.lr_mean_end;
    let lr_scale = args.lr_scale;
    let lr_opac = args.lr_opac;
    let cosine_lr = args.cosine_lr;
    // ---- 密度控制 / refine ------------------------------------------------
    let growth_frac = args.growth_frac;
    let refine_every = args.refine_every;
    let max_splats = args.max_splats;
    // 动态梯度阈值
    let xray_refine_thr = if let Some(pct) = args.dyn_grad_percentile {
        if pct <= 0.0 || pct >= 1.0 {
            panic!("--densify-grad-percentile must be in (0,1)");
        }
        XRayRefineGradThreshold::Dynamic(pct)
    } else {
        XRayRefineGradThreshold::Fixed(args.fixed_grad_thr)
    };
    let enable_split = args.split;
    let percent_dense = args.percent_dense;
    let split_scale = args.split_scale;
    let bound_factor = args.bound_factor;
    let cull_density = args.cull_density;
    let max_screen_size = args.max_screen_size;
    let cull_contribution = args.cull_contribution && !args.no_cull_contribution;
    let cull_percentile = args.cull_percentile;
    let cull_floor = args.cull_floor;
    let min_splats = args.min_splats;
    let density_reset_interval = args.density_reset;
    // ---- 损失 -------------------------------------------------------------
    let loss_type = args.loss;
    let loss_eps = args.loss_eps;
    let loss_delta = args.loss_delta;
    let proj_weight = args.proj_weight;
    let proj_ssim_weight = args.proj_ssim_weight;
    let multiscale_weight = args.multiscale_weight;
    let window_weight = args.window_weight;
    let grad_weight = args.grad_weight;
    let grad_ramp_from = args.grad_ramp_from;
    let grad_ramp_to = args.grad_ramp_to;
    let grad_edge_scale = args.grad_edge_scale;
    // ---- 评估与输出 --------------------------------------------------------
    let eval_every = args.eval_every;
    let eval_split_every = args.eval_split_every;
    let eval_views_count = args.eval_views;
    let out = args.out.clone();
    let log_csv = args.log_csv.clone();

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
        roi,
        max_scene_batch_cache_size: 1 << 30,
    };
    let result = brush_dataset::load_dataset(vfs, &load_config).await?;
    let dataset = result.dataset;
    // 从 DICOM header 取 SDD (源到探测器距离), 用于场景尺度定义。
    let bytes = std::fs::read(&dcm)?;
    let dcm_meta = brush_dicom::parse_dicom(&bytes)?;
    let sdd = dcm_meta.geometry.sdd as f32;
    let g0 = dataset.train.views[0].gray_image.as_ref().expect("gray");
    println!(
        "{} loaded {} views, frame0 {}x{}",
        ts(),
        dataset.train.views.len(),
        g0.width,
        g0.height
    );

    // 心动相位诊断: 打印 train 视图的 phase 范围, 确认 (0071,1010) 数据有效。
    {
        let phases: Vec<f32> = dataset.train.views.iter().map(|v| v.phase).collect();
        let (min_p, max_p) = phases
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p), hi.max(p))
            });
        let n_uniq = {
            let mut s: Vec<f32> = phases.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            s.dedup();
            s.len()
        };
        println!(
            "{} cardiac phase: {} views, range [{:.3}, {:.3}], {} unique values",
            ts(),
            phases.len(),
            min_p,
            max_p,
            n_uniq
        );
        if n_uniq <= 1 {
            log::warn!(
                "phase appears constant ({} unique) — check the (0071,1010) private tag",
                n_uniq
            );
        }
    }

    // 物理时间诊断: 帧时间应随帧号递增 (t = f/fps 或 FrameTimeVector)。
    {
        let t0 = dataset.train.views[0].time;
        let t1 = dataset.train.views.last().map(|v| v.time).unwrap_or(t0);
        println!(
            "{} real time: [{:.3}, {:.3}] s ({} frames, {} s total)",
            ts(),
            t0,
            t1,
            dataset.train.views.len(),
            t1 - t0
        );
        if t1 - t0 <= 1e-6 {
            log::warn!("time is constant — check FrameTimeVector / fps fallback");
        }
    }

    // 球半径: 显式 `--scene-extent` 优先, 否则按相机几何自动计算等中心 FOV
    // 半径 (半 FOV = (W/2)·SOD/fx), 乘 1.05 留边距 —— 保证点云覆盖整个视野。
    let scene_extent = match scene_extent {
        Some(r) => r,
        None => {
            // 物理上界: C-arm 旋转所能容纳的最大长度 = min(SOD, SDD-SOD),
            // x0.6 安全冗余 + deform 网格分辨率折中 (0.8→11mm/单元过粗,
            // 0.6→8.3mm/单元)。SOD = 源到等中心, SDD-SOD = 等中心到探测器。
            let sod = dataset.train.views[0].camera.position.length();
            let r = sod.min(sdd - sod) * 0.6;
            println!(
                "{} auto scene_extent = {r:.1} mm (min(SOD {sod:.0}, SDD-SOD {:.0}) x 0.6)",
                ts(),
                sdd - sod
            );
            r
        }
    };
    if gamma_target.is_some() {
        // 打印实际用的 gamma(由 dicom.rs 在加载时计算)。
        if let Some(g) = result.gamma {
            println!("{} auto gamma = {g:.3} (median -> {:.2})", ts(), gamma_target.unwrap());
        }
    }

    // ---- Trainer: 随机初始化 + deform 模式 + 启用 refine -----------------
    // `create_xray_trainer` 做随机球内点云(KNN scale + μ_water 密度初始化),
    // 并把 config.refine.scene_extent 设为 scene_extent; `enable_deform=true`
    // 时按 `deform_backend` 创建 HexPlane (默认) 或 hash-grid 形变网络
    // (coord_scale = scene_extent)。
    let mut cfg = XRayTrainConfig::default();
    cfg.init_density = init_density;
    cfg.lr_mean = lr_mean;
    cfg.lr_mean_end = lr_mean_end;
    cfg.total_iters = iters;
    cfg.lr_scale = lr_scale;
    cfg.lr_opac = lr_opac;
    cfg.proj_weight = proj_weight;
    cfg.proj_ssim_weight = proj_ssim_weight;
    cfg.loss_type = loss_type;
    cfg.loss_eps = loss_eps;
    cfg.loss_delta = loss_delta;
    cfg.cosine_lr = cosine_lr;
    cfg.multiscale_weight = multiscale_weight;
    cfg.window_weight = window_weight;
    cfg.grad_weight = grad_weight;
    cfg.grad_ramp_from = grad_ramp_from;
    cfg.grad_ramp_to = grad_ramp_to;
    cfg.grad_edge_scale = grad_edge_scale;
    cfg.refine = XRayRefineConfig {
        refine_every,
        scene_extent,
        growth_select_fraction: growth_frac,
        grad_threshold: xray_refine_thr,
        enable_split,
        density_reset_interval,
        percent_dense: percent_dense,
        split_scale_factor: split_scale,
        max_bound_factor: bound_factor,
        cull_density_threshold: cull_density,
        max_splats,
        max_screen_size: max_screen_size,
        cull_contribution,
        cull_contribution_percentile: cull_percentile,
        cull_contribution_floor: cull_floor,
        min_splats,
        ..XRayRefineConfig::default()
    };
    // 相机几何: R0 = half_w (W/2 世界), half_h, sod。
    let img_size = glam::uvec2(g0.width, g0.height);
    let cam0 = &dataset.train.views[0].camera;
    let focal = cam0.focal(img_size);
    let sod = cam0.position.length();
    let half_w = (g0.width as f32 * 0.5) * sod / focal.x;
    let half_h = (g0.height as f32 * 0.5) * sod / focal.y;
    // R0 = W/2 世界 (FOV 圆柱半径): 追踪 >R0 的点 (应是空气, refine 后大量 prune)。
    let r0 = half_w;
    // 初始采样区域: ball (球, scene_extent) 或 cylinder (绕 Z, R=half_w×scale,
    // 高度=可视高度上界 2·half_h·(1+R/SOD), 密度恒定 → 点数按体积比缩放)。
    let init = if init_shape == "cylinder" {
        let r0 = half_w;
        let r = r0 * init_radius_scale;
        let half_h_cyl = if init_height_factor > 0.0 {
            half_h * init_height_factor
        } else {
            half_h * (1.0 + r / sod)
        };
        brush_train::xray_train::InitRegion::Cylinder {
            radius: r,
            half_height: half_h_cyl,
        }
    } else {
        brush_train::xray_train::InitRegion::Ball { radius: scene_extent }
    };
    // FOV 过滤初始化: 默认只保留至少在一个视角内投影的点 (消离群点);
    // --no-fov-filter 关闭 (圆柱旋转中会重新入视野, 见实验)。
    let train_cams: Vec<_> = dataset.train.views.iter().map(|v| v.camera).collect();
    let fov = Some((train_cams.as_slice(), glam::uvec2(g0.width, g0.height)));
    let mut trainer = create_xray_trainer(cfg, points, scene_extent, init, &device, fov, None);
    // 梯度诊断只在 eval 步收集(打印 + CSV 用), 见训练循环。

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
    // 验证产物分目录: eval/nrrd (投影对比) | eval/bin (原始参数) | eval/ply (点云)。
    let eval_nrrd = out.join("eval/nrrd");
    let eval_bin = out.join("eval/bin");
    let eval_ply = out.join("eval/ply");
    std::fs::create_dir_all(&eval_nrrd)?;
    std::fs::create_dir_all(&eval_bin)?;
    std::fs::create_dir_all(&eval_ply)?;

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
        save_stack(&eval_nrrd, 0, &pairs);
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

    // ---- 训练循环(与标准流程一致, 含 density control + deform) -----------
    for iter in 0..iters {
        let step = iter + 1;
        // 梯度诊断只在 eval 步需要(打印 + CSV): 其余步关闭, 省掉每步 4 次
        // GPU→CPU readback(原先硬编码开启时的固定开销)。
        trainer.set_collect_grads(step % eval_every == 0 || step == iters);
        // loss 标量同样只在 eval 步读回: 关闭时 CPU 提交可与 GPU 执行重叠。
        trainer.set_collect_loss(step % eval_every == 0 || step == iters);
        let batch = dataloader.next_batch().await;
        let stats = trainer.step(&batch).await;

        // Density control (skip the very first step so gradients accumulate)。
        if step > 1
            && let Some(refine_stats) = trainer.maybe_refine(step).await
        {
            let (beyond, total, md) = trainer.splats_beyond_radius(r0).await;
            println!(
                "{} refine iter {step}: {} splats (added {}, split {}, pruned {}) grad_thr={} | >R0={beyond}/{total} ({:.1}%), meanμ={md:.5}",
                ts(),
                refine_stats.total_splats,
                refine_stats.num_added,
                refine_stats.num_split,
                refine_stats.num_pruned,
                refine_stats
                    .grad_threshold
                    .map_or(-1.0f32, |t| t),
                beyond as f32 / total.max(1) as f32 * 100.0,
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
            save_stack(&eval_nrrd, step, &pairs);
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
    let ply_path = eval_ply.join("canonical_final.ply");
    std::fs::write(&ply_path, ply)?;
    println!("{} exported {}", ts(), ply_path.display());

    // ---- 原始参数 .bin 导出 (无损, 无 PLY 激活域往返) --------------------
    // transforms [N,10] (means+quats+log_scales) + raw_opac [N], 均为 raw
    // 域 f32; 供 gs2volume --bin= 直接消费, 与 PLY 结果三方对比。
    {
        use burn::tensor::TensorData;
        let c = trainer.canonical();
        let t_data: Vec<f32> = c
            .transforms
            .val()
            .into_data_async()
            .await?
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("transforms read: {e}"))?;
        let o_data: Vec<f32> = c
            .raw_opacities
            .val()
            .into_data_async()
            .await?
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("raw read: {e}"))?;
        let mut tb = Vec::with_capacity(t_data.len() * 4);
        for v in &t_data {
            tb.extend_from_slice(&v.to_le_bytes());
        }
        let mut ob = Vec::with_capacity(o_data.len() * 4);
        for v in &o_data {
            ob.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(eval_bin.join("canonical_final_transforms.bin"), tb)?;
        std::fs::write(eval_bin.join("canonical_final_raw.bin"), ob)?;
        println!("{} exported {} (transforms+raw, 无激活往返)", ts(), out.display());
    }

    println!("{} done -> {}", ts(), out.display());
    Ok(())
}
