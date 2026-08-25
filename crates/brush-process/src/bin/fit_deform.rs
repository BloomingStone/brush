//! 心动相位驱动的动态 X-ray 重建: 拟合 `images/RXA_pig_with_phase.dcm`。
//!
//! 与 `fit_static` 的区别:
//!   1. **启用 deform 模式** (`enable_deform=true`): 训练器创建
//!      [`DeformNetwork`] 生成形变场, 以心动相位 (DICOM 私有标签
//!      `(0071,1010)`, creator `(0071,0010)`
//!      `YOUR_INSTITUTION_PHASE_1.0`) 为条件, 预测每个 splat 的
//!      `(d_xyz, d_scaling, d_rotation)`, 生成动态场。
//!      - 默认后端 **HexPlane** (`--deform-backend=hexplane`): 6 个 2D 特征
//!        平面 (XY/XZ/YZ + XT/YT/ZT) 双线性采样求和 + 轻量 MLP; 时间轴
//!        **环形回绕** (phase 0 与 1 是同一相位)。比 hash-grid 快 (select
//!        56→24、MLP 9 层→5 层), 更适合低频心脏运动与 wgpu 后端。
//!      - `--deform-backend=hashgrid` 切回多分辨率 hash-grid + skip-MLP。
//!   2. **保质量形变** (默认): 不预测 `d_scaling` (`--predict-scaling` 开启),
//!      形变场只有位移+旋转 —— splat 积分吸收正比于 scale, 缩放会改变总
//!      吸收; 局部密度变化应由高斯点移动产生。
//!   3. **AST (asynchronous time) 噪声**: 训练时给 phase 加随时间衰减的
//!      高斯噪声 (参考项目 `get_linear_noise_func`), 增强相位泛化。
//!   4. **warm-up**: 前 `warm_up` 步不施加形变 (dummy 梯度), 让 canonical
//!      splats 先收敛到静态结构。
//!   5. **可学习时间条件化** (`--enable-time`, 默认关): 形变网络额外以真实
//!      物理时间 `t = f/fps` (FrameTimeVector 缺失/退化时) 为条件, 通过一个
//!      **可学习频率的傅里叶编码** 自动拟合数据中的呼吸 (及其它非周期)
//!      运动频率 —— 无需预知呼吸频率范围。心脏 phase 保持已知圆环轴。
//!
//! 每 `eval_every` 步对 held-out 视图做 eval (每个视图用其真实 phase),
//! 打印 PSNR/SSIM/LPIPS 并保存 GT|pred 拼接 NRRD stack; 训练结束导出
//! canonical splats 的 PLY。
//!
//! 用法:
//!   cargo run -p brush-process --bin fit_deform -- \
//!     images/RXA_pig_with_phase.dcm \
//!     --iters=10000 --points=30000 --refine-every=400 \
//!     --eval-split-every=5 --eval-views=8 --eval-every=500 \
//!     --fixed-grad-thr=1e-6 --split --out=target/fit_deform
//!   # HexPlane 参数 (默认): --deform-backend=hexplane \
//!   #   --hex-res=64 --hex-time-res=32 --hex-features=16 \
//!   #   --hex-mlp-width=128 --hex-mlp-layers=2

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene::SceneView;
use brush_dataset::scene_loader::SceneLoader;
use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::xray_eval::save_gray_nrrd_f32_stack;
use brush_train::xray_refine::XRayRefineConfig;
use brush_train::xray_train::{DeformBackend, XRayTrainConfig, create_xray_trainer};use brush_vfs::BrushVfs;
use brush_xray::XRaySplats;
use burn::tensor::{Device, TensorData};

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
    let args: Vec<String> = std::env::args().collect();
    let mut dcm: Option<PathBuf> = None;
    let mut iters = 10_000u32;
    // 初始化点数: 5000 (2026-08-21 扫描最优: 比 30000 少, PSNR 反而更高
    // 41.12 vs 40.63, 且快 ~1.5×)。densify 会按梯度阈值补足, 初始种子少
    // → 空气区噪声少、放置更高效。
    let mut points = 5_000u32;
    // None → 从相机几何自动计算(等中心 FOV 半径), 保证点云覆盖整个视野。
    let mut scene_extent: Option<f32> = None;
    // 自动 gamma: 让全局强度中位数映射到该目标灰度(0.5 = 中灰)。
    let mut gamma_target: Option<f32> = Some(0.5);
    // 初始密度 = 0.01 mm⁻¹ (5×水): 2026-08-21 扫描 points=5000 下甜点
    // (PSNR 41.32)。之前 0.02 (10×水) 让初始球成高密度雾, 空气 splat 密度
    // 饱和永不衰减 → pruned≈0; 降到 0.01 后空气点可衰减并被密度裁剪。
    let mut init_density = 0.01f32;
    let mut lr_mean = 2e-5f64;
    // 末期不冻结: 默认 2e-6 (cosine min / 指数末段)。
    let mut lr_mean_end = 2e-6f64;
    let mut lr_scale = 5e-3f64;
    let mut lr_opac = 0.012f64;
    // Deform 网络 LR (线性衰减到 lr_deform_end)。
    let mut lr_deform = 1e-3f64;
    let mut lr_deform_end = 1e-4f64;
    // AST (asynchronous time) 噪声: 训练时给 phase 加随时间衰减的高斯噪声。
    let mut enable_ast = true;
    // Warm-up 步数: 前 N 步不施加形变 (dummy 梯度), 让 canonical 先收敛。
    let mut warm_up = 300u32;
    // Deform 后端: hexplane (默认) 或 hashgrid。
    let mut deform_backend = DeformBackend::HexPlane;
    // HexPlane 超参 (默认: spatial=64 / time=32 / features=16 / mlp 128x2)。
    let mut hex_res = 64u32;
    let mut hex_time_res = 32u32;
    let mut hex_features = 16usize;
    let mut hex_mlp_width = 128usize;
    let mut hex_mlp_layers = 2usize;
    // HexPlane 特征平面空间 TV 权重 (0=关): 强制形变场低频/平滑, 防止
    // 退化为带限周期模式拟合投影噪声 (2026-08-25 形变场诊断)。
    let mut plane_tv_weight = 0.0f32;
    // 保质量形变 (默认): 不预测 d_scaling, 局部密度变化由位移/旋转产生。
    let mut predict_scaling = false;
    // 可学习时间条件化 (默认开): 形变网络用可学习傅里叶频率拟合呼吸等
    // 非周期运动 (无需预知呼吸频率), 与心脏 phase 圆环轴互补。呼吸明显
    // 数据 +4~5dB, 呼吸弱数据中性 (+0.04dB), 无害。
    let mut enable_time = true;
    // 时间编码参数: 频率个数 / 初始化范围 (Hz, 对数间隔)。紧生理带
    // [0.2,1.5] 最优 (2026-08-20 扫描: 39.22 vs 默认 38.90)。
    let mut time_freqs = 10usize;
    let mut time_min_freq = 0.2f32;
    let mut time_max_freq = 1.5f32;
    // 时间抖动 (秒, 高斯std; 0=关): 对 time 条件加噪, 强制形变场时间局部平滑
    // (连续视频式序列, 相邻帧形变小)。类似 AST 相位噪声。
    let mut time_jitter = 0.0f32;
    // 时间 TV 正则权重 (0=关): 惩罚 deform 在 (phase+dp,time+dt) 与
    // (phase,time) 的位移差 → 编码"相邻帧形变小", 提升 held-out 泛化。
    let mut time_tv_weight = 0.0f32;
    let mut time_tv_dp = 0.0f32;       // 相位步长 (每帧心搏推进)
    let mut time_tv_dt = 0.0125f32;    // 时间步长 = 1帧 @80fps
    let mut time_tv_sample = 1024usize; // TV 子集 splat 数
    // 每次 refine 只 densify 25% 的过阈值 splat → 平衡增长与速度。
    let mut growth_frac = 0.25f32;
    let mut refine_every = 400u32;
    // 硬性 splat 数上限: 到顶后只 prune 不再增 (原 1M, 10k 步中期就可能顶到)。
    let mut max_splats = 300_000u32;
    let mut eval_every = 100u32;
    // 保存形变场: deform 网络 ckpt (.bin) + 4D 形变场 NIfTI (d_xyz over
    // [x,y,z,phase,3])。0 = 不保存 (诊断用)。
    let mut save_deform = true;
    // 验证集: `--eval-split-every=N` 每 N 帧扣一个 held-out 视图;
    // `--eval-views=M` 每次 eval 采 M 个验证视图(均匀)。
    let mut eval_split_every: Option<usize> = None;
    let mut eval_views_count = 8usize;
    // 固定 densify 梯度阈值 (默认 5e-6 = 2026-08-24 修正: 1e-5 质量过差,
    // 5e-6 ~40k splats, LPIPS 0.192 vs 1e-5 的 0.222, 性价比甜点);
    // None = 动态百分位 (0.98pct 仅 ~16k)。
    let mut fixed_grad_thr: Option<f32> = Some(5e-6);
    // 启用 oversized 高梯度点拆分(clone-only → clone+split, 参考 RGB refine_splats)。
    let mut enable_split = false;
    // proj 域损失权重 (在 -ln(intensity) 域比较; 默认 1.0 已作为最优默认)。
    let mut proj_weight = 1.0f32;
    // proj 域 SSIM 权重 (0 = 关闭, proj 损失保持纯 L1)。
    let mut proj_ssim_weight = 0.0f32;
    // 像素级损失类型: l1 | charbonnier | huber | l2。Charbonnier 最优
    // (2026-08-21: PSNR +0.20dB, LPIPS -0.007 vs L1), X-ray 噪声更稳。
    let mut loss_type = brush_loss::gray::GrayLossType::Charbonnier;
    let mut loss_eps = 1e-3f32; // Charbonnier ε
    let mut loss_delta = 0.1f32; // Huber δ
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
    // 贡献裁剪 (默认关): 剪掉 density×屏幕面积×可见性 都低且处于最低百分位
    // 的 splat (微小/永不可见废点, 密度裁剪剪不掉)。
    let mut cull_contribution = false;
    let mut cull_percentile = 0.05f32;
    let mut cull_floor = 1e-3f32;
    let mut min_splats = 0u32;
    // 多尺度(金字塔)损失权重 (默认 0.5, 最强项)。
    let mut multiscale_weight = 0.5f32;
    // 多窗宽窗位损失权重 (默认 0.5, LPIPS 感知增强)。
    let mut window_weight = 0.5f32;
    // 梯度(Sobel 差分)损失权重 (0 = 关闭)。
    let mut grad_weight = 0.0f32;
    // 边缘加权梯度损失 ramp: [from,to] 内权重 0→grad_weight (smoothstep)。
    // 默认 3000 起步、到训练末全权 → 前期 L1/SSIM 主导, 后期集中推边缘。
    let mut grad_ramp_from = 3_000u32;
    let mut grad_ramp_to = 0u32; // 0 = total_iters
    // GT 边缘幅度加权: clamp(|∇gt|/scale, 0, 1); 0 = 纯梯度损失。
    // 默认 0.03: 高于平坦区噪声底(~0.002) ~10x, 强边缘(p99 0.02-0.05)满权。
    let mut grad_edge_scale = 0.03f32;
    // 分阶段双场训练: N 步后开训时间条件呼吸场 (学残差运动), 心电场纯相位。
    // 0 = 单场训练 (当前行为)。默认冻结心电场 (--no-respi-freeze 改为联合训练)。
    let mut respi_after = 0u32;
    let mut respi_freeze = true;
    // 密度软重置间隔 (0 = 关闭; 参考项目用 2000)。
    let mut density_reset_interval = 0u32;
    let mut out = PathBuf::from("target/fit_deform");
    // ROI 截取 (默认四边各裁 20px 去 FOV 暗边; `--roi=no` 关闭, `--roi=N`
    // 四边各裁 N, `--roi=x0,y0,w,h` 显式矩形)。像素与相机内参同步调整。
    let mut roi: brush_dataset::config::RoiSpec = brush_dataset::config::RoiSpec::Inset(20);
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
        } else if let Some(v) = a.strip_prefix("--lr-deform=") {
            lr_deform = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--lr-deform-end=") {
            lr_deform_end = v.parse()?;
        } else if a == "--no-ast" {
            enable_ast = false;
        } else if let Some(v) = a.strip_prefix("--warm-up=") {
            warm_up = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--deform-backend=") {
            deform_backend = match v {
                "hexplane" => DeformBackend::HexPlane,
                "hashgrid" => DeformBackend::HashGrid,
                _ => anyhow::bail!("invalid --deform-backend '{v}' (hexplane|hashgrid)"),
            };
        } else if let Some(v) = a.strip_prefix("--hex-res=") {
            hex_res = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--hex-time-res=") {
            hex_time_res = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--hex-features=") {
            hex_features = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--hex-mlp-width=") {
            hex_mlp_width = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--hex-mlp-layers=") {
            hex_mlp_layers = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--plane-tv-weight=") {
            plane_tv_weight = v.parse()?;
        } else if a == "--predict-scaling" {
            predict_scaling = true;
        } else if a == "--no-predict-scaling" {
            predict_scaling = false;
        } else if a == "--enable-time" {
            enable_time = true;
        } else if a == "--no-time" {
            enable_time = false;
        } else if let Some(v) = a.strip_prefix("--time-freqs=") {
            time_freqs = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-min-freq=") {
            time_min_freq = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-max-freq=") {
            time_max_freq = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-jitter=") {
            time_jitter = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-tv-weight=") {
            time_tv_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-tv-dp=") {
            time_tv_dp = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-tv-dt=") {
            time_tv_dt = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--time-tv-sample=") {
            time_tv_sample = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--growth-frac=") {
            growth_frac = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--refine-every=") {
            refine_every = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--max-splats=") {
            max_splats = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--eval-every=") {
            eval_every = v.parse()?;
        } else if a == "--no-save-deform" {
            save_deform = false;
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
        } else if let Some(v) = a.strip_prefix("--loss=") {
            loss_type = match v {
                "l1" => brush_loss::gray::GrayLossType::L1,
                "charbonnier" => brush_loss::gray::GrayLossType::Charbonnier,
                "huber" => brush_loss::gray::GrayLossType::Huber,
                "l2" => brush_loss::gray::GrayLossType::L2,
                _ => anyhow::bail!("invalid --loss '{v}' (l1|charbonnier|huber|l2)"),
            };
        } else if let Some(v) = a.strip_prefix("--loss-eps=") {
            loss_eps = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--loss-delta=") {
            loss_delta = v.parse()?;
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
        } else if a == "--cull-contribution" {
            cull_contribution = true;
        } else if a == "--no-cull-contribution" {
            cull_contribution = false;
        } else if let Some(v) = a.strip_prefix("--cull-percentile=") {
            cull_percentile = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--cull-floor=") {
            cull_floor = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--min-splats=") {
            min_splats = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--multiscale-weight=") {
            multiscale_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--window-weight=") {
            window_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--grad-weight=") {
            grad_weight = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--grad-ramp-from=") {
            grad_ramp_from = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--grad-ramp-to=") {
            grad_ramp_to = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--grad-edge-scale=") {
            grad_edge_scale = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--respi-after=") {
            respi_after = v.parse()?;
        } else if a == "--no-respi-freeze" {
            respi_freeze = false;
        } else if let Some(v) = a.strip_prefix("--density-reset=") {
            density_reset_interval = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--log-csv=") {
            log_csv = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if let Some(v) = a.strip_prefix("--roi=") {
            // --roi=no | N | x0,y0,w,h (像素): 截取图像 ROI 并同步调整相机内参
            // (fov/主点), 用于去除 FOV 暗边或局部区域重建。默认四边各裁 20。
            roi = brush_dataset::config::parse_roi(v).map_err(anyhow::Error::msg)?;
        } else if dcm.is_none() {
            dcm = Some(PathBuf::from(a));
        }
        i += 1;
    }
    let dcm = dcm.expect(
        "usage: fit_deform <dcm> [--iters=N] [--points=N] [--scene-extent=MM] \
         [--gamma-target=G] [--init-density=MU] [--lr-mean=LR] \
         [--lr-mean-end=LR] [--lr-scale=LR] [--lr-opac=LR] \
         [--lr-deform=LR] [--lr-deform-end=LR] [--no-ast] [--warm-up=N] \
         [--deform-backend=hexplane|hashgrid] [--hex-res=N] \
         [--hex-time-res=N] [--hex-features=N] [--hex-mlp-width=N] \
         [--hex-mlp-layers=N] [--plane-tv-weight=W] [--predict-scaling|--no-predict-scaling] \
         [--growth-frac=F] [--refine-every=N] [--max-splats=N] \
         [--eval-split-every=N] \
         [--eval-views=M] [--fixed-grad-thr=F] [--split] [--proj-weight=W] \
         [--proj-ssim-weight=S] [--cosine-lr] [--percent-dense=F] \
         [--split-scale=F] [--bound-factor=F] [--cull-density=MU] \
         [--max-screen-size=PX] [--multiscale-weight=W] [--window-weight=W] \
         [--grad-weight=W] [--grad-ramp-from=N] [--grad-ramp-to=N] \
         [--grad-edge-scale=S] [--respi-after=N] [--no-respi-freeze] \
         [--time-jitter=S] [--time-tv-weight=W] [--time-tv-dp=S] \
         [--time-tv-dt=S] [--time-tv-sample=N] [--density-reset=N] [--eval-every=N] \
         [--roi=no|N|x0,y0,w,h] [--log-csv=FILE] [--out=DIR]",
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
        roi,
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

    // ---- Trainer: 随机初始化 + deform 模式 + 启用 refine -----------------
    // `create_xray_trainer` 做随机球内点云(KNN scale + μ_water 密度初始化),
    // 并把 config.refine.scene_extent 设为 scene_extent; `enable_deform=true`
    // 时按 `deform_backend` 创建 HexPlane (默认) 或 hash-grid 形变网络
    // (coord_scale = scene_extent)。
    let mut cfg = XRayTrainConfig::default();
    cfg.total_iters = iters;
    cfg.enable_deform = true; // 心动相位驱动的动态场
    cfg.enable_ast = enable_ast; // 异步时间噪声 (默认开)
    cfg.warm_up = warm_up; // 前 N 步不施加形变
    cfg.deform_backend = deform_backend;
    cfg.predict_scaling = predict_scaling; // 保质量形变 (默认关缩放)
    cfg.enable_time = enable_time; // 可学习时间条件化 (默认关)
    cfg.time_jitter = time_jitter; // 时间抖动 (连续视频平滑)
    cfg.time_tv_weight = time_tv_weight; // 时间 TV 正则 (相邻帧形变小)
    cfg.time_tv_dp = time_tv_dp;
    cfg.time_tv_dt = time_tv_dt;
    cfg.time_tv_sample = time_tv_sample;
    cfg.time_enc = brush_deform::TimeEncodingConfig {
        n_freqs: time_freqs,
        min_freq: time_min_freq,
        max_freq: time_max_freq,
        ..brush_deform::TimeEncodingConfig::default()
    };
    cfg.hex_plane = HexPlaneDeformConfig {
        hex_plane: HexPlaneConfig {
            n_feature_dim: hex_features,
            spatial_resolution: hex_res,
            time_resolution: hex_time_res,
            ..HexPlaneConfig::default()
        },
        mlp_hidden: hex_mlp_width,
        mlp_layers: hex_mlp_layers,
        predict_scaling,
        enable_time,
        time_enc: brush_deform::TimeEncodingConfig {
            n_freqs: time_freqs,
            min_freq: time_min_freq,
            max_freq: time_max_freq,
            ..brush_deform::TimeEncodingConfig::default()
        },
        plane_tv_weight,
    };
    cfg.init_density = init_density;
    cfg.lr_mean = lr_mean;
    cfg.lr_mean_end = lr_mean_end;
    cfg.lr_scale = lr_scale;
    cfg.lr_opac = lr_opac;
    cfg.lr_deform = lr_deform;
    cfg.lr_deform_end = lr_deform_end;
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
    cfg.respi_after = respi_after;
    cfg.respi_freeze = respi_freeze;
    cfg.time_jitter = time_jitter;
    cfg.time_tv_weight = time_tv_weight;
    cfg.time_tv_dp = time_tv_dp;
    cfg.time_tv_dt = time_tv_dt;
    cfg.time_tv_sample = time_tv_sample;
    cfg.refine = XRayRefineConfig {
        refine_every,
        scene_extent,
        growth_select_fraction: growth_frac,
        fixed_grad_threshold: fixed_grad_thr,
        enable_split,
        density_reset_interval,
        percent_dense: percent_dense.unwrap_or(0.0003),
        split_scale_factor: split_scale.unwrap_or(std::f32::consts::FRAC_1_SQRT_2),
        max_bound_factor: bound_factor.unwrap_or(3.0),
        // prune 阈值 5e-4 (25% 水密度): 2026-08-21 高阈值扫描 LPIPS 最优
        // (0.4658 vs 2e-4 的 0.4719), PSNR 持平, 且剪掉更多空气废点。
        cull_density_threshold: cull_density.unwrap_or(5e-4),
        max_splats,
        max_screen_size: max_screen_size.unwrap_or(0.0),
        cull_contribution,
        cull_contribution_percentile: cull_percentile,
        cull_contribution_floor: cull_floor,
        min_splats,
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
    let backend_name = match deform_backend {
        DeformBackend::HexPlane => {
            format!(
                "hexplane rs={} rt={} C={} mlp {}x{}",
                hex_res, hex_time_res, hex_features, hex_mlp_width, hex_mlp_layers
            )
        }
        DeformBackend::HashGrid => "hashgrid".to_owned(),
    };
    println!(
        "{} init splats: {} (random ball r={}mm, init μ={} mm⁻¹, lr_mean={}->{}, lr_deform={}->{}), deform={} (predict_scaling={}, enable_time={}, time_freqs={}[{}-{}Hz]), refine every {}{}",
        ts(),
        trainer.num_splats(),
        scene_extent,
        init_density,
        lr_mean,
        lr_mean_end,
        lr_deform,
        lr_deform_end,
        backend_name,
        predict_scaling,
        enable_time,
        time_freqs,
        time_min_freq,
        time_max_freq,
        refine_every,
        if respi_after > 0 {
            format!(", respi staged @ {respi_after} (cardiac phase-only, respi time-only)")
        } else {
            String::new()
        }
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

    // ---- 导出形变场 (诊断) ---------------------------------------------
    // deform 网络 ckpt (.bin) + 每个相位一个 4D 形变场 NIfTI
    // [nx,ny,nz,3] 向量场 (位移 = 世界坐标 mm), affine 随 nii 保存。
    // 检查形变场是否合理建模运动 (参考 GS-dev-contrast-flow x_ray_saver.py)。
    if save_deform
        && let Some(deform) = trainer.deform()
    {
        use burn::module::Module;
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
        let ckpt = out.join("deform_final.bin");
        let record = deform.clone().into_record();
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(record, ckpt.clone())
            .map_err(|e| anyhow::anyhow!("save deform ckpt: {e}"))?;
        println!("{} saved deform ckpt -> {}", ts(), ckpt.display());

        // 网格范围: canonical splat 均值包围盒 (+5% 边距)。
        let means = trainer.canonical().means().into_data_async().await?;
        let mv: Vec<f32> = means.to_vec()?;
        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
        for c in mv.chunks_exact(3) {
            for d in 0..3 {
                mn[d] = mn[d].min(c[d]);
                mx[d] = mx[d].max(c[d]);
            }
        }
        let pad = 0.05;
        let grid = [64usize, 64, 48];
        for d in 0..3 {
            let span = (mx[d] - mn[d]).max(1e-3);
            mn[d] -= span * pad;
            mx[d] += span * pad;
        }
        // affine: 由 bbox 构造 (origin=mn, 间距=(mx-mn)/(n-1)), 随 nii 保存。
        let spacing = [
            (mx[0] - mn[0]) / (grid[0] - 1) as f32,
            (mx[1] - mn[1]) / (grid[1] - 1) as f32,
            (mx[2] - mn[2]) / (grid[2] - 1) as f32,
        ];
        // 采样 8 个相位 × 固定 time=0, 每相位单独存 4D [nx,ny,nz,3]。
        let n_phase = 8usize;
        let per_xyz = grid[0] * grid[1] * grid[2];
        let deform_dev = trainer.canonical().device().clone();
        let deform_ad = deform_dev.autodiff();
        let time = 0.0f32;
        use burn::tensor::Tensor;
        // 预构建网格点 (固定)。
        let mut pts: Vec<f32> = Vec::with_capacity(per_xyz * 3);
        for iz in 0..grid[2] {
            let z = mn[2] + (mx[2] - mn[2]) * iz as f32 / (grid[2] - 1) as f32;
            for iy in 0..grid[1] {
                let y = mn[1] + (mx[1] - mn[1]) * iy as f32 / (grid[1] - 1) as f32;
                for ix in 0..grid[0] {
                    let x = mn[0] + (mx[0] - mn[0]) * ix as f32 / (grid[0] - 1) as f32;
                    pts.extend_from_slice(&[x, y, z]);
                }
            }
        }
        let n_pts = pts.len() / 3;
        let xyz_t = Tensor::<2>::from_data(TensorData::new(pts, [n_pts, 3]), &deform_ad);
        for p in 0..n_phase {
            let phase = p as f32 / n_phase as f32;
            let phase_t = Tensor::<2>::from_data(
                TensorData::new(vec![phase; n_pts], [n_pts, 1]),
                &deform_ad,
            );
            let time_t = Tensor::<2>::from_data(
                TensorData::new(vec![time; n_pts], [n_pts, 1]),
                &deform_ad,
            );
            let d = deform.forward(xyz_t.clone(), phase_t, time_t).d_xyz;
            let dv: Vec<f32> = d.into_data_async().await?.to_vec()?;
            debug_assert_eq!(dv.len(), per_xyz * 3, "deform field size mismatch");
            let nii = out.join(format!("deform_field_phase{p:02}.nii.gz"));
            write_nifti_vec4d(&nii, &dv, grid, spacing, mn)?;
            println!("{} saved deform field (phase={phase:.3}) -> {}", ts(), nii.display());
        }
    }

    // 可学习时间频率诊断: 训练后网络把频率收敛到数据中的真实运动频率
    // (如 ~0.8Hz 呼吸)。对照人工统计核验。
    if enable_time
        && let Some(freqs) = trainer.learned_time_freqs().await
    {
        let sorted = {
            let mut s = freqs.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            s
        };
        println!(
            "{} learned time freqs (Hz): {}",
            ts(),
            sorted
                .iter()
                .map(|f| format!("{f:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    // 分阶段模式: 打印呼吸场学习到的频率 (诊断呼吸收敛)。
    if respi_after > 0
        && let Some(freqs) = trainer.respi_learned_time_freqs().await
    {
        let sorted = {
            let mut s = freqs.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            s
        };
        println!(
            "{} respi learned time freqs (Hz): {}",
            ts(),
            sorted
                .iter()
                .map(|f| format!("{f:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    println!("{} done -> {}", ts(), out.display());
    Ok(())
}

/// 写一个 NIfTI-1 (little-endian) 5D float32 位移场 `[nx, ny, nz, 1, 3]`
/// (第4维 = 单例时间 1, 第5维 = 3 个位移分量)。此结构让工具识别为
/// **向量/位移场** (参考 ASOCA dvf/phase_00.nii.gz: dims [x,y,z,1,3])。
/// 位移为世界坐标 mm; affine 随 nii 保存 (sform)。
fn write_nifti_vec4d(
    path: &Path,
    data: &[f32],
    grid: [usize; 3],
    spacing: [f32; 3],
    origin: [f32; 3],
) -> anyhow::Result<()> {
    let [nx, ny, nz] = grid;
    assert_eq!(data.len(), nx * ny * nz * 3);
    let mut hdr = [0u8; 348];
    hdr[0..4].copy_from_slice(&(348i32).to_le_bytes()); // sizeof_hdr
    hdr[40..42].copy_from_slice(&(5i16).to_le_bytes()); // dim[0] = 5
    let dims = [nx as i16, ny as i16, nz as i16, 1i16, 3i16];
    for (i, d) in dims.iter().enumerate() {
        hdr[42 + i * 2..44 + i * 2].copy_from_slice(&d.to_le_bytes()); // dim[1..5]
    }
    hdr[70..72].copy_from_slice(&(16i16).to_le_bytes()); // datatype = float32
    hdr[72..74].copy_from_slice(&(32i16).to_le_bytes()); // bitpix
    hdr[76..80].copy_from_slice(&1.0f32.to_le_bytes()); // pixdim[0] = qfac 1
    let pixdims = [
        1.0f32, // pixdim[0]
        spacing[0], spacing[1], spacing[2],
        1.0f32, // 时间轴
        1.0f32, // 分量轴
        1.0f32, 1.0f32,
    ];
    for (i, v) in pixdims.iter().enumerate() {
        hdr[76 + i * 4..80 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    hdr[108..112].copy_from_slice(&(352f32).to_le_bytes()); // vox_offset
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes()); // scl_slope
    // 仿射随 nii 保存: sform_code=2 (aligned), qform_code=0 (与参考一致)。
    hdr[252..254].copy_from_slice(&(0i16).to_le_bytes()); // qform_code
    hdr[254..256].copy_from_slice(&(2i16).to_le_bytes()); // sform_code
    let rows = [
        [spacing[0], 0.0f32, 0.0f32, origin[0]],
        [0.0f32, spacing[1], 0.0f32, origin[1]],
        [0.0f32, 0.0f32, spacing[2], origin[2]],
    ];
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            hdr[280 + r * 16 + c * 4..284 + r * 16 + c * 4]
                .copy_from_slice(&v.to_le_bytes());
        }
    }
    hdr[268..272].copy_from_slice(&origin[0].to_le_bytes()); // qoffset_x
    hdr[272..276].copy_from_slice(&origin[1].to_le_bytes()); // qoffset_y
    hdr[276..280].copy_from_slice(&origin[2].to_le_bytes()); // qoffset_z
    hdr[344..348].copy_from_slice(b"n+1\0"); // magic

    // 写 .nii.gz (gzip)
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut gz = GzEncoder::new(f, Compression::default());
    gz.write_all(&hdr)?;
    gz.write_all(&[0u8; 4])?; // pad to vox_offset 352
    for v in data {
        gz.write_all(&v.to_le_bytes())?;
    }
    gz.finish()?;
    Ok(())
}