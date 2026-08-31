//! 训练配置: 单一 `FitConfig` (serde) 覆盖 fit_static / fit_deform 的全部
//! 参数 (FDK 相关剔除)。CLI 参数 → 构造 FitConfig; C FFI 传 JSON → serde 反
//! 序列化。模式敏感字段用 `Option` (None → 按模式默认), 与 fit_static /
//! fit_deform 的既有默认一致。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 训练模式: 静态重建 (无 deform) 或相位驱动动态重建。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum FitMode {
    #[default]
    Static,
    Deform,
}

/// 全部训练 / 导出参数。字段默认值 = 现有 fit_static / fit_deform CLI 默认;
/// 模式敏感字段为 `Option`, `None` 时按 [`FitConfig::resolve`] 取模式默认。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FitConfig {
    pub mode: FitMode,
    /// DICOM 序列文件路径。
    pub dcm: PathBuf,
    /// 输出目录。
    pub out: PathBuf,

    // ---- 数据预处理 ------------------------------------------------------
    /// 自动 gamma 目标灰度 (None + no_gamma=false → 0.5)。
    pub gamma_target: Option<f32>,
    /// 关闭自动 gamma。
    pub no_gamma: bool,
    /// ROI: "no" | "N" | "x0,y0,w,h" (默认 "20")。
    pub roi: String,
    /// 场景半径 (mm); None = 相机几何自动。
    pub scene_extent: Option<f32>,
    /// 加载最大分辨率 (默认 1920)。
    pub max_resolution: u32,
    /// 初始化 FOV 过滤: 只保留至少在一个视角内投影的点 (默认开)。
    pub fov_filter: bool,

    // ---- 初始化 ----------------------------------------------------------
    /// 初始化 splat 点数。
    pub points: u32,
    /// 初始化采样区域 "ball" | "cylinder" (None → static: cylinder, deform: ball)。
    pub init_shape: Option<String>,
    /// 圆柱半径 = R0 × scale (None → static: 1.5, deform: 1.0)。
    pub init_radius_scale: Option<f32>,
    /// cylinder 体积比点数基准 (R0 处点数, 默认 5000)。
    pub init_density_base: u32,
    /// >0 覆盖自动高度 (默认 0)。
    pub init_height_factor: f32,
    /// 初始密度 mm⁻¹ (None → static: 0.002, deform: 0.01)。
    pub init_density: Option<f32>,

    // ---- 优化 / 学习率 ----------------------------------------------------
    pub iters: u32,
    pub lr_mean: f64,
    pub lr_mean_end: f64,
    pub lr_scale: f64,
    pub lr_opac: f64,
    pub lr_deform: f64,
    pub lr_deform_end: f64,
    pub cosine_lr: bool,

    // ---- 形变网络 (仅 deform) ---------------------------------------------
    /// 关闭 AST 相位噪声 (默认关 = 噪声开)。
    pub no_ast: bool,
    pub warm_up: u32,
    /// "hexplane" | "hashgrid"。
    pub deform_backend: String,
    /// 预测 d_scaling (默认不预测 = 保质量形变)。
    pub predict_scaling: bool,

    // ---- HexPlane 超参 ----------------------------------------------------
    pub hex_res: u32,
    pub hex_time_res: u32,
    pub hex_features: usize,
    pub hex_mlp_width: usize,
    pub hex_mlp_layers: usize,
    pub plane_tv_weight: f32,
    pub rigid_anchor_weight: f32,

    // ---- 密度控制 / refine ------------------------------------------------
    pub growth_frac: f32,
    pub refine_every: u32,
    pub max_splats: u32,
    pub refine_until_frac: f32,
    /// 固定 densify 梯度阈值 (None → 5e-6; 与 dyn_grad_percentile 互斥)。
    pub fixed_grad_thr: Option<f32>,
    /// 动态 densify 梯度百分位 (0~1, 覆盖 fixed_grad_thr)。
    pub dyn_grad_percentile: Option<f32>,
    pub split: bool,
    /// None → static: 0.0005, deform: 0.0003。
    pub percent_dense: Option<f32>,
    /// None → 1/√2。
    pub split_scale: Option<f32>,
    /// None → 1.0 (× scene_extent)。
    pub bound_factor: Option<f32>,
    /// None → 5e-4。
    pub cull_density: Option<f32>,
    /// ≤0 = 关闭 (None → 0)。
    pub max_screen_size: Option<f32>,
    pub cull_contribution: bool,
    pub cull_percentile: f32,
    pub cull_floor: f32,
    pub min_splats: u32,
    pub density_reset: u32,

    // ---- scale 约束 (细长条抑制) -------------------------------------------
    /// 屏幕面积惩罚权重 (Brush #479, 默认 0.1 = ab_cap10_pen01 最优配置)。
    pub screen_area_penalty: f32,
    /// log 空间各向异性正则权重 (默认 0 关; 0.02 轻量版实验可选)。
    pub scale_aniso_weight: f32,
    /// log-scale 软上限 (mm; 默认 10)。
    pub scale_cap_mm: f32,
    /// log-scale 软上限正则权重 (默认 0.5)。
    pub scale_cap_weight: f32,

    // ---- 损失 --------------------------------------------------------------
    /// "l1" | "charbonnier" | "huber" | "l2"。
    pub loss: String,
    pub loss_eps: f32,
    pub loss_delta: f32,
    pub proj_weight: f32,
    pub proj_ssim_weight: f32,
    pub multiscale_weight: f32,
    pub window_weight: f32,
    pub grad_weight: f32,
    pub grad_ramp_from: u32,
    pub grad_ramp_to: u32,
    pub grad_edge_scale: f32,

    // ---- 评估与输出 ---------------------------------------------------------
    /// None → static: 500, deform: 100。
    pub eval_every: Option<u32>,
    /// 每 N 帧扣一个 held-out 视图 (None = 用 train view 0)。
    pub eval_split_every: Option<usize>,
    pub eval_views: usize,
    /// 保存 eval GT|pred NRRD stack (默认开; --no-eval 时忽略)。
    pub save_eval: bool,
    /// 关闭 eval: 不 eval/不建 eval 目录/不写 metrics.csv (省 VGG 推理与
    /// readback; 训练结束无验证指标)。
    pub eval_enabled: bool,
    /// 导出 canonical PLY 点云 (默认关)。
    pub save_ply: bool,
    /// deform 模式导出 deform_final.bin + 每相位网格场 nii.gz (默认关)。
    pub save_deform: bool,
    /// 导出原始参数 .bin (transforms+raw, 供 gs2volume --bin= 复用; 默认关)。
    pub save_bin: bool,
    /// 指标 CSV 路径 (None → <out>/metrics.csv; "off" 关闭)。
    pub log_csv: Option<PathBuf>,

    // ---- volume 导出 ---------------------------------------------------------
    /// 体素尺寸 mm (默认 1.0)。
    pub voxel_mm: f32,
    /// 显式 XY / Z 范围 mm (None → 自适应 splat bbox, cap 400mm)。
    pub extent_xy: Option<f32>,
    pub extent_z: Option<f32>,
    /// deform 网格场相位数 (默认 8)。
    pub n_deform_phases: u32,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            mode: FitMode::Static,
            dcm: PathBuf::new(),
            out: PathBuf::from("target/fit"),
            gamma_target: None,
            no_gamma: false,
            roi: "20".to_owned(),
            scene_extent: None,
            max_resolution: 1920,
            fov_filter: true,
            points: 5_000,
            init_shape: None,
            init_radius_scale: None,
            init_density_base: 5_000,
            init_height_factor: 0.0,
            init_density: None,
            iters: 10_000,
            lr_mean: 2e-5,
            lr_mean_end: 2e-6,
            lr_scale: 5e-3,
            lr_opac: 0.012,
            lr_deform: 1e-3,
            lr_deform_end: 1e-4,
            cosine_lr: false,
            no_ast: false,
            warm_up: 300,
            deform_backend: "hexplane".to_owned(),
            predict_scaling: false,
            hex_res: 64,
            hex_time_res: 32,
            hex_features: 16,
            hex_mlp_width: 128,
            hex_mlp_layers: 2,
            plane_tv_weight: 0.0,
            rigid_anchor_weight: 0.0,
            growth_frac: 1.0,
            refine_every: 400,
            max_splats: 300_000,
            refine_until_frac: 0.9,
            fixed_grad_thr: None,
            dyn_grad_percentile: None,
            split: true,
            percent_dense: None,
            split_scale: None,
            bound_factor: None,
            cull_density: None,
            max_screen_size: None,
            cull_contribution: false,
            cull_percentile: 0.05,
            cull_floor: 1e-3,
            min_splats: 0,
            density_reset: 0,
            screen_area_penalty: 0.1,
            scale_aniso_weight: 0.0,
            scale_cap_mm: 10.0,
            scale_cap_weight: 0.5,
            loss: "charbonnier".to_owned(),
            loss_eps: 1e-3,
            loss_delta: 0.1,
            proj_weight: 1.0,
            proj_ssim_weight: 0.0,
            multiscale_weight: 0.5,
            window_weight: 0.5,
            grad_weight: 0.0,
            grad_ramp_from: 3_000,
            grad_ramp_to: 0,
            grad_edge_scale: 0.03,
            eval_every: None,
            eval_split_every: None,
            eval_views: 8,
            save_eval: true,
            eval_enabled: true,
            save_ply: false,
            save_deform: false,
            save_bin: false,
            log_csv: None,
            voxel_mm: 1.0,
            extent_xy: None,
            extent_z: None,
            n_deform_phases: 8,
        }
    }
}

/// 模式敏感字段的最终值 (None → 模式默认, 与 fit_static / fit_deform 一致)。
pub struct Resolved {
    pub gamma_target: Option<f32>,
    pub init_shape: String,
    pub init_radius_scale: f32,
    pub init_density: f32,
    pub eval_every: u32,
    pub grad_threshold: crate::train::GradThr,
    pub percent_dense: f32,
    pub split_scale: f32,
    pub bound_factor: f32,
    pub cull_density: f32,
    pub max_screen_size: f32,
    pub enable_ast: bool,
    pub predict_scaling: bool,
    pub loss_type: brush_loss::gray::GrayLossType,
    pub deform_backend: brush_train::xray_train::DeformBackend,
    pub roi: brush_dataset::config::RoiSpec,
}

impl FitConfig {
    /// 解析所有模式敏感字段 (含字符串枚举)。校验失败返回 Err。
    pub fn resolve(&self) -> anyhow::Result<Resolved> {
        let deform = self.mode == FitMode::Deform;
        let gamma_target = if self.no_gamma {
            None
        } else {
            self.gamma_target.or(Some(0.5))
        };
        let init_shape = self
            .init_shape
            .clone()
            .unwrap_or_else(|| if deform { "ball" } else { "cylinder" }.to_owned());
        let init_radius_scale = self.init_radius_scale.unwrap_or(if deform { 1.0 } else { 1.5 });
        let init_density = self.init_density.unwrap_or(if deform { 0.01 } else { 0.002 });
        let eval_every = self.eval_every.unwrap_or(if deform { 100 } else { 500 });
        let grad_threshold = if let Some(pct) = self.dyn_grad_percentile {
            if !(0.0..1.0).contains(&pct) {
                anyhow::bail!("dyn_grad_percentile must be in (0,1)");
            }
            crate::train::GradThr::Dynamic(pct)
        } else {
            crate::train::GradThr::Fixed(self.fixed_grad_thr.unwrap_or(5e-6))
        };
        // clone 恢复 (2026-08-28): mm 尺度场景下 0.0003/0.0005 的 clone 阈值
        // (scene_extent×pd ≈ 0.13mm) 比 splat 实际尺度小两个数量级 → clone 死
        // 代码; 0.02 与真实尺度同量级 (实验最优 pd002_gf100)。
        let percent_dense = self.percent_dense.unwrap_or(0.02);
        let split_scale = self
            .split_scale
            .unwrap_or(std::f32::consts::FRAC_1_SQRT_2);
        let bound_factor = self.bound_factor.unwrap_or(1.0);
        let cull_density = self.cull_density.unwrap_or(5e-4);
        let max_screen_size = self.max_screen_size.unwrap_or(0.0);
        let enable_ast = !self.no_ast;
        let predict_scaling = self.predict_scaling;
        let loss_type = match self.loss.as_str() {
            "l1" => brush_loss::gray::GrayLossType::L1,
            "charbonnier" => brush_loss::gray::GrayLossType::Charbonnier,
            "huber" => brush_loss::gray::GrayLossType::Huber,
            "l2" => brush_loss::gray::GrayLossType::L2,
            other => anyhow::bail!("invalid loss '{other}' (l1|charbonnier|huber|l2)"),
        };
        let deform_backend = match self.deform_backend.as_str() {
            "hexplane" => brush_train::xray_train::DeformBackend::HexPlane,
            "hashgrid" => brush_train::xray_train::DeformBackend::HashGrid,
            other => anyhow::bail!("invalid deform_backend '{other}' (hexplane|hashgrid)"),
        };
        let roi = brush_dataset::config::parse_roi(&self.roi)
            .map_err(|e| anyhow::anyhow!("invalid roi: {e}"))?;
        Ok(Resolved {
            gamma_target,
            init_shape,
            init_radius_scale,
            init_density,
            eval_every,
            grad_threshold,
            percent_dense,
            split_scale,
            bound_factor,
            cull_density,
            max_screen_size,
            enable_ast,
            predict_scaling,
            loss_type,
            deform_backend,
            roi,
        })
    }
}
