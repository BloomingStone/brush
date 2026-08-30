//! 形变场导出工具 (独立进程, 干净 GPU 上下文): 加载 deform_final.bin,
//! 在 `2·scene_extent` 立方网格上按相位 forward, 输出行主序 npy + 5D nii.gz。
//!
//! 训练后内存池状态不稳定 (整网格 matmul autotune 曾 OOM / 内存池损坏,
//! 见 fit_deform 注释), 因此由 brush-fit 训练结束时 spawn 本工具:
//!   brush-fit-dump --ckpt=deform_final.bin --scene-extent=264 --spacing=4.12 \
//!       --n-phase=8 --out=<stem> [--hex-* --predict-scaling --enable-time ...]
//!
//! 输出: <out>_phase{p:02}.npy (行主序 [nx,ny,nz,3]) + .nii.gz (5D [x,y,z,1,3])。

use std::path::PathBuf;

use anyhow::Result;
use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig, HexPlaneDeformModel};
use brush_train::xray_train::DeformNetwork;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Device, Tensor, TensorData};

use brush_fit::export::{write_nifti_vec5d_f32, write_npy_f32};

fn parse_args() -> anyhow::Result<DumpArgs> {
    let mut a = DumpArgs::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let s = &args[i];
        if let Some(v) = s.strip_prefix("--ckpt=") {
            a.ckpt = PathBuf::from(v);
        } else if let Some(v) = s.strip_prefix("--scene-extent=") {
            a.scene_extent = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--spacing=") {
            a.spacing = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--n-phase=") {
            a.n_phase = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--out=") {
            a.out = PathBuf::from(v);
        } else if let Some(v) = s.strip_prefix("--hex-res=") {
            a.hex_res = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--hex-time-res=") {
            a.hex_time = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--hex-features=") {
            a.hex_feat = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--mlp-width=") {
            a.mlp_w = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--mlp-layers=") {
            a.mlp_l = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--predict-scaling") {
            a.predict_scaling = true;
        } else if let Some(v) = s.strip_prefix("--time-freqs=") {
            a.time_freqs = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--time-min-freq=") {
            a.time_min_freq = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--time-max-freq=") {
            a.time_max_freq = v.parse()?;
        } else if let Some(v) = s.strip_prefix("--no-time") {
            a.enable_time = false;
        }
        i += 1;
    }
    anyhow::ensure!(!a.ckpt.as_os_str().is_empty(), "need --ckpt=...");
    Ok(a)
}

struct DumpArgs {
    ckpt: PathBuf,
    scene_extent: f32,
    spacing: f32,
    out: PathBuf,
    n_phase: usize,
    hex_res: u32,
    hex_time: u32,
    hex_feat: usize,
    mlp_w: usize,
    mlp_l: usize,
    predict_scaling: bool,
    enable_time: bool,
    time_freqs: usize,
    time_min_freq: f32,
    time_max_freq: f32,
}

impl Default for DumpArgs {
    fn default() -> Self {
        Self {
            ckpt: PathBuf::new(),
            scene_extent: 153.0,
            spacing: 1.2,
            out: PathBuf::from("/tmp/dump"),
            n_phase: 8,
            hex_res: 64,
            hex_time: 32,
            hex_feat: 16,
            mlp_w: 128,
            mlp_l: 2,
            predict_scaling: false,
            enable_time: true,
            time_freqs: 10,
            time_min_freq: 0.2,
            time_max_freq: 1.5,
        }
    }
}

fn main() -> Result<()> {
    let a = parse_args()?;

    // 训练模式会跳过 wgpu 析构 panic, 这里正常结束即可。
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio");
    rt.block_on(run(&a))
}

async fn run(a: &DumpArgs) -> Result<()> {
    let wgpu = brush_fit::burn_init_setup().await;
    let device: Device = wgpu.into();
    let device_ad = device.clone().autodiff();

    let cfg = HexPlaneDeformConfig {
        hex_plane: HexPlaneConfig {
            n_feature_dim: a.hex_feat,
            spatial_resolution: a.hex_res,
            time_resolution: a.hex_time,
            coord_scale: a.scene_extent,
            ..HexPlaneConfig::default()
        },
        mlp_hidden: a.mlp_w,
        mlp_layers: a.mlp_l,
        predict_scaling: a.predict_scaling,
        enable_time: a.enable_time,
        time_enc: brush_deform::TimeEncodingConfig {
            n_freqs: a.time_freqs,
            min_freq: a.time_min_freq,
            max_freq: a.time_max_freq,
            ..Default::default()
        },
        plane_tv_weight: 0.0,
        rigid_anchor_weight: 0.0,
    };
    let model = HexPlaneDeformModel::new(cfg, &device_ad);
    // ckpt 是 DeformNetwork 枚举记录, 先按枚举加载再解包 HexPlane 子记录。
    type DeformRec = <DeformNetwork as burn::module::Module>::Record;
    let rec: DeformRec =
        BinFileRecorder::<FullPrecisionSettings>::new().load(a.ckpt.clone(), &device_ad)
            .map_err(|e| anyhow::anyhow!("load {:?}: {e}", a.ckpt))?;
    let hex_rec = match rec {
        DeformRec::HexPlane(r) => r,
        _ => anyhow::bail!("ckpt is not a HexPlane deform (got HashGrid)"),
    };
    let model = model.load_record(hex_rec);
    println!(
        "[dump] loaded {} (coord_scale={}, rs={})",
        a.ckpt.display(),
        a.scene_extent,
        a.hex_res
    );

    // 网格: 包围盒 = scene_extent*2 立方, 由 spacing 决定分辨率。
    let span = 2.0 * a.scene_extent;
    let grid = [
        (span / a.spacing).ceil() as usize,
        (span / a.spacing).ceil() as usize,
        (span / a.spacing).ceil() as usize,
    ];
    let origin = [-a.scene_extent, -a.scene_extent, -a.scene_extent];
    println!(
        "[dump] grid {}^3, spacing {}mm (cell {:.2}mm, 每单元 {:.1} 采样)",
        grid[0],
        a.spacing,
        2.0 * a.scene_extent / a.hex_res as f32,
        (2.0 * a.scene_extent / a.hex_res as f32) / a.spacing
    );

    let per_xyz = grid[0] * grid[1] * grid[2];
    let mut pts = Vec::with_capacity(per_xyz * 3);
    for iz in 0..grid[2] {
        let z = origin[2] + span * iz as f32 / (grid[2] - 1) as f32;
        for iy in 0..grid[1] {
            let y = origin[1] + span * iy as f32 / (grid[1] - 1) as f32;
            for ix in 0..grid[0] {
                let x = origin[0] + span * ix as f32 / (grid[0] - 1) as f32;
                pts.extend_from_slice(&[x, y, z]);
            }
        }
    }
    // 整网格一次 forward (与 fit_deform 的 dump_deform 一致; 分批 forward
    // 的多次 autotune 反而在训练后内存池状态下不稳定)。
    let mut paths = Vec::new();
    for p in 0..a.n_phase {
        let ph = if a.n_phase > 1 {
            p as f32 / a.n_phase as f32
        } else {
            0.0
        };
        let xyz_t = Tensor::<2>::from_data(TensorData::new(pts.clone(), [per_xyz, 3]), &device_ad);
        let phase_t = Tensor::<2>::from_data(TensorData::new(vec![ph; per_xyz], [per_xyz, 1]), &device_ad);
        let time_t = Tensor::<2>::from_data(TensorData::new(vec![0.0f32; per_xyz], [per_xyz, 1]), &device_ad);
        let d = model.forward(xyz_t, phase_t, time_t).d_xyz;
        let dv: Vec<f32> = d.into_data_async().await?.to_vec()?;
        let stem = a
            .out
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "deform_field".to_owned());
        let stem = format!("{stem}_phase{p:02}");
        let npy = a.out.with_file_name(format!("{stem}.npy"));
        write_npy_f32(&npy, &dv, &[grid[0], grid[1], grid[2], 3])?;
        let nii = a.out.with_file_name(format!("{stem}.nii.gz"));
        write_nifti_vec5d_f32(&nii, &dv, grid, [a.spacing; 3], origin)?;
        println!("[dump] phase={ph:.3} -> {} / {}", npy.display(), nii.display());
        paths.push(nii);
    }
    println!("[dump] done -> {}", a.out.display());
    Ok(())
}
