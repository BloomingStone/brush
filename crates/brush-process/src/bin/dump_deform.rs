//! 诊断工具: 加载训练的 deform ckpt (deform_final.bin), 以任意网格分辨率
//! 重导出形变场, 判断"高频"是真实还是导出欠采样 (HexPlane 单元 vs 采样间距)。
//!
//! 用法:
//!   env -u DISPLAY CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(0)' \
//!     ./target/release/dump_deform --ckpt=experiments/output/.../deform_final.bin \
//!     --scene-extent=153 --spacing=1.2 --out=/tmp/dump_1p2mm
//!
//! 输出: <out>_phase{p:02}.nii.gz + 平滑度统计 (相邻体素差/位移比, 自相关)。

use std::path::PathBuf;

use anyhow::{Context, Result};
use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig, HexPlaneDeformModel};
use brush_train::xray_train::DeformNetwork;
use burn::module::{Module, Param};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Device, Tensor, TensorData};

fn write_nifti_vec4d_gz(
    path: &std::path::Path,
    data: &[f32],
    grid: [usize; 3],
    spacing: [f32; 3],
    origin: [f32; 3],
) -> Result<()> {
    let [nx, ny, nz] = grid;
    let mut hdr = [0u8; 348];
    hdr[0..4].copy_from_slice(&(348i32).to_le_bytes());
    hdr[40..42].copy_from_slice(&(5i16).to_le_bytes());
    let dims = [nx as i16, ny as i16, nz as i16, 1i16, 3i16];
    for (i, d) in dims.iter().enumerate() {
        hdr[42 + i * 2..44 + i * 2].copy_from_slice(&d.to_le_bytes());
    }
    hdr[70..72].copy_from_slice(&(16i16).to_le_bytes());
    hdr[72..74].copy_from_slice(&(32i16).to_le_bytes());
    let pixdims = [1.0f32, spacing[0], spacing[1], spacing[2], 1.0, 1.0, 1.0, 1.0];
    for (i, v) in pixdims.iter().enumerate() {
        hdr[76 + i * 4..80 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    hdr[108..112].copy_from_slice(&(352f32).to_le_bytes());
    hdr[112..116].copy_from_slice(&1.0f32.to_le_bytes());
    hdr[252..254].copy_from_slice(&(0i16).to_le_bytes());
    hdr[254..256].copy_from_slice(&(2i16).to_le_bytes());
    let rows = [
        [spacing[0], 0.0, 0.0, origin[0]],
        [0.0, spacing[1], 0.0, origin[1]],
        [0.0, 0.0, spacing[2], origin[2]],
    ];
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            hdr[280 + r * 16 + c * 4..284 + r * 16 + c * 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    hdr[268..272].copy_from_slice(&origin[0].to_le_bytes());
    hdr[272..276].copy_from_slice(&origin[1].to_le_bytes());
    hdr[276..280].copy_from_slice(&origin[2].to_le_bytes());
    hdr[344..348].copy_from_slice(b"n+1\0");
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let f = std::fs::File::create(path)?;
    let mut gz = GzEncoder::new(f, Compression::default());
    gz.write_all(&hdr)?;
    gz.write_all(&[0u8; 4])?;
    for v in data {
        gz.write_all(&v.to_le_bytes())?;
    }
    gz.finish()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut ckpt = PathBuf::new();
    let mut scene_extent = 153.0f32;
    let mut spacing = 1.2f32;
    let mut out = PathBuf::from("/tmp/dump");
    let mut hex_res = 64usize;
    let mut hex_time = 32usize;
    let mut hex_feat = 16usize;
    let mut mlp_w = 128usize;
    let mut mlp_l = 2usize;
    let mut phase = 0.0f32;
    let mut time = 0.0f32;
    let mut n_phase = 1usize;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(s) = a.strip_prefix("--ckpt=") { ckpt = PathBuf::from(s); }
        else if let Some(s) = a.strip_prefix("--scene-extent=") { scene_extent = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--spacing=") { spacing = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--out=") { out = PathBuf::from(s); }
        else if let Some(s) = a.strip_prefix("--hex-res=") { hex_res = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--hex-time-res=") { hex_time = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--hex-features=") { hex_feat = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--mlp-width=") { mlp_w = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--mlp-layers=") { mlp_l = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--phase=") { phase = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--time=") { time = s.parse()?; }
        else if let Some(s) = a.strip_prefix("--n-phase=") { n_phase = s.parse()?; }
        i += 1;
    }
    anyhow::ensure!(!ckpt.as_os_str().is_empty(), "need --ckpt=...");

    let wgpu = brush_process::burn_init_setup().await;
    let device: Device = wgpu.into();
    let device_ad = device.clone().autodiff();

    let cfg = HexPlaneDeformConfig {
        hex_plane: HexPlaneConfig {
            n_feature_dim: hex_feat,
            spatial_resolution: hex_res as u32,
            time_resolution: hex_time as u32,
            coord_scale: scene_extent,
            ..HexPlaneConfig::default()
        },
        mlp_hidden: mlp_w,
        mlp_layers: mlp_l,
        predict_scaling: false,
        enable_time: true,
        time_enc: brush_deform::TimeEncodingConfig {
            n_freqs: 10,
            min_freq: 0.2,
            max_freq: 1.5,
            ..Default::default()
        },
        plane_tv_weight: 0.0,
    };
    let model = HexPlaneDeformModel::new(cfg, &device_ad);
    // ckpt 是 DeformNetwork 枚举记录 (含 HexPlane/HashGrid variant tag),
    // 先按枚举记录加载, 再解包出 HexPlane 子记录。
    type DeformRec = <DeformNetwork as burn::module::Module>::Record;
    let rec: DeformRec =
        BinFileRecorder::<FullPrecisionSettings>::new().load(ckpt.clone(), &device_ad)
            .map_err(|e| anyhow::anyhow!("load {ckpt:?}: {e}"))?;
    let hex_rec = match rec {
        DeformRec::HexPlane(r) => r,
        _ => anyhow::bail!("ckpt is not a HexPlane deform (got HashGrid)"),
    };
    let model = model.load_record(hex_rec);
    println!("loaded ckpt {} (coord_scale={scene_extent}, rs={hex_res})", ckpt.display());

    // 网格: 包围盒 = scene_extent*2 立方 (与训练一致), 由 spacing 决定分辨率。
    let span = 2.0 * scene_extent;
    let grid = [
        (span / spacing).ceil() as usize,
        (span / spacing).ceil() as usize,
        (span / spacing).ceil() as usize,
    ];
    let origin = [-scene_extent, -scene_extent, -scene_extent];
    println!("grid {}^3, spacing {spacing}mm (HexPlane cell {:.2}mm, 每单元 {:.1} 采样)",
        grid[0], 2.0 * scene_extent / hex_res as f32, (2.0 * scene_extent / hex_res as f32) / spacing);

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
    let n_pts = pts.len() / 3;
    let xyz_t = Tensor::<2>::from_data(TensorData::new(pts, [n_pts, 3]), &device_ad);

    for p in 0..n_phase {
        let ph = phase + p as f32 / n_phase.max(1) as f32 * 0.25;
        let phase_t = Tensor::<2>::from_data(TensorData::new(vec![ph; n_pts], [n_pts, 1]), &device_ad);
        let time_t = Tensor::<2>::from_data(TensorData::new(vec![time; n_pts], [n_pts, 1]), &device_ad);
        let d = model.forward(xyz_t.clone(), phase_t, time_t).d_xyz;
        let dv: Vec<f32> = d.into_data_async().await?.to_vec()?;
        let nii = out.with_file_name(format!("{}_phase{p:02}.nii.gz", out.file_name().unwrap().to_string_lossy()));
        write_nifti_vec4d_gz(&nii, &dv, grid, [spacing; 3], origin)?;
        // 平滑度: 相邻体素位移差 / 平均位移, 及自相关
        let mut mag: Vec<f64> = dv.chunks_exact(3).map(|c| ((c[0] as f64).powi(2) + (c[1] as f64).powi(2) + (c[2] as f64).powi(2)).sqrt()).collect();
        let mean_mag = mag.iter().sum::<f64>() / mag.len() as f64;
        // x 方向相邻差
        let (nx_, ny_, nz_) = (grid[0], grid[1], grid[2]);
        let mut diff_sum = 0.0; let mut cnt = 0;
        for iz in 0..nz_ { for iy in 0..ny_ { for ix in 0..nx_.saturating_sub(1) {
            let a = (iz*ny_ + iy)*nx_ + ix;
            diff_sum += (mag[a+1] - mag[a]).abs(); cnt += 1;
        }}}
        // 自相关 (x 方向 lag1)
        let dx: Vec<f64> = dv.chunks_exact(3).map(|c| c[0] as f64).collect();
        let m = dx.iter().sum::<f64>() / dx.len() as f64;
        let mut var=0.0; let mut cov=0.0;
        for iz in 0..nz_ { for iy in 0..ny_ { for ix in 0..nx_.saturating_sub(1) {
            let a=(iz*ny_+iy)*nx_+ix; var+=(dx[a]-m).powi(2); cov+=(dx[a]-m)*(dx[a+1]-m);
        }}}
        var += (dx[dx.len()-1]-m).powi(2);
        let ac = cov/var;
        println!("phase={ph:.3}: 平均位移 {mean_mag:.2}mm, 相邻差 {:.3}mm (比 {:.0}%), 自相关 lag1 {ac:.3}",
            diff_sum/cnt as f64, 100.0*diff_sum/cnt as f64/mean_mag);
        mag.clear();
    }
    println!("done -> {}", out.display());
    Ok(())
}
