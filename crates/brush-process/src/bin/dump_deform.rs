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

use anyhow::Result;
use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig, HexPlaneDeformModel};
use brush_train::xray_train::DeformNetwork;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Device, Tensor, TensorData};

/// 写一个 NumPy .npy (version 1.0) float32 数组, **行主序 (C-order)**。
fn write_npy_f32(path: &std::path::Path, data: &[f32], shape: &[usize]) -> Result<()> {
    use std::io::Write;
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "npy data len mismatch");
    let shape_str = format!(
        "({}{})",
        shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
        if shape.len() == 1 { "," } else { "" }
    );
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    let pad = (16 - (10 + header.len() + 1) % 16) % 16;
    for _ in 0..pad {
        header.push(' ');
    }
    header.push('\n');
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    out.write_all(b"\x93NUMPY")?;
    out.write_all(&[1u8, 0u8])?;
    out.write_all(&(header.len() as u16).to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    for v in data {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

/// nifti-rs: 写 3D float32 卷 `[a, b, c]` (特征平面: 前两维空间, 第三维通道)。
fn write_nifti_3d_f32(path: &std::path::Path, data: &[f32], grid: [usize; 3]) -> Result<()> {
    use nifti::writer::WriterOptions;
    use nifti::NiftiHeader;
    let [a, b, c] = grid;
    let arr = ndarray::Array3::from_shape_vec((a, b, c), data.to_vec())
        .map_err(|e| anyhow::anyhow!("ndarray shape: {e}"))?;
    let mut hdr = NiftiHeader::default();
    hdr.dim[0] = 3;
    hdr.dim[1] = a as u16;
    hdr.dim[2] = b as u16;
    hdr.dim[3] = c as u16;
    hdr.datatype = nifti::NiftiType::Float32 as i16;
    hdr.bitpix = 32;
    hdr.pixdim[0] = 1.0;
    hdr.pixdim[1] = 1.0;
    hdr.pixdim[2] = 1.0;
    hdr.pixdim[3] = 1.0;
    hdr.qform_code = 0;
    hdr.sform_code = 0;
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .map_err(|e| anyhow::anyhow!("write nifti: {e}"))?;
    Ok(())
}

/// nifti-rs: 写位移场 5D `[nx, ny, nz, 1, 3]` (同 ASOCA dvf), affine 由
/// spacing/origin 构造 (sform_code=2)。
fn write_nifti_vec5d_f32(
    path: &std::path::Path,
    data: &[f32],
    grid: [usize; 3],
    spacing: [f32; 3],
    origin: [f32; 3],
) -> Result<()> {
    use nifti::writer::WriterOptions;
    use nifti::NiftiHeader;
    let [nx, ny, nz] = grid;
    assert_eq!(data.len(), nx * ny * nz * 3);
    let arr = ndarray::Array5::from_shape_vec((nx, ny, nz, 1usize, 3usize), data.to_vec())
        .map_err(|e| anyhow::anyhow!("ndarray shape: {e}"))?;
    let mut hdr = NiftiHeader::default();
    hdr.dim[0] = 5;
    hdr.dim[1] = nx as u16;
    hdr.dim[2] = ny as u16;
    hdr.dim[3] = nz as u16;
    hdr.dim[4] = 1;
    hdr.dim[5] = 3;
    hdr.datatype = nifti::NiftiType::Float32 as i16;
    hdr.bitpix = 32;
    hdr.pixdim[0] = 1.0;
    hdr.pixdim[1] = spacing[0];
    hdr.pixdim[2] = spacing[1];
    hdr.pixdim[3] = spacing[2];
    hdr.pixdim[4] = 1.0;
    hdr.pixdim[5] = 1.0;
    hdr.qform_code = 0;
    hdr.sform_code = 2;
    hdr.srow_x = [spacing[0], 0.0, 0.0, origin[0]];
    hdr.srow_y = [0.0, spacing[1], 0.0, origin[1]];
    hdr.srow_z = [0.0, 0.0, spacing[2], origin[2]];
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .map_err(|e| anyhow::anyhow!("write nifti: {e}"))?;
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
    let mut dump_planes: Option<PathBuf> = None;
    let mut selftest: Option<PathBuf> = None;
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
        else if let Some(s) = a.strip_prefix("--dump-planes=") { dump_planes = Some(PathBuf::from(s)); }
        else if let Some(s) = a.strip_prefix("--selftest=") { selftest = Some(PathBuf::from(s)); }
        i += 1;
    }
    anyhow::ensure!(!ckpt.as_os_str().is_empty(), "need --ckpt=...");

    let wgpu = brush_process::burn_init_setup().await;
    let device: Device = wgpu.into();
    let device_ad = device.clone().autodiff();

    // ---- 布局自检: 已知值张量 [2,3,4] (值 = i*100+j*10+k) ----
    if let Some(dir) = &selftest {
        use burn::tensor::Tensor;
        let h = 2usize; let w = 3usize; let c = 4usize;
        let mut vals = Vec::with_capacity(h * w * c);
        for i in 0..h { for j in 0..w { for k in 0..c {
            vals.push((i * 100 + j * 10 + k) as f32);
        }}}
        let t = Tensor::<3>::from_data(TensorData::new(vals.clone(), [h, w, c]), &device_ad);
        let back: Vec<f32> = t.clone().into_data_async().await?.to_vec()?;
        println!("SELFTEST: 输入 values (生成顺序 i,j,k): {vals:?}");
        println!("SELFTEST: to_vec() 输出:              {back:?}");
        println!("SELFTEST: 期望 row-major [2,3,4] = 输入; 若输出不同说明 burn 布局非 row-major");
        std::fs::create_dir_all(dir)?;
        let p = dir.join("selftest.npy");
        write_npy_f32(&p, &back, &[h, w, c])?;
        println!("SELFTEST: saved {p:?}");
        return Ok(());
    }

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

    // ---- 可选: 输出 HexPlane 特征平面 (棋盘格诊断) ---------------------
    if let Some(dir) = dump_planes {
        std::fs::create_dir_all(&dir)?;
        for (name, plane) in model.planes() {
            let dims = plane.dims(); // [a, b, C]
            let data: Vec<f32> = plane.clone().into_data_async().await?.to_vec()?;
            let [a, b, c] = [dims[0], dims[1], dims[2]];
            // 相邻单元相关 (轴0/轴1): 展平对计算 Pearson r
            let pearson = |pairs: &[(f32, f32)]| -> f64 {
                let n = pairs.len() as f64;
                if n < 2.0 { return f64::NAN; }
                let mx = pairs.iter().map(|p| p.0 as f64).sum::<f64>() / n;
                let my = pairs.iter().map(|p| p.1 as f64).sum::<f64>() / n;
                let cov = pairs.iter().map(|p| (p.0 as f64 - mx) * (p.1 as f64 - my)).sum::<f64>() / n;
                let vx = pairs.iter().map(|p| (p.0 as f64 - mx).powi(2)).sum::<f64>() / n;
                let vy = pairs.iter().map(|p| (p.1 as f64 - my).powi(2)).sum::<f64>() / n;
                if vx <= 0.0 || vy <= 0.0 { f64::NAN } else { cov / (vx * vy).sqrt() }
            };
            let mut p0 = Vec::new(); let mut p1 = Vec::new();
            for iy in 0..b {
                for ix in 0..a.saturating_sub(1) {
                    for ch in 0..c {
                        let idx = (iy * a + ix) * c + ch;
                        p0.push((data[idx], data[idx + c]));
                    }
                }
            }
            for ix in 0..a {
                for iy in 0..b.saturating_sub(1) {
                    for ch in 0..c {
                        let idx = (iy * a + ix) * c + ch;
                        p1.push((data[idx], data[idx + a * c]));
                    }
                }
            }
            println!("plane {name} [{a}x{b}x{c}]: 相邻单元相关 轴0={:.3} 轴1={:.3} (平滑→+1, 棋盘格→-1), 幅度 std={:.4}",
                pearson(&p0), pearson(&p1),
                (data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / data.len() as f64).sqrt());
            // 保存 .npy [a, b, C] (行主序) + .nii.gz (nifti-rs, [a,b,C] 3D 卷)。
            let path = dir.join(format!("plane_{name}.npy"));
            write_npy_f32(&path, &data, &[a, b, c])?;
            let nii = dir.join(format!("plane_{name}.nii.gz"));
            write_nifti_3d_f32(&nii, &data, [a, b, c])?;
            println!("  saved -> {} / {}", path.display(), nii.display());
        }
        return Ok(());
    }

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
        let ph = if n_phase > 1 { p as f32 / n_phase as f32 } else { phase };
        let phase_t = Tensor::<2>::from_data(TensorData::new(vec![ph; n_pts], [n_pts, 1]), &device_ad);
        let time_t = Tensor::<2>::from_data(TensorData::new(vec![time; n_pts], [n_pts, 1]), &device_ad);
        let d = model.forward(xyz_t.clone(), phase_t, time_t).d_xyz;
        let dv: Vec<f32> = d.into_data_async().await?.to_vec()?;
        let stem = format!("{}_phase{p:02}", out.file_name().unwrap().to_string_lossy());
        let npy = out.with_file_name(format!("{stem}.npy"));
        write_npy_f32(&npy, &dv, &[grid[0], grid[1], grid[2], 3])?;
        let nii = out.with_file_name(format!("{stem}.nii.gz"));
        write_nifti_vec5d_f32(&nii, &dv, grid, [spacing; 3], origin)?;
        println!("  saved deform field -> {} / {}", npy.display(), nii.display());
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
