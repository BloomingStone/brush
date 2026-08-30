//! 导出: ①必须的 phase=0 volume (.nii.gz, 3D) — deform 模式先过 phase=0
//! 形变场; ②可选 PLY 点云; ③可选 deform 网络权重 + 每相位网格场; ④可选
//! 原始参数 .bin (供 gs2volume --bin= 复用)。

use std::path::{Path, PathBuf};

use brush_deform::{deform_splats, Deforms};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::xray_train::XRayTrainer;
use brush_voxel::{VoxelSettings, voxelize_forward};
use brush_xray::XRaySplats;
use burn::tensor::{Tensor, TensorData};
use nifti::writer::WriterOptions;
use nifti::{NiftiHeader, NiftiType};

use crate::config::{FitConfig, FitMode};

/// 内部 x-major 布局 → 3D nifti (列优先磁盘, dims (X,Y,Z)), 对角 srow。
pub fn write_nifti_volume(
    path: &Path,
    data: &[f32],
    vol_x: usize,
    vol_y: usize,
    vol_z: usize,
    rx: f32,
    ry: f32,
    rz: f32,
) -> anyhow::Result<()> {
    let arr = ndarray::Array3::from_shape_vec((vol_x, vol_y, vol_z), data.to_vec())
        .map_err(|e| anyhow::anyhow!("ndarray shape: {e}"))?;
    let sx = 2.0 * rx / vol_x as f32;
    let sy = 2.0 * ry / vol_y as f32;
    let sz = 2.0 * rz / vol_z as f32;
    let mut hdr = NiftiHeader::default();
    hdr.datatype = NiftiType::Float32 as i16;
    hdr.bitpix = 32;
    hdr.qform_code = 0;
    hdr.sform_code = 2;
    hdr.srow_x = [sx, 0.0, 0.0, -rx];
    hdr.srow_y = [0.0, sy, 0.0, -ry];
    hdr.srow_z = [0.0, 0.0, sz, -rz];
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .map_err(|e| anyhow::anyhow!("write nifti: {e}"))?;
    Ok(())
}

/// 5D 位移场 `[nx,ny,nz,1,3]` (同 ASOCA dvf), sform_code=2。
pub fn write_nifti_vec5d_f32(
    path: &Path,
    data: &[f32],
    grid: [usize; 3],
    spacing: [f32; 3],
    origin: [f32; 3],
) -> anyhow::Result<()> {
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
    hdr.datatype = NiftiType::Float32 as i16;
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

/// NumPy v1.0 float32, 行主序 (C-order)。
pub fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> anyhow::Result<()> {
    use std::io::Write;
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "npy data len mismatch");
    let shape_str = format!(
        "({}{})",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "),
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

/// Convert canonical [`brush_xray::XRaySplats`] into a PLY-able [`Splats`]
/// (SH degree 0 → grayscale; the X-ray renderer is SH-free).
fn xray_to_splats(canonical: &XRaySplats, device: &burn::tensor::Device) -> Splats {
    let n = canonical.num_splats() as usize;
    let means = canonical.means();
    let rots = canonical.rotations();
    let log_scales = canonical.log_scales();
    let opac = canonical.raw_opacities.val();
    let sh = burn::tensor::Tensor::<3>::zeros([n, 1, 3], device);
    Splats::from_tensor_data(means, rots, log_scales, sh, opac, SplatRenderMode::Default)
}

/// 自适应体积网格: 每轴 p99.9 的 |mean|+3σ (与显式 extent 取 max, cap
/// 400mm) — 保证 voxelizer 不截断 splat 密度 (DRR ≈ GS)。
pub fn adaptive_grid(
    means: &[f32],
    log_scales: &[f32],
    cfg: &FitConfig,
) -> (f32, f32) {
    let n = means.len() / 3;
    let mut half_w: Option<f32> = cfg.extent_xy.map(|e| 0.5 * e);
    let mut half_z: Option<f32> = cfg.extent_z.map(|e| 0.5 * e);
    if cfg.extent_xy.is_none() && cfg.extent_z.is_none() && n > 0 {
        let sig: Vec<f32> = log_scales.iter().map(|&l| l.exp()).collect();
        let mut p999 = [0.0f32; 3];
        let mut cmax = [0.0f32; 3];
        for ax in 0..3 {
            let mut vals: Vec<f32> = (0..n)
                .map(|i| means[i * 3 + ax].abs() + 3.0 * sig[i * 3 + ax])
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).expect("f32"));
            p999[ax] = vals[((0.999 * vals.len() as f32) as usize).min(vals.len() - 1)];
            cmax[ax] = vals[vals.len() - 1];
        }
        let cap = 400.0f32;
        let w = p999[0].min(cap).max(p999[1].min(cap));
        let z = p999[2].min(cap);
        half_w = Some(half_w.unwrap_or(0.0).max(w));
        half_z = Some(half_z.unwrap_or(0.0).max(z));
        println!(
            "[volume] adaptive grid: splat |mean|+3σ p99.9 = ({:.0},{:.0},{:.0}) max = ({:.0},{:.0},{:.0})mm -> half XY {:.0}mm Z {:.0}mm (cap {cap:.0})",
            p999[0], p999[1], p999[2], cmax[0], cmax[1], cmax[2],
            half_w.unwrap(), half_z.unwrap(),
        );
    }
    (half_w.unwrap_or(1.0), half_z.unwrap_or(1.0))
}

/// 导出必须的 phase=0 volume: ①deform 模式: canonical 经 phase=0/time=0
/// 形变场 → deformed splats; ②自适应网格 → voxelize → <out>/volume_phase00.nii.gz。
pub async fn export_volume_phase0(
    cfg: &FitConfig,
    trainer: &XRayTrainer,
    device: &burn::tensor::Device,
    out: &Path,
) -> anyhow::Result<PathBuf> {
    let canonical = trainer.canonical();
    let (means, rots, log_scales, raw) = read_canonical(canonical).await?;
    let n = means.len() / 3;

    // deform 模式: phase=0, time=0 的形变 (网络 forward, 含 d_rotation)。
    let (means, rots) = if cfg.mode == FitMode::Deform {
        use burn::module::Module;
        let deform = trainer
            .deform()
            .ok_or_else(|| anyhow::anyhow!("deform mode requires a trained deform network"))?;
        let device_ad = device.clone().autodiff();
        let xyz =
            Tensor::<2>::from_data(TensorData::new::<f32, _>(means.clone(), [n, 3]), &device_ad);
        let phase_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![0.0; n], [n, 1]), &device_ad);
        let time_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![0.0; n], [n, 1]), &device_ad);
        let deforms: Deforms = deform.forward(xyz.clone(), phase_t, time_t);
        let canonical_ad =
            XRaySplats::from_raw(means.clone(), rots.clone(), log_scales.clone(), raw.clone(), &device_ad);
        let def_ad = deform_splats(&canonical_ad, &deforms);
        let dmeans: Vec<f32> = def_ad.means().into_data_async().await?.to_vec()?;
        let drots: Vec<f32> = def_ad.rotations().into_data_async().await?.to_vec()?;
        let dv: Vec<f32> = deforms.d_xyz.into_data_async().await?.to_vec()?;
        let dmax = (0..n)
            .map(|k| glam::Vec3::new(dv[k * 3], dv[k * 3 + 1], dv[k * 3 + 2]).length())
            .fold(0.0f32, f32::max);
        println!("[volume] deform phase=0, time=0: max |d| = {dmax:.2} mm (含 d_rotation)");
        (dmeans, drots)
    } else {
        (means, rots)
    };

    let (half_w, half_z) = adaptive_grid(&means, &log_scales, cfg);
    let voxel_mm = cfg.voxel_mm.max(1e-3);
    let n_xy = (2.0 * half_w / voxel_mm).round().max(1.0) as usize;
    let n_z = (2.0 * half_z / voxel_mm).round().max(1.0) as usize;
    println!(
        "[volume] grid {n_xy}x{n_xy}x{n_z} @ {voxel_mm:.2}mm (span XY {:.0}mm Z {:.0}mm)",
        2.0 * half_w,
        2.0 * half_z
    );

    let splats = XRaySplats::from_raw(means, rots, log_scales, raw, device);
    let settings = VoxelSettings::new(
        glam::uvec3(n_xy as u32, n_xy as u32, n_z as u32),
        glam::vec3(2.0 * half_w, 2.0 * half_w, 2.0 * half_z),
        glam::Vec3::ZERO,
    );
    let v_vol = voxelize_forward(&splats, &settings).await;
    let vol_vec: Vec<f32> = v_vol.into_data_async().await?.to_vec()?;

    let vol_path = out.join("volume_phase00.nii.gz");
    write_nifti_volume(&vol_path, &vol_vec, n_xy, n_xy, n_z, half_w, half_w, half_z)?;
    println!("[export] volume -> {}", vol_path.display());
    Ok(vol_path)
}

/// canonical splats → CPU (means / rots / log_scales / raw opac)。
pub async fn read_canonical(
    canonical: &XRaySplats,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let t: Vec<f32> = canonical.transforms.val().into_data_async().await?.to_vec()?;
    let o: Vec<f32> = canonical.raw_opacities.val().into_data_async().await?.to_vec()?;
    let n = o.len();
    let mut means = Vec::with_capacity(n * 3);
    let mut rots = Vec::with_capacity(n * 4);
    let mut log_scales = Vec::with_capacity(n * 3);
    for i in 0..n {
        means.extend_from_slice(&t[i * 10..i * 10 + 3]);
        rots.extend_from_slice(&t[i * 10 + 3..i * 10 + 7]);
        log_scales.extend_from_slice(&t[i * 10 + 7..i * 10 + 10]);
    }
    Ok((means, rots, log_scales, o))
}

/// 导出 canonical PLY 点云 (激活域, up=Y)。
pub async fn export_ply(
    trainer: &XRayTrainer,
    device: &burn::tensor::Device,
    out: &Path,
) -> anyhow::Result<PathBuf> {
    let splats = xray_to_splats(trainer.canonical(), device);
    let ply = brush_serde::splat_to_ply(splats, Some(glam::Vec3::Y)).await?;
    let ply_path = out.join("canonical_final.ply");
    std::fs::write(&ply_path, ply)?;
    println!("[export] ply -> {}", ply_path.display());
    Ok(ply_path)
}

/// 导出原始参数 .bin (transforms [N,10] + raw_opac [N], raw 域 f32)。
pub async fn export_bin(
    trainer: &XRayTrainer,
    out: &Path,
) -> anyhow::Result<PathBuf> {
    let (means, rots, log_scales, raw) = read_canonical(trainer.canonical()).await?;
    let n = raw.len();
    let mut t = Vec::with_capacity(n * 10);
    for i in 0..n {
        t.extend_from_slice(&means[i * 3..i * 3 + 3]);
        t.extend_from_slice(&rots[i * 4..i * 4 + 4]);
        t.extend_from_slice(&log_scales[i * 3..i * 3 + 3]);
    }
    let mut tb = Vec::with_capacity(t.len() * 4);
    for v in &t {
        tb.extend_from_slice(&v.to_le_bytes());
    }
    let mut ob = Vec::with_capacity(raw.len() * 4);
    for v in &raw {
        ob.extend_from_slice(&v.to_le_bytes());
    }
    let prefix = out.join("canonical_final");
    std::fs::write(out.join("canonical_final_transforms.bin"), tb)?;
    std::fs::write(out.join("canonical_final_raw.bin"), ob)?;
    println!("[export] bin -> {}_transforms.bin / _raw.bin", prefix.display());
    Ok(prefix)
}

/// 导出 deform 网络权重 (deform_final.bin) + 每相位网格形变场
/// (deform_field_phase{p:02}.npy 行主序 [nx,ny,nz,3] + .nii.gz 5D)。
/// 进程内用训练器自身的 deform 网络 (与 ckpt 结构天然一致)。
pub async fn export_deform(
    cfg: &FitConfig,
    trainer: &XRayTrainer,
    scene_extent: f32,
    out: &Path,
) -> anyhow::Result<(PathBuf, Vec<PathBuf>)> {
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

    let deform = trainer
        .deform()
        .ok_or_else(|| anyhow::anyhow!("no deform network to export"))?;
    let ckpt = out.join("deform_final.bin");
    let record = deform.clone().into_record();
    BinFileRecorder::<FullPrecisionSettings>::new()
        .record(record, ckpt.clone())
        .map_err(|e| anyhow::anyhow!("save deform ckpt: {e}"))?;
    println!("[export] deform ckpt -> {}", ckpt.display());

    // 采样间距 = 半单元 (对齐 HexPlane 单元, 避免混叠; 同 dump_deform)。
    let hex_res = trainer.config().hex_plane.hex_plane.spatial_resolution;
    let cell = 2.0 * scene_extent / hex_res as f32;
    let spacing = (cell / 2.0).max(0.5);
    println!(
        "[export] deform field via brush-fit-dump (spacing {spacing:.2}mm, cell {cell:.2}mm, 每单元 {:.1} 采样)...",
        cell / spacing
    );

    // 训练后内存池状态不稳定 (整网格 matmul autotune 曾 OOM / 内存池损坏),
    // 网格场导出委托独立进程 brush-fit-dump (干净设备)。C 嵌入场景需把
    // brush-fit-dump 放在宿主程序同目录或 PATH; 可用 BRUSH_FIT_DUMP_BIN 覆盖。
    let exe = std::env::var_os("BRUSH_FIT_DUMP_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .map(|p| p.with_file_name("brush-fit-dump"))
                .unwrap_or_else(|| PathBuf::from("brush-fit-dump"))
        });
    let hex = &cfg;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg(format!("--ckpt={}", ckpt.display()))
        .arg(format!("--scene-extent={scene_extent}"))
        .arg(format!("--spacing={spacing}"))
        .arg(format!("--n-phase={}", cfg.n_deform_phases.max(1)))
        .arg(format!("--out={}", out.join("deform_field").display()))
        .arg(format!("--hex-res={}", hex.hex_res))
        .arg(format!("--hex-time-res={}", hex.hex_time_res))
        .arg(format!("--hex-features={}", hex.hex_features))
        .arg(format!("--mlp-width={}", hex.hex_mlp_width))
        .arg(format!("--mlp-layers={}", hex.hex_mlp_layers))
        .arg(format!("--time-freqs={}", hex.time_freqs))
        .arg(format!("--time-min-freq={}", hex.time_min_freq))
        .arg(format!("--time-max-freq={}", hex.time_max_freq));
    if hex.predict_scaling {
        cmd.arg("--predict-scaling");
    }
    let time_on = hex.enable_time && !hex.no_time;
    if !time_on {
        cmd.arg("--no-time");
    }
    let st = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("spawn brush-fit-dump ({}): {e}", exe.display()))?;
    if !st.success() {
        anyhow::bail!("brush-fit-dump 导出失败 (status={st}); 确保 {exe:?} 存在 (cargo build --release -p brush-fit)");
    }
    let mut fields = Vec::new();
    for p in 0..cfg.n_deform_phases.max(1) {
        fields.push(out.join(format!("deform_field_phase{p:02}.nii.gz")));
    }
    Ok((ckpt, fields))
}

