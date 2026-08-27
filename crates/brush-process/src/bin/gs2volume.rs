//! GS2volume: 把训练好的 GS 点云 (canonical_final.ply) 体素化 (brush-voxel),
//! 与 FDK 体积相加, 输出为 nii.gz + NRRD 供目视监察 GS 重建是否有真实 3D 结构
//! (vs 仅靠投影拟合的条纹)。
//!
//! 输出网格: XY padding 到 N (默认 326), z 保持 FDK 的 183, spacing 同 FDK
//! (0.927mm)。voxelizer 输出布局 [x][y][z] (z 最快) → 转成公共布局 [y][z][x]
//! (x 最快, 同 DRR 内核 / FDK)。

use brush_cube::{MU_WATER, silu};
use brush_serde::import::load_splat_from_ply;
use brush_voxel::{VoxelSettings, voxelize_forward};
use brush_xray::XRaySplats;
use burn::tensor::{Device, Tensor, TensorData};
use std::path::{Path, PathBuf};

fn read_nifti_volume(path: &Path) -> anyhow::Result<(Vec<f32>, usize, usize, usize)> {
    use nifti::{NiftiObject, ReaderOptions};
    let obj = ReaderOptions::new().read_file(path)?;
    let dims = obj.header().dim;
    let (vx, vy, vz) = (dims[1] as usize, dims[2] as usize, dims[3] as usize);
    let volume = obj.into_volume();
    let data: Vec<f32> = volume.into_nifti_typed_data()?;
    Ok((data, vx, vy, vz))
}

fn write_nifti_volume(
    path: &Path,
    data: &[f32],
    vol_x: usize,
    vol_y: usize,
    vol_z: usize,
    rx: f32,
    ry: f32,
    rz: f32,
) -> anyhow::Result<()> {
    use nifti::writer::WriterOptions;
    use nifti::{NiftiHeader, NiftiType};
    let arr = ndarray::Array3::from_shape_vec((vol_y, vol_z, vol_x), data.to_vec())
        .map_err(|e| anyhow::anyhow!("ndarray shape: {e}"))?;
    let sx = 2.0 * rx / vol_x as f32;
    let sy = 2.0 * ry / vol_y as f32;
    let sz = 2.0 * rz / vol_z as f32;
    let mut hdr = NiftiHeader::default();
    hdr.datatype = NiftiType::Float32 as i16;
    hdr.bitpix = 32;
    hdr.qform_code = 0;
    hdr.sform_code = 2;
    hdr.srow_x = [0.0, 0.0, sx, -rx];
    hdr.srow_y = [sy, 0.0, 0.0, -ry];
    hdr.srow_z = [0.0, sz, 0.0, -rz];
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .map_err(|e| anyhow::anyhow!("write nifti: {e}"))?;
    Ok(())
}

/// 写 3D NRRD (dimension 3, sizes: x z y, 布局 [y][z][x], x 最快)。
fn write_nrrd_3d(path: &Path, data: &[f32], vol_x: usize, vol_y: usize, vol_z: usize) -> anyhow::Result<()> {
    let mut payload = Vec::with_capacity(data.len() * 4);
    for v in data {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let header = format!(
        "NRRD0004\n\
         # GS2volume (float32, 布局 [y][z][x], x 最快)\n\
         type: float\n\
         dimension: 3\n\
         sizes: {vol_x} {vol_z} {vol_y}\n\
         encoding: raw\n\
         endian: little\n\
         \n"
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = Vec::with_capacity(header.len() + payload.len());
    file.extend_from_slice(header.as_bytes());
    file.append(&mut payload);
    std::fs::write(path, file)?;
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut ply: Option<PathBuf> = None;
    let mut fdk_vol: Option<PathBuf> = None;
    let mut fdk_meta: Option<PathBuf> = None;
    let mut fdk_calib: Option<PathBuf> = None;
    let mut out = PathBuf::from("target/gs2volume");
    let mut n_xy = 326usize;
    let mut signed = false;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--ply=") {
            ply = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--fdk=") {
            fdk_vol = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--fdk-meta=") {
            fdk_meta = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--fdk-calib=") {
            fdk_calib = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if let Some(v) = a.strip_prefix("--n-xy=") {
            n_xy = v.parse()?;
        } else if a == "--signed" {
            signed = true;
        }
        i += 1;
    }
    let ply = ply.expect("usage: gs2volume --ply=<canonical_final.ply> --fdk=<nii> --fdk-meta=<json> --fdk-calib=<json> --out=<dir>");
    let fdk_vol = fdk_vol.expect("--fdk required");
    std::fs::create_dir_all(&out)?;

    // ---- 后端 + 设备 ----
    let wgpu = brush_process::burn_init_setup().await;
    let device: Device = wgpu.into();
    let device_ad = device.clone().autodiff();

    // ---- 加载 GS ply ----
    let file = tokio::fs::File::open(&ply).await?;
    let msg = load_splat_from_ply(file, None).await?;
    let data = msg.data;
    let n = data.num_splats();
    let means = data.means;
    let rots = data.rotations.unwrap_or_else(|| {
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n { v.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]); }
        v
    });
    let log_scales = data.log_scales.unwrap();
    let raw = data.raw_opacities.unwrap();
    println!("loaded {} splats from {ply:?}", n);

    // ---- raw → sigmoid 域 (让 voxelizer 输出 = μ 场) ----
    // voxelizer opac = sigmoid(raw2); 我们想要 opac = μ = MU_WATER·(silu(raw) 或 raw)。
    let mut raw2 = Vec::with_capacity(n);
    for &r in &raw {
        let mu = if signed {
            MU_WATER * r
        } else {
            MU_WATER * silu(r)
        };
        // sigmoid(raw2) = mu → raw2 = ln(mu/(1-mu)); 负值 (signed) 无法表示 → clamp。
        let mu = mu.max(1e-7).min(1.0 - 1e-7);
        raw2.push((mu / (1.0 - mu)).ln());
    }

    // ---- FDK 体积 + meta + 转置修正 ----
    let meta_path = fdk_meta.unwrap_or_else(|| fdk_vol.with_file_name("meta.json"));
    let calib_path = fdk_calib.unwrap_or_else(|| fdk_vol.with_file_name("calib.json"));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let rx0 = meta["rx"].as_f64().unwrap_or(118.6) as f32;
    let rz0 = meta["rz"].as_f64().unwrap_or(84.7) as f32;
    let (mut fdk_vec, _hx, _hy, _hz) = read_nifti_volume(&fdk_vol)?;
    let fdk_x = meta["vol_x"].as_u64().unwrap_or(_hx as u64) as usize;
    let fdk_y = meta["vol_y"].as_u64().unwrap_or(_hy as u64) as usize;
    let fdk_z = meta["vol_z"].as_u64().unwrap_or(_hz as u64) as usize;
    assert_eq!(fdk_vec.len(), fdk_x * fdk_y * fdk_z, "FDK size mismatch");
    // nifti data.t() 转置修正 (x<->y)。
    let sy = fdk_x * fdk_z;
    let sz = fdk_x;
    let mut out_v = vec![0.0f32; fdk_vec.len()];
    for iy in 0..fdk_y {
        for iz in 0..fdk_z {
            for ix in 0..fdk_x {
                let src = iy * sy + iz * sz + ix;
                let dst = ix * sy + iz * sz + iy;
                out_v[dst] = fdk_vec[src];
            }
        }
    }
    fdk_vec = out_v;

    // ---- GS 体素化 (网格 = FDK XY padding 到 n_xy, spacing 保持) ----
    let vox_mm = 2.0 * rx0 / fdk_x as f32; // 0.927mm
    let rx = vox_mm * n_xy as f32 / 2.0;   // n_xy 世界半宽
    let rz = rz0;                          // z 不变 (FDK 183)
    let n_z = fdk_z;
    let settings = VoxelSettings::new(
        glam::uvec3(n_xy as u32, n_xy as u32, n_z as u32),
        glam::vec3(2.0 * rx, 2.0 * rx, 2.0 * rz),
        glam::Vec3::ZERO,
    );
    println!(
        "voxelize GS: grid {n_xy}x{n_xy}x{n_z}, voxel {vox_mm:.3}mm, world {rx:.1}x{rx:.1}x{rz:.1}mm, signed={signed}"
    );
    let splats = XRaySplats::from_raw(means, rots, log_scales, raw2, &device);
    let v_vol = voxelize_forward(&splats, &settings).await; // [n_x, n_y, n_z], z 最快
    let gs_vol: Vec<f32> = v_vol.into_data().to_vec()?;

    // 公共布局 [y][z][x] (x 最快): gs_common[y][z][x] = gs[x][y][z]
    let mut gs_common = vec![0.0f32; n_xy * n_xy * n_z];
    for ix in 0..n_xy {
        for iy in 0..n_xy {
            for iz in 0..n_z {
                let src = ix * n_xy * n_z + iy * n_z + iz;   // voxelizer [x][y][z]
                let dst = iy * n_xy * n_z + iz * n_xy + ix; // common [y][z][x]
                gs_common[dst] = gs_vol[src];
            }
        }
    }

    // ---- FDK 居中 padding 到 n_xy (XY), 公共布局 [y][z][x] ----
    let ox = (n_xy - fdk_x) / 2;
    let oy = (n_xy - fdk_y) / 2;
    let mut fdk_pad = vec![0.0f32; n_xy * n_xy * n_z];
    for iy in 0..fdk_y {
        for iz in 0..fdk_z {
            for ix in 0..fdk_x {
                let src = iy * fdk_x * fdk_z + iz * fdk_x + ix;
                let dst = (iy + oy) * n_xy * n_z + iz * n_xy + (ix + ox);
                fdk_pad[dst] = fdk_vec[src];
            }
        }
    }

    // ---- 相加 + 保存 ----
    let mut total = Vec::with_capacity(gs_common.len());
    for k in 0..gs_common.len() {
        total.push(gs_common[k] + fdk_pad[k]);
    }
    let tag = if signed { "signed" } else { "uns" };
    write_nifti_volume(&out.join(format!("gs_{tag}.nii.gz")), &gs_common, n_xy, n_xy, n_z, rx, rx, rz)?;
    write_nifti_volume(&out.join("fdk_padded.nii.gz"), &fdk_pad, n_xy, n_xy, n_z, rx, rx, rz)?;
    write_nifti_volume(&out.join(format!("total_{tag}.nii.gz")), &total, n_xy, n_xy, n_z, rx, rx, rz)?;
    write_nrrd_3d(&out.join(format!("gs_{tag}.nrrd")), &gs_common, n_xy, n_xy, n_z)?;
    write_nrrd_3d(&out.join("fdk_padded.nrrd"), &fdk_pad, n_xy, n_xy, n_z)?;
    write_nrrd_3d(&out.join(format!("total_{tag}.nrrd")), &total, n_xy, n_xy, n_z)?;
    println!(
        "saved -> {out:?}: gs_{tag}.nrrd / fdk_padded.nrrd / total_{tag}.nrrd (+nii.gz)"
    );
    println!("  gs range [{:.4},{:.4}] fdk [{:.4},{:.4}] total [{:.4},{:.4}]",
        gs_common.iter().cloned().fold(f32::INFINITY, f32::min),
        gs_common.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        fdk_pad.iter().cloned().fold(f32::INFINITY, f32::min),
        fdk_pad.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        total.iter().cloned().fold(f32::INFINITY, f32::min),
        total.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    Ok(())
}
