//! GS2volume: 把训练好的 GS 点云 (canonical_final.ply) 体素化 (brush-voxel),
//! 与 FDK 体积相加, 输出为 nii.gz + NRRD 供目视监察 GS 重建是否有真实 3D 结构
//! (vs 仅靠投影拟合的条纹)。
//!
//! 输出网格: XY padding 到 N (默认 326), z 保持 FDK 的 183, spacing 同 FDK
//! (0.927mm)。内部统一 x-major 布局 `idx(x,y,z) = x*(ny*nz) + y*nz + z`
//! (同 voxelizer 输出 / DRR 内核 / R2 fields); nifti/nrrd 写入时转磁盘
//! 列优先 (x 最快, NIfTI-1/NRRD 标准, 文件 dims 自然 (X,Y,Z))。

use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig, HexPlaneDeformModel, deform_splats};
use brush_train::xray_train::DeformNetwork;
use brush_serde::import::load_splat_from_ply;
use brush_voxel::{VoxelSettings, voxelize_forward};
use brush_xray::XRaySplats;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::{Path, PathBuf};

fn read_nifti_volume(path: &Path) -> anyhow::Result<(Vec<f32>, usize, usize, usize)> {
    use nifti::{NiftiObject, ReaderOptions};
    let obj = ReaderOptions::new().read_file(path)?;
    let dims = obj.header().dim;
    let (vx, vy, vz) = (dims[1] as usize, dims[2] as usize, dims[3] as usize);
    let volume = obj.into_volume();
    let data: Vec<f32> = volume.into_nifti_typed_data()?;
    // nifti-rs 读回磁盘列优先缓冲 (x 最快) → 内部 x-major 布局。
    Ok((brush_process::volume_layout::to_xmajor(&data, vx, vy, vz), vx, vy, vz))
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
    // 内部 x-major → 数组形状 (vol_x, vol_y, vol_z); nifti-rs 自动转磁盘
    // 列优先, 文件 dims (X,Y,Z) 自然顺序, 对角 srow。
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

/// 写 3D NRRD (dimension 3, sizes 自然 (X,Y,Z), 轴0 = x 最快 = 磁盘标准)。
/// 内部 x-major 缓冲先转磁盘列优先再写。
fn write_nrrd_3d(path: &Path, data: &[f32], vol_x: usize, vol_y: usize, vol_z: usize) -> anyhow::Result<()> {
    let payload = brush_process::volume_layout::from_xmajor(data, vol_x, vol_y, vol_z);
    let mut bytes = Vec::with_capacity(payload.len() * 4);
    for v in payload {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let header = format!(
        "NRRD0004\n\
         # GS2volume (float32, 磁盘列优先 x 最快, sizes (X,Y,Z))\n\
         type: float\n\
         dimension: 3\n\
         sizes: {vol_x} {vol_y} {vol_z}\n\
         encoding: raw\n\
         endian: little\n\
         \n"
    );
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = Vec::with_capacity(header.len() + bytes.len());
    file.extend_from_slice(header.as_bytes());
    file.append(&mut bytes);
    std::fs::write(path, file)?;
    Ok(())
}

/// 读 .npy (v1.0, float32, C-order) 返回 flat 数据。
fn read_npy_f32(path: &Path) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a npy file");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = String::from_utf8(bytes[10..10 + header_len].to_vec())?;
    let shape: Vec<usize> = {
        let Some(open) = header.find("'shape':") else { return Ok((vec![], vec![])) };
        let sub = &header[open..];
        let Some(lb) = sub.find('(') else { return Ok((vec![], vec![])) };
        let sub = &sub[lb + 1..];
        let Some(rb) = sub.find(')') else { return Ok((vec![], vec![])) };
        sub[..rb]
            .split(',')
            .map(|x| x.trim().parse::<usize>().unwrap_or(0))
            .filter(|&x| x > 0)
            .collect()
    };
    let total: usize = shape.iter().product();
    let data_start = 10 + header_len;
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let o = data_start + i * 4;
        out.push(f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]));
    }
    Ok((out, shape))
}

/// 在形变场网格 `[nx,ny,nz,3]` (row-major, 坐标 `-ext + 2*ext*i/(n-1)`) 三线性插值。
fn sample_deform_field(field: &[f32], n: [usize; 3], ext: f32, p: glam::Vec3) -> glam::Vec3 {
    let [nx, ny, nz] = n;
    let f = |x: f32, n: usize| -> (f32, usize, usize) {
        let g = (x + ext) / (2.0 * ext) * (n as f32 - 1.0);
        let i0 = g.floor().clamp(0.0, (n - 1) as f32) as usize;
        let i1 = (i0 + 1).min(n - 1);
        (g - i0 as f32, i0, i1)
    };
    let (tx, ix0, ix1) = f(p.x, nx);
    let (ty, iy0, iy1) = f(p.y, ny);
    let (tz, iz0, iz1) = f(p.z, nz);
    let at = |ix: usize, iy: usize, iz: usize, a: usize| -> f32 {
        // dump_deform 实际布局: pts 按 z 最慢、x 最快 (for iz { for iy { for ix }}),
        // flat = (iz*ny + iy)*nx + ix (尽管 npy 声明 [nx,ny,nz,3] 是误导的)。
        field[((iz * ny + iy) * nx + ix) * 3 + a]
    };
    let mut d = glam::Vec3::ZERO;
    for a in 0..3 {
        let c000 = at(ix0, iy0, iz0, a);
        let c100 = at(ix1, iy0, iz0, a);
        let c010 = at(ix0, iy1, iz0, a);
        let c110 = at(ix1, iy1, iz0, a);
        let c001 = at(ix0, iy0, iz1, a);
        let c101 = at(ix1, iy0, iz1, a);
        let c011 = at(ix0, iy1, iz1, a);
        let c111 = at(ix1, iy1, iz1, a);
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        d[a] = c0 * (1.0 - tz) + c1 * tz;
    }
    d
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
    let mut deform_field: Option<PathBuf> = None;
    let mut deform_extent = 264.0f32;
    let mut ckpt: Option<PathBuf> = None; // HexPlane 网络权重 (deform_final.bin)
    let mut no_fdk = false;
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
        } else if let Some(v) = a.strip_prefix("--deform-field=") {
            deform_field = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--ckpt=") {
            ckpt = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--deform-extent=") {
            deform_extent = v.parse()?;
        } else if a == "--no-fdk" {
            no_fdk = true;
        }
        i += 1;
    }
    let ply = ply.expect("usage: gs2volume --ply=<canonical_final.ply> --fdk=<nii> --fdk-meta=<json> --fdk-calib=<json> --out=<dir>");
    let fdk_vol = fdk_vol.expect("--fdk required (or --no-fdk)");
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
    // 形变场 (phase 0, time 0): canonical → deformed (实际渲染位置)。
    let mut means = means;
    let mut rots = rots;
    // 形变: 优先用网络权重 (--ckpt, 含 d_xyz+d_rotation, 精确); 否则用网格场 (有损)。
    if let Some(cp) = &ckpt {
        let device_ad = device.clone().autodiff();
        let cfg = HexPlaneDeformConfig {
            hex_plane: HexPlaneConfig {
                n_feature_dim: 16,
                spatial_resolution: 64,
                time_resolution: 32,
                coord_scale: deform_extent,
                ..HexPlaneConfig::default()
            },
            mlp_hidden: 128,
            mlp_layers: 2,
            predict_scaling: false,
            enable_time: true,
            time_enc: brush_deform::TimeEncodingConfig {
                n_freqs: 10,
                min_freq: 0.2,
                max_freq: 1.5,
                ..Default::default()
            },
            plane_tv_weight: 0.0,
            rigid_anchor_weight: 0.0,
        };
        let model = HexPlaneDeformModel::new(cfg, &device_ad);
        type DeformRec = <DeformNetwork as burn::module::Module>::Record;
        let rec: DeformRec =
            BinFileRecorder::<FullPrecisionSettings>::new().load(cp.clone(), &device_ad)
                .map_err(|e| anyhow::anyhow!("load {cp:?}: {e}"))?;
        let hex_rec = match rec {
            DeformRec::HexPlane(r) => r,
            _ => anyhow::bail!("ckpt is not a HexPlane deform (got HashGrid)"),
        };
        let model = model.load_record(hex_rec);
        let n_s = means.len() / 3;
        let xyz = Tensor::<2>::from_data(TensorData::new::<f32, _>(means.clone(), [n_s, 3]), &device_ad);
        let phase_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![0.0; n_s], [n_s, 1]), &device_ad);
        let time_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![0.0; n_s], [n_s, 1]), &device_ad);
        let deforms = model.forward(xyz.clone(), phase_t, time_t);
        let canonical_ad = XRaySplats::from_raw(means.clone(), rots.clone(), log_scales.clone(), raw.clone(), &device_ad);
        let def_ad = deform_splats(&canonical_ad, &deforms);
        let dmeans: Vec<f32> = def_ad.means().into_data_async().await?.to_vec()?;
        let drots: Vec<f32> = def_ad.rotations().into_data_async().await?.to_vec()?;
        let dv: Vec<f32> = deforms.d_xyz.into_data_async().await?.to_vec()?;
        let mut dmax = 0.0f32;
        for k in 0..n_s {
            dmax = dmax.max(glam::Vec3::new(dv[k*3], dv[k*3+1], dv[k*3+2]).length());
        }
        println!("deform via network ckpt {cp:?}: max|d|={dmax:.2}mm (含 d_rotation)");
        means = dmeans;
        rots = drots;
    } else if let Some(df) = &deform_field {
        let (field, shape) = read_npy_f32(df)?;
        let n3 = [shape[0], shape[1], shape[2]];
        assert_eq!(shape.len(), 4, "deform field must be [nx,ny,nz,3]");
        assert_eq!(field.len(), n3[0] * n3[1] * n3[2] * 3, "deform field size");
        let mut dmax = 0.0f32;
        let canon0 = if n > 0 { [means[0], means[1], means[2]] } else { [0.0; 3] };
        for i in 0..n {
            let p = glam::Vec3::new(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]);
            let d = sample_deform_field(&field, n3, deform_extent, p);
            means[i * 3] = p.x + d.x;
            means[i * 3 + 1] = p.y + d.y;
            means[i * 3 + 2] = p.z + d.z;
            dmax = dmax.max(d.length());
        }
        if n > 0 {
            println!("deform sample0: canon=({:.4},{:.4},{:.4}) -> def=({:.4},{:.4},{:.4})",
                canon0[0], canon0[1], canon0[2], means[0], means[1], means[2]);
        }
        println!("applied deform field {df:?} (extent {deform_extent}mm), max |d|={dmax:.2}mm");
    }

    // ---- 密度域: raw logits 直通 —— voxelizer 内核激活 (与 brush-xray
    //      同约定: unsigned = MU_WATER·silu(raw), --signed = MU_WATER·raw) ----
    // (旧绕行在 host 反解 logit, 对 μ>1 / 负密度有损 clamp; 现已移除。)

    // ---- FDK 体积 + meta (read_nifti_volume 已转内部 x-major 布局) ----
    let mut rx0 = 118.6f32;
    let mut rz0 = 84.7f32;
    let mut fdk_x = 256usize;
    let mut fdk_y = 256usize;
    let mut fdk_z = 183usize;
    let mut fdk_vec: Vec<f32> = Vec::new();
    if !no_fdk {
        let meta_path = fdk_meta.unwrap_or_else(|| fdk_vol.with_file_name("meta.json"));
        let _calib_path = fdk_calib.unwrap_or_else(|| fdk_vol.with_file_name("calib.json"));
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        rx0 = meta["rx"].as_f64().unwrap_or(118.6) as f32;
        rz0 = meta["rz"].as_f64().unwrap_or(84.7) as f32;
        let (fdk_vec0, _hx, _hy, _hz) = read_nifti_volume(&fdk_vol)?;
        fdk_x = meta["vol_x"].as_u64().unwrap_or(_hx as u64) as usize;
        fdk_y = meta["vol_y"].as_u64().unwrap_or(_hy as u64) as usize;
        fdk_z = meta["vol_z"].as_u64().unwrap_or(_hz as u64) as usize;
        assert_eq!(fdk_vec0.len(), fdk_x * fdk_y * fdk_z, "FDK size mismatch");
        fdk_vec = fdk_vec0;
    }

    // ---- GS 体素化 (网格 = FDK XY padding 到 n_xy, spacing 保持) ----
    let vox_mm = 2.0 * rx0 / fdk_x as f32; // 0.927mm
    let rx = vox_mm * n_xy as f32 / 2.0;   // n_xy 世界半宽
    let rz = rz0;                          // z 不变 (FDK 183)
    let n_z = fdk_z;
    let mut settings = VoxelSettings::new(
        glam::uvec3(n_xy as u32, n_xy as u32, n_z as u32),
        glam::vec3(2.0 * rx, 2.0 * rx, 2.0 * rz),
        glam::Vec3::ZERO,
    );
    if signed {
        settings = settings.with_signed_opac(true);
    }
    println!(
        "voxelize GS: grid {n_xy}x{n_xy}x{n_z}, voxel {vox_mm:.3}mm, world {rx:.1}x{rx:.1}x{rz:.1}mm, signed={signed} (kernel MU_WATER·silu/raw)"
    );
    let splats = XRaySplats::from_raw(means, rots, log_scales, raw, &device);
    let v_vol = voxelize_forward(&splats, &settings).await; // x-major [n_x, n_y, n_z]
    let gs_vol: Vec<f32> = v_vol.into_data().to_vec()?;

    // dump 原生布局 (x-major, 与 DRR/FDK 一致) 供对照
    write_nrrd_3d(&out.join("gs_native_raw.nrrd"), &gs_vol, n_xy, n_xy, n_z)?;
    // 公共布局 = x-major: voxelizer 输出即公共布局, 零转置。
    let gs_common = gs_vol;

    // ---- FDK 居中 padding 到 n_xy (XY), 公共布局 x-major; no-fdk 全 0 ----
    let ox = (n_xy - fdk_x) / 2;
    let oy = (n_xy - fdk_y) / 2;
    let mut fdk_pad = vec![0.0f32; n_xy * n_xy * n_z];
    if !no_fdk {
        for ix in 0..fdk_x {
            for iy in 0..fdk_y {
                for iz in 0..fdk_z {
                    let src = ix * (fdk_y * fdk_z) + iy * fdk_z + iz;
                    let dst = (ix + ox) * (n_xy * n_z) + (iy + oy) * n_z + iz;
                    fdk_pad[dst] = fdk_vec[src];
                }
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
