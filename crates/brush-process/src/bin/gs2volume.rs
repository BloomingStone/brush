//! GS2volume: 把训练好的 GS 点云 (canonical_final.ply) 体素化 (brush-voxel),
//! 与 FDK 体积相加 (可选), 输出为 nii.gz + NRRD 供目视监察 GS 重建是否有
//! 真实 3D 结构 (vs 仅靠投影拟合的条纹); 可选 `--compare-drr` 用 DRR 渲染
//! 体积并与 GS 直接投影对比。
//!
//! 体积网格: 默认跟随 FDK (XY padding 到 N, spacing 同 FDK);
//! **独立尺寸**: `--ref-dcm=<dcm>` 从 DICOM 几何推断 (XY = 1.5× 图像等中心
//! 宽度, Z = 图像等中心高度, 与 fit_static 初始 GS 点范围一致), 或显式
//! `--extent-xy=MM --extent-z=MM`, 配合 `--voxel-mm` (默认 1.0)。
//! 内部统一 x-major 布局 `idx(x,y,z) = x*(ny*nz) + y*nz + z`
//! (同 voxelizer 输出 / DRR 内核 / R2 fields); nifti/nrrd 写入时转磁盘
//! 列优先 (x 最快, NIfTI-1/NRRD 标准, 文件 dims 自然 (X,Y,Z))。

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, RoiSpec, XRayOrientation};
use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig, HexPlaneDeformModel, deform_splats};
use brush_drr::{DrrOps, DrrSettings};
use brush_train::xray_train::DeformNetwork;
use brush_serde::import::load_splat_from_ply;
use brush_voxel::{VoxelSettings, voxelize_forward};
use brush_xray::{XRaySplats, render_xray_forward};
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
    let mut n_z = 0usize; // 0 = 自动 (FDK z 或 extent/voxel-mm)
    let mut n_xy_given = false;
    let mut n_z_given = false;
    let mut signed = false;
    let mut deform_field: Option<PathBuf> = None;
    let mut deform_extent = 264.0f32;
    let mut ckpt: Option<PathBuf> = None; // HexPlane 网络权重 (deform_final.bin)
    let mut no_fdk = false;
    // 独立体积尺寸: --ref-dcm 从 DICOM 推断 (XY=1.5×图宽世界, Z=图高世界),
    // --extent-xy/--extent-z 显式指定世界范围 (mm), --voxel-mm 体素尺寸。
    let mut ref_dcm: Option<PathBuf> = None;
    let mut extent_xy: Option<f32> = None;
    let mut extent_z: Option<f32> = None;
    let mut voxel_mm = 1.0f32;
    let mut compare_drr = false;
    // --eval-split-every=N: 与训练一致的 held-out 视图选择 (否则用 train 视图)。
    let mut eval_split: Option<usize> = None;
    let mut dump_isects = false;
    // --bin=<prefix>: 直接从 fit_static 导出的原始参数 .bin 加载
    // (<prefix>_transforms.bin [N,10] f32 + <prefix>_raw.bin [N] f32, raw 域,
    // 无 PLY 激活域往返), 与 --ply= 互斥。
    let mut bin_prefix: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--bin=") {
            bin_prefix = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--ply=") {
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
            n_xy_given = true;
        } else if let Some(v) = a.strip_prefix("--n-z=") {
            n_z = v.parse()?;
            n_z_given = true;
        } else if let Some(v) = a.strip_prefix("--ref-dcm=") {
            ref_dcm = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--extent-xy=") {
            extent_xy = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--extent-z=") {
            extent_z = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--voxel-mm=") {
            voxel_mm = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--eval-split-every=") {
            eval_split = Some(v.parse()?);
        } else if a == "--compare-drr" {
            compare_drr = true;
        } else if a == "--dump-isects" {
            dump_isects = true;
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
    // 输入源: --bin=<prefix> (fit_static 原始参数, 无 PLY 往返) 或 --ply=。
    let mut means: Vec<f32>;
    let mut rots: Vec<f32>;
    let mut log_scales: Vec<f32>;
    let mut raw: Vec<f32>;
    let mut n;
    if let Some(prefix) = bin_prefix {
        let t_path = prefix.with_file_name(format!(
            "{}_transforms.bin",
            prefix.file_name().unwrap_or_default().to_string_lossy()
        ));
        let o_path = prefix.with_file_name(format!(
            "{}_raw.bin",
            prefix.file_name().unwrap_or_default().to_string_lossy()
        ));
        let t_bytes = std::fs::read(&t_path)?;
        let o_bytes = std::fs::read(&o_path)?;
        assert_eq!(t_bytes.len() % 40, 0, "transforms.bin must be [N,10] f32");
        let t: Vec<f32> = t_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let o: Vec<f32> = o_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        n = t.len() / 10;
        assert_eq!(o.len(), n, "raw.bin size mismatch");
        means = t.chunks_exact(10).flat_map(|c| c[0..3].to_vec()).collect();
        rots = t.chunks_exact(10).flat_map(|c| c[3..7].to_vec()).collect();
        log_scales = t.chunks_exact(10).flat_map(|c| c[7..10].to_vec()).collect();
        raw = o;
        println!("loaded {n} splats from raw .bin ({t_path:?} + raw)");
    } else {
        let ply = ply.expect(
            "usage: gs2volume [--bin=<prefix> | --ply=<canonical_final.ply>] [--fdk=<nii> --fdk-meta=<json> --fdk-calib=<json> | --no-fdk] \
             [--ref-dcm=<dcm> | --extent-xy=MM --extent-z=MM] [--voxel-mm=MM] [--n-xy=N] [--n-z=N] \
             [--compare-drr] [--signed] [--out=<dir>]",
        );
        // ---- 加载 GS ply ----
        let file = tokio::fs::File::open(&ply).await?;
        let msg = load_splat_from_ply(file, None).await?;
        let data = msg.data;
        n = data.num_splats();
        means = data.means;
        rots = data.rotations.unwrap_or_else(|| {
            let mut v = Vec::with_capacity(n * 4);
            for _ in 0..n {
                v.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            }
            v
        });
        log_scales = data.log_scales.unwrap();
        raw = data.raw_opacities.unwrap();
        println!("loaded {n} splats from {ply:?}");
    }
    std::fs::create_dir_all(&out)?;

    // ---- 后端 + 设备 ----
    let wgpu = brush_process::burn_init_setup().await;
    let device: Device = wgpu.into();

    // ---- 形变场 (phase 0, time 0): canonical → deformed (实际渲染位置) ----
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
    // FDK 仅用于旧式网格 (spacing 跟随 FDK) 或叠加参考; --ref-dcm /
    // --extent-* 独立尺寸模式下列为可叠加项 (spacing 不同时警告并跳过)。
    let mut rx0 = 118.6f32;
    let mut rz0 = 84.7f32;
    let mut fdk_x = 256usize;
    let mut fdk_y = 256usize;
    let mut fdk_z = 183usize;
    let mut fdk_vec: Vec<f32> = Vec::new();
    if let Some(fdk_vol_path) = fdk_vol.clone() {
        let meta_path = fdk_meta.unwrap_or_else(|| fdk_vol_path.with_file_name("meta.json"));
        let _calib_path = fdk_calib.unwrap_or_else(|| fdk_vol_path.with_file_name("calib.json"));
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        rx0 = meta["rx"].as_f64().unwrap_or(118.6) as f32;
        rz0 = meta["rz"].as_f64().unwrap_or(84.7) as f32;
        let (fdk_vec0, _hx, _hy, _hz) = read_nifti_volume(&fdk_vol_path)?;
        fdk_x = meta["vol_x"].as_u64().unwrap_or(_hx as u64) as usize;
        fdk_y = meta["vol_y"].as_u64().unwrap_or(_hy as u64) as usize;
        fdk_z = meta["vol_z"].as_u64().unwrap_or(_hz as u64) as usize;
        assert_eq!(fdk_vec0.len(), fdk_x * fdk_y * fdk_z, "FDK size mismatch");
        fdk_vec = fdk_vec0;
    } else if !no_fdk {
        eprintln!(
            "note: no --fdk and no --no-fdk; exporting pure GS volume (FDK overlay skipped)"
        );
    }

    // ---- 体积几何: 独立尺寸 (--ref-dcm / --extent-*) 或 FDK 跟随 ----
    // --ref-dcm: XY = 1.5× 图像等中心宽度, Z = 图像等中心高度 (与 fit_static
    // 初始 cylinder 半径 half_w·1.5 及图高一致)。
    let mut half_w: Option<f32> = None;
    let mut half_z: Option<f32> = None;
    if let Some(dcm_path) = &ref_dcm {
        let bytes = std::fs::read(dcm_path)?;
        let meta = brush_dicom::parse_dicom(&bytes)?;
        let g = meta.geometry;
        let w_world = g.width as f32 * g.delx as f32 * g.sod as f32 / g.sdd as f32;
        let h_world = g.height as f32 * g.dely as f32 * g.sod as f32 / g.sdd as f32;
        half_w = Some(0.75 * w_world); // XY extent = 1.5 × 宽 → 半宽
        half_z = Some(0.5 * h_world);
        println!(
            "ref-dcm {dcm_path:?}: {}x{}px, iso w={w_world:.1}mm h={h_world:.1}mm -> volume XY={:.1}mm Z={h_world:.1}mm",
            g.width, g.height, 1.5 * w_world,
        );
    }
    if let Some(e) = extent_xy {
        half_w = Some(0.5 * e);
    }
    if let Some(e) = extent_z {
        half_z = Some(0.5 * e);
    }

    // 网格: n_xy/n_z 显式 > 由 extent/voxel-mm 推出 > FDK 跟随。
    let (n_xy, n_z, rx, rz, vox_mm) = match (half_w, half_z) {
        (Some(hw), Some(hz)) => {
            assert!(voxel_mm > 0.0, "--voxel-mm must be > 0");
            let n_xy = if n_xy_given {
                n_xy
            } else {
                (2.0 * hw / voxel_mm).round().max(1.0) as usize
            };
            let n_z = if n_z_given {
                n_z
            } else {
                (2.0 * hz / voxel_mm).round().max(1.0) as usize
            };
            let rx = n_xy as f32 * voxel_mm / 2.0;
            let rz = n_z as f32 * voxel_mm / 2.0;
            println!(
                "independent volume grid: {n_xy}x{n_xy}x{n_z}, voxel {voxel_mm:.3}mm, world {:.1}x{:.1}x{:.1}mm",
                2.0 * rx, 2.0 * rx, 2.0 * rz
            );
            (n_xy, n_z, rx, rz, voxel_mm)
        }
        _ => {
            // FDK 跟随 (旧行为): spacing = FDK voxel, XY padding 到 n_xy。
            let vox_mm = 2.0 * rx0 / fdk_x as f32;
            let rx = vox_mm * n_xy as f32 / 2.0;
            let n_z = if n_z_given { n_z } else { fdk_z };
            let rz = vox_mm * n_z as f32 / 2.0;
            println!(
                "FDK-following grid: {n_xy}x{n_xy}x{n_z}, voxel {vox_mm:.3}mm, world {:.1}x{:.1}x{:.1}mm",
                2.0 * rx, 2.0 * rx, 2.0 * rz
            );
            (n_xy, n_z, rx, rz, vox_mm)
        }
    };
    // FDK 叠加仅在 spacing 匹配时有效。
    let fdk_ok = !fdk_vec.is_empty()
        && (half_w.is_none() && half_z.is_none()
            || (fdk_x as f32 * vox_mm - 2.0 * rx0).abs() < 1e-3
                && (fdk_z as f32 * vox_mm - 2.0 * rz0).abs() < 1e-3);
    if !fdk_vec.is_empty() && !fdk_ok {
        eprintln!("warning: FDK spacing ({}mm) != GS voxel ({vox_mm:.3}mm); FDK overlay skipped", 2.0 * rx0 / fdk_x as f32);
    }

    let mut settings = VoxelSettings::new(
        glam::uvec3(n_xy as u32, n_xy as u32, n_z as u32),
        glam::vec3(2.0 * rx, 2.0 * rx, 2.0 * rz),
        glam::Vec3::ZERO,
    );
    if signed {
        settings = settings.with_signed_opac(true);
    }
    println!(
        "voxelize GS: grid {n_xy}x{n_xy}x{n_z}, world {:.1}x{:.1}x{:.1}mm, signed={signed} (kernel MU_WATER·silu/raw)",
        2.0 * rx, 2.0 * rx, 2.0 * rz
    );
    let splats = XRaySplats::from_raw(means, rots, log_scales, raw, &device);
    let v_vol = if dump_isects {
        // 诊断: 直连 raw pipeline, 落盘中间数组 (cube_offsets / isect / projected)。
        use brush_voxel::{VoxelOps, VoxelPass};
        use burn::backend::ops::{FloatTensorOps, IntTensorOps};
        let transforms = brush_render::burn_glue::unwrap_wgpu_float(splats.transforms.val());
        let raw_opac = brush_render::burn_glue::unwrap_wgpu_float(splats.raw_opacities.val());
        let transforms_dump = transforms.clone();
        let raw_dump = raw_opac.clone();
        let vout = <brush_cube::MainBackend as VoxelOps>::voxelize(
            &settings,
            transforms,
            raw_opac,
            VoxelPass::Forward,
        )
        .await;
        let vol_data = brush_cube::MainBackend::float_into_data(vout.out_volume.clone())
            .await
            .expect("volume readback");
        let cube_offsets = brush_cube::MainBackend::int_into_data(vout.aux.cube_offsets)
            .await
            .expect("cube_offsets readback");
        let isect = brush_cube::MainBackend::int_into_data(vout.compact_gid_from_isect)
            .await
            .expect("isect readback");
        let projected = brush_cube::MainBackend::float_into_data(vout.projected_splats)
            .await
            .expect("projected readback");
        let g2c = brush_cube::MainBackend::int_into_data(vout.global_from_compact_gid)
            .await
            .expect("g2c readback");
        let transforms_data = brush_cube::MainBackend::float_into_data(transforms_dump)
            .await
            .expect("transforms readback");
        let raw_data = brush_cube::MainBackend::float_into_data(raw_dump)
            .await
            .expect("raw readback");
        println!(
            "dump-isects: num_visible={} num_intersections={}",
            vout.aux.num_visible, vout.aux.num_intersections
        );
        let save_raw = |name: &str, bytes: &[u8]| {
            std::fs::write(out.join(name), bytes).expect("dump write");
        };
        save_raw("dbg_cube_offsets.bin", cube_offsets.as_bytes());
        save_raw("dbg_isect.bin", isect.as_bytes());
        save_raw("dbg_projected.bin", projected.as_bytes());
        save_raw("dbg_g2c.bin", g2c.as_bytes());
        save_raw("dbg_volume.bin", vol_data.as_bytes());
        save_raw("dbg_transforms.bin", transforms_data.as_bytes());
        save_raw("dbg_raw.bin", raw_data.as_bytes());
        // 转回 burn Tensor 供后续统一路径 (compare 等)。
        use brush_render::burn_glue::wrap_wgpu_float;
        wrap_wgpu_float::<3>(vout.out_volume)
    } else {
        voxelize_forward(&splats, &settings).await // x-major [n_x, n_y, n_z]
    };
    let gs_vol: Vec<f32> = v_vol.clone().into_data().to_vec()?;

    // dump 原生布局 (x-major, 与 DRR/FDK 一致) 供对照
    write_nrrd_3d(&out.join("gs_native_raw.nrrd"), &gs_vol, n_xy, n_xy, n_z)?;
    // 公共布局 = x-major: voxelizer 输出即公共布局, 零转置。
    let gs_common = gs_vol;

    // ---- FDK 居中 padding 到 n_xy (XY), 公共布局 x-major; 无 FDK 全 0 ----
    let ox = (n_xy.saturating_sub(fdk_x)) / 2;
    let oy = (n_xy.saturating_sub(fdk_y)) / 2;
    let mut fdk_pad = vec![0.0f32; n_xy * n_xy * n_z];
    if fdk_ok {
        let (fx0, fy0) = (fdk_x.min(n_xy), fdk_y.min(n_xy));
        for ix in 0..fx0 {
            for iy in 0..fy0 {
                for iz in 0..fdk_z.min(n_z) {
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

    // ---- DRR vs GS 直接投影对比 (--compare-drr, 需 --ref-dcm) ----
    // 对均匀抽样的视图: GS xray 直接渲染 vs 体积 DRR 渲染 (两者都是
    // proj = ∫μ dl 域), 输出 GT|exp(-gs)|exp(-drr) NRRD stack + 指标。
    if compare_drr {
        let dcm_path = ref_dcm.as_ref().expect("--compare-drr requires --ref-dcm=<dcm>");
        use brush_vfs::BrushVfs;
        use std::sync::Arc;
        let file = tokio::fs::File::open(dcm_path).await?;
        let name = dcm_path
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
            max_resolution: 1920,
            eval_split_every: eval_split,
            subsample_frames: None,
            subsample_points: None,
            alpha_mode: None,
            dicom_orientation: XRayOrientation::Ap,
            dicom_normalization: DicomNormalization::Minmax,
            dicom_gamma: None,
            // 与 fit_static 默认一致的 gamma (中位数→0.5), 否则 compare 的
            // GT 子图偏黑, 与训练 eval 栈不可比。
            dicom_gamma_target: Some(0.5),
            roi: RoiSpec::Inset(20),
            max_scene_batch_cache_size: 1 << 30,
        };
        let result = brush_dataset::load_dataset(vfs, &load_config).await?;
        // 与 fit_static 相同的 eval 视图选择: --eval-split-every 时用
        // held-out 集合 (与 gt_pred 栈同帧), 否则 train 视图均匀抽样。
        let views = if let Some(eval_scene) = &result.dataset.eval
            && !eval_scene.views.is_empty()
        {
            println!(
                "eval-split-every={:?}: 用 held-out eval 集 ({} views, 与训练 gt_pred 同帧)",
                eval_split,
                eval_scene.views.len()
            );
            &eval_scene.views
        } else {
            &result.dataset.train.views
        };
        let n_views = views.len();
        let sel: Vec<usize> = (0..8usize.min(n_views)).map(|i| i * n_views / 8).collect();
        println!("DRR vs GS compare: {} views (of {n_views})", sel.len());

        // DRR 步数: dt ≲ 0.5·voxel。
        let ray_half = (rx * rx + rx * rx + rz * rz).sqrt();
        let steps = ((2.0 * ray_half) / (0.5 * vox_mm)).ceil().max(256.0) as u32;

        let mut stack_imgs: Vec<Vec<f32>> = Vec::new();
        let mut stack_gs_proj: Vec<Vec<f32>> = Vec::new();
        let mut stack_drr_proj: Vec<Vec<f32>> = Vec::new();
        let mut h0 = 0usize;
        let mut w0 = 0usize;
        let mut metrics: Vec<String> = Vec::new();
        metrics.push("view_idx,mean_abs_diff,max_abs_diff,mean_rel,gs_sum,drr_sum".to_string());
        for &vi in &sel {
            let view = &views[vi];
            let gray = view.gray_image.as_ref().expect("gray GT");
            let (h, w) = (gray.height as usize, gray.width as usize);
            h0 = h;
            w0 = w;
            let img = glam::uvec2(w as u32, h as u32);
            let cam = view.camera;
            // GS 直接渲染 (raw logits, unsigned 内核激活)。
            let gs_proj = render_xray_forward(&splats, &cam, img, 1.0).await;
            let gs_v: Vec<f32> = gs_proj
                .to_data_async()
                .await
                .expect("gs proj readback")
                .as_slice::<f32>()
                .expect("f32")
                .to_vec();
            // 体积 DRR (x-major 直连, scale=1 bias=0)。
            let drr_settings = DrrSettings::new(
                &cam,
                img.x,
                img.y,
                n_xy as u32,
                n_xy as u32,
                n_z as u32,
                steps,
                rx,
                rx,
                rz,
                1.0,
                0.0,
            );
            let vol_ft =
                brush_render::burn_glue::unwrap_wgpu_float(v_vol.clone());
            let drr_proj = <brush_cube::MainBackend as DrrOps>::drr_forward(&drr_settings, vol_ft)
                .await;
            let drr_t = brush_render::burn_glue::wrap_wgpu_float::<2>(drr_proj);
            let drr_v: Vec<f32> = drr_t
                .to_data_async()
                .await
                .expect("drr readback")
                .as_slice::<f32>()
                .expect("f32")
                .to_vec();

            // 指标 (proj 域)。
            let mut max_abs = 0.0f32;
            let mut sum_abs = 0.0f32;
            let mut sum_rel = 0.0f32;
            let mut n_m = 0usize;
            let (mut gs_sum, mut drr_sum) = (0.0f32, 0.0f32);
            for k in 0..gs_v.len() {
                let d = (drr_v[k] - gs_v[k]).abs();
                sum_abs += d;
                if d > max_abs {
                    max_abs = d;
                }
                if gs_v[k].abs() > 1e-4 {
                    sum_rel += d / gs_v[k].abs();
                    n_m += 1;
                }
                gs_sum += gs_v[k];
                drr_sum += drr_v[k];
            }
            let mean_abs = sum_abs / gs_v.len().max(1) as f32;
            let mean_rel = if n_m > 0 { sum_rel / n_m as f32 } else { 0.0 };
            println!(
                "  view {vi:3}: mean|drr-gs|={mean_abs:.4} max={max_abs:.4} mean_rel={mean_rel:.4} gs_sum={gs_sum:.3} drr_sum={drr_sum:.3}"
            );
            metrics.push(format!(
                "{vi},{mean_abs:.5},{max_abs:.5},{mean_rel:.5},{gs_sum:.3},{drr_sum:.3}"
            ));

            // stack 行: 逐行交错 [GT | exp(-gs) | exp(-drr)], 行序自底向上
            // (与 save_gray_nrrd_f32_stack 的 NRRD 显示约定一致, 避免上下颠倒)。
            let to_int = |p: &[f32]| -> Vec<f32> {
                p.iter().map(|&v| (-v.clamp(1e-3, 14.0)).exp()).collect()
            };
            let gs_i = to_int(&gs_v);
            let drr_i = to_int(&drr_v);
            let mut row = Vec::with_capacity(h * w * 3);
            for y in (0..h).rev() {
                let (gt_line, gs_line, drr_line) = (
                    &gray.data.as_ref()[y * w..(y + 1) * w],
                    &gs_i[y * w..(y + 1) * w],
                    &drr_i[y * w..(y + 1) * w],
                );
                row.extend_from_slice(gt_line);
                row.extend_from_slice(gs_line);
                row.extend_from_slice(drr_line);
            }
            stack_imgs.push(row);
            // raw proj 转储也用自底向上行序。
            let mut gs_line = Vec::with_capacity(h * w);
            let mut drr_line = Vec::with_capacity(h * w);
            for y in (0..h).rev() {
                gs_line.extend_from_slice(&gs_v[y * w..(y + 1) * w]);
                drr_line.extend_from_slice(&drr_v[y * w..(y + 1) * w]);
            }
            stack_gs_proj.push(gs_line);
            stack_drr_proj.push(drr_line);
        }
        // NRRD stack [N, H, 3W] (z = 视图)。
        {
            let mut payload = Vec::with_capacity(stack_imgs.len() * h0 * w0 * 3 * 4);
            for img in &stack_imgs {
                for v in img {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
            }
            let header = format!(
                "NRRD0004\n# DRR vs GS compare: GT | exp(-GS proj) | exp(-DRR proj), 视图 z\n\
                 type: float\ndimension: 3\nsizes: {} {} {}\nencoding: raw\nendian: little\n\n",
                w0 * 3,
                h0,
                stack_imgs.len()
            );
            let mut f = Vec::with_capacity(header.len() + payload.len());
            f.extend_from_slice(header.as_bytes());
            f.append(&mut payload);
            std::fs::write(out.join("compare_drr_vs_gs.nrrd"), f)?;
        }
        std::fs::write(out.join("compare_drr_vs_gs.csv"), metrics.join("\n"))?;
        // Raw proj 转储 (诊断): gs_proj / drr_proj 每视图 [H,W]。
        for (tag, stack) in [("gs_proj", &stack_gs_proj), ("drr_proj", &stack_drr_proj)] {
            let mut payload = Vec::with_capacity(stack.len() * h0 * w0 * 4);
            for img in stack {
                for v in img {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
            }
            let header = format!(
                "NRRD0004\n# raw proj {tag} (x-ray / DRR), 视图 z\n\
                 type: float\ndimension: 3\nsizes: {w0} {h0} {}\nencoding: raw\nendian: little\n\n",
                stack.len()
            );
            let mut f = Vec::with_capacity(header.len() + payload.len());
            f.extend_from_slice(header.as_bytes());
            f.append(&mut payload);
            std::fs::write(out.join(format!("compare_{tag}.nrrd")), f)?;
        }
        println!("saved -> {out:?}/compare_drr_vs_gs.nrrd (GT|GS|DRR) + .csv + raw proj");
    }
    Ok(())
}
