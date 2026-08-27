//! FDK cone-beam 重建 → 各向异性静态先验体积 + 逐视角 DRR 预计算 + LSQ 标定。
//!
//! 用途: 作为 GS 残差建模的静态先验。本 bin:
//!   1. 每帧投影 `p = -ln(gray)`, 2D cosine 加权 + 频域 ramp 滤波 (Ram-Lak+Hann)。
//!   2. FDK 反投影到**各向异性**圆柱 FOV: XY(旋转平面) 域 `[-rx,rx]^2`
//!      (rx = half_w × pad_xy, padding 容纳投影内角落结构), Z 方向 `[-rz,rz]`
//!      (rz = half_h, 只需与投影高度平齐)。**只填充全采样圆柱**
//!      (`x²+y² < half_w²`), 部分采样环带置 0 (240° 短扫描下伪影 > 缺失)。
//!   3. 存 `<out>/volume.nii.gz` (float32, affine 由 rx/ry/rz 推导) + meta.json。
//!   4. 正向投影 (射线步进 ∫μ dl) 若干视图 → 测耗时 + LSQ 标定。
//!
//! 用法:
//!   fdk_volume images/pig-data-cor-new-phase.dcm --vol-x=256 [--pad-xy=1.3]
//!     [--roi=40] [--mu-scale=0.00021] --out=.../fdk

use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene::SceneView;
use brush_vfs::BrushVfs;
use glam::{Affine3A, Vec3};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex, Fft};

/// 频域 ramp 滤波 (Ram-Lak + Hann 窗)。
fn ramp_filter_row_fft(row: &[f32], fft: &Arc<dyn Fft<f32>>, ifft: &Arc<dyn Fft<f32>>) -> Vec<f32> {
    let n = row.len();
    let n2 = n * 2;
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n2];
    for i in 0..n {
        buf[i] = Complex::new(row[i], 0.0);
    }
    fft.process(&mut buf);
    let half = n2 / 2;
    for k in 0..n2 {
        let freq = if k <= half { k as f32 } else { (n2 - k) as f32 };
        let hann = 0.5 + 0.5 * (PI * freq / half as f32).cos();
        buf[k] *= freq * hann;
    }
    ifft.process(&mut buf);
    let inv = 1.0 / n2 as f32;
    buf[..n].iter().map(|c| c.re * inv).collect()
}

/// 单帧 FDK 数据。
struct Frame {
    w2l: Affine3A,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    sod: f32,
    width: usize,
    height: usize,
    filtered: Vec<f32>,
}

/// 对每帧构建 `proj = -ln(gray)`, 2D cosine 加权 + ramp 滤波 + Parker 加权。
fn build_frames(views: &[SceneView], delx: f32) -> Vec<Frame> {
    let mut planner = FftPlanner::<f32>::new();
    let w0 = views[0].gray_image.as_ref().expect("gray GT").width as usize;
    let fft = planner.plan_fft_forward(2 * w0);
    let ifft = planner.plan_fft_inverse(2 * w0);
    let fft = Arc::new(fft);
    let ifft = Arc::new(ifft);

    let angles: Vec<f32> = views
        .iter()
        .map(|v| v.camera.position.x.atan2(v.camera.position.y))
        .collect();
    let a_start = angles.iter().cloned().fold(f32::INFINITY, f32::min);
    let a_end = angles.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let delta = a_end - a_start;
    let g0 = views[0].gray_image.as_ref().expect("gray GT");
    let fx0 = views[0].camera.focal(glam::uvec2(g0.width, g0.height)).x;
    let fan_half = ((g0.width as f32 * 0.5) / fx0).atan();
    let parker_span = delta + 2.0 * fan_half;
    let is_short = delta < 6.0;
    println!(
        "Parker: Δ={:.1}° fan_half={:.1}° span={:.1}° applied={is_short}",
        delta.to_degrees(),
        fan_half.to_degrees(),
        parker_span.to_degrees()
    );

    views
        .par_iter()
        .zip(angles.par_iter())
        .map(|(view, &alpha)| {
            let parker = if is_short {
                let u = ((alpha - a_start) / parker_span).clamp(0.0, 1.0);
                (PI * u).sin().powi(2)
            } else {
                1.0
            };
            let gray = view.gray_image.as_ref().expect("gray GT");
            let (w, h) = (gray.width as usize, gray.height as usize);
            let img_size = glam::uvec2(w as u32, h as u32);
            let cam = &view.camera;
            let focal = cam.focal(img_size);
            let center = cam.center(img_size);
            let sod = cam.position.length();
            let (fx, fy) = (focal.x, focal.y);
            let (cx, cy) = (center.x, center.y);

            let mut proj = vec![0.0f32; w * h];
            for (i, &g) in gray.data.iter().enumerate() {
                proj[i] = -g.clamp(1e-3, 1.0).ln();
            }
            let det_px = delx;
            let mut weighted = vec![0.0f32; w * h];
            for y in 0..h {
                let v_det = (y as f32 - cy) * det_px;
                for x in 0..w {
                    let u_det = (x as f32 - cx) * det_px;
                    let cos_w = sod / (sod * sod + u_det * u_det + v_det * v_det).sqrt();
                    weighted[y * w + x] = proj[y * w + x] * cos_w;
                }
            }
            let du_world = sod / fx;
            let mut filtered = vec![0.0f32; w * h];
            for y in 0..h {
                let row = &weighted[y * w..(y + 1) * w];
                let filt = ramp_filter_row_fft(row, &fft, &ifft);
                for x in 0..w {
                    filtered[y * w + x] = filt[x] * du_world * parker;
                }
            }
            Frame {
                w2l: cam.world_to_local(),
                fx,
                fy,
                cx,
                cy,
                sod,
                width: w,
                height: h,
                filtered,
            }
        })
        .collect()
}

/// 各向异性 FDK 反投影: 网格 `[vol_x, vol_y, vol_z]`, 世界域
/// `[-rx,rx]x[-ry,ry]x[-rz,rz]`, **只填充全采样圆柱** `x²+y² < mask_r²`
/// (mask_r = half_w, 部分采样环带置 0)。内存布局 x-major:
/// `idx(x,y,z) = x*(vol_y*vol_z) + y*vol_z + z` (x 最慢, 同 DRR 内核 /
/// brush-voxel)。
fn backproject_aniso(
    frames: &[Frame],
    vol_x: usize,
    vol_y: usize,
    vol_z: usize,
    rx: f32,
    ry: f32,
    rz: f32,
    mask_r: f32,
) -> Vec<f32> {
    let dx = 2.0 * rx / vol_x as f32;
    let dy = 2.0 * ry / vol_y as f32;
    let dz = 2.0 * rz / vol_z as f32;
    let mask_r2 = mask_r * mask_r;
    let sy = vol_y * vol_z;
    let mut volume = vec![0.0f32; vol_x * vol_y * vol_z];
    volume
        .par_chunks_mut(sy)
        .enumerate()
        .for_each(|(ix, slice)| {
            let x = -rx + (ix as f32 + 0.5) * dx;
            for iy in 0..vol_y {
                let y = -ry + (iy as f32 + 0.5) * dy;
                if x * x + y * y > mask_r2 {
                    continue;
                }
                let row_base = iy * vol_z;
                for iz in 0..vol_z {
                    let z = -rz + (iz as f32 + 0.5) * dz;
                    let p_w = Vec3::new(x, y, z);
                    let mut acc = 0.0f32;
                    for f in frames {
                        let p_c = f.w2l.transform_point3(p_w);
                        let dzp = p_c.z;
                        if dzp <= 1.0 {
                            continue;
                        }
                        let u = f.fx * p_c.x / dzp + f.cx;
                        let v = f.fy * p_c.y / dzp + f.cy;
                        if u < 0.0 || v < 0.0 {
                            continue;
                        }
                        let ui = u as usize;
                        let vi = v as usize;
                        if ui + 1 >= f.width || vi + 1 >= f.height {
                            continue;
                        }
                        let fu = u - ui as f32;
                        let fv = v - vi as f32;
                        let row = vi * f.width;
                        let p00 = f.filtered[row + ui];
                        let p10 = f.filtered[row + ui + 1];
                        let p01 = f.filtered[row + f.width + ui];
                        let p11 = f.filtered[row + f.width + ui + 1];
                        let val = p00 * (1.0 - fu) * (1.0 - fv)
                            + p10 * fu * (1.0 - fv)
                            + p01 * (1.0 - fu) * fv
                            + p11 * fu * fv;
                        let wgt = f.sod / dzp;
                        acc += wgt * wgt * val;
                    }
                    slice[row_base + iz] = acc;
                }
            }
        });
    volume
}

/// 三线性插值采样各向异性体积 (x-major 布局, 同 DRR 内核)。
#[inline]
fn sample_vol_aniso(
    volume: &[f32],
    vol_x: usize,
    vol_y: usize,
    vol_z: usize,
    rx: f32,
    ry: f32,
    rz: f32,
    p: Vec3,
) -> f32 {
    let inv_dx = vol_x as f32 / (2.0 * rx);
    let inv_dy = vol_y as f32 / (2.0 * ry);
    let inv_dz = vol_z as f32 / (2.0 * rz);
    let fx = (p.x + rx) * inv_dx - 0.5;
    let fy = (p.y + ry) * inv_dy - 0.5;
    let fz = (p.z + rz) * inv_dz - 0.5;
    if fx < 0.0 || fy < 0.0 || fz < 0.0 {
        return 0.0;
    }
    let ix = fx as usize;
    let iy = fy as usize;
    let iz = fz as usize;
    if ix + 1 >= vol_x || iy + 1 >= vol_y || iz + 1 >= vol_z {
        return 0.0;
    }
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let tz = fz - iz as f32;
    // x-major: flat(x,y,z) = x·(vy·vz) + y·vz + z (z fastest).
    let sx = vol_y * vol_z;
    let sy = vol_z;
    let base = ix * sx + iy * sy + iz;
    let c000 = volume[base];
    let c100 = volume[base + sx];
    let c010 = volume[base + sy];
    let c110 = volume[base + sx + sy];
    let c001 = volume[base + 1];
    let c101 = volume[base + sx + 1];
    let c011 = volume[base + sy + 1];
    let c111 = volume[base + sx + sy + 1];
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    c0 * (1.0 - tz) + c1 * tz
}

/// 正向投影 (CPU 射线步进, 各向异性体积)。
fn forward_project_view(
    volume: &[f32],
    vol_x: usize,
    vol_y: usize,
    vol_z: usize,
    rx: f32,
    ry: f32,
    rz: f32,
    view: &SceneView,
    steps: usize,
) -> Vec<f32> {
    let gray = view.gray_image.as_ref().expect("gray GT");
    let (w, h) = (gray.width as usize, gray.height as usize);
    let img_size = glam::uvec2(w as u32, h as u32);
    let cam = &view.camera;
    let focal = cam.focal(img_size);
    let center = cam.center(img_size);
    let (fx, fy) = (focal.x, focal.y);
    let (cx, cy) = (center.x, center.y);
    let sod = cam.position.length();
    let t_near = sod - rx;
    let t_far = sod + rx;
    let dt = (t_far - t_near) / steps as f32;

    (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..w).map(move |x| {
                let dir = Vec3::new((x as f32 - cx) / fx, (y as f32 - cy) / fy, 1.0);
                let mut acc = 0.0f32;
                for s in 0..steps {
                    let t = t_near + (s as f32 + 0.5) * dt;
                    let p_local = dir * t;
                    let p_world = cam.local_to_world().transform_point3(p_local);
                    acc += sample_vol_aniso(volume, vol_x, vol_y, vol_z, rx, ry, rz, p_world) * dt;
                }
                acc
            })
        })
        .collect()
}

/// 写 3D 体积 .nii.gz (nifti-rs, sform_code=2)。各向异性 affine:
/// 内部布局 x-major `idx(x,y,z) = x*(vol_y*vol_z) + y*vol_z + z`, 数组形状
/// (vol_x, vol_y, vol_z); nifti-rs 内部 data.t() 转成磁盘列优先 (x 最快),
/// 文件 dims 即自然 (X,Y,Z), srow 为标准对角。
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
    // 数组轴 = [x,y,z] (x-major), dim[1..3] = (X,Y,Z) 自然顺序, 对角 srow。
    hdr.srow_x = [sx, 0.0, 0.0, -rx];
    hdr.srow_y = [0.0, sy, 0.0, -ry];
    hdr.srow_z = [0.0, 0.0, sz, -rz];
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .map_err(|e| anyhow::anyhow!("write nifti: {e}"))?;
    Ok(())
}

/// 写 .npy (v1.0, float32, C-order) — DRR 目检图。
fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> anyhow::Result<()> {
    use std::io::Write;
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "npy data len mismatch");
    let shape_str = format!(
        "({}{})",
        shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "),
        if shape.len() == 1 { "," } else { "" }
    );
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}", shape_str);
    let header_len = header.len() + 1;
    let pad = (16 - (10 + header_len) % 16) % 16;
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

/// LSQ 密度标定: 静态区 (低 GT 时序方差) 拟合 `proj_gt ≈ s·proj_fdk + b`。
fn lsq_calibrate(
    proj_fdk: &[f32],
    gts: &[&[f32]],
    n_pix: usize,
    min_fdk: f32,
    n_static: usize,
) -> (f32, f32, f32) {
    let nvar = gts.len().min(4);
    let mut var = vec![f32::INFINITY; n_pix];
    for i in 0..n_pix {
        let mut v = 0.0f32;
        for k in 0..nvar.saturating_sub(1) {
            let d = gts[k][i] - gts[k + 1][i];
            v += d * d;
        }
        var[i] = v;
    }
    let mut order: Vec<usize> = (0..n_pix).collect();
    order.sort_by(|&a, &b| var[a].total_cmp(&var[b]));

    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    let mut used = 0usize;
    for &idx in order.iter() {
        if proj_fdk[idx] > min_fdk {
            let x = proj_fdk[idx] as f64;
            let y = gts[0][idx] as f64;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            used += 1;
            if used >= n_static {
                break;
            }
        }
    }
    let n = used.max(1) as f64;
    let denom = n * sxx - sx * sx;
    let s = if denom.abs() > 1e-12 {
        (n * sxy - sx * sy) / denom
    } else {
        1.0
    };
    let b = (sy - s * sx) / n;
    let mut err = 0.0f64;
    for (&p, &g) in proj_fdk.iter().zip(gts[0].iter()) {
        let y = s as f64 * p as f64 + b as f64;
        err += (y - g as f64).abs();
    }
    (s as f32, b as f32, (err / n_pix.max(1) as f64) as f32)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut dcm: Option<PathBuf> = None;
    let mut vol_x = 256usize;
    let mut vol_z: Option<usize> = None;
    // 重建域内缩系数 (默认 1.0): 圆柱 mask 半径 = half_w × mask_scale。
    // 亮边实为标定 bias (padded 版 b<0 使圆柱外空气发亮); mask=1.0 无
    // padding 时 b≈0, 亮边可忽略, 内缩反而切掉真实结构掉 PSNR。
    let mut mask_scale = 1.0f32;
    let mut steps = 256usize;
    let mut calib_views = 12usize;
    let mut save_drr = 2usize;
    let mut mu_scale = 0.00021f32;
    let mut out = PathBuf::from("target/fdk/volume");
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--vol-x=") {
            vol_x = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--vol-z=") {
            vol_z = Some(v.parse()?);
        } else if let Some(v) = a.strip_prefix("--mask-scale=") {
            mask_scale = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--steps=") {
            steps = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--calib-views=") {
            calib_views = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--save-drr=") {
            save_drr = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--mu-scale=") {
            mu_scale = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if dcm.is_none() {
            dcm = Some(PathBuf::from(a));
        }
        i += 1;
    }
    let dcm = dcm.expect("usage: fdk_volume <dcm> [--vol-x=N] [--pad-xy=F] [--roi=N]");

    // ---- 加载 DICOM (roi 裁暗边, minmax + 自动 gamma) ----
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
    let load_config = LoadDatasetConfig {
        max_frames: None,
        max_resolution: 1920,
        eval_split_every: None,
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        dicom_orientation: XRayOrientation::Ap,
        dicom_normalization: DicomNormalization::Minmax,
        dicom_gamma: None,
        dicom_gamma_target: Some(0.5),
        max_scene_batch_cache_size: 1 << 30,
        roi: brush_dataset::config::RoiSpec::Inset(20),
    };
    let result = brush_dataset::load_dataset(vfs, &load_config).await?;
    let views = &result.dataset.train.views;
    println!("loaded {} views (roi inset=20)", views.len());

    // 几何。
    let bytes = std::fs::read(&dcm)?;
    let meta = brush_dicom::parse_dicom(&bytes)?;
    let delx = meta.geometry.delx as f32;
    let g0 = views[0].gray_image.as_ref().expect("gray");
    let img_size = glam::uvec2(g0.width, g0.height);
    let cam = &views[0].camera;
    let focal = cam.focal(img_size);
    let sod = cam.position.length();
    let half_w = (g0.width as f32 * 0.5) * sod / focal.x;
    let half_h = (g0.height as f32 * 0.5) * sod / focal.y;
    // 域: XY = half_w (无 padding), Z = half_h (只与投影高度平齐)。
    let rx = half_w;
    let ry = half_w;
    let rz = half_h;
    let vol_y = vol_x; // XY 正方形
    let spacing_xy = 2.0 * rx / vol_x as f32;
    let vol_z = vol_z.unwrap_or(((2.0 * rz) / spacing_xy).round() as usize);
    println!(
        "geometry: sod={sod:.1} frame {}x{} delx={delx:.3}\n  domain rx=ry={rx:.1} rz={rz:.1}  mask_r={:.1} (half_w {half_w:.1} x {mask_scale})\n  grid {vol_x}x{vol_y}x{vol_z}  spacing_xy={spacing_xy:.3} spacing_z={:.3}",
        g0.width, g0.height, half_w * mask_scale, 2.0 * rz / vol_z as f32
    );

    // ---- ramp 滤波 + FDK 反投影 (只填全采样圆柱 half_w) ----
    let frames = build_frames(views, delx);
    println!("built {} filtered frames", frames.len());
    let t0 = std::time::Instant::now();
    let mask_r = half_w * mask_scale;
    let mut volume = backproject_aniso(&frames, vol_x, vol_y, vol_z, rx, ry, rz, mask_r);
    println!("FDK backproject {vol_x}x{vol_y}x{vol_z} in {:.1}s", t0.elapsed().as_secs_f32());
    let max_v = volume.iter().copied().fold(0.0f32, f32::max);
    let n_neg = volume.iter().filter(|&&v| v < 0.0).count();
    println!(
        "volume max={max_v:.4} neg={n_neg} ({:.1}%), mu_scale={mu_scale}",
        n_neg as f32 / volume.len() as f32 * 100.0
    );
    for v in &mut volume {
        *v *= mu_scale;
    }

    // ---- 保存体积 + 元数据 ----
    std::fs::create_dir_all(&out)?;
    let vol_path = out.join("volume.nii.gz");
    write_nifti_volume(&vol_path, &volume, vol_x, vol_y, vol_z, rx, ry, rz)?;
    let meta_json = format!(
        "{{\"vol_x\":{vol_x},\"vol_y\":{vol_y},\"vol_z\":{vol_z},\"rx\":{rx},\"ry\":{ry},\"rz\":{rz},\"delx\":{delx},\"sod\":{sod}}}"
    );
    std::fs::write(out.join("meta.json"), meta_json)?;
    println!("saved {vol_path:?} + meta.json");

    // ---- 正向投影耗时 + LSQ 标定 ----
    let sel: Vec<usize> = (0..views.len())
        .step_by((views.len() / calib_views).max(1))
        .take(calib_views)
        .collect();
    let t1 = std::time::Instant::now();
    let projs: Vec<Vec<f32>> = sel
        .iter()
        .map(|&vi| forward_project_view(&volume, vol_x, vol_y, vol_z, rx, ry, rz, &views[vi], steps))
        .collect();
    let dt = t1.elapsed();
    let per_view = dt.as_secs_f32() / projs.len() as f32;
    println!(
        "forward project {}/{} views ({steps} steps) in {:.1}s → {:.2}s/view, 全 ≈ {:.0}s",
        projs.len(),
        views.len(),
        dt.as_secs_f32(),
        per_view,
        per_view * views.len() as f32
    );

    let gts: Vec<Vec<f32>> = sel
        .iter()
        .map(|&vi| {
            let g = views[vi].gray_image.as_ref().expect("gray GT");
            g.data.iter().map(|&v| -v.clamp(1e-3, 1.0).ln()).collect()
        })
        .collect();
    let n_pix = projs[0].len();
    let gt_refs: Vec<&[f32]> = gts.iter().map(|g| g.as_slice()).collect();
    let mut fdk_sorted = projs[0].clone();
    fdk_sorted.sort_by(|a, b| a.total_cmp(b));
    let min_fdk = fdk_sorted[fdk_sorted.len() / 2].max(1e-4);
    let (s, b, mean_err) = lsq_calibrate(&projs[0], &gt_refs, n_pix, min_fdk, n_pix / 50);
    println!("LSQ 标定: s={s:.6}, b={b:.6} (min_fdk={min_fdk:.4}, mean |err|={mean_err:.4})");
    std::fs::write(out.join("calib.json"), format!("{{\"s\":{s},\"b\":{b}}}"))?;

    // FDK 先验强度 PSNR。
    let mut se = 0.0f64;
    let mut n = 0u64;
    for (i, (&p, &g)) in projs[0].iter().zip(gts[0].iter()).enumerate() {
        let y = s as f64 * p as f64 + b as f64;
        let pred = (-y).exp().clamp(0.0, 1.0);
        let gi = (-(g as f64)).exp().clamp(0.0, 1.0);
        let d = pred - gi;
        se += d * d;
        n += 1;
    }
    let psnr = if se / n.max(1) as f64 > 1e-12 {
        10.0 * (1.0 / (se / n.max(1) as f64)).log10()
    } else {
        100.0
    };
    println!("FDK 先验 (校准后, 视图0): PSNR_int={psnr:.2} dB");

    // 保存校准后 DRR 目检。
    for (vi, &idx) in sel.iter().enumerate().take(save_drr) {
        let mut drr = vec![0.0f32; n_pix];
        for (k, &p) in projs[vi].iter().enumerate() {
            drr[k] = (-(s as f64 * p as f64 + b as f64)).exp().clamp(0.0, 1.0) as f32;
        }
        let g = views[idx].gray_image.as_ref().expect("gray");
        let (w, h) = (g.width as usize, g.height as usize);
        write_npy_f32(&out.join(format!("fdk_drr_v{vi:02}.npy")), &drr, &[h, w])?;
        let gt: Vec<f32> = g.data.iter().copied().collect();
        write_npy_f32(&out.join(format!("gt_drr_v{vi:02}.npy")), &gt, &[h, w])?;
    }
    println!("done -> {out:?}");
    Ok(())
}
