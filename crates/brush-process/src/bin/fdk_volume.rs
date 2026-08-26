//! FDK cone-beam 重建 → 静态先验体积 + 逐视角 DRR 预计算 + LSQ 密度标定。
//!
//! 用途: 作为 GS 残差建模的静态先验 (FDK 静态结构高度吻合, 动态部分由
//! 带符号残差 GS 修正)。本 bin:
//!   1. 每帧投影 `p = -ln(gray)`, 2D cosine 加权 + 频域 ramp 滤波 (Ram-Lak+Hann)。
//!   2. FDK 反投影到等中心圆柱 FOV (XY 圆内, 旋转轴 Z) 体素网格。
//!   3. 存 `<out>/volume.npy` (float32, 世界坐标 [-r,r]³) + `<out>/meta.json`。
//!   4. 正向投影 (射线步进 ∫μ dl) 若干视图 → 测耗时 (P1a 闸门) + LSQ 标定。
//!
//! 用法:
//!   cargo run --release -p brush-process --bin fdk_volume -- \
//!     images/pig-data-cor-new-phase.dcm --vol=256 --out=experiments/output/.../fdk

use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_dataset::scene::SceneView;
use brush_vfs::BrushVfs;
use glam::{Affine3A, Vec3};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex, Fft};

/// 频域 ramp 滤波 (Ram-Lak + Hann 窗)。对每行 FFT, 乘 |k|/N · Hann, IFFT。
fn ramp_filter_row_fft(row: &[f32], fft: &Arc<dyn Fft<f32>>, ifft: &Arc<dyn Fft<f32>>) -> Vec<f32> {
    let n = row.len();
    let n2 = n * 2; // zero-padding 避免周期性边界振铃
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

/// 单帧 FDK 数据: 相机局部变换 + ramp 滤波后的投影。
struct Frame {
    w2l: Affine3A,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    sod: f32,
    width: usize,
    height: usize,
    /// Ramp 滤波后的投影 `[H, W]`。
    filtered: Vec<f32>,
}

/// 对每帧构建 `proj = -ln(gray)`, 2D cosine 加权 + ramp 滤波。
fn build_frames(views: &[SceneView], sdd: f32) -> Vec<Frame> {
    let mut planner = FftPlanner::<f32>::new();
    let w0 = views[0].gray_image.as_ref().expect("gray GT").width as usize;
    let fft = planner.plan_fft_forward(2 * w0);
    let ifft = planner.plan_fft_inverse(2 * w0);
    let fft = Arc::new(fft);
    let ifft = Arc::new(ifft);
    views
        .par_iter()
        .map(|view| {
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

            // 2D cosine 加权 (锥束倾斜补偿)。
            let det_px = sdd / fx;
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
                    filtered[y * w + x] = filt[x] * du_world;
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

/// FDK 反投影: 体素网格 `vol^3`, 世界范围 `[-cyl_radius, cyl_radius]^3`。
/// C-arm 绕世界 Z 轴旋转, 圆柱 FOV 沿 Z 轴延伸, 圆截面在 XY 平面。
fn backproject(frames: &[Frame], vol: usize, cyl_radius: f32, cyl_scale: f32) -> Vec<f32> {
    let delta = 2.0 * cyl_radius / vol as f32;
    let cyl_r2 = (cyl_radius * cyl_scale).powi(2);
    let mut volume = vec![0.0f32; vol * vol * vol];
    volume
        .par_chunks_mut(vol * vol)
        .enumerate()
        .for_each(|(iy, slice)| {
            let y = -cyl_radius + (iy as f32 + 0.5) * delta;
            let y2 = y * y;
            for iz in 0..vol {
                let z = -cyl_radius + (iz as f32 + 0.5) * delta;
                let row_base = iz * vol;
                for ix in 0..vol {
                    let x = -cyl_radius + (ix as f32 + 0.5) * delta;
                    if x * x + y2 > cyl_r2 {
                        continue;
                    }
                    let p_w = Vec3::new(x, y, z);
                    let mut acc = 0.0f32;
                    for f in frames {
                        let p_c = f.w2l.transform_point3(p_w);
                        let dz = p_c.z;
                        if dz <= 1.0 {
                            continue;
                        }
                        let u = f.fx * p_c.x / dz + f.cx;
                        let v = f.fy * p_c.y / dz + f.cy;
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
                        let wgt = f.sod / dz;
                        acc += wgt * wgt * val;
                    }
                    slice[row_base + ix] = acc;
                }
            }
        });
    volume
}

/// 三线性插值采样体积 `vol^3`, 世界范围 `[-r, r]^3`。圆柱外返回 0。
#[inline]
fn sample_vol(volume: &[f32], vol: usize, r: f32, p: Vec3) -> f32 {
    let delta = 2.0 * r / vol as f32;
    let fx = (p.x + r) / delta - 0.5;
    let fy = (p.y + r) / delta - 0.5;
    let fz = (p.z + r) / delta - 0.5;
    if fx < 0.0 || fy < 0.0 || fz < 0.0 {
        return 0.0;
    }
    let ix = fx as usize;
    let iy = fy as usize;
    let iz = fz as usize;
    if ix + 1 >= vol || iy + 1 >= vol || iz + 1 >= vol {
        return 0.0;
    }
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let tz = fz - iz as f32;
    let idx = |x: usize, y: usize, z: usize| (y * vol + z) * vol + x;
    let c000 = volume[idx(ix, iy, iz)];
    let c100 = volume[idx(ix + 1, iy, iz)];
    let c010 = volume[idx(ix, iy + 1, iz)];
    let c110 = volume[idx(ix + 1, iy + 1, iz)];
    let c001 = volume[idx(ix, iy, iz + 1)];
    let c101 = volume[idx(ix + 1, iy, iz + 1)];
    let c011 = volume[idx(ix, iy + 1, iz + 1)];
    let c111 = volume[idx(ix + 1, iy + 1, iz + 1)];
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    c0 * (1.0 - tz) + c1 * tz
}

/// 正向投影 (射线步进 ∫μ dl): 对给定视图, 累加穿过体积的密度积分。
/// 返回原始 proj 积分 `[H, W]` (未经 exp 映射, 也未经密度尺度缩放)。
fn forward_project_view(
    volume: &[f32],
    vol: usize,
    r: f32,
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
    let t_near = sod - r;
    let t_far = sod + r;
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
                    acc += sample_vol(volume, vol, r, p_world) * dt;
                }
                acc
            })
        })
        .collect()
}

/// 写 .npy (v1.0, float32, C-order)。
fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> anyhow::Result<()> {
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

/// LSQ 密度标定: 在静态区像素拟合 `proj_gt ≈ s·proj_fdk + b` (最小二乘闭式)。
/// 静态区 = 真实图像时序方差最低的像素 (低时序方差 → 骨/背景, 无运动)。
/// 用前 `nvar` 个 GT 帧估方差; 只保留 proj_fdk > min_fdk 的组织区像素。
/// 返回 `(s, b, mean|err| over all pixels)`。
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
    let mut vol = 256usize;
    let mut cyl_scale = 1.0f32;
    let mut steps = 256usize;
    let mut calib_views = 12usize;
    // 保存校准后 DRR 视图数 (0 = 不存) 用于目检 FDK 先验质量。
    let mut save_drr = 2usize;
    // FDK 体积 → 真实 μ(mm⁻¹) 的缩放 (无量纲累加 × mu_scale = 线性衰减系数)。
    let mut mu_scale = 0.00021f32;
    let mut out = PathBuf::from("target/fdk/volume");
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--vol=") {
            vol = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--cyl-scale=") {
            cyl_scale = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--steps=") {
            steps = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--calib-views=") {
            calib_views = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--mu-scale=") {
            mu_scale = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--save-drr=") {
            save_drr = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if dcm.is_none() {
            dcm = Some(PathBuf::from(a));
        }
        i += 1;
    }
    let dcm = dcm.expect("usage: fdk_volume <dcm> [--vol=N] [--cyl-scale=F] [--steps=N] [--out=DIR]");

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
    // 与 fit_deform 相同: roi=Inset(20) (608x434), minmax 归一化 + 自动 gamma。
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
    println!("loaded {} views", views.len());

    // 从 DICOM header 取探测器像素间距 delx (cosine 加权用)。
    let bytes = std::fs::read(&dcm)?;
    let meta = brush_dicom::parse_dicom(&bytes)?;
    let delx = meta.geometry.delx as f32;
    println!("delx = {delx:.4} mm (detector pixel spacing)");

    // 几何: 圆柱半径 = 探测器内切圆在等中心处的半宽。
    let g0 = views[0].gray_image.as_ref().expect("gray");
    let img_size = glam::uvec2(g0.width, g0.height);
    let cam = &views[0].camera;
    let focal = cam.focal(img_size);
    let sod = cam.position.length();
    let half_w = (g0.width as f32 * 0.5) * sod / focal.x;
    let half_h = (g0.height as f32 * 0.5) * sod / focal.y;
    let cyl_radius = half_w.min(half_h);
    println!(
        "SOD={sod:.1} mm, frame {}x{}, cyl radius={cyl_radius:.1} mm (half-w {half_w:.1}, half-h {half_h:.1})",
        g0.width, g0.height
    );

    // ---- ramp 滤波 + FDK 反投影 ----
    let frames = build_frames(views, delx);
    println!("built {} filtered frames", frames.len());
    let t0 = std::time::Instant::now();
    let mut volume = backproject(&frames, vol, cyl_radius, cyl_scale);
    println!("FDK backproject {}^3 in {:.1}s", vol, t0.elapsed().as_secs_f32());

    let max_v = volume.iter().copied().fold(0.0f32, f32::max);
    let n_neg = volume.iter().filter(|&&v| v < 0.0).count();
    println!(
        "volume {vol}^3 (voxel {:.2} mm), raw_max={max_v:.4}, neg={n_neg} ({:.1}%), mu_scale={mu_scale}",
        2.0 * cyl_radius / vol as f32,
        n_neg as f32 / volume.len() as f32 * 100.0
    );
    // 应用 mu_scale: 体积现在为真实 μ (mm^-1), 线积分 = 真实光程。
    for v in &mut volume {
        *v *= mu_scale;
    }

    // ---- 保存体积 + 元数据 ----
    std::fs::create_dir_all(&out)?;
    let vol_path = out.join("volume.npy");
    write_npy_f32(&vol_path, &volume, &[vol, vol, vol])?;
    let meta_json = format!(
        "{{\"vol\":{vol},\"cyl_radius\":{cyl_radius},\"delx\":{delx},\"sod\":{sod},\"world_range\":{cyl_radius}}}"
    );
    std::fs::write(out.join("meta.json"), meta_json)?;
    println!("saved {vol_path:?} + meta.json");

    // ---- 正向投影耗时 (P1a 闸门) + LSQ 标定 ----
    // 均匀取 calib_views 个视图。
    let sel: Vec<usize> = (0..views.len())
        .step_by((views.len() / calib_views).max(1))
        .take(calib_views)
        .collect();
    let t1 = std::time::Instant::now();
    let projs: Vec<Vec<f32>> = sel
        .iter()
        .map(|&vi| forward_project_view(&volume, vol, cyl_radius, &views[vi], steps))
        .collect();
    let dt = t1.elapsed();
    let per_view = dt.as_secs_f32() / projs.len() as f32;
    let est_all = per_view * views.len() as f32;
    println!(
        "forward project {}/{} views ({} steps) in {:.1}s → {:.2}s/view, 全 {}/帧 ≈ {:.0}s",
        projs.len(),
        views.len(),
        steps,
        dt.as_secs_f32(),
        per_view,
        views.len(),
        est_all
    );

    // LSQ 标定: 静态像素按 GT 时序方差选 (低方差 = 骨/背景), 只取组织区。
    let gts: Vec<Vec<f32>> = sel
        .iter()
        .map(|&vi| {
            let g = views[vi].gray_image.as_ref().expect("gray GT");
            g.data
                .iter()
                .map(|&v| -v.clamp(1e-3, 1.0).ln())
                .collect()
        })
        .collect();
    let n_pix = projs[0].len();
    let gt_refs: Vec<&[f32]> = gts.iter().map(|g| g.as_slice()).collect();
    // min_fdk: 组织区阈值 = p50 of proj_fdk (避开空气)。
    let mut fdk_sorted = projs[0].clone();
    fdk_sorted.sort_by(|a, b| a.total_cmp(b));
    let min_fdk = fdk_sorted[fdk_sorted.len() / 2].max(1e-4);
    let (s, b, mean_err) = lsq_calibrate(&projs[0], &gt_refs, n_pix, min_fdk, n_pix / 50);
    println!(
        "LSQ 标定: s={s:.6}, b={b:.6} (min_fdk={min_fdk:.4}, mean |err|={mean_err:.4} over all pixels)"
    );
    std::fs::write(out.join("calib.json"), format!("{{\"s\":{s},\"b\":{b}}}"))?;

    // ---- FDK 先验质量验证: 校准后 DRR (exp(-(s·proj_fdk+b))) vs GT ----
    // 全图 PSNR + 静态区 PSNR (低时序方差像素)。
    let nvar = gt_refs.len().min(4);
    let mut static_mask = vec![false; n_pix];
    let mut svar = vec![f32::INFINITY; n_pix];
    for i in 0..n_pix {
        let mut v = 0.0f32;
        for k in 0..nvar.saturating_sub(1) {
            let d = gt_refs[k][i] - gt_refs[k + 1][i];
            v += d * d;
        }
        svar[i] = v;
    }
    let mut order: Vec<usize> = (0..n_pix).collect();
    order.sort_by(|&a, &b| svar[a].total_cmp(&svar[b]));
    for &idx in order[..(n_pix / 5)].iter() {
        static_mask[idx] = true;
    }
    // 重新用下标算 (上面对 static_mask 的用法错误)。
    let mut se_all = 0.0f64;
    let mut se_st = 0.0f64;
    let (mut n_all, mut n_st) = (0u64, 0u64);
    for (i, (&p, &g)) in projs[0].iter().zip(gt_refs[0].iter()).enumerate() {
        let y = s as f64 * p as f64 + b as f64;
        let pred = (-y).exp().clamp(0.0, 1.0);
        let gt_int = (-g as f64).exp().clamp(0.0, 1.0);
        let d = pred - gt_int;
        se_all += d * d;
        n_all += 1;
        if static_mask[i] {
            se_st += d * d;
            n_st += 1;
        }
    }
    let mse_all = se_all / n_all.max(1) as f64;
    let mse_st = se_st / n_st.max(1) as f64;
    let psnr_all = if mse_all > 1e-12 { 10.0 * (1.0 / mse_all).log10() } else { 100.0 };
    let psnr_st = if mse_st > 1e-12 { 10.0 * (1.0 / mse_st).log10() } else { 100.0 };
    println!(
        "FDK 先验 (校准后, 视图0): PSNR_all={psnr_all:.2} dB, PSNR_static={psnr_st:.2} dB (n_st={n_st}/{n_all})"
    );
    println!("done -> {out:?}");
    // 保存校准后 DRR (intensity = exp(-(s·proj_fdk+b))) 供目检。
    for (vi, &idx) in sel.iter().enumerate().take(save_drr) {
        let mut drr = vec![0.0f32; n_pix];
        for (i, &p) in projs[vi].iter().enumerate() {
            drr[i] = (-(s as f64 * p as f64 + b as f64)).exp().clamp(0.0, 1.0) as f32;
        }
        let g = views[idx].gray_image.as_ref().expect("gray");
        let (w, h) = (g.width as usize, g.height as usize);
        write_npy_f32(&out.join(format!("fdk_drr_v{vi:02}.npy")), &drr, &[h, w])?;
        let gt: Vec<f32> = g.data.iter().copied().collect();
        write_npy_f32(&out.join(format!("gt_drr_v{vi:02}.npy")), &gt, &[h, w])?;
    }
    if save_drr > 0 {
        println!("saved calibrated DRR + GT for {} views (目检先验质量)", save_drr.min(sel.len()));
    }
    Ok(())
}
