//! `fit_volume`: gradient-refine an FDK volume against the DICOM projections
//! via the differentiable DRR (cubecl kernels in `brush-drr`).
//!
//! Analogue of DiffVox / DiffDRR volume refinement. Starts from the FDK
//! volume, minimises `mean((proj - proj_gt)^2)` (proj domain) over the
//! training views with Adam. The volume + Adam state + loss stay **on GPU**
//! (fusion backend; the DRR kernels bridge via brush-drr's fusion glue), so
//! there are no per-step CPU round-trips. Outputs a refined volume that is
//! a cleaner static prior (fewer FDK artifacts) for the residual-GS
//! pipeline.
//!
//! Usage:
//!   fit_volume images/pig-data-cor-new-phase.dcm \
//!     --volume=.../volume.nii.gz --meta=.../meta.json --calib=.../calib.json \
//!     [--iters=N] [--lr=F] [--batch=K] [--steps=S] [--out=DIR]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use brush_cube::test_helpers::test_device;
use brush_dataset::config::{DicomNormalization, LoadDatasetConfig, XRayOrientation};
use brush_drr::DrrSettings;
use brush_vfs::BrushVfs;
use burn::tensor::{Tensor, TensorData};
use brush_render::burn_glue::{unwrap_wgpu_float, wrap_wgpu_float};
/// 读 3D 体积 .nii.gz (nifti-rs), 返回 (float32 数据, vol)。
fn read_nifti_volume(path: &Path) -> anyhow::Result<(Vec<f32>, usize)> {
    use nifti::{NiftiObject, ReaderOptions};
    let obj = ReaderOptions::new().read_file(path)?;
    let dims = obj.header().dim;
    let vol = dims[1] as usize;
    let volume = obj.into_volume();
    let data: Vec<f32> = volume.into_nifti_typed_data()?;
    Ok((data, vol))
}

/// 写 3D 体积为 .nii.gz (nifti-rs, sform_code=2)。世界范围 `[-half_r, half_r]^3`。
fn write_nifti_volume(path: &Path, data: &[f32], vol: usize, half_r: f32) -> anyhow::Result<()> {
    use nifti::writer::WriterOptions;
    use nifti::{NiftiHeader, NiftiType};
    let arr = ndarray::Array3::from_shape_vec((vol, vol, vol), data.to_vec())
        .map_err(|e| anyhow::anyhow!("ndarray shape: {e}"))?;
    let spacing = 2.0 * half_r / vol as f32;
    let mut hdr = NiftiHeader::default();
    hdr.dim[0] = 3;
    hdr.dim[1] = vol as u16;
    hdr.dim[2] = vol as u16;
    hdr.dim[3] = vol as u16;
    hdr.datatype = NiftiType::Float32 as i16;
    hdr.bitpix = 32;
    hdr.pixdim[1] = spacing;
    hdr.pixdim[2] = spacing;
    hdr.pixdim[3] = spacing;
    hdr.qform_code = 0;
    hdr.sform_code = 2;
    // nifti-rs 内部 data.t() (Fortran 序) → 文件轴 [X,Z,Y]:
    // axis0=X→x(对角i), axis1=Z 需→z, axis2=Y 需→y。所以 srow_y 用 k, srow_z 用 j。
    hdr.srow_x = [spacing, 0.0, 0.0, -half_r];
    hdr.srow_y = [0.0, 0.0, spacing, -half_r];
    hdr.srow_z = [0.0, spacing, 0.0, -half_r];
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .map_err(|e| anyhow::anyhow!("write nifti: {e}"))?;
    Ok(())
}

/// Minimal .npy (v1.0, float32, C-order) reader.
fn read_npy_f32(path: &Path) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a npy file");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = String::from_utf8(bytes[10..10 + header_len].to_vec())?;
    let shape = parse_npy_shape(&header);
    let total: usize = shape.iter().product();
    let data_start = 10 + header_len;
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let o = data_start + i * 4;
        out.push(f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]));
    }
    Ok((out, shape))
}

/// Extract the `shape=(...)` tuple from an npy header (robust to padding).
fn parse_npy_shape(header: &str) -> Vec<usize> {
    let Some(open) = header.find("'shape':") else { return vec![] };
    let Some(rel) = header[open..].find('(') else { return vec![] };
    let open_paren = open + rel;
    let Some(rel2) = header[open_paren..].find(')') else { return vec![] };
    let close_paren = open_paren + rel2;
    header[open_paren + 1..close_paren]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// Minimal .npy (v1.0, float32, C-order) writer.
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

/// Fusion DRR forward: high-level `Tensor` in/out, bridges via brush-drr.
async fn drr_forward_t(settings: &DrrSettings, volume: Tensor<3>) -> Tensor<2> {
    let ft = unwrap_wgpu_float(volume);
    let out = <brush_cube::MainBackend as brush_drr::DrrOps>::drr_forward(settings, ft).await;
    wrap_wgpu_float::<2>(out)
}

/// Fusion DRR backward: high-level `Tensor` in/out.
async fn drr_backward_t(settings: &DrrSettings, v_proj: Tensor<2>) -> Tensor<3> {
    let ft = unwrap_wgpu_float(v_proj);
    let out = <brush_cube::MainBackend as brush_drr::DrrOps>::drr_backward(settings, ft).await;
    wrap_wgpu_float::<3>(out)
}

/// 平方 TV 梯度: TV = Σ(μ_{i+1}-μ_i)² (三轴), dTV/dμ = 离散 Laplacian×2
/// (边界自动为 0 项)。加到体积梯度。
fn tv_grad(v: &Tensor<3>) -> Tensor<3> {
    use burn::tensor::s;
    // x 轴 (dim2)
    let prev_x = Tensor::cat(vec![v.clone().slice(s![.., .., 0..1]), v.clone().slice(s![.., .., 0..-1])], 2);
    let next_x = Tensor::cat(vec![v.clone().slice(s![.., .., 1..]), v.clone().slice(s![.., .., -1..])], 2);
    let mut g = v.clone().sub(prev_x).mul_scalar(2.0).add(v.clone().sub(next_x).mul_scalar(2.0));
    // y 轴 (dim1)
    let prev_y = Tensor::cat(vec![v.clone().slice(s![.., 0..1, ..]), v.clone().slice(s![.., 0..-1, ..])], 1);
    let next_y = Tensor::cat(vec![v.clone().slice(s![.., 1.., ..]), v.clone().slice(s![.., -1.., ..])], 1);
    g = g.add(v.clone().sub(prev_y).mul_scalar(2.0)).add(v.clone().sub(next_y).mul_scalar(2.0));
    // z 轴 (dim0)
    let prev_z = Tensor::cat(vec![v.clone().slice(s![0..1, .., ..]), v.clone().slice(s![0..-1, .., ..])], 0);
    let next_z = Tensor::cat(vec![v.clone().slice(s![1.., .., ..]), v.clone().slice(s![-1.., .., ..])], 0);
    g.add(v.clone().sub(prev_z).mul_scalar(2.0)).add(v.clone().sub(next_z).mul_scalar(2.0))
}

/// 平方 TV 值 (日志用)。
fn tv_value(v: &Tensor<3>) -> f32 {
    use burn::tensor::s;
    let dx = v.clone().slice(s![.., .., 1..]).sub(v.clone().slice(s![.., .., ..-1])).powf_scalar(2.0).sum();
    let dy = v.clone().slice(s![.., 1.., ..]).sub(v.clone().slice(s![.., ..-1, ..])).powf_scalar(2.0).sum();
    let dz = v.clone().slice(s![1.., .., ..]).sub(v.clone().slice(s![..-1, .., ..])).powf_scalar(2.0).sum();
    dx.add(dy).add(dz).into_scalar()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut dcm: Option<PathBuf> = None;
    let mut volume_path = PathBuf::from("experiments/output/fdk-residual/fdk/volume.npy");
    let mut meta_path = PathBuf::from("experiments/output/fdk-residual/fdk/meta.json");
    let mut calib_path = PathBuf::from("experiments/output/fdk-residual/fdk/calib.json");
    let mut iters = 500usize;
    let mut lr = 1e-4f32;
    let mut batch = 16usize;
    let mut steps = 256usize;
    let mut out = PathBuf::from("experiments/output/fdk-residual/fit_volume");
    let mut selftest = false;
    let mut eval_only = false;
    let mut tv = 0.0f32;
    let mut motion_mask = false;
    let mut mask_sigma = 0.25f32; // 残差掩膜尺度 (proj 单位)
    let mut mask_warmup = 150usize;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--volume=") {
            volume_path = PathBuf::from(v);
        } else if let Some(v) = a.strip_prefix("--meta=") {
            meta_path = PathBuf::from(v);
        } else if let Some(v) = a.strip_prefix("--calib=") {
            calib_path = PathBuf::from(v);
        } else if let Some(v) = a.strip_prefix("--iters=") {
            iters = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--lr=") {
            lr = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--batch=") {
            batch = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--steps=") {
            steps = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--out=") {
            out = PathBuf::from(v);
        } else if let Some(v) = a.strip_prefix("--tv=") {
            tv = v.parse()?;
        } else if a == "--motion-mask" {
            motion_mask = true;
        } else if let Some(v) = a.strip_prefix("--mask-sigma=") {
            mask_sigma = v.parse()?;
        } else if let Some(v) = a.strip_prefix("--mask-warmup=") {
            mask_warmup = v.parse()?;
        } else if a == "--selftest" {
            selftest = true;
        } else if a == "--eval-only" {
            eval_only = true;
        } else if dcm.is_none() {
            dcm = Some(PathBuf::from(a));
        }
        i += 1;
    }
    let dcm = dcm.expect("usage: fit_volume <dcm> --volume=<npy> --meta=<json> --calib=<json>");

    // ---- 加载 DICOM (与 fdk_volume / fit_deform 一致: roi=20) ----
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
    println!("loaded {} views", views.len());

    // ---- 加载体积 + 元数据 + 标定 ----
    let (vol_vec, vol_shape) = read_nifti_volume(&volume_path)?;
    let vol = vol_shape;
    assert_eq!(vol_vec.len(), vol * vol * vol, "volume must be cubic");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let cyl_radius = meta["cyl_radius"].as_f64().unwrap_or(0.0) as f32;
    let calib: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&calib_path)?)?;
    let scale = calib["s"].as_f64().unwrap_or(1.0) as f32;
    let bias = calib["b"].as_f64().unwrap_or(0.0) as f32;
    println!(
        "volume {vol}^3 (half_r={cyl_radius:.1}mm), calib s={scale:.5} b={bias:.5}, iters={iters} lr={lr} batch={batch}"
    );

    let device: burn::tensor::Device = test_device().await.into();
    let vol_total = vol * vol * vol;

    // ---- 投影 GT (proj_gt = -ln(gray)) + 每视图 DrrSettings + GT 张量 ----
    let mut gts: Vec<Vec<f32>> = Vec::with_capacity(views.len());
    let mut settings: Vec<DrrSettings> = Vec::with_capacity(views.len());
    let mut gt_tensors: Vec<Tensor<2>> = Vec::with_capacity(views.len());
    let g0 = views[0].gray_image.as_ref().expect("gray");
    let (img_w, img_h) = (g0.width as u32, g0.height as u32);
    let n_pix = (img_w * img_h) as f32;
    for v in views.iter() {
        let gray = v.gray_image.as_ref().expect("gray GT");
        let gt: Vec<f32> = gray.data.iter().map(|&g| -g.clamp(1e-3, 1.0).ln()).collect();
        gts.push(gt.clone());
        settings.push(DrrSettings::new(
            &v.camera,
            img_w,
            img_h,
            vol as u32,
            steps as u32,
            cyl_radius,
            scale,
            bias,
        ));
        gt_tensors.push(Tensor::<2>::from_data(
            TensorData::new::<f32, _>(gt, [img_h as usize, img_w as usize]),
            &device,
        ));
    }
    println!("n_views={} img={}x{} n_pix={n_pix:.0}", views.len(), img_w, img_h);

    // ---- 有限差分自检: backward 解析梯度 vs 数值梯度 ----
    if selftest {
        let vol: u32 = 16;
        let img: u32 = 48;
        let half_r = 100.0f32;
        let steps: u32 = 64;
        let mut v: Vec<f32> = (0..vol * vol * vol).map(|_| rand::random::<f32>() * 0.01).collect();
        let st = DrrSettings::new(
            &views[0].camera,
            img,
            img,
            vol,
            steps,
            half_r,
            1.0,
            0.0,
        );
        let fwd = |vv: Vec<f32>| {
            let t = Tensor::<3>::from_data(TensorData::new::<f32, _>(vv, [vol as usize; 3]), &device);
            let p = drr_forward_t(&st, t);
            p
        };
        // 解析: backward with v_proj=ones (L = Σ proj)
        let proj0 = fwd(v.clone());
        let ones = Tensor::<2>::ones([img as usize; 2], &device);
        let v_vol = drr_backward_t(&st, ones).await;
        let v_vol_cpu: Vec<f32> = v_vol.into_data().into_vec::<f32>().unwrap();
        let eps = 1e-4f32;
        for idx in [0usize, 100, 1000, vol as usize * vol as usize / 2 + 17, v.len() / 2] {
            let orig = v[idx];
            v[idx] = orig + eps;
            let p2 = drr_forward_t(&st, Tensor::<3>::from_data(TensorData::new::<f32, _>(v.clone(), [vol as usize; 3]), &device)).await;
            let s2: f32 = p2.sum().into_scalar();
            v[idx] = orig - eps;
            let p3 = drr_forward_t(&st, Tensor::<3>::from_data(TensorData::new::<f32, _>(v.clone(), [vol as usize; 3]), &device)).await;
            let s3: f32 = p3.sum().into_scalar();
            v[idx] = orig;
            let num = (s2 - s3) / (2.0 * eps);
            let ana = v_vol_cpu[idx];
            println!("selftest voxel {idx}: num={num:.6} analytic={ana:.6} ratio={:.3}", num / ana.max(1e-12));
        }
        return Ok(());
    }

    // ---- 体积 + Adam 状态 (全程 GPU) ----
    let mut volume = Tensor::<3>::from_data(
        TensorData::new::<f32, _>(vol_vec, [vol, vol, vol]),
        &device,
    );

    // ---- eval-only: 评估已保存体积 (含强度 PSNR) ----
    if eval_only {
        let mut se = 0.0f64;
        let mut sei = 0.0f64;
        let mut n = 0u64;
        for k in 0..8 {
            let vi = (k * views.len() / 8) % views.len();
            let proj = drr_forward_t(&settings[vi], volume.clone()).await;
            let pred: Vec<f32> = proj.into_data().into_vec::<f32>().unwrap();
            for (p, &g) in pred.iter().zip(gts[vi].iter()) {
                let d = (*p - g) as f64;
                se += d * d;
                let pi = (-(*p as f64)).exp().clamp(0.0, 1.0);
                let gi = (-(g as f64)).exp().clamp(0.0, 1.0);
                sei += (pi - gi) * (pi - gi);
                n += 1;
            }
        }
        let mse = se / n.max(1) as f64;
        let msei = sei / n.max(1) as f64;
        let psnr = if mse > 1e-12 { 10.0 * (1.0 / mse).log10() } else { 100.0 };
        let psnri = if msei > 1e-12 { 10.0 * (1.0 / msei).log10() } else { 100.0 };
        println!("EVAL: psnr(proj)={psnr:.2} dB  psnr(int)={psnri:.2} dB");
        return Ok(());
    }

    let mut m = volume.clone().mul_scalar(0.0);
    let mut v = volume.clone().mul_scalar(0.0);
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    let eps = 1e-8f32;
    let mut t_step = 0usize;

    // ---- 训练循环 (全部 GPU 算子) ----
    let t0 = std::time::Instant::now();
    for it in 0..iters {
        let start = (it * batch) % views.len();
        let mut grad_acc: Option<Tensor<3>> = None;
        let mut loss_acc = 0.0f32;
        for b in 0..batch {
            let vi = (start + b) % views.len();
            let s = &settings[vi];
            let proj = drr_forward_t(s, volume.clone()).await;
            let diff = proj.sub(gt_tensors[vi].clone());
            // 残差自适应运动掩膜: 静态区收敛后残差→0 权重→1, 心脏残差大→0。
            // w = 1/(1+(|diff|/σ)²), 在 it>=mask_warmup 时启用。
            let (diff_w, loss) = if motion_mask && it >= mask_warmup {
                let r = diff.clone().abs();
                let w = r.mul_scalar(1.0 / mask_sigma).powf_scalar(2.0).add_scalar(1.0).recip();
                let md = diff.clone().mul(w.clone());
                (md.clone(), md.mul(diff.clone()).mean())
            } else {
                (diff.clone(), diff.powf_scalar(2.0).mean())
            };
            loss_acc += loss.clone().into_scalar::<f32>();
            let v_proj = diff_w.mul_scalar(2.0 / n_pix);
            let v_vol = drr_backward_t(s, v_proj).await;
            grad_acc = Some(match grad_acc {
                Some(g) => g.add(v_vol),
                None => v_vol,
            });
        }
        let mut grad = grad_acc.expect("batch grad");
        // TV 正则: d(λ·TV)/dμ 加入梯度。
        if tv > 0.0 {
            grad = grad.add(tv_grad(&volume).mul_scalar(tv * batch as f32));
        }

        // Adam (GPU 张量运算)。
        t_step += 1;
        let b1_t = b1.powf(t_step as f32);
        let b2_t = b2.powf(t_step as f32);
        m = m.mul_scalar(b1).add(grad.clone().mul_scalar(1.0 - b1));
        v = v.mul_scalar(b2).add(grad.clone().powf_scalar(2.0).mul_scalar(1.0 - b2));
        let m_hat = m.clone().div_scalar(1.0 - b1_t);
        let v_hat = v.clone().div_scalar(1.0 - b2_t);
        let step = m_hat.div(v_hat.sqrt().add_scalar(eps));
        volume = volume.sub(step.mul_scalar(lr));

        if it % 10 == 0 || it == iters - 1 {
            // eval: 8 均匀视图 PSNR (proj 域 + 强度域 exp(-proj) vs GT gray)。
            let mut se = 0.0f64;
            let mut sei = 0.0f64;
            let mut n = 0u64;
            for k in 0..8 {
                let vi = (k * views.len() / 8) % views.len();
                let proj = drr_forward_t(&settings[vi], volume.clone()).await;
                let pred: Vec<f32> = proj.into_data().into_vec::<f32>().unwrap();
                for (p, &g) in pred.iter().zip(gts[vi].iter()) {
                    let d = (p - g) as f64;
                    se += d * d;
                    let pi = (-(*p as f64)).exp().clamp(0.0, 1.0);
                    let gi = (-(g as f64)).exp().clamp(0.0, 1.0);
                    sei += (pi - gi) * (pi - gi);
                    n += 1;
                }
            }
            let mse = se / n.max(1) as f64;
            let psnr = if mse > 1e-12 { 10.0 * (1.0 / mse).log10() } else { 100.0 };
            let msei = sei / n.max(1) as f64;
            let psnri = if msei > 1e-12 { 10.0 * (1.0 / msei).log10() } else { 100.0 };
            println!(
                "iter {it}/{iters} loss={:.6} psnr(proj)={psnr:.2} psnr(int)={psnri:.2} dB  ({:.1}s)",
                loss_acc / batch as f32,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    println!("done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- 保存 refined 体积 ----
    std::fs::create_dir_all(&out)?;
    let vol_cpu: Vec<f32> = volume.clone().into_data().into_vec::<f32>().unwrap();
    write_nifti_volume(&out.join("volume_refined.nii.gz"), &vol_cpu, vol, cyl_radius)?;
    println!("saved {out:?}/volume_refined.nii.gz");
    Ok(())
}
