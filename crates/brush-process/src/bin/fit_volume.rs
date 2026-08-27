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
fn read_nifti_volume(path: &Path) -> anyhow::Result<(Vec<f32>, usize, usize, usize)> {
    use nifti::{NiftiObject, ReaderOptions};
    let obj = ReaderOptions::new().read_file(path)?;
    let dims = obj.header().dim;
    let (vx, vy, vz) = (dims[1] as usize, dims[2] as usize, dims[3] as usize);
    let volume = obj.into_volume();
    let data: Vec<f32> = volume.into_nifti_typed_data()?;
    Ok((data, vx, vy, vz))
}

/// 沿 `y=-x` 反射体积 (诊断 FDK 世界系镜像): `(x,y,z) → (-y,-x,z)`。
/// 索引: new(ix'=vol_y-1-iy, iz, iy'=vol_x-1-ix) = old(ix, iy, iz) (方网格
/// vol_x==vol_y 下无插值)。布局 `idx = iy*vol_x*vol_z + iz*vol_x + ix`。
fn mirror_volume_negxy(vol: &mut Vec<f32>, vol_x: usize, vol_y: usize, vol_z: usize) {
    assert_eq!(vol_x, vol_y, "y=-x mirror needs square xy grid");
    let sy = vol_x * vol_z;
    let sz = vol_x;
    let mut out = vec![0.0f32; vol.len()];
    for iy in 0..vol_y {
        for iz in 0..vol_z {
            for ix in 0..vol_x {
                let src = iy * sy + iz * sz + ix;
                let (ix2, iy2) = (vol_y - 1 - iy, vol_x - 1 - ix);
                let dst = iy2 * sy + iz * sz + ix2;
                out[dst] = vol[src];
            }
        }
    }
    *vol = out;
}

/// XY 平面转置 (x↔y 交换, 反射跨 y=x): `(x,y,z) → (y,x,z)`。
/// 索引: new(ix2, iy2, iz) = old(iy2, ix2, iz) — 方网格下无插值。
/// FDK 重建世界系与训练/GT 差了跨 y=x 的反射 → 转置修正。
fn transpose_volume_xy(vol: &mut Vec<f32>, vol_x: usize, vol_y: usize, vol_z: usize) {
    assert_eq!(vol_x, vol_y, "xy transpose needs square xy grid");
    let sy = vol_x * vol_z;
    let sz = vol_x;
    let mut out = vec![0.0f32; vol.len()];
    for iy in 0..vol_y {
        for iz in 0..vol_z {
            for ix in 0..vol_x {
                let src = iy * sy + iz * sz + ix;
                let dst = ix * sy + iz * sz + iy;
                out[dst] = vol[src];
            }
        }
    }
    *vol = out;
}

/// 写 3D 体积为 .nii.gz (nifti-rs, sform_code=2)。世界范围 `[-half_r, half_r]^3`。
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
    // 内核布局 [y,z,x] (y 最慢, x 最快) → 数组形状必须 (vol_y, vol_z, vol_x)。
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
    // 数组轴 = [y,z,x], nifti-rs 视为 [i,j,k] → i→y, j→z, k→x。
    hdr.srow_x = [0.0, 0.0, sx, -rx];
    hdr.srow_y = [sy, 0.0, 0.0, -ry];
    hdr.srow_z = [0.0, sz, 0.0, -rz];
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

/// L1 TV 梯度 (边缘保持): TV = Σ|μ_{i+1}-μ_i|, dTV/dμ = sign 差和
/// (smooth tanh 近似, ε 控制平滑)。
fn tv_l1_grad(v: &Tensor<3>, eps: f32) -> Tensor<3> {
    use burn::tensor::s;
    let sn = |t: Tensor<3>| t.clone().mul_scalar(1.0 / eps).tanh();
    // x 轴 (dim2): 前向/后向差分的符号。
    let dx_f = v.clone().slice(s![.., .., 1..]).sub(v.clone().slice(s![.., .., ..-1])); // μ_{i+1}-μ_i
    // μ_i 的贡献: 对前向差 (μ_i - μ_{i-1}) 是 -sign(dx_f[.., .., i-1]); 简化用移位。
    let prev_x = Tensor::cat(vec![v.clone().slice(s![.., .., 0..1]), v.clone().slice(s![.., .., 0..-1])], 2);
    let next_x = Tensor::cat(vec![v.clone().slice(s![.., .., 1..]), v.clone().slice(s![.., .., -1..])], 2);
    let mut g = sn(v.clone().sub(prev_x)).add(sn(v.clone().sub(next_x)));
    let prev_y = Tensor::cat(vec![v.clone().slice(s![.., 0..1, ..]), v.clone().slice(s![.., 0..-1, ..])], 1);
    let next_y = Tensor::cat(vec![v.clone().slice(s![.., 1.., ..]), v.clone().slice(s![.., -1.., ..])], 1);
    g = g.add(sn(v.clone().sub(prev_y))).add(sn(v.clone().sub(next_y)));
    let prev_z = Tensor::cat(vec![v.clone().slice(s![0..1, .., ..]), v.clone().slice(s![0..-1, .., ..])], 0);
    let next_z = Tensor::cat(vec![v.clone().slice(s![1.., .., ..]), v.clone().slice(s![-1.., .., ..])], 0);
    g.add(sn(v.clone().sub(prev_z))).add(sn(v.clone().sub(next_z)))
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
    let mut dump_drr: Option<PathBuf> = None;
    let mut dump_views = 8usize;
    let mut volume_mirror = false;
    let mut volume_transpose = false;
    let mut save_volume: Option<PathBuf> = None;
    let mut tv = 0.0f32;
    let mut tv_type = "l1".to_string();
    let mut tv_eps = 0.01f32;
    let mut motion_mask = false;
    let mut mask_sigma = 0.25f32; // 残差掩膜尺度 (proj 单位)
    let mut mask_warmup = 150usize;
    // 训练中周期保存验证 NRRD 序列 (pred+gt 拼接, 目视验证方向/质量)。
    let mut save_val_nrrd: Option<PathBuf> = None;
    let mut save_val_every = 500usize;
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
        } else if let Some(v) = a.strip_prefix("--tv-type=") {
            tv_type = v.to_string();
        } else if let Some(v) = a.strip_prefix("--tv-eps=") {
            tv_eps = v.parse()?;
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
        } else if let Some(v) = a.strip_prefix("--dump-drr=") {
            dump_drr = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--dump-views=") {
            dump_views = v.parse()?;
        } else if a == "--volume-mirror" {
            volume_mirror = true;
        } else if a == "--volume-transpose" {
            volume_transpose = true;
        } else if let Some(v) = a.strip_prefix("--save-volume=") {
            save_volume = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--save-val-nrrd=") {
            save_val_nrrd = Some(PathBuf::from(v));
        } else if let Some(v) = a.strip_prefix("--save-val-every=") {
            save_val_every = v.parse()?;
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
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let rx = meta["rx"].as_f64().unwrap_or(118.6) as f32;
    let ry = meta["ry"].as_f64().unwrap_or(rx as f64) as f32;
    let rz = meta["rz"].as_f64().unwrap_or(84.7) as f32;
    let (mut vol_vec, _hx, _hy, _hz) = read_nifti_volume(&volume_path)?;
    // 维度以 meta.json 为准 (nii header dims 是置换后的 [y,z,x])。
    let vol_x = meta["vol_x"].as_u64().unwrap_or(_hx as u64) as usize;
    let vol_y = meta["vol_y"].as_u64().unwrap_or(_hy as u64) as usize;
    let vol_z = meta["vol_z"].as_u64().unwrap_or(_hz as u64) as usize;
    assert_eq!(vol_vec.len(), vol_x * vol_y * vol_z, "volume size mismatch");
    let calib: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&calib_path)?)?;
    let scale = calib["s"].as_f64().unwrap_or(1.0) as f32;
    let bias = calib["b"].as_f64().unwrap_or(0.0) as f32;
    println!(
        "volume {vol_x}x{vol_y}x{vol_z} (rx={rx:.1} ry={ry:.1} rz={rz:.1}mm), calib s={scale:.5} b={bias:.5}, iters={iters} lr={lr} batch={batch}"
    );

    let device: burn::tensor::Device = test_device().await.into();
    let vol_total = vol_x * vol_y * vol_z;

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
            vol_x as u32,
            vol_y as u32,
            vol_z as u32,
            steps as u32,
            rx,
            ry,
            rz,
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
        let volx: u32 = 16;
        let voly: u32 = 16;
        let volz: u32 = 12;
        let img: u32 = 48;
        let rx: f32 = 100.0;
        let ry: f32 = 100.0;
        let rz: f32 = 75.0;
        let steps: u32 = 64;
        let mut v: Vec<f32> = (0..volx * voly * volz).map(|_| rand::random::<f32>() * 0.01).collect();
        let st = DrrSettings::new(
            &views[0].camera,
            img,
            img,
            volx,
            voly,
            volz,
            steps,
            rx,
            ry,
            rz,
            1.0,
            0.0,
        );
        let fwd = |vv: Vec<f32>| {
            let t = Tensor::<3>::from_data(TensorData::new::<f32, _>(vv, [volx as usize, voly as usize, volz as usize]), &device);
            let p = drr_forward_t(&st, t);
            p
        };
        // 解析: backward with v_proj=ones (L = Σ proj)
        let proj0 = fwd(v.clone());
        let ones = Tensor::<2>::ones([img as usize; 2], &device);
        let v_vol = drr_backward_t(&st, ones).await;
        let v_vol_cpu: Vec<f32> = v_vol.into_data().into_vec::<f32>().unwrap();
        let eps = 1e-4f32;
        for idx in [0usize, 100, 1000, volx as usize * voly as usize / 2 + 17, v.len() / 2] {
            let orig = v[idx];
            v[idx] = orig + eps;
            let p2 = drr_forward_t(&st, Tensor::<3>::from_data(TensorData::new::<f32, _>(v.clone(), [volx as usize, voly as usize, volz as usize]), &device)).await;
            let s2: f32 = p2.sum().into_scalar();
            v[idx] = orig - eps;
            let p3 = drr_forward_t(&st, Tensor::<3>::from_data(TensorData::new::<f32, _>(v.clone(), [volx as usize, voly as usize, volz as usize]), &device)).await;
            let s3: f32 = p3.sum().into_scalar();
            v[idx] = orig;
            let num = (s2 - s3) / (2.0 * eps);
            let ana = v_vol_cpu[idx];
            println!("selftest voxel {idx}: num={num:.6} analytic={ana:.6} ratio={:.3}", num / ana.max(1e-12));
        }
        return Ok(());
    }

    // ---- 体积 + Adam 状态 (全程 GPU) ----
    // 诊断: 沿 y=-x 反射体积 (修正 FDK 世界系镜像)。
    if volume_mirror {
        mirror_volume_negxy(&mut vol_vec, vol_x, vol_y, vol_z);
        println!("volume mirrored across y=-x (FDK world-frame fix)");
    }
    if volume_transpose {
        transpose_volume_xy(&mut vol_vec, vol_x, vol_y, vol_z);
        println!("volume transposed in xy (x<->y, FDK y=x-reflection fix)");
    }
    if let Some(sp) = &save_volume {
        if let Some(parent) = sp.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_nifti_volume(sp, &vol_vec, vol_x, vol_y, vol_z, rx, ry, rz)?;
        println!("saved volume (after transform) -> {sp:?}");
        return Ok(());
    }
    let mut volume = Tensor::<3>::from_data(
        TensorData::new::<f32, _>(vol_vec, [vol_x, vol_y, vol_z]),
        &device,
    );

    // ---- eval-only: 评估已保存体积 (含强度 PSNR) ----
    if eval_only {
        let mut se = 0.0f64;
        let mut sei = 0.0f64;
        let mut n = 0u64;
        // 跨所有帧均匀采样 (每个旋转角度一段); --dump-views=0 = 全部帧。
        let n_eval = if dump_views == 0 {
            views.len()
        } else {
            dump_views.max(1)
        };
        // 一次性输出多帧序列: 所有采样视角叠成一个 [N,H,W] 3D NRRD。
        let mut stack_fdk: Vec<f32> = Vec::new();
        let mut stack_gt: Vec<f32> = Vec::new();
        let mut stack_info: Vec<String> = Vec::new();
        for k in 0..n_eval {
            let vi = (k * views.len() / n_eval) % views.len();
            let p = views[vi].camera.position;
            let ang = p.x.atan2(p.y).to_degrees();
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
            if let Some(dir) = &dump_drr {
                stack_fdk.extend(pred.iter().map(|p| (-(*p as f64)).exp().clamp(0.0, 1.0) as f32));
                stack_gt.extend(gts[vi].iter().map(|&g| (-(g as f64)).exp().clamp(0.0, 1.0) as f32));
                stack_info.push(format!("view{k:02}_idx{vi:03}_ang{ang:+.0}°"));
            }
        }
        // 写 3D 多帧 NRRD (一次输出整个序列)。
        if let Some(dir) = &dump_drr {
            use brush_train::xray_eval::save_gray_nrrd_f32_stack;
            std::fs::create_dir_all(dir)?;
            let (w, h) = (img_w as usize, img_h as usize);
            let mut info_path = dir.join("stack_views.txt");
            std::fs::write(&info_path, stack_info.join("\n") + "\n")?;
            let fm = TensorData::new::<f32, _>(stack_fdk, [n_eval, h, w]);
            let gm = TensorData::new::<f32, _>(stack_gt, [n_eval, h, w]);
            save_gray_nrrd_f32_stack(&dir.join("fdk_drr_gpu_stack.nrrd"), &fm)?;
            save_gray_nrrd_f32_stack(&dir.join("gt_drr_stack.nrrd"), &gm)?;
            println!(
                "dumped {n_eval}-frame NRRD stack -> {dir:?} (fdk_drr_gpu_stack.nrrd / gt_drr_stack.nrrd), view list in stack_views.txt"
            );
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
        // TV 正则: d(λ·TV)/dμ 加入梯度 (l1 边缘保持 / l2 平滑)。
        if tv > 0.0 {
            let g = if tv_type == "l2" {
                tv_grad(&volume)
            } else {
                tv_l1_grad(&volume, tv_eps)
            };
            grad = grad.add(g.mul_scalar(tv * batch as f32));
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
            // 周期保存验证 NRRD 序列 (8 视角, pred|gt 拼接)。
            if let Some(dir) = &save_val_nrrd
                && (it % save_val_every == 0 || it == iters - 1)
            {
                use brush_train::xray_eval::save_gray_nrrd_f32_stack;
                std::fs::create_dir_all(dir)?;
                let (w, h) = (img_w as usize, img_h as usize);
                let mut merged: Vec<f32> = Vec::with_capacity(8 * h * w * 2);
                for k in 0..8 {
                    let vi = (k * views.len() / 8) % views.len();
                    let proj = drr_forward_t(&settings[vi], volume.clone()).await;
                    let pred: Vec<f32> = proj.into_data().into_vec::<f32>().unwrap();
                    // 水平拼接 [GT row | pred row] (非逐像素交错, 避免竖条纹)。
                    for y in 0..h {
                        let row_g = y * w;
                        for x in 0..w {
                            let g = gts[vi][row_g + x];
                            merged.push((-(g as f64)).exp().clamp(0.0, 1.0) as f32);
                        }
                        for x in 0..w {
                            let p = pred[row_g + x];
                            merged.push((-(p as f64)).exp().clamp(0.0, 1.0) as f32);
                        }
                    }
                }
                let td = TensorData::new::<f32, _>(merged, [8, h, w * 2]);
                save_gray_nrrd_f32_stack(&dir.join(format!("val_{it:05}.nrrd")), &td)?;
            }
        }
    }
    println!("done in {:.1}s", t0.elapsed().as_secs_f32());

    // ---- 保存 refined 体积 ----
    std::fs::create_dir_all(&out)?;
    let vol_cpu: Vec<f32> = volume.clone().into_data().into_vec::<f32>().unwrap();
    write_nifti_volume(&out.join("volume_refined.nii.gz"), &vol_cpu, vol_x, vol_y, vol_z, rx, ry, rz)?;
    println!("saved {out:?}/volume_refined.nii.gz");
    Ok(())
}
