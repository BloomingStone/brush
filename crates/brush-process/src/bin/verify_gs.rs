//! verify_gs: 验证保存的 GS 产物 (canonical_final.ply + deform_field_phase{p}.npy)
//! 能否复现实验的重建结果。
//!
//! 加载 ply 的 canonical splats, 用形变场网格三线性插值到 phase 0, 在
//! eval 视图 (phase≈0 的那些) 上投影 (brush-xray forward), 与 GT 和训练时
//! 保存的 pred (gt_pred_*.nrrd 对应位置) 对比 PSNR。
//!
//! 纯 GS 不需要 FDK: 只需要 ply + 形变场 + shape/spacing/AABB。

use brush_deform::{HexPlaneConfig, HexPlaneDeformConfig, HexPlaneDeformModel, deform_splats};
use brush_render::camera::Camera;
use brush_render::kernels::camera_model::CameraModel;
use brush_serde::import::load_splat_from_ply;
use brush_train::xray_train::DeformNetwork;
use brush_xray::XRaySplats;
use brush_xray_bwd::render_xray;
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Device, Tensor, TensorData};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    // dump_deform 实际布局: pts 按 z 最慢、x 最快 (for iz { for iy { for ix }}),
    // flat = (iz*ny + iy)*nx + ix (尽管 npy 声明 [nx,ny,nz,3] 是误导的)。
    let at = |ix: usize, iy: usize, iz: usize, a: usize| -> f32 {
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

fn psnr(a: &[f32], b: &[f32]) -> f32 {
    let mut se = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64);
        se += d * d;
    }
    let mse = se / a.len().max(1) as f64;
    if mse > 1e-12 { 10.0 * (1.0 / mse).log10() as f32 } else { 99.0 }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut ply: Option<PathBuf> = None;
    let mut deform_field: Option<PathBuf> = None;
    let mut deform_extent = 264.0f32;
    let mut dcm: Option<PathBuf> = None;
    let mut out = PathBuf::from("target/verify_gs");
    let mut eval_views_n = 8usize;
    let mut ckpt: Option<PathBuf> = None;
    let mut hex_res = 64u32;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if let Some(v) = a.strip_prefix("--ply=") { ply = Some(PathBuf::from(v)); }
        else if let Some(v) = a.strip_prefix("--deform-field=") { deform_field = Some(PathBuf::from(v)); }
        else if let Some(v) = a.strip_prefix("--deform-extent=") { deform_extent = v.parse()?; }
        else if let Some(v) = a.strip_prefix("--out=") { out = PathBuf::from(v); }
        else if let Some(v) = a.strip_prefix("--eval-views=") { eval_views_n = v.parse()?; }
        else if let Some(v) = a.strip_prefix("--ckpt=") { ckpt = Some(PathBuf::from(v)); }
        else if let Some(v) = a.strip_prefix("--hex-res=") { hex_res = v.parse()?; }
        else if dcm.is_none() { dcm = Some(PathBuf::from(a)); }
        i += 1;
    }
    let ply = ply.expect("usage: verify_gs <dcm> --ply=... [--deform-field=...] --out=...");
    let dcm = dcm.expect("usage: verify_gs <dcm> ...");
    std::fs::create_dir_all(&out)?;

    let wgpu = brush_process::burn_init_setup().await;
    let device: Device = wgpu.into();
    let device_ad = device.clone().autodiff();

    // ---- 加载 ply -> canonical splats ----
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
    let canonical = XRaySplats::from_raw(means.clone(), rots.clone(), log_scales.clone(), raw.clone(), &device);

    // ---- 加载形变场 + 插值 -> deformed ----
    let mut deformed = None;
    if let Some(df) = &deform_field {
        let (field, shape) = read_npy_f32(df)?;
        let n3 = [shape[0], shape[1], shape[2]];
        assert_eq!(shape.len(), 4, "deform field must be [nx,ny,nz,3]");
        assert_eq!(field.len(), n3[0] * n3[1] * n3[2] * 3, "deform field size");
        let mut dmeans = means.clone();
        let mut dmax = 0.0f32;
        for k in 0..n {
            let p = glam::Vec3::new(means[k * 3], means[k * 3 + 1], means[k * 3 + 2]);
            let d = sample_deform_field(&field, n3, deform_extent, p);
            dmeans[k * 3] = p.x + d.x;
            dmeans[k * 3 + 1] = p.y + d.y;
            dmeans[k * 3 + 2] = p.z + d.z;
            dmax = dmax.max(d.length());
        }
        println!("deform field {df:?}: max|d|={dmax:.2}mm");
        deformed = Some(XRaySplats::from_raw(dmeans, rots.clone(), log_scales.clone(), raw.clone(), &device));
    }

    // ---- 可选: 直接加载 deform 网络 (精确求位移, 不走网格) ----
    let mut net_model: Option<HexPlaneDeformModel> = None;
    let mut net_deformed = None;
    if let Some(cp) = &ckpt {
        let device_ad = device.clone().autodiff();
        let cfg = HexPlaneDeformConfig {
            hex_plane: HexPlaneConfig {
                n_feature_dim: 16,
                spatial_resolution: hex_res,
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
        net_model = Some(model.clone());
        let n_s = means.len() / 3;
        let xyz = Tensor::<2>::from_data(TensorData::new::<f32, _>(means.clone(), [n_s, 3]), &device_ad);
        let phase_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![0.0; n_s], [n_s, 1]), &device_ad);
        let time_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![0.0; n_s], [n_s, 1]), &device_ad);
        let deforms = model.forward(xyz.clone(), phase_t, time_t);
        // 完整形变: 位移 + 旋转 (predict_scaling=false, d_scaling=0 → 尺度不变)。
        let dv: Vec<f32> = deforms.d_xyz.clone().into_data_async().await?.to_vec()?;
        let mut dmax = 0.0f32;
        for k in 0..n_s {
            dmax = dmax.max(glam::Vec3::new(dv[k*3], dv[k*3+1], dv[k*3+2]).length());
        }
        println!("network deform (phase0,time0) from {cp:?}: max|d|={dmax:.2}mm (含 rotation)");
        let canonical_ad = XRaySplats::from_raw(
            means.clone(), rots.clone(), log_scales.clone(), raw.clone(), &device_ad,
        );
        let deformed_ad = deform_splats(&canonical_ad, &deforms);
        let dmeans: Vec<f32> = deformed_ad.means().into_data_async().await?.to_vec()?;
        let drots: Vec<f32> = deformed_ad.rotations().into_data_async().await?.to_vec()?;
        net_deformed = Some(XRaySplats::from_raw(dmeans, drots, log_scales.clone(), raw.clone(), &device));
    }

    // ---- 加载数据集 (同 fit_deform 配置) ----
    let file = tokio::fs::File::open(&dcm).await?;
    let name = dcm.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or("input.dcm".into());
    let vfs = Arc::new(
        brush_vfs::BrushVfs::from_reader(tokio::io::BufReader::new(file), Some(name))
            .await.expect("construct vfs"),
    );
    let load_config = brush_dataset::config::LoadDatasetConfig {
        max_frames: None,
        max_resolution: 1920,
        eval_split_every: Some(5),
        subsample_frames: None,
        subsample_points: None,
        alpha_mode: None,
        dicom_orientation: brush_dataset::config::XRayOrientation::Ap,
        dicom_normalization: brush_dataset::config::DicomNormalization::Minmax,
        dicom_gamma: None,
        dicom_gamma_target: Some(0.5),
        max_scene_batch_cache_size: 1 << 30,
        roi: brush_dataset::config::RoiSpec::Inset(20),
    };
    let result = brush_dataset::load_dataset(vfs, &load_config).await?;
    let eval_views: Vec<&brush_dataset::scene::SceneView> = if let Some(eval_scene) = &result.dataset.eval {
        let n_ev = eval_scene.views.len();
        (0..eval_views_n).map(|i| &eval_scene.views[i * n_ev / eval_views_n]).collect()
    } else {
        vec![&result.dataset.train.views[0]]
    };
    println!("eval views: {}", eval_views.len());

    // ---- 逐视图渲染 (canonical vs deformed) + 对比 GT ----
    for (vi, view) in eval_views.iter().enumerate() {
        let gray = view.gray_image.as_ref().expect("gray GT");
        let img_size = glam::uvec2(gray.width, gray.height);
        let cam = &view.camera;
        let gt: Vec<f32> = gray.data.as_ref().to_vec();
        let phase = view.phase;
        let time = view.time;
        let mut line = format!("view {vi}: phase={phase:.3} time={time:.4}");

        async fn render(splats: &XRaySplats, cam: &Camera, img_size: glam::UVec2, device_ad: &Device) -> Tensor<2> {
            // 与训练 eval_view 一致: lift 到 autodiff + render_xray (Backward pass)。
            let ad = brush_xray_bwd::lift_xray_splats_to_autodiff(splats.clone());
            let out = render_xray(ad, cam, img_size, 1.0, false).await;
            out.img
        }

        // canonical (未变形)
        let proj_c = render(&canonical, cam, img_size, &device_ad).await;
        let int_c: Vec<f32> = proj_c.into_data().to_vec::<f32>()?.iter()
            .map(|&p| (-(p as f64).clamp(1e-3, 14.0)).exp() as f32).collect();
        line.push_str(&format!(" | canonical_psnr={:.2}", psnr(&int_c, &gt)));

        // deformed (phase0 形变场网格)
        let mut int_d: Option<Vec<f32>> = None;
        if let Some(def) = &deformed {
            let proj_d = render(def, cam, img_size, &device_ad).await;
            int_d = Some(proj_d.into_data().to_vec::<f32>()?.iter()
                .map(|&p| (-(p as f64).clamp(1e-3, 14.0)).exp() as f32).collect());
            line.push_str(&format!(" | def_grid_psnr={:.2}", psnr(int_d.as_ref().unwrap(), &gt)));
        }
        // deformed (网络精确)
        let mut int_n: Option<Vec<f32>> = None;
        if let Some(def) = &net_deformed {
            let proj_d = render(def, cam, img_size, &device_ad).await;
            int_n = Some(proj_d.into_data().to_vec::<f32>()?.iter()
                .map(|&p| (-(p as f64).clamp(1e-3, 14.0)).exp() as f32).collect());
            line.push_str(&format!(" | def_net_psnr={:.2}", psnr(int_n.as_ref().unwrap(), &gt)));
        }
        println!("{line}");

        // 每视图精确复现: 网络在 view 的 phase/time 上形变 (同训练 eval_view)。
        let mut int_net_phase: Option<Vec<f32>> = None;
        if let Some(model) = &net_model {
            let n_s = means.len() / 3;
            let xyz = Tensor::<2>::from_data(TensorData::new::<f32, _>(means.clone(), [n_s, 3]), &device_ad);
            let phase_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![phase; n_s], [n_s, 1]), &device_ad);
            let time_t = Tensor::<2>::from_data(TensorData::new::<f32, _>(vec![time; n_s], [n_s, 1]), &device_ad);
            let deforms = model.forward(xyz, phase_t, time_t);
            let canonical_ad = XRaySplats::from_raw(
                means.clone(), rots.clone(), log_scales.clone(), raw.clone(), &device_ad,
            );
            let def_ad = deform_splats(&canonical_ad, &deforms);
            let dmeans: Vec<f32> = def_ad.means().into_data_async().await?.to_vec()?;
            let drots: Vec<f32> = def_ad.rotations().into_data_async().await?.to_vec()?;
            let def_sp = XRaySplats::from_raw(dmeans, drots, log_scales.clone(), raw.clone(), &device);
            let proj_d = render(&def_sp, cam, img_size, &device_ad).await;
            let int_p = proj_d.into_data().to_vec::<f32>()?.iter()
                .map(|&p| (-(p as f64).clamp(1e-3, 14.0)).exp() as f32).collect::<Vec<_>>();
            line.push_str(&format!(" | def_phase_psnr={:.2}", psnr(&int_p, &gt)));
            int_net_phase = Some(int_p);
        }
        println!("{line}");

        // 保存对比图: GT | pred(网络) | 差值 (PNG + NRRD)
        if let Some(int_n) = int_net_phase.as_ref().or(int_n.as_ref()) {
            let h = gray.height as usize;
            let w = gray.width as usize;
            let save = |name: &str, pred: &[f32]| -> anyhow::Result<()> {
                use image::{GrayImage, Luma};
                let mut img = GrayImage::new((w * 3) as u32, h as u32);
                for y in 0..h {
                    for x in 0..w {
                        let gi = (gt[y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
                        let pi = (pred[y * w + x].clamp(0.0, 1.0) * 255.0) as u8;
                        let di = ((gt[y * w + x] - pred[y * w + x]).abs().clamp(0.0, 1.0) * 255.0) as u8;
                        img.put_pixel(x as u32, y as u32, Luma([gi]));
                        img.put_pixel((w + x) as u32, y as u32, Luma([pi]));
                        img.put_pixel((2 * w + x) as u32, y as u32, Luma([di]));
                    }
                }
                img.save(out.join(format!("{name}_v{vi:02}.png")))?;
                Ok(())
            };
            save("verify", &int_n)?;
            // 也存 NRRD (pred vs GT)
            use brush_train::xray_eval::save_gray_nrrd_f32;
            let mut pred_td = TensorData::new::<f32, _>(int_n.clone(), [h, w]);
            let mut gt_td = TensorData::new::<f32, _>(gt.clone(), [h, w]);
            save_gray_nrrd_f32(&out.join(format!("pred_v{vi:02}.nrrd")), &pred_td)?;
            save_gray_nrrd_f32(&out.join(format!("gt_v{vi:02}.nrrd")), &gt_td)?;
        }
    }
    println!("done -> {out:?}");
    Ok(())
}
