//! Backward of the X-ray projection: cov2d-cone backward (conic + mu →
//! cov3D + mean through the 3x3 J), pixel-projection VJP (xy → mean), and
//! computeCov3D backward (cov3D → scale / quat). Transcribed from
//! R2-Gaussian `backward.cu` (`computeCov2DCUDA` cone branch,
//! `preprocessCUDA`, `computeCov3D`).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{MU_WATER, Mat3, Quat, Sym2, Sym3, Vec3A, compute_cov3d, dnormvdv4, is_finite_f32, sigmoid};
use brush_xray::kernels::helpers::{
    XRAY_LANES, cone_geometry, read_quat_unorm_xray, read_scale_xray,
};
use brush_xray::kernels::types::XRayProjectUniforms;

pub const WG_SIZE: u32 = 256;

/// Cov2D cone-beam backward: from `v_conic` (2x2) and `v_mu` to the 3D
/// covariance grads (6) and the mean grad flowing through the projection
/// Jacobian / ray-direction row. Mirrors R2 `computeCov2DCUDA` (mode=1).
#[cube]
#[allow(clippy::approx_constant)]
fn compute_cov2d_cone_bwd(
    mean_c: Vec3A,
    scale: Vec3A,
    quat: Quat,
    u: XRayProjectUniforms,
    v_conic: Sym2,
    v_mu: f32,
) -> (Sym3, Vec3A) {
    let (_conic, mu, cov3, _j, m, tx, ty, txtz, tytz) =
        cone_geometry(mean_c, scale, quat, u);
    let vrk = compute_cov3d(scale, quat);

    let hata = cov3.c00;
    let hatb = cov3.c01;
    let hatc = cov3.c02;
    let hatd = cov3.c11;
    let hate = cov3.c12;
    let hatf = cov3.c22;

    let denom = hata * hatd - hatb * hatb;
    let denom2inv = 1.0f32 / (denom * denom + 1.0e-7f32);
    let circ = hata * hatd * hatf + 2.0f32 * hatb * hatc * hate - hata * hate * hate
        - hatf * hatb * hatb - hatd * hatc * hatc;
    let pi = 3.14159265358979323846f32;
    let pi_mu = pi / (mu + 1.0e-7f32);
    let circ_diamond = circ / denom;

    // 2x2 conic grads.
    let mut dL_dhata = denom2inv
        * (-hatd * hatd * v_conic.c00 + hatb * hatd * v_conic.c01
            + (denom - hata * hatd) * v_conic.c11);
    let mut dL_dhatd = denom2inv
        * (-hata * hata * v_conic.c11 + hata * hatb * v_conic.c01
            + (denom - hata * hatd) * v_conic.c00);
    let mut dL_dhatb = denom2inv
        * (2.0f32 * hatb * hatd * v_conic.c00 - (denom + 2.0f32 * hatb * hatb) * v_conic.c01
            + 2.0f32 * hata * hatb * v_conic.c11);
    // mu grads.
    dL_dhata += pi_mu * ((hatd * hatf - hate * hate) / denom - hatd * circ_diamond / denom) * v_mu;
    dL_dhatb += pi_mu
        * ((2.0f32 * hatc * hate - 2.0f32 * hatf * hatb) / denom
            + 2.0f32 * hatb * circ_diamond / denom)
        * v_mu;
    let dL_dhatc = pi_mu * ((2.0f32 * hatb * hate - 2.0f32 * hatd * hatc) / denom) * v_mu;
    dL_dhatd += pi_mu * ((hata * hatf - hatc * hatc) / denom - hata * circ_diamond / denom) * v_mu;
    let dL_dhate = pi_mu * ((2.0f32 * hatb * hatc - 2.0f32 * hata * hate) / denom) * v_mu;
    let dL_dhatf = pi_mu * ((hata * hatd - hatb * hatb) / denom) * v_mu;

    // D = dL/dcov2 (symmetric 3x3) from the six cov2d grads.
    // G = full 3x3 cov2d gradient (symmetric-vector -> full: off-diag halved).
    let dg = Mat3::from_cols(
        Vec3A::new(dL_dhata, 0.5f32 * dL_dhatb, 0.5f32 * dL_dhatc),
        Vec3A::new(0.5f32 * dL_dhatb, dL_dhatd, 0.5f32 * dL_dhate),
        Vec3A::new(0.5f32 * dL_dhatc, 0.5f32 * dL_dhate, dL_dhatf),
    );

    // v_cov3D = S = symmetric-vector gradient of Vrk (matches R2's dL_dcov).
    // cov3 = mᵀ·Vrk·m ⇒ full-gradient X = m·G·mᵀ; the 6 stored cov entries
    // are symmetric, so S[i,i] = X[i,i] and S[i,j] = 2·X[i,j] (i≠j).
    let active = mu != 0.0f32;
    let dr0 = dg.mul_vec3(m.row0());
    let dr1 = dg.mul_vec3(m.row1());
    let dr2 = dg.mul_vec3(m.row2());
    let v_c00 = select(active, m.row0().dot(dr0), 0.0f32);
    let v_c01 = select(active, 2.0f32 * m.row0().dot(dr1), 0.0f32);
    let v_c02 = select(active, 2.0f32 * m.row0().dot(dr2), 0.0f32);
    let v_c11 = select(active, m.row1().dot(dr1), 0.0f32);
    let v_c12 = select(active, 2.0f32 * m.row1().dot(dr2), 0.0f32);
    let v_c22 = select(active, m.row2().dot(dr2), 0.0f32);

    // Mean grads through J: dL/dm = (Vrk·m)·D' with D' = D diagonal doubled
    // (symmetric-adjoint of m ↦ mᵀ·Vrk·m; matches R2's dL_dM), dL/dJ = Wᵀ·dL/dm.
    let dm_diag2 = Mat3::from_cols(
        Vec3A::new(2.0f32 * dL_dhata, dL_dhatb, dL_dhatc),
        Vec3A::new(dL_dhatb, 2.0f32 * dL_dhatd, dL_dhate),
        Vec3A::new(dL_dhatc, dL_dhate, 2.0f32 * dL_dhatf),
    );
    let dl_dm = vrk.mul_mat3(m).mul_mat3(dm_diag2);
    let w = u.view_rotation();
    let dl_dj = Mat3::from_cols(
        w.transpose_mul_vec3(dl_dm.col0()),
        w.transpose_mul_vec3(dl_dm.col1()),
        w.transpose_mul_vec3(dl_dm.col2()),
    );

    // J element grads (my J is column-major; [r][c] = c{c}_{x,y,z}[r]):
    //   J00 = fx/tz, J20 = -fx·tx/tz², J11 = fy/tz, J21 = -fy·ty/tz²,
    //   J02 = tx/l, J12 = ty/l, J22 = tz/l.
    let dL_dJ00 = dl_dj.c0_x;
    let dL_dJ20 = dl_dj.c0_z;
    let dL_dJ11 = dl_dj.c1_y;
    let dL_dJ21 = dl_dj.c1_z;
    let dL_dJ02 = dl_dj.c2_x;
    let dL_dJ12 = dl_dj.c2_y;
    let dL_dJ22 = dl_dj.c2_z;

    let tz = mean_c.z();
    let inv_tz = 1.0f32 / tz;
    let inv_tz2 = inv_tz * inv_tz;
    let inv_tz3 = inv_tz2 * inv_tz;
    let l = f32::sqrt(tx * tx + ty * ty + tz * tz);
    let inv_l3 = 1.0f32 / (l * l * l);

    let x_grad_mul = select(txtz > u.lim_pos_x || txtz < u.lim_neg_x, 0.0f32, 1.0f32);
    let y_grad_mul = select(tytz > u.lim_pos_y || tytz < u.lim_neg_y, 0.0f32, 1.0f32);

    // dL/dt from J's derivatives w.r.t. (tx, ty, tz).
    let dL_dtx = x_grad_mul
        * (-u.focal_x * inv_tz2 * dL_dJ20
            + (l * l - tx * tx) * inv_l3 * dL_dJ02
            - tx * ty * inv_l3 * dL_dJ12
            - tx * tz * inv_l3 * dL_dJ22);
    let dL_dty = y_grad_mul
        * (-u.focal_y * inv_tz2 * dL_dJ21
            - tx * ty * inv_l3 * dL_dJ02
            + (l * l - ty * ty) * inv_l3 * dL_dJ12
            - ty * tz * inv_l3 * dL_dJ22);
    let dL_dtz = -u.focal_x * inv_tz2 * dL_dJ00
        + 2.0f32 * u.focal_x * tx * inv_tz3 * dL_dJ20
        - u.focal_y * inv_tz2 * dL_dJ11
        + 2.0f32 * u.focal_y * ty * inv_tz3 * dL_dJ21
        - tx * tz * inv_l3 * dL_dJ02
        - ty * tz * inv_l3 * dL_dJ12
        + (l * l - tz * tz) * inv_l3 * dL_dJ22;

    // Camera-space mean grad (t *is* the camera-space point).
    let v_mean_c_cov = Vec3A::new(dL_dtx, dL_dty, dL_dtz);

    (
        Sym3 {
            c00: v_c00,
            c01: v_c01,
            c02: v_c02,
            c11: v_c11,
            c12: v_c12,
            c22: v_c22,
        },
        v_mean_c_cov,
    )
}

/// Backward of `computeCov3D`. Brush forward is
/// `Sigma = R_std·S²·R_stdᵀ` (= `(S·R_r2)ᵀ(S·R_r2)` with R2's transposed
/// rotation `R_r2 = R_stdᵀ`). Transcribed from R2 `backward.cu`
/// `computeCov3D`, expressed in brush terms (R2's `R = R_r2 = transpose(r)`).
#[cube]
fn compute_cov3d_bwd(scale: Vec3A, quat: Quat, v_cov3d: Sym3) -> (Vec3A, Quat) {
    let r = quat.to_mat3();
    // msr = R_std·S — each column of R scaled by the matching scalar.
    let msr = Mat3::from_cols(
        r.col0().scale(scale.x()),
        r.col1().scale(scale.y()),
        r.col2().scale(scale.z()),
    );
    // R2's M = S·R_r2 = (R_std·S)ᵀ = msrᵀ.
    let m_r2 = Mat3::from_cols(msr.row0(), msr.row1(), msr.row2());
    // dL_dSigma (symmetric 3x3) from the 6 grads.
    let dl_dsigma = Mat3 {
        c0_x: v_cov3d.c00,
        c0_y: 0.5f32 * v_cov3d.c01,
        c0_z: 0.5f32 * v_cov3d.c02,
        c1_x: 0.5f32 * v_cov3d.c01,
        c1_y: v_cov3d.c11,
        c1_z: 0.5f32 * v_cov3d.c12,
        c2_x: 0.5f32 * v_cov3d.c02,
        c2_y: 0.5f32 * v_cov3d.c12,
        c2_z: v_cov3d.c22,
    };
    // dL_dM = 2·M·dL_dSigma (R2's M = S·R_r2).
    let dl_dm_raw = m_r2.mul_mat3(dl_dsigma);
    let dl_dm = Mat3::from_cols(
        dl_dm_raw.col0().scale(2.0f32),
        dl_dm_raw.col1().scale(2.0f32),
        dl_dm_raw.col2().scale(2.0f32),
    );

    // v_scale = (R_r2.row0·dL_dM.row0, ...) = (r.col0·dL_dM.row0, ...)
    // (R_r2.row_k = r.col_k since R_r2 = rᵀ). Empirically verified vs R2 `_C`.
    let v_scale = Vec3A::new(
        r.col0().dot(dl_dm.row0()),
        r.col1().dot(dl_dm.row1()),
        r.col2().dot(dl_dm.row2()),
    );

    // mt = dL_dM with ROWS scaled by s (R2's effective dL_dMt, empirically
    // verified vs R2 `_C` — NOT the transpose).
    let mt00 = dl_dm.c0_x * scale.x();
    let mt01 = dl_dm.c1_x * scale.x();
    let mt02 = dl_dm.c2_x * scale.x();
    let mt10 = dl_dm.c0_y * scale.y();
    let mt11 = dl_dm.c1_y * scale.y();
    let mt12 = dl_dm.c2_y * scale.y();
    let mt20 = dl_dm.c0_z * scale.z();
    let mt21 = dl_dm.c1_z * scale.z();
    let mt22 = dl_dm.c2_z * scale.z();

    let w = quat.w();
    let qx = quat.x();
    let qy = quat.y();
    let qz = quat.z();
    // R2: r=q.x(=w), x=q.y, y=q.z, z=q.w.
    let dq_w = 2.0f32 * qz * (mt01 - mt10) + 2.0f32 * qy * (mt20 - mt02)
        + 2.0f32 * qx * (mt12 - mt21);
    let dq_x = 2.0f32 * qy * (mt10 + mt01) + 2.0f32 * qz * (mt20 + mt02)
        + 2.0f32 * w * (mt12 - mt21) - 4.0f32 * qx * (mt22 + mt11);
    let dq_y = 2.0f32 * qx * (mt10 + mt01) + 2.0f32 * w * (mt20 - mt02)
        + 2.0f32 * qz * (mt12 + mt21) - 4.0f32 * qy * (mt22 + mt00);
    let dq_z = 2.0f32 * w * (mt01 - mt10) + 2.0f32 * qx * (mt20 + mt02)
        + 2.0f32 * qy * (mt12 + mt21) - 4.0f32 * qz * (mt11 + mt00);

    (v_scale, Quat::new(dq_w, dq_x, dq_y, dq_z))
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_xray_bwd_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    v_combined: &Tensor<f32>,
    v_transforms: &mut Tensor<f32>,
    v_raw_opac: &mut Tensor<f32>,
    v_refine_weight: &mut Tensor<f32>,
    u: XRayProjectUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];
    let base = (global_gid * 10u32) as usize;
    let vbase = (compact_gid * XRAY_LANES) as usize;

    // Upstream grads from the rasterize backward.
    let v_xy = Vec3A::new(v_combined[vbase], v_combined[vbase + 1], 0.0f32);
    let v_conic = Sym2 {
        c00: v_combined[vbase + 2],
        c01: v_combined[vbase + 3],
        c11: v_combined[vbase + 4],
    };
    let v_opac = v_combined[vbase + 5];
    let v_mu = v_combined[vbase + 6];

    // Inputs.
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    let scale = read_scale_xray(transforms, base, u.scale_modifier);
    let quat_unorm = read_quat_unorm_xray(transforms, base);
    let quat = quat_unorm.normalize();
    let mean_c = u.world_to_cam(mean);
    let raw_opac = raw_opacities[global_gid as usize];

    // Cov2D cone backward → v_cov3D + v_mean_c_cov.
    let (v_cov3d, v_mean_c_cov) = compute_cov2d_cone_bwd(mean_c, scale, quat, u, v_conic, v_mu);

    // Pixel projection VJP: xy (pixel) → mean_c.
    let inv_z = 1.0f32 / mean_c.z();
    let v_mcx = u.focal_x * inv_z * v_xy.x();
    let v_mcy = u.focal_y * inv_z * v_xy.y();
    let v_mcz = -u.focal_x * mean_c.x() * inv_z * inv_z * v_xy.x()
        - u.focal_y * mean_c.y() * inv_z * inv_z * v_xy.y();
    let v_mean_c = Vec3A::new(
        v_mean_c_cov.x() + v_mcx,
        v_mean_c_cov.y() + v_mcy,
        v_mean_c_cov.z() + v_mcz,
    );
    let v_mean = u.view_rotation().transpose_mul_vec3(v_mean_c);

    // Cov3D backward → v_scale + v_quat (normalized), then normalize VJP.
    // dnormvdv must take the UNNORMALIZED quat (the input to `.normalize()`);
    // using the normalized one drops the 1/||q|| factor.
    let (v_scale, v_quat_norm) = compute_cov3d_bwd(scale, quat, v_cov3d);
    let v_quat = dnormvdv4(quat_unorm, v_quat_norm);

    // Opacity: activated density is `MU_WATER · silu(raw)` (exp6), so
    // d opac/d raw = MU_WATER · silu'(raw), silu'(x) = σ(x) + x·σ(x)(1−σ(x)).
    let sig = sigmoid(raw_opac);
    let silu_deriv = sig + raw_opac * sig * (1.0f32 - sig);
    let v_raw = v_opac * MU_WATER * silu_deriv;

    // Refine weight: viewspace (mean2D) gradient norm per splat, used by the
    // density controller for densification (mirrors the RGB render path's
    // `v_refine_weight`). Clamped to stay finite / non-negative.
    let refine_norm = f32::sqrt(v_xy.x() * v_xy.x() + v_xy.y() * v_xy.y());
    let refine_clean = select(is_finite_f32(refine_norm), refine_norm, 0.0f32);
    v_refine_weight[global_gid as usize] = clamp(refine_clean, 0.0f32, 1.0e32f32);

    // Write dense outputs (global-gid indexed).
    v_transforms[base] = v_mean.x();
    v_transforms[base + 1] = v_mean.y();
    v_transforms[base + 2] = v_mean.z();
    // Quaternion (w,x,y,z).
    v_transforms[base + 3] = v_quat.w();
    v_transforms[base + 4] = v_quat.x();
    v_transforms[base + 5] = v_quat.y();
    v_transforms[base + 6] = v_quat.z();
    // log-scale: dL/dlog = dL/dscale · scale.
    v_transforms[base + 7] = v_scale.x() * scale.x();
    v_transforms[base + 8] = v_scale.y() * scale.y();
    v_transforms[base + 9] = v_scale.z() * scale.z();

    v_raw_opac[global_gid as usize] = v_raw;
}
