//! Backward of the voxelizer preprocess: 3D-conic inverse backward
//! (`dL/dcov` through `cov_voxel = Mᵀ·Vrk·M`), the covariance → scale /
//! quaternion backward, the sigmoid opacity VJP, and the (already
//! dVoxel-compensated) mean grad. Mirrors R2 `computeCov3DCUDA` +
//! `preprocessCUDA` backward (voxelizer).

#![allow(non_snake_case)]

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{Mat3, Quat, Vec3A, compute_cov3d, dnormvdv4, sigmoid};

use super::helpers::{read_quat, read_scale_mod, read_scale_raw};
use super::render_bwd::BWD_LANES;
use super::types::VoxelUniforms;

pub const WG_SIZE: u32 = 256;

/// Covariance → scale / (raw) quaternion backward, transcribed from R2's
/// `computeCov3D` (voxelizer variant — no quaternion-normalization VJP).
/// Brush forward is `Sigma = R_std·S²·R_stdᵀ`; R2's R is the transpose of
/// the standard matrix, so here `R_r2 = rᵀ`, `M = S·R_r2`. Empirically
/// verified against R2 `_C` for anisotropic scales + non-identity quats.
#[cube]
fn compute_cov3d_bwd(scale: Vec3A, quat: Quat, v_cov3d: brush_cube::Sym3) -> (Vec3A, Quat) {
    let r = quat.to_mat3();
    // msr = R_std·S — each column of R scaled by the matching scalar.
    let msr = Mat3::from_cols(
        r.col0().scale(scale.x()),
        r.col1().scale(scale.y()),
        r.col2().scale(scale.z()),
    );
    // R2's M = S·R_r2 = (R_std·S)ᵀ = msrᵀ.
    let m_r2 = Mat3::from_cols(msr.row0(), msr.row1(), msr.row2());
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

    // v_scale = (R_r2.row0·dL_dM.row0, ...) = (r.col0·dL_dM.row0, ...).
    let v_scale = Vec3A::new(
        r.col0().dot(dl_dm.row0()),
        r.col1().dot(dl_dm.row1()),
        r.col2().dot(dl_dm.row2()),
    );

    // mt = dL_dM with ROWS scaled by s (R2's effective dL_dMt, NOT transpose).
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
pub fn preprocess_voxel_bwd_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    v_combined: &Tensor<f32>,
    v_transforms: &mut Tensor<f32>,
    v_raw_opac: &mut Tensor<f32>,
    u: VoxelUniforms,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];
    let base = (global_gid * 10u32) as usize;
    let vbase = (compact_gid * BWD_LANES) as usize;

    let v_mean = Vec3A::new(v_combined[vbase], v_combined[vbase + 1], v_combined[vbase + 2]);
    let v_conic_a = v_combined[vbase + 3];
    let v_conic_b = v_combined[vbase + 4];
    let v_conic_c = v_combined[vbase + 5];
    let v_conic_d = v_combined[vbase + 6];
    let v_conic_e = v_combined[vbase + 7];
    let v_conic_f = v_combined[vbase + 8];
    let v_opac = v_combined[vbase + 9];

    // Recompute the voxel-space covariance. The voxelizer normalizes the
    // quaternion in the forward, so the recompute (and the quat VJP below)
    // must use the normalized quat too. dnormvdv takes the UNNORMALIZED
    // quat (the input to `.normalize()`); using the normalized one drops
    // the 1/||q|| factor.
    let scale = read_scale_mod(transforms, base, u.scale_modifier);
    let scale_raw = read_scale_raw(transforms, base);
    let quat_unorm = read_quat(transforms, base);
    let quat = quat_unorm.normalize();
    let vrk = compute_cov3d(scale, quat);

    let ix = u.inv_d_voxel_x;
    let iy = u.inv_d_voxel_y;
    let iz = u.inv_d_voxel_z;
    let hata = vrk.c00 * ix * ix;
    let hatb = vrk.c01 * ix * iy;
    let hatc = vrk.c02 * ix * iz;
    let hatd = vrk.c11 * iy * iy;
    let hate = vrk.c12 * iy * iz;
    let hatf = vrk.c22 * iz * iz;

    let denom = hata * hatd * hatf + 2.0f32 * hatb * hatc * hate - hata * hate * hate
        - hatf * hatb * hatb - hatd * hatc * hatc;
    let denom2inv = 1.0f32 / (denom * denom + 1.0e-7f32);

    // 3D conic inverse backward (R2 `computeCov3DCUDA`).
    let denom_da = hatd * hatf - hate * hate;
    let denom_db = 2.0f32 * hatc * hate - 2.0f32 * hatf * hatb;
    let denom_dc = 2.0f32 * hatb * hate - 2.0f32 * hatd * hatc;
    let denom_dd = hata * hatf - hatc * hatc;
    let denom_de = 2.0f32 * hatb * hatc - 2.0f32 * hata * hate;
    let denom_df = hata * hatd - hatb * hatb;

    let ce_bf = hatc * hate - hatb * hatf;
    let be_cd = hatb * hate - hatc * hatd;
    let bc_ae = hatb * hatc - hata * hate;

    let dL_dhata = denom2inv
        * (-denom_da * denom_da * v_conic_a - ce_bf * denom_da * v_conic_b
            - be_cd * denom_da * v_conic_c
            + (hatf * denom - denom_dd * denom_da) * v_conic_d
            + (-hate * denom - bc_ae * denom_da) * v_conic_e
            + (hatd * denom - denom_df * denom_da) * v_conic_f);
    let dL_dhatb = denom2inv
        * (-denom_da * denom_db * v_conic_a
            + (-hatf * denom - ce_bf * denom_db) * v_conic_b
            + (hate * denom - be_cd * denom_db) * v_conic_c
            - denom_dd * denom_db * v_conic_d
            + (hatc * denom - bc_ae * denom_db) * v_conic_e
            + (-2.0f32 * hatb * denom - denom_df * denom_db) * v_conic_f);
    let dL_dhatc = denom2inv
        * (-denom_da * denom_dc * v_conic_a
            + (hate * denom - ce_bf * denom_dc) * v_conic_b
            + (-hatd * denom - be_cd * denom_dc) * v_conic_c
            + (-2.0f32 * hatc * denom - denom_dd * denom_dc) * v_conic_d
            + (hatb * denom - bc_ae * denom_dc) * v_conic_e
            - denom_df * denom_dc * v_conic_f);
    let dL_dhatd = denom2inv
        * ((hatf * denom - denom_da * denom_dd) * v_conic_a - ce_bf * denom_dd * v_conic_b
            + (-hatc * denom - be_cd * denom_dd) * v_conic_c
            - denom_dd * denom_dd * v_conic_d
            - bc_ae * denom_dd * v_conic_e
            + (hata * denom - denom_df * denom_dd) * v_conic_f);
    let dL_dhate = denom2inv
        * ((-2.0f32 * hate * denom - denom_da * denom_de) * v_conic_a
            + (hatc * denom - ce_bf * denom_de) * v_conic_b
            + (hatb * denom - be_cd * denom_de) * v_conic_c
            - denom_dd * denom_de * v_conic_d
            + (-hata * denom - bc_ae * denom_de) * v_conic_e
            - denom_df * denom_de * v_conic_f);
    let dL_dhatf = denom2inv
        * ((hatd * denom - denom_da * denom_df) * v_conic_a
            + (-hatb * denom - ce_bf * denom_df) * v_conic_b
            - be_cd * denom_df * v_conic_c
            + (hata * denom - denom_dd * denom_df) * v_conic_d
            - bc_ae * denom_df * v_conic_e
            - denom_df * denom_df * v_conic_f);

    // dL/dVrk through cov_voxel = Mᵀ·Vrk·M with M = diag(1/dVoxel).
    let v_cov00 = ix * ix * dL_dhata;
    let v_cov01 = ix * iy * dL_dhatb;
    let v_cov02 = ix * iz * dL_dhatc;
    let v_cov11 = iy * iy * dL_dhatd;
    let v_cov12 = iy * iz * dL_dhate;
    let v_cov22 = iz * iz * dL_dhatf;
    let v_cov3d = brush_cube::Sym3 {
        c00: v_cov00,
        c01: v_cov01,
        c02: v_cov02,
        c11: v_cov11,
        c12: v_cov12,
        c22: v_cov22,
    };

    // Covariance → scale / quat, then quaternion-normalize VJP.
    let (v_scale, v_quat_raw) = compute_cov3d_bwd(scale, quat, v_cov3d);
    let v_quat = dnormvdv4(quat_unorm, v_quat_raw);

    // Mean: R2's `dL_dmeans = dL_dmean3D_norm` (already dVoxel-compensated).
    let v_mean_world = v_mean;

    // Opacity: sigmoid → logit.
    let opac = sigmoid(raw_opacities[global_gid as usize]);
    let v_raw = v_opac * opac * (1.0f32 - opac);

    // log-scale: dL/dlog = dL/dscale · scale.
    let v_log = Vec3A::new(v_scale.x() * scale_raw.x(), v_scale.y() * scale_raw.y(), v_scale.z() * scale_raw.z());

    // Write dense outputs (global-gid indexed), interleaved [N,10].
    v_transforms[base] = v_mean_world.x();
    v_transforms[base + 1] = v_mean_world.y();
    v_transforms[base + 2] = v_mean_world.z();
    v_transforms[base + 3] = v_quat.w();
    v_transforms[base + 4] = v_quat.x();
    v_transforms[base + 5] = v_quat.y();
    v_transforms[base + 6] = v_quat.z();
    v_transforms[base + 7] = v_log.x();
    v_transforms[base + 8] = v_log.y();
    v_transforms[base + 9] = v_log.z();

    v_raw_opac[global_gid as usize] = v_raw;
}
