//! Fused HexPlane query kernels (Stage 2): the six-plane bilinear gather +
//! sum collapses to ONE forward kernel and ONE backward kernel (atomic
//! scatter to the plane grids + xyz VJP), replacing ~170 tiny burn-op
//! kernel launches per step.
//!
//! Matches the pure-burn reference (`hex_plane::HexPlane::forward_pure`)
//! numerically and in autodiff semantics: `floor` carries zero gradient
//! (burn's `float_floor` backward returns zeros), so the xyz gradient flows
//! only through the interpolation fraction, and the coordinate clamp
//! zeroes the xyz gradient for out-of-range points.

use super::atomic::AtomicAddF32;
use crate::hex_plane::HexPlaneConfig;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Kernel uniforms: plane dims + normalization. Also the host mirror saved
/// in the autodiff state for the backward pass.
#[derive(CubeLaunch, CubeType, Debug, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct HexPlaneKernelUniforms {
    /// Number of splats.
    pub n: u32,
    /// Spatial plane resolution.
    pub rs: u32,
    /// Time axis resolution.
    pub rt: u32,
    /// Feature channels per plane cell.
    pub c: u32,
    pub coord_scale: f32,
    pub phase_min: f32,
    pub phase_max: f32,
}

impl HexPlaneKernelUniforms {
    pub fn from_config(cfg: &HexPlaneConfig, n: u32) -> Self {
        Self {
            n,
            rs: cfg.spatial_resolution,
            rt: cfg.time_resolution,
            c: cfg.n_feature_dim as u32,
            coord_scale: cfg.coord_scale,
            phase_min: cfg.phase_min,
            phase_max: cfg.phase_max,
        }
    }
}

/// Per-axis interpolation indices `(lo, hi, frac)` for a clamped (spatial)
/// or circular (time) axis of resolution `res`. `coord` is in `[0, 1]`.
#[cube]
fn axis_indices(coord: f32, res: u32, wrap: bool) -> (u32, u32, f32) {
    let scaled = if wrap {
        coord * res as f32
    } else {
        coord * (res as f32 - 1.0f32)
    };
    let lo_f = f32::floor(scaled);
    let frac = scaled - lo_f;
    let lo = if wrap {
        (lo_f as u32) % res
    } else {
        lo_f as u32
    };
    let hi = if wrap {
        (lo + 1u32) % res
    } else {
        select(lo + 1u32 >= res, res - 1u32, lo + 1u32)
    };
    (lo, hi, frac)
}

/// Bilinear query of one plane (flattened `[h_axis, w_axis, C]`, row-major)
/// at normalized coords `(u, v)`, feature channel `c`.
#[cube]
fn query_plane(
    plane: &Tensor<f32>,
    u: f32,
    v: f32,
    h_axis: u32,
    w_axis: u32,
    wrap_v: bool,
    c: u32,
    u_cfg: &HexPlaneKernelUniforms,
) -> f32 {
    let (u0, u1, fu) = axis_indices(u, h_axis, false);
    let (v0, v1, fv) = axis_indices(v, w_axis, wrap_v);

    let f00 = plane[((u0 * w_axis + v0) * u_cfg.c + c) as usize];
    let f01 = plane[((u0 * w_axis + v1) * u_cfg.c + c) as usize];
    let f10 = plane[((u1 * w_axis + v0) * u_cfg.c + c) as usize];
    let f11 = plane[((u1 * w_axis + v1) * u_cfg.c + c) as usize];

    let w00 = (1.0f32 - fu) * (1.0f32 - fv);
    let w01 = (1.0f32 - fu) * fv;
    let w10 = fu * (1.0f32 - fv);
    let w11 = fu * fv;

    f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11
}

/// One thread per `(splat, feature)` pair: compute the six plane queries
/// and sum them into `out[i, c]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn hex_plane_query_fwd_kernel(
    xyz: &Tensor<f32>,
    phase: &Tensor<f32>,
    xy: &Tensor<f32>,
    xz: &Tensor<f32>,
    yz: &Tensor<f32>,
    xt: &Tensor<f32>,
    yt: &Tensor<f32>,
    zt: &Tensor<f32>,
    out: &mut Tensor<f32>,
    u_cfg: HexPlaneKernelUniforms,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= u_cfg.n * u_cfg.c {
        terminate!();
    }
    let i = gid / u_cfg.c;
    let c = gid % u_cfg.c;
    let base = (i * 3u32) as usize;

    let x = clamp((xyz[base] / u_cfg.coord_scale + 1.0f32) * 0.5f32, 0.0f32, 1.0f32);
    let y = clamp(
        (xyz[base + 1] / u_cfg.coord_scale + 1.0f32) * 0.5f32,
        0.0f32,
        1.0f32,
    );
    let z = clamp(
        (xyz[base + 2] / u_cfg.coord_scale + 1.0f32) * 0.5f32,
        0.0f32,
        1.0f32,
    );
    let t = clamp(
        (phase[i as usize] - u_cfg.phase_min) / (u_cfg.phase_max - u_cfg.phase_min),
        0.0f32,
        1.0f32,
    );

    let f_xy = query_plane(xy, x, y, u_cfg.rs, u_cfg.rs, false, c, &u_cfg);
    let f_xz = query_plane(xz, x, z, u_cfg.rs, u_cfg.rs, false, c, &u_cfg);
    let f_yz = query_plane(yz, y, z, u_cfg.rs, u_cfg.rs, false, c, &u_cfg);
    let f_xt = query_plane(xt, x, t, u_cfg.rs, u_cfg.rt, true, c, &u_cfg);
    let f_yt = query_plane(yt, y, t, u_cfg.rs, u_cfg.rt, true, c, &u_cfg);
    let f_zt = query_plane(zt, z, t, u_cfg.rs, u_cfg.rt, true, c, &u_cfg);

    out[gid as usize] = f_xy + f_xz + f_yz + f_xt + f_yt + f_zt;
}

/// One thread per `(splat, feature)` pair: scatter `g = v_feat[i, c]` to the
/// four bilinear corners of each of the six planes (atomic adds) and
/// accumulate the xyz VJP through the interpolation fractions. Returns
/// nothing — gradients accumulate directly into the `v_*` buffers.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn hex_plane_query_bwd_kernel<A: AtomicAddF32>(
    xyz: &Tensor<f32>,
    phase: &Tensor<f32>,
    v_feat: &Tensor<f32>,
    xy: &Tensor<f32>,
    xz: &Tensor<f32>,
    yz: &Tensor<f32>,
    xt: &Tensor<f32>,
    yt: &Tensor<f32>,
    zt: &Tensor<f32>,
    v_xyz: &mut Tensor<Atomic<A::Storage>>,
    v_xy: &mut Tensor<Atomic<A::Storage>>,
    v_xz: &mut Tensor<Atomic<A::Storage>>,
    v_yz: &mut Tensor<Atomic<A::Storage>>,
    v_xt: &mut Tensor<Atomic<A::Storage>>,
    v_yt: &mut Tensor<Atomic<A::Storage>>,
    v_zt: &mut Tensor<Atomic<A::Storage>>,
    u_cfg: HexPlaneKernelUniforms,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= u_cfg.n * u_cfg.c {
        terminate!();
    }
    let i = gid / u_cfg.c;
    let c = gid % u_cfg.c;
    let g = v_feat[gid as usize];
    let base = (i * 3u32) as usize;

    // Raw (pre-clamp) normalized coords for the clamp mask.
    let raw_x = (xyz[base] / u_cfg.coord_scale + 1.0f32) * 0.5f32;
    let raw_y = (xyz[base + 1] / u_cfg.coord_scale + 1.0f32) * 0.5f32;
    let raw_z = (xyz[base + 2] / u_cfg.coord_scale + 1.0f32) * 0.5f32;
    let x = clamp(raw_x, 0.0f32, 1.0f32);
    let y = clamp(raw_y, 0.0f32, 1.0f32);
    let z = clamp(raw_z, 0.0f32, 1.0f32);
    let t = clamp(
        (phase[i as usize] - u_cfg.phase_min) / (u_cfg.phase_max - u_cfg.phase_min),
        0.0f32,
        1.0f32,
    );
    // Clamp backward masks out-of-range axes (matches burn's clamp).
    // (cubecl can't lower `..=` ranges, so keep the explicit comparisons.)
    #[allow(clippy::manual_range_contains)]
    let mx = select(raw_x < 0.0f32 || raw_x > 1.0f32, 0.0f32, 1.0f32);
    #[allow(clippy::manual_range_contains)]
    let my = select(raw_y < 0.0f32 || raw_y > 1.0f32, 0.0f32, 1.0f32);
    #[allow(clippy::manual_range_contains)]
    let mz = select(raw_z < 0.0f32 || raw_z > 1.0f32, 0.0f32, 1.0f32);

    let (g_xy_u, g_xy_v) = scatter_plane_grads::<A>(
        v_xy, xy, x, y, u_cfg.rs, u_cfg.rs, false, c, g, &u_cfg,
    );
    let (g_xz_u, g_xz_v) = scatter_plane_grads::<A>(
        v_xz, xz, x, z, u_cfg.rs, u_cfg.rs, false, c, g, &u_cfg,
    );
    let (g_yz_u, g_yz_v) = scatter_plane_grads::<A>(
        v_yz, yz, y, z, u_cfg.rs, u_cfg.rs, false, c, g, &u_cfg,
    );
    let (g_xt_u, _g_xt_v) = scatter_plane_grads::<A>(
        v_xt, xt, x, t, u_cfg.rs, u_cfg.rt, true, c, g, &u_cfg,
    );
    let (g_yt_u, _g_yt_v) = scatter_plane_grads::<A>(
        v_yt, yt, y, t, u_cfg.rs, u_cfg.rt, true, c, g, &u_cfg,
    );
    let (g_zt_u, _g_zt_v) = scatter_plane_grads::<A>(
        v_zt, zt, z, t, u_cfg.rs, u_cfg.rt, true, c, g, &u_cfg,
    );

    // xyz VJP: d(feat)/d(frac) × axis scale (floor carries zero gradient,
    // so the fraction derivative is exactly the axis scale) ×
    // d(coord)/d(world) = 0.5 / coord_scale, masked by the clamp, all
    // scaled by the upstream gradient `g` for this (splat, feature) thread.
    let scale_s = (u_cfg.rs - 1u32) as f32;
    let s = 0.5f32 / u_cfg.coord_scale;
    let vx = (g_xy_u + g_xz_u + g_xt_u) * g * scale_s * s * mx;
    let vy = (g_xy_v + g_yz_u + g_yt_u) * g * scale_s * s * my;
    let vz = (g_xz_v + g_yz_v + g_zt_u) * g * scale_s * s * mz;
    A::add(&v_xyz[base], vx);
    A::add(&v_xyz[base + 1], vy);
    A::add(&v_xyz[base + 2], vz);
}

/// Scatter `g` to the four bilinear corners of one plane (atomic adds) and
/// return `(d_feat/d_frac_u, d_feat/d_frac_v)` for the xyz VJP.
#[cube]
fn scatter_plane_grads<A: AtomicAddF32>(
    v_plane: &mut Tensor<Atomic<A::Storage>>,
    plane: &Tensor<f32>,
    u: f32,
    v: f32,
    h_axis: u32,
    w_axis: u32,
    wrap_v: bool,
    c: u32,
    g: f32,
    u_cfg: &HexPlaneKernelUniforms,
) -> (f32, f32) {
    let (u0, u1, fu) = axis_indices(u, h_axis, false);
    let (v0, v1, fv) = axis_indices(v, w_axis, wrap_v);

    let row00 = ((u0 * w_axis + v0) * u_cfg.c + c) as usize;
    let row01 = ((u0 * w_axis + v1) * u_cfg.c + c) as usize;
    let row10 = ((u1 * w_axis + v0) * u_cfg.c + c) as usize;
    let row11 = ((u1 * w_axis + v1) * u_cfg.c + c) as usize;

    let f00 = plane[row00];
    let f01 = plane[row01];
    let f10 = plane[row10];
    let f11 = plane[row11];

    let w00 = (1.0f32 - fu) * (1.0f32 - fv);
    let w01 = (1.0f32 - fu) * fv;
    let w10 = fu * (1.0f32 - fv);
    let w11 = fu * fv;

    A::add(&v_plane[row00], g * w00);
    A::add(&v_plane[row01], g * w01);
    A::add(&v_plane[row10], g * w10);
    A::add(&v_plane[row11], g * w11);

    // d(feat)/d(frac_u) = (1-fv)(f10 - f00) + fv(f11 - f01)
    let d_fu = (1.0f32 - fv) * (f10 - f00) + fv * (f11 - f01);
    // d(feat)/d(frac_v) = (1-fu)(f01 - f00) + fu(f11 - f10)
    let d_fv = (1.0f32 - fu) * (f01 - f00) + fu * (f11 - f10);
    (d_fu, d_fv)
}
