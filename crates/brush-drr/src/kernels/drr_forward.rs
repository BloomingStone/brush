//! DRR forward: cone-beam line integral through the volume.
//!
//! One thread per output pixel. Each ray marches `steps` samples from
//! `sod - half_r` to `sod + half_r`, trilinearly samples the volume
//! (`[-half_r, half_r]^3`, world coords) and accumulates `∫μ dl`.
//! Output `proj = scale * integral + bias` (calibrated optical depth).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{Mat3, Vec3A};

use super::types::DrrUniforms;

pub const WG_SIZE: u32 = 256;

/// Trilinear sample of the `[vol,vol,vol]` volume at world point `p`.
/// Memory layout matches the FDK volume (see fdk_volume.rs):
/// `idx(x,y,z) = (y*vol + z)*vol + x`. Outside the grid → 0.
#[cube]
pub fn drr_trilinear(volume: &Tensor<f32>, vol: u32, half_r: f32, inv_delta: f32, p: Vec3A) -> f32 {
    let vx = (p.x() + half_r) * inv_delta - 0.5;
    let vy = (p.y() + half_r) * inv_delta - 0.5;
    let vz = (p.z() + half_r) * inv_delta - 0.5;
    let mut out = 0.0f32;
    if vx >= 0.0 && vy >= 0.0 && vz >= 0.0 {
        let ix = vx as u32;
        let iy = vy as u32;
        let iz = vz as u32;
        if ix + 1 < vol && iy + 1 < vol && iz + 1 < vol {
            let tx = vx - ix as f32;
            let ty = vy - iy as f32;
            let tz = vz - iz as f32;

            let base = (iy * vol + iz) * vol + ix;
            let sx = vol * vol; // stride per y
            let sz = vol; // stride per z

            let c000 = volume[base as usize];
            let c100 = volume[(base + 1) as usize];
            let c010 = volume[(base + sz) as usize];
            let c110 = volume[(base + sz + 1) as usize];
            let c001 = volume[(base + sx) as usize];
            let c101 = volume[(base + sx + 1) as usize];
            let c011 = volume[(base + sx + sz) as usize];
            let c111 = volume[(base + sx + sz + 1) as usize];

            let c00 = c000 * (1.0 - tx) + c100 * tx;
            let c10 = c010 * (1.0 - tx) + c110 * tx;
            let c01 = c001 * (1.0 - tx) + c101 * tx;
            let c11 = c011 * (1.0 - tx) + c111 * tx;
            let c0 = c00 * (1.0 - ty) + c10 * ty;
            let c1 = c01 * (1.0 - ty) + c11 * ty;
            out = c0 * (1.0 - tz) + c1 * tz;
        }
    }
    out
}

#[cube(launch)]
pub fn drr_forward_kernel(
    volume: &Tensor<f32>,
    out_proj: &mut Tensor<f32>,
    u: DrrUniforms,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= u.img_w * u.img_h {
        terminate!();
    }
    let x = gid % u.img_w;
    let y = gid / u.img_w;

    let cam_rot = Mat3::from_cols(
        Vec3A::new(u.rot_c0_x, u.rot_c0_y, u.rot_c0_z),
        Vec3A::new(u.rot_c1_x, u.rot_c1_y, u.rot_c1_z),
        Vec3A::new(u.rot_c2_x, u.rot_c2_y, u.rot_c2_z),
    );
    let cam_pos = Vec3A::new(u.cam_x, u.cam_y, u.cam_z);
    let dir = Vec3A::new((x as f32 - u.cx) / u.fx, (y as f32 - u.cy) / u.fy, 1.0);

    let t_near = u.sod - u.half_r;
    let t_far = u.sod + u.half_r;
    let dt = (t_far - t_near) / u.steps as f32;

    let mut acc = 0.0f32;
    for s in 0..u.steps {
        let t = t_near + (s as f32 + 0.5) * dt;
        let p_local = dir.scale(t);
        let p_world = cam_rot.mul_vec3(p_local).add(cam_pos);
        acc += drr_trilinear(volume, u.vol, u.half_r, u.inv_delta, p_world) * dt;
    }

    let idx = (y * u.img_w + x) as usize;
    out_proj[idx] = acc * u.scale + u.bias;
}
