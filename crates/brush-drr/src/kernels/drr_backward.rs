//! DRR backward: scatter the per-pixel projection gradient back along each
//! ray into the volume (trilinear weights, atomic add).
//!
//! `proj = scale * ∫ μ dl + bias` → `dL/dμ_voxel = v_proj[x,y] * scale * dt * w`
//! where `w` are the trilinear interpolation weights of the sample.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;
use brush_cube::{Mat3, Vec3A};

use super::atomic::AtomicAddF32;
use super::types::DrrUniforms;

pub const WG_SIZE: u32 = 256;

#[cube(launch)]
pub fn drr_backward_kernel<A: AtomicAddF32>(
    v_proj: &Tensor<f32>,
    v_volume: &mut Tensor<Atomic<A::Storage>>,
    u: DrrUniforms,
) {
    let gid = ABSOLUTE_POS as u32;
    if gid >= u.img_w * u.img_h {
        terminate!();
    }
    let x = gid % u.img_w;
    let y = gid / u.img_w;
    let vp = v_proj[(y * u.img_w + x) as usize];

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

    for s in 0..u.steps {
        let t = t_near + (s as f32 + 0.5) * dt;
        let p_local = dir.scale(t);
        let p_world = cam_rot.mul_vec3(p_local).add(cam_pos);

        let vx = (p_world.x() + u.half_r) * u.inv_delta - 0.5;
        let vy = (p_world.y() + u.half_r) * u.inv_delta - 0.5;
        let vz = (p_world.z() + u.half_r) * u.inv_delta - 0.5;
        if vx >= 0.0 && vy >= 0.0 && vz >= 0.0 {
            let ix = vx as u32;
            let iy = vy as u32;
            let iz = vz as u32;
            if ix + 1 < u.vol && iy + 1 < u.vol && iz + 1 < u.vol {
                let tx = vx - ix as f32;
                let ty = vy - iy as f32;
                let tz = vz - iz as f32;
                let wx0 = 1.0 - tx;
                let wx1 = tx;
                let wy0 = 1.0 - ty;
                let wy1 = ty;
                let wz0 = 1.0 - tz;
                let wz1 = tz;

                let base = (iy * u.vol + iz) * u.vol + ix;
                let sx = u.vol * u.vol;
                let sz = u.vol;
                let val = vp * dt * u.scale;

                A::add(&v_volume[base as usize], val * wx0 * wy0 * wz0);
                A::add(&v_volume[(base + 1) as usize], val * wx1 * wy0 * wz0);
                A::add(&v_volume[(base + sz) as usize], val * wx0 * wy1 * wz0);
                A::add(&v_volume[(base + sz + 1) as usize], val * wx1 * wy1 * wz0);
                A::add(&v_volume[(base + sx) as usize], val * wx0 * wy0 * wz1);
                A::add(&v_volume[(base + sx + 1) as usize], val * wx1 * wy0 * wz1);
                A::add(&v_volume[(base + sx + sz) as usize], val * wx0 * wy1 * wz1);
                A::add(
                    &v_volume[(base + sx + sz + 1) as usize],
                    val * wx1 * wy1 * wz1,
                );
            }
        }
    }
}
