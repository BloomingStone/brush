//! f32-atomic-add abstraction (mirrors `brush-render-bwd` / `brush-voxel`).

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

#[cube]
pub trait AtomicAddF32: Send + Sync + 'static {
    type Storage: Numeric;
    fn add(target: &Atomic<Self::Storage>, val: f32);
}

#[derive(CubeType)]
pub struct HfAtomicAdd;

#[derive(CubeType)]
pub struct CasAtomicAdd;

#[cube]
impl AtomicAddF32 for HfAtomicAdd {
    type Storage = f32;
    fn add(target: &Atomic<f32>, val: f32) {
        Atomic::fetch_add(target, val);
    }
}

#[cube]
impl AtomicAddF32 for CasAtomicAdd {
    type Storage = u32;
    fn add(target: &Atomic<u32>, val: f32) {
        let mut old_value = Atomic::load(target);
        let mut done = false;
        while !done {
            let new_bits = u32::reinterpret(f32::reinterpret(old_value) + val);
            let actual = Atomic::compare_exchange_weak(target, old_value, new_bits);
            if actual == old_value {
                done = true;
            } else {
                old_value = actual;
            }
        }
    }
}
