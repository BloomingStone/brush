//! Multi-resolution hash-grid encoding (Instant-NGP / tiny-cuda-nn style),
//! implemented with differentiable burn tensor ops (floor / gather / lerp).
//!
//! Positions are expected normalized to `[0, 1]` (the caller normalizes world
//! coordinates by the scene scale before querying). Each level has its own
//! resolution `base * (max/base)^(level/(n_levels-1))`; the level's 3D lattice
//! is hashed into a `2^log2_hashmap_size` table and looked up with trilinear
//! interpolation. All ops are differentiable, so the table is trainable.

use burn::module::{Module, Param, ParamId};
use burn::tensor::{Device, Int, Tensor, TensorData};
use rand::{RngExt, SeedableRng};

/// Configuration for the hash-grid encoding (defaults match the Python
/// `HashGridDefromModel`: `x_multires` levels of `n_features_per_level`
/// features, `log2_hashmap_size` table).
#[derive(Debug, Clone)]
pub struct HashGridConfig {
    /// Number of resolution levels.
    pub n_levels: u32,
    /// Features per level.
    pub n_features_per_level: u32,
    /// Log2 of the hash table size.
    pub log2_hashmap_size: u32,
    /// Base (coarsest) resolution.
    pub base_resolution: u32,
    /// Max (finest) resolution.
    pub max_resolution: u32,
    /// Init range for the table values (uniform `[-init_scale, init_scale]`).
    pub init_scale: f32,
    /// RNG seed for the table initialization.
    pub seed: u64,
}

impl Default for HashGridConfig {
    fn default() -> Self {
        Self {
            n_levels: 7,
            n_features_per_level: 4,
            log2_hashmap_size: 7,
            base_resolution: 16,
            max_resolution: 128,
            init_scale: 0.1,
            seed: 0,
        }
    }
}

impl HashGridConfig {
    /// Total output feature dimension (`n_levels * n_features_per_level`).
    pub fn output_channels(&self) -> u32 {
        self.n_levels * self.n_features_per_level
    }
}

/// Multi-resolution hash-grid encoding module. A single trainable table of
/// shape `[n_levels * table_size, n_features_per_level]`.
#[derive(Module, Debug)]
pub struct HashGrid {
    params: Param<Tensor<2>>,
    #[module(skip)]
    cfg: HashGridConfig,
    table_size: u32,
}

impl HashGrid {
    pub fn new(cfg: HashGridConfig, device: &Device) -> Self {
        let table_size = 1u32 << cfg.log2_hashmap_size;
        let rows = cfg.n_levels * table_size;
        let cols = cfg.n_features_per_level;
        let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
        let values: Vec<f32> = (0..rows as usize * cols as usize)
            .map(|_| rng.random_range(-cfg.init_scale..cfg.init_scale))
            .collect();
        let t = Tensor::<2>::from_data(
            TensorData::new(values, [rows as usize, cols as usize]),
            device,
        )
        .detach()
        .require_grad();
        Self {
            params: Param::initialized(ParamId::new(), t),
            cfg,
            table_size,
        }
    }

    pub fn output_channels(&self) -> u32 {
        self.cfg.output_channels()
    }

    /// Resolution for a level.
    fn resolution_for_level(&self, level: u32) -> f32 {
        if self.cfg.n_levels == 1 {
            return self.cfg.max_resolution as f32;
        }
        let t = level as f32 / (self.cfg.n_levels - 1) as f32;
        let base = self.cfg.base_resolution as f32;
        let max = self.cfg.max_resolution as f32;
        base * (max / base).powf(t)
    }

    /// Encode `xyz` (`[N, 3]`, normalized to `[0, 1]`) into
    /// `[N, n_levels * n_features_per_level]`.
    pub fn forward(&self, xyz: Tensor<2>) -> Tensor<2> {
        let table = self.params.val();
        let mut out: Vec<Tensor<2>> = Vec::with_capacity(self.cfg.n_levels as usize);

        for level in 0..self.cfg.n_levels {
            let res = self.resolution_for_level(level);
            let scaled = xyz.clone() * (res - 1.0); // [N, 3] in [0, res-1]
            let lo = scaled.clone().floor().int(); // [N, 3] Int
            let frac = scaled - lo.clone().float(); // [N, 3] in [0, 1)

            // 8 trilinear corners.
            let mut acc: Option<Tensor<2>> = None;
            for dz in 0..2i32 {
                for dy in 0..2i32 {
                    for dx in 0..2i32 {
                        let corner = lo.clone() + Tensor::<2, Int>::from_data(
                            TensorData::new(vec![dx, dy, dz], [1, 3]),
                            &xyz.device(),
                        );
                        let row = self.hash_level(level, corner); // [N]
                        let feat = table.clone().select(0, row); // [N, feats]
                        let w = trilinear_weight(frac.clone(), dx, dy, dz);
                        let term = feat * w;
                        acc = Some(match acc {
                            None => term,
                            Some(a) => a + term,
                        });
                    }
                }
            }
            out.push(acc.expect("8 corners"));
        }

        Tensor::cat(out, 1)
    }

    /// Flat table row for a level's integer corner coordinates `[N, 3]`.
    fn hash_level(&self, level: u32, corner: Tensor<2, Int>) -> Tensor<1, Int> {
        let n = corner.dims()[0];
        let x = corner.clone().slice([0..n, 0..1]).squeeze_dim::<1>(1);
        let y = corner.clone().slice([0..n, 1..2]).squeeze_dim::<1>(1);
        let z = corner.slice([0..n, 2..3]).squeeze_dim::<1>(1);
        // Coords are non-negative here, so plain `%` is fine.
        let h = x.mul_scalar(73856093i64)
            + y.mul_scalar(19349663i64)
            + z.mul_scalar(83492791i64);
        let h = h % self.table_size as i64;
        h + (level * self.table_size) as i64
    }
}

/// Trilinear weight for corner `(dx, dy, dz)`: product over axes of
/// `frac` (when the corner is `+1` on that axis) or `1 - frac` otherwise.
fn trilinear_weight(frac: Tensor<2>, dx: i32, dy: i32, dz: i32) -> Tensor<2> {
    let n = frac.dims()[0];
    let one = Tensor::<2>::ones_like(&frac);

    let fx = frac.clone().slice([0..n, 0..1]);
    let fy = frac.clone().slice([0..n, 1..2]);
    let fz = frac.slice([0..n, 2..3]);

    let wx = if dx == 1 { fx } else { one.clone().slice([0..n, 0..1]) - fx };
    let wy = if dy == 1 { fy } else { one.clone().slice([0..n, 1..2]) - fy };
    let wz = if dz == 1 { fz } else { one.slice([0..n, 2..3]) - fz };

    wx * wy * wz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_channels_matches_config() {
        let cfg = HashGridConfig::default();
        assert_eq!(cfg.output_channels(), 7 * 4);
    }
}
