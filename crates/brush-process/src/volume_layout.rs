//! Volume flat-layout helpers.
//!
//! Two flat conventions coexist in this repo:
//! - **x-major** (internal, R2-Gaussian `fields` / brush-voxel / DRR):
//!   `flat(x,y,z) = x*(ny*nz) + y*nz + z` (x slowest, z fastest).
//! - **disk standard** (NIfTI-1 / NRRD column-major):
//!   `flat(x,y,z) = x + y*nx + z*nx*ny` (x fastest).
//!
//! [`to_xmajor`] converts a disk-standard buffer to the internal layout
//! (nifti-rs returns the raw disk buffer on read); [`from_xmajor`] is the
//! inverse (needed for the manual NRRD writer; nifti-rs's writer performs
//! the transpose itself when given an `(nx, ny, nz)` ndarray).

/// Convert a disk-standard (x-fastest, column-major) flat volume into the
/// internal x-major layout.
pub fn to_xmajor(vol: &[f32], nx: usize, ny: usize, nz: usize) -> Vec<f32> {
    assert_eq!(vol.len(), nx * ny * nz, "to_xmajor size mismatch");
    let mut out = vec![0.0f32; vol.len()];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                // disk: x + y*nx + z*nx*ny
                let src = ix + iy * nx + iz * nx * ny;
                // x-major: x*(ny*nz) + y*nz + z
                let dst = ix * (ny * nz) + iy * nz + iz;
                out[dst] = vol[src];
            }
        }
    }
    out
}

/// Convert an internal x-major flat volume into the disk-standard
/// (x-fastest, column-major) layout.
pub fn from_xmajor(vol: &[f32], nx: usize, ny: usize, nz: usize) -> Vec<f32> {
    assert_eq!(vol.len(), nx * ny * nz, "from_xmajor size mismatch");
    let mut out = vec![0.0f32; vol.len()];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let src = ix * (ny * nz) + iy * nz + iz;
                let dst = ix + iy * nx + iz * nx * ny;
                out[dst] = vol[src];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(x: usize, y: usize, z: usize, nx: usize, ny: usize, nz: usize) -> usize {
        // x-major flat of a volume whose value encodes (x,y,z):
        // v = x*1000 + y*10 + z, placed at the x-major flat index.
        x * (ny * nz) + y * nz + z
    }

    #[test]
    fn round_trip_identity() {
        let (nx, ny, nz) = (3usize, 4, 5);
        let n = nx * ny * nz;
        let mut xm = vec![0.0f32; n];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    xm[idx(ix, iy, iz, nx, ny, nz)] = (ix * 1000 + iy * 10 + iz) as f32;
                }
            }
        }
        let disk = from_xmajor(&xm, nx, ny, nz);
        // Spot-check the disk layout: x fastest, then y, then z.
        // disk flat(x,y,z) = x + y*nx + z*nx*ny; value = x*1000 + y*10 + z.
        assert_eq!(disk[0], 0.0);
        assert_eq!(disk[1], 1000.0); // x+1
        assert_eq!(disk[nx], 10.0); // y+1
        assert_eq!(disk[nx * ny], 1.0); // z+1
        let back = to_xmajor(&disk, nx, ny, nz);
        assert_eq!(xm, back, "x-major round trip");
    }

    #[test]
    fn anisotropic_dims() {
        // Exercise non-square dims (vx != vy != vz).
        let (nx, ny, nz) = (2usize, 5, 3);
        let n = nx * ny * nz;
        let xm: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let disk = from_xmajor(&xm, nx, ny, nz);
        let back = to_xmajor(&disk, nx, ny, nz);
        assert_eq!(xm, back);
    }
}
