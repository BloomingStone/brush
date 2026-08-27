//! NIfTI-1 round-trip test for the internal x-major volume convention.
//!
//! Internal layout (x-major, R2 fields / brush-voxel / DRR):
//! `flat(x,y,z) = x*(ny*nz) + y*nz + z` (x slowest, z fastest).
//! Disk (NIfTI-1 standard, column-major): `flat(x,y,z) = x + y*nx + z*nx*ny`
//! (x fastest). nifti-rs's writer transposes an `(nx, ny, nz)` ndarray
//! automatically; its reader returns the raw disk buffer, which we convert
//! with `to_xmajor`. This test locks the whole chain: write → read → convert
//! must be the identity, and the on-disk header must carry natural
//! `(X,Y,Z)` dims with a diagonal sform.

use nifti::writer::WriterOptions;
use nifti::{NiftiHeader, NiftiType, NiftiObject, ReaderOptions};

fn write_vol(path: &std::path::Path, data: &[f32], nx: usize, ny: usize, nz: usize) {
    let arr = ndarray::Array3::from_shape_vec((nx, ny, nz), data.to_vec()).unwrap();
    let mut hdr = NiftiHeader::default();
    hdr.datatype = NiftiType::Float32 as i16;
    hdr.bitpix = 32;
    hdr.qform_code = 0;
    hdr.sform_code = 2;
    hdr.srow_x = [0.5, 0.0, 0.0, -1.0];
    hdr.srow_y = [0.0, 0.6, 0.0, -1.2];
    hdr.srow_z = [0.0, 0.0, 0.7, -1.4];
    WriterOptions::new(path)
        .reference_header(&hdr)
        .write_nifti(&arr)
        .unwrap();
}

fn read_vol(path: &std::path::Path) -> (Vec<f32>, usize, usize, usize) {
    let obj = ReaderOptions::new().read_file(path).unwrap();
    let dims = obj.header().dim;
    let (vx, vy, vz) = (dims[1] as usize, dims[2] as usize, dims[3] as usize);
    let volume = obj.into_volume();
    let data: Vec<f32> = volume.into_nifti_typed_data().unwrap();
    (data, vx, vy, vz)
}

#[test]
fn nifti_round_trip_xmajor() {
    let (nx, ny, nz) = (3usize, 4, 5); // anisotropic, non-square
    let n = nx * ny * nz;
    // x-major volume whose value encodes its (x,y,z) position.
    let xm: Vec<f32> = {
        let mut v = vec![0.0f32; n];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    v[ix * (ny * nz) + iy * nz + iz] = (ix * 1000 + iy * 10 + iz) as f32;
                }
            }
        }
        v
    };

    let dir = std::env::temp_dir().join("brush_nifti_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vol.nii.gz");
    write_vol(&path, &xm, nx, ny, nz);

    let obj = ReaderOptions::new().read_file(&path).unwrap();
    let dims = obj.header().dim;
    // Natural dims: dim[1..3] = (X, Y, Z).
    assert_eq!((dims[1] as usize, dims[2] as usize, dims[3] as usize), (nx, ny, nz));
    // Diagonal sform (world axes aligned with file axes).
    let srow_x = obj.header().srow_x;
    let srow_y = obj.header().srow_y;
    let srow_z = obj.header().srow_z;
    assert_eq!((srow_x[0], srow_x[1], srow_x[2]), (0.5, 0.0, 0.0));
    assert_eq!((srow_y[0], srow_y[1], srow_y[2]), (0.0, 0.6, 0.0));
    assert_eq!((srow_z[0], srow_z[1], srow_z[2]), (0.0, 0.0, 0.7));

    // Raw disk buffer must be column-major (x fastest):
    // flat(x,y,z) = x + y*nx + z*nx*ny, value = x*1000 + y*10 + z.
    let (raw, _vx, _vy, _vz) = read_vol(&path);
    assert_eq!(raw[0], 0.0);
    assert_eq!(raw[1], 1000.0); // x+1 fastest
    assert_eq!(raw[nx], 10.0); // then y
    assert_eq!(raw[nx * ny], 1.0); // then z

    // Read-back (disk order) converted to x-major must equal the original.
    let back = brush_process::volume_layout::to_xmajor(&raw, nx, ny, nz);
    assert_eq!(xm, back, "x-major nifti round trip");

    // And from_xmajor must be the exact inverse (what the NRRD writer uses).
    assert_eq!(brush_process::volume_layout::from_xmajor(&xm, nx, ny, nz), raw);

    std::fs::remove_dir_all(&dir).ok();
}
