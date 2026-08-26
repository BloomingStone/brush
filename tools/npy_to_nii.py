#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "nibabel>=5.4.2",
#     "numpy>=2.5.2",
# ]
# ///
"""Convert an FDK volume .npy (world-coordinate cylindrical FOV) to .nii.gz.

The FDK volume (from `fdk_volume`) lives on a uniform grid spanning
`[-cyl_radius, cyl_radius]^3` in C-arm world coordinates. This writes a
NIfTI-1 file with the matching affine so it can be viewed in Slicer /
ITK-SNAP / etc.

Usage:
  python3 tools/npy_to_nii.py <volume.npy> [--meta meta.json] [-o out.nii.gz]

If `--meta` is given (fdk_volume writes `<out>/meta.json` with `vol` and
`cyl_radius`), the spacing/origin are derived automatically. Otherwise pass
`--spacing=MM` (isotropic, default 1.0) and the volume is assumed centered at
the origin.
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("--meta", default=None, help="fdk_volume meta.json (vol/cyl_radius)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--spacing", type=float, default=None, help="isotropic voxel mm (if no meta)")
    ap.add_argument("--mu-scale", type=float, default=None,
                    help="optional density scale to apply (mm^-1); volume.npy is already scaled")
    args = ap.parse_args()

    vol = np.load(args.npy)
    vol = np.ascontiguousarray(vol, dtype=np.float32)
    n = vol.shape[0]
    assert vol.shape == (n, n, n), f"expected cube, got {vol.shape}"

    if args.meta and os.path.exists(args.meta):
        meta = json.load(open(args.meta))
        r = float(meta["cyl_radius"])
        spacing = 2.0 * r / n
        origin = np.array([-r, -r, -r])
        print(f"meta: cyl_radius={r:.2f}mm -> spacing={spacing:.3f}mm, origin={origin}")
    else:
        spacing = args.spacing or 1.0
        origin = np.array([-spacing * n / 2.0] * 3)
        print(f"no meta: spacing={spacing}mm, origin={origin} (assume centered)")

    if args.mu_scale is not None:
        vol = vol * args.mu_scale

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = spacing
    affine[:3, 3] = origin

    out = args.out or os.path.splitext(args.npy)[0] + ".nii.gz"
    img = nib.Nifti1Image(vol, affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, out)
    print(f"saved {out}  shape={vol.shape}  affine diag={affine.diagonal()[:3]}")


if __name__ == "__main__":
    main()
