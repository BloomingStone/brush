"""Post-train evaluation for rigid-anchor sweep.

Usage:
  python3 tools/eval_rigid_anchor.py experiments/output/rigid-anchor <cfg1> <cfg2> ...
Prints final LPIPS/PSNR/SSIM (from log) + deform-field statistics
(global mean displacement, mean after offset removal, radial profile peak).
"""
import glob
import os
import re
import sys

import nibabel as nib
import numpy as np


def final_metrics(log):
    best = None
    with open(log) as f:
        for line in f:
            m = re.search(r"iter (\d+) loss=.*?psnr=\s*([\d.]+) ssim=\s*([\d.]+) lpips=\s*([\d.]+)", line)
            if m:
                best = (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return best


def field_stats(nii_path):
    d = nib.load(nii_path).get_fdata()[..., 0, :]
    mean_d = d.mean(axis=(0, 1, 2))
    off = np.linalg.norm(mean_d)
    d0 = d - mean_d
    mag = np.linalg.norm(d0, axis=-1)
    n = d.shape[0]
    yy, xx, zz = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2 + (zz - (n - 1) / 2) ** 2)
    peak, peak_r = 0.0, 0
    for rr in range(0, n // 2, 4):
        ring = (r >= rr) & (r < rr + 4)
        v = mag[ring].mean()
        if v > peak:
            peak, peak_r = v, rr + 2
    return {
        "offset_mm": off,
        "mean_residual_mm": mag.mean(),
        "p95_residual_mm": np.percentile(mag, 95),
        "peak_r_mm": peak_r * 2.4,
        "peak_mm": peak,
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "experiments/output/rigid-anchor"
    cfgs = sys.argv[2:] if len(sys.argv) > 2 else sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    print(f"{'cfg':<12} {'iter':>5} {'PSNR':>6} {'SSIM':>7} {'LPIPS':>7} | {'offset':>6} {'resid':>6} {'p95':>6} | peak@r")
    for c in cfgs:
        log = os.path.join(root, c, f"{c}.log")
        if not os.path.exists(log):
            log = os.path.join(root, f"{c}.log")  # logs at exp root
        if not os.path.exists(log):
            log = glob.glob(os.path.join(root, c, "*.log"))
            log = log[0] if log else None
        if log:
            fm = final_metrics(log)
            if not fm:
                continue
            row = f"{c:<12} {fm[0]:>5} {fm[1]:>6.2f} {fm[2]:>7.4f} {fm[3]:>7.4f}"
        else:
            row = f"{c:<12} {'-':>5} {'-':>6} {'-':>7} {'-':>7}"
        nii = glob.glob(os.path.join(root, c, "deform_field_phase00.nii.gz"))
        if nii:
            s = field_stats(nii[0])
            row += f" | {s['offset_mm']:>6.2f} {s['mean_residual_mm']:>6.2f} {s['p95_residual_mm']:>6.2f} | {s['peak_mm']:.2f}@{s['peak_r_mm']:.0f}mm"
        print(row)


if __name__ == "__main__":
    main()
