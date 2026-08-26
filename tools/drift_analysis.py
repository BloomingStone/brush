#!/usr/bin/env python3
"""Analyze point-cloud drift: initial sampling region vs final canonical splats.

For each init-cylinder experiment config, read the initial region params
(cylinder radius / half-height from the log) and the final canonical_final.ply
splat means, and measure:
  - mean |final position| (how far points ended from isocenter)
  - fraction of points beyond R0, beyond the init radius, beyond the deform
    grid (scene_extent 153mm)
  - the point density profile vs radius (drift direction)

Usage: python3 tools/drift_analysis.py <experiments/output/2026-08-26_init-cylinder>
"""
import glob
import os
import re
import sys

import numpy as np


def read_ply_means(path):
    """Read x y z of 'vertex' elements from an ASCII/binary PLY."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            header.append(line)
            if line.startswith(b"end_header"):
                break
    text = b"".join(header).decode()
    n = None
    for line in text.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[2])
    if n is None:
        raise ValueError("no vertex count")
    fmt = "ascii" if "format ascii" in text else "binary"
    with open(path, "rb") as f:
        for _ in header:
            f.readline()
        body = f.read()
    if fmt == "ascii":
        pts = []
        for ln in body.decode().strip().splitlines()[:n]:
            parts = ln.split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(pts)
    else:
        # binary_little_endian: per-vertex fixed-size records.
        # offset of x/y/z = number of float properties before them × 4.
        props = []
        in_vertex = False
        for line in text.splitlines():
            if line.startswith("element vertex"):
                in_vertex = True
                continue
            if line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line.startswith("property"):
                props.append(line.split()[2])
        xyz_idx = [props.index("x"), props.index("y"), props.index("z")]
        rec_bytes = len(props) * 4
        recs = np.frombuffer(body[: n * rec_bytes], dtype="<f4").reshape(n, len(props))
        return recs[:, xyz_idx].astype(np.float64)


def init_region_from_log(log):
    m = re.search(r"cylinder init: R=([\d.]+) .* half_h=([\d.]+), N=(\d+)", log)
    if m:
        return float(m.group(1)), float(m.group(2)), int(m.group(3))
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "experiments/output/2026-08-26_init-cylinder"
    print(f"{'cfg':<10} {'R(mm)':>7} {'H/2':>6} {'N':>6} | {'mean|r|':>8} {'p50':>6} {'p95':>6} {'max':>6} | {'%>R0':>6} {'%>R':>6} {'%>153':>6}")
    for cfg in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(cfg):
            continue
        name = os.path.basename(cfg)
        log = os.path.join(cfg, name + ".log")
        if not os.path.exists(log):
            log = os.path.join(root, name + ".log")
        if not os.path.exists(log):
            continue
        region = init_region_from_log(open(log).read())
        ply = os.path.join(cfg, "canonical_final.ply")
        if not os.path.exists(ply) or region is None:
            continue
        R, hh, N = region
        pts = read_ply_means(ply)
        r = np.linalg.norm(pts, axis=1)
        print(f"{name:<10} {R:>7.1f} {hh:>6.1f} {N:>6} | {r.mean():>8.1f} {np.median(r):>6.1f} "
              f"{np.percentile(r,95):>6.1f} {r.max():>6.1f} | {(r>R).mean()*100:>5.1f}% "
              f"{(r>R*1.0).mean()*100:>5.1f}% {(r>153).mean()*100:>5.1f}%")


if __name__ == "__main__":
    main()
