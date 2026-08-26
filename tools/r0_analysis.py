#!/usr/bin/env python3
"""Analyze points with R > R0 (118.6mm) from the cylinder-init experiments.

Reads canonical_final.ply (raw opacity logits), computes activated density
`mu = 0.002 * silu(raw)`, splits points by R vs R0, and plots the density
histogram of the >R0 points (to judge whether they are air or structure).

Usage: python3 tools/r0_analysis.py <exp_dir>
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MU = 0.002
R0 = 118.6


def read_ply(path):
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
    fmt = "ascii" if "format ascii" in text else "binary"
    with open(path, "rb") as f:
        for _ in header:
            f.readline()
        body = f.read()
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
    rec_bytes = len(props) * 4
    recs = np.frombuffer(body[: n * rec_bytes], dtype="<f4").reshape(n, len(props))
    idx = {p: i for i, p in enumerate(props)}
    return recs[:, [idx["x"], idx["y"], idx["z"], idx["opacity"]]]


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def analyze(cfg, ax):
    ply = os.path.join(cfg, "canonical_final.ply")
    if not os.path.exists(ply):
        return None
    name = os.path.basename(cfg)
    d = read_ply(ply)
    r = np.linalg.norm(d[:, :3], axis=1)
    raw = d[:, 3]
    mu = MU * silu(raw)
    outer = r > R0
    n_out = outer.sum()
    # air-fraction estimate: activated density below 2x water-noise floor?
    # report percentiles + fraction "low" (< 0.004, ~2x MU_WATER)
    mu_o = mu[outer]
    frac_low = (mu_o < 0.004).mean() * 100
    frac_tiny = (mu_o < 0.002).mean() * 100
    ax.hist(mu_o, bins=60, alpha=0.7, label=f"{name} (n={n_out}, {outer.mean()*100:.0f}%)")
    ax.axvline(np.median(mu_o), color="k", ls="--", lw=1)
    ax.axvline(0.004, color="r", ls=":", lw=1, label="0.004 (2x μ_water)")
    return dict(
        name=name,
        n=n_out,
        frac=outer.mean() * 100,
        median=float(np.median(mu_o)),
        p25=float(np.percentile(mu_o, 25)),
        p75=float(np.percentile(mu_o, 75)),
        frac_low=frac_low,
        frac_tiny=frac_tiny,
        total=len(r),
    )


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "experiments/output/2026-08-26_init-cylinder"
    out_png = os.path.join(root, "r0_density_hist.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for cfg in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(cfg) and os.path.exists(os.path.join(cfg, "canonical_final.ply")):
            rows.append(analyze(cfg, ax))
    ax.set_xlabel("activated density mu (mm^-1) of R>R0 points")
    ax.set_ylabel("count")
    ax.set_title(f"R>R0({R0}mm) point density distribution")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"saved {out_png}")
    print(f"{'cfg':<10} {'n>R0':>6} {'frac%':>6} | {'median':>7} {'p25':>7} {'p75':>7} | {'<0.004%':>7} {'<0.002%':>7}")
    for r in rows:
        if r:
            print(f"{r['name']:<10} {r['n']:>6} {r['frac']:>6.1f} | {r['median']:>7.4f} "
                  f"{r['p25']:>7.4f} {r['p75']:>7.4f} | {r['frac_low']:>6.1f}% {r['frac_tiny']:>6.1f}%")


if __name__ == "__main__":
    main()
