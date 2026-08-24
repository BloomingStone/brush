#!/usr/bin/env python3
"""GT 梯度图输出 + 噪声影响量化。

动机: 边缘加权梯度损失的权重 map 用 |∇gt|。若 GT 有噪声, 原始 Sobel 梯度会
被噪声污染 (平坦区梯度≈噪声级), 权重可能集中在噪声而非真实边缘。

输出:
  1. 表格: 梯度幅度统计 (p50/p90/p95/p99/max) + 噪声级估计 (平坦区梯度中位)
     - 原始 Sobel vs 先高斯平滑再 Sobel
  2. PNG montage: GT | 原始梯度 | 平滑后梯度 | 高频残差 (gt - blur(gt))

用法:
  uv run --with numpy --with pillow python3 tools/gt_gradmap.py \
    target/exp/pig_opt40k/gt_pred_40000.nrrd target/exp/dsa_opt40k/gt_pred_40000.nrrd
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_nrrd_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    hdr_end = data.find(b"\n\n")
    header = data[hdr_end - 1024 if hdr_end > 1024 else 0:hdr_end].decode("ascii", "replace")
    sizes = None
    for ln in header.splitlines():
        ln = ln.strip()
        if ln.startswith("sizes:"):
            sizes = [int(x) for x in ln.split(":")[1].split()]
        if ln.startswith("endian:"):
            endian = ln.split(":")[1].strip()
    arr = np.frombuffer(data[hdr_end + 2:], dtype=np.dtype(np.float32).newbyteorder(
        ">" if endian == "big" else "<"))
    return arr.reshape(list(reversed(sizes)))


def split_gt_pred(vol):
    n, h, w2 = vol.shape
    return vol[:, :, : w2 // 2], vol[:, :, w2 // 2:]


def sobel_mag(img):
    p = np.pad(img, 1, mode="reflect")
    gx = (-p[0:-2, 0:-2] + p[0:-2, 2:] - 2 * p[1:-1, 0:-2]
          + 2 * p[1:-1, 2:] - p[2:, 0:-2] + p[2:, 2:])
    gy = (-p[0:-2, 0:-2] + p[2:, 0:-2] - 2 * p[0:-2, 1:-1]
          + 2 * p[2:, 1:-1] - p[0:-2, 2:] + p[2:, 2:])
    return np.hypot(gx, gy) / 4.0


def blur(img, sigma=1.5):
    n = 9
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = n // 2
    out = np.apply_along_axis(
        lambda m: np.convolve(np.pad(m, pad, mode="reflect"), k, mode="valid"), 1, img)
    return np.apply_along_axis(
        lambda m: np.convolve(np.pad(m, pad, mode="reflect"), k, mode="valid"), 0, out)


def flat_noise(img, gmag, low_frac=0.25):
    """低梯度掩码区域 (平坦区) 的梯度幅度中位 ≈ 噪声级."""
    thr = np.percentile(gmag, 100 * low_frac)
    mask = gmag <= thr
    return float(gmag[mask].mean()) if mask.any() else 0.0


def analyze(path, outdir):
    vol = load_nrrd_f32(path)
    gt, _ = split_gt_pred(vol)
    n = gt.shape[0]
    stats = []
    for i in range(n):
        g = gt[i]
        gm = sobel_mag(g)
        gm_s = sobel_mag(blur(g))
        hf = g - blur(g)
        stats.append(dict(
            raw=gm, sm=g.mm_s if hasattr(g, "mm_s") else gm_s, hf=hf))
        stats[-1]["sm"] = gm_s
    # 汇总
    pcts = (50, 90, 95, 99)
    g_raw = np.concatenate([s["raw"].ravel() for s in stats])
    g_sm = np.concatenate([s["sm"].ravel() for s in stats])
    print(f"== {Path(path).name}: {n} views")
    print(f"{'':12} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8} {'平坦区噪声':>10}")
    for name, arr in (("原始Sobel", g_raw), ("平滑后", g_sm)):
        q = np.percentile(arr, pcts)
        flat = flat_noise(np.concatenate([s["raw"].ravel() for s in stats]) if name == "原始Sobel" else arr, arr)
        print(f"{name:12} {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f} {q[3]:8.4f} {arr.max():8.4f} {flat:10.4f}")
    # 噪声级 = 平坦区原始梯度 (近似). 单独算
    g_raw_flat = np.percentile(g_raw, 25)
    print(f"   p25 原始梯度 ≈ {g_raw_flat:.4f} (平坦区噪声参考)")
    # 信号/噪声: 强边缘 (p99) vs 平坦 (p25)
    print(f"   p99/p25 = {np.percentile(g_raw, 99) / max(g_raw_flat, 1e-9):.1f}x")
    # montage 第一视图 (存到各自实验目录)
    g, gm, gm_s, hf = gt[0], stats[0]["raw"], stats[0]["sm"], stats[0]["hf"]
    out = Path(path).parent / (Path(path).stem + "_gt_grad.png")
    _save(g, gm, gm_s, hf, out)


def _save(g, gm, gm_s, hf, out):
    h, w = g.shape
    panels = [g, gm, gm_s, hf]
    names = ["GT", "raw|grad|", "sm|grad|", "hf(gt-blur)"]
    canvas = Image.new("L", (w * 4 + 3 * 6, h + 16), 0)
    for k, arr in enumerate(panels):
        a = np.abs(arr)
        im = a / (np.percentile(a, 99) + 1e-9)
        im = Image.fromarray(np.clip(im * 255, 0, 255).astype(np.uint8))
        canvas.paste(im, (k * (w + 6), 16))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    for p in args:
        analyze(p, p.rsplit("/", 1)[0])
