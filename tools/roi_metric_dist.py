#!/usr/bin/env python3
"""按 ROI 大小记录验证图像 GT|pred 的 L1 / PSNR / SSIM 分布。

检验假设: 大体结构重建完毕后, 背景/平滑区域(相同值)主导损失, 差异
(边缘/解剖细节)被淹没 → 内部窄窗指标应明显差于全图。

用法:
  python3 tools/roi_metric_dist.py target/exp/pig_opt40k/gt_pred_40000.nrrd [roi_sizes...]

默认 ROI 边长比例: 1.0 1/2 1/4 1/8 (居中窗); 同时报告最内窗外的边框环。
只依赖 numpy (自解析 NRRD raw float32, 自实现全局 SSIM)。
"""
import sys
from pathlib import Path

import numpy as np


def load_nrrd_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    # 头部以空行结束
    hdr_end = data.find(b"\n\n")
    header = data[hdr_end - 1024 if hdr_end > 1024 else 0:hdr_end].decode("ascii", "replace")
    lines = header.splitlines()
    sizes = None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("sizes:"):
            sizes = [int(x) for x in ln.split(":")[1].split()]
        if ln.startswith("type:"):
            typ = ln.split(":")[1].strip()
        if ln.startswith("endian:"):
            endian = ln.split(":")[1].strip()
    if not sizes:
        raise ValueError(f"no sizes in {path}")
    arr = np.frombuffer(data[hdr_end + 2:], dtype=np.dtype(np.float32).newbyteorder(
        ">" if endian == "big" else "<"))
    # NRRD "sizes:" 是 x,y,z 顺序 (最后一维最快); numpy reshape 用 z,y,x
    return arr.reshape(list(reversed(sizes)))


def split_gt_pred(vol):
    n, h, w2 = vol.shape
    w = w2 // 2
    return vol[:, :, :w], vol[:, :, w:]


def global_ssim(gt, pred, c1=0.01, c2=0.03):
    """全局 SSIM (同一数据范围归一化). gt/pred 均在 [0,1]."""
    m1, m2 = gt.mean(), pred.mean()
    v1, v2 = gt.var(), pred.var()
    cov = ((gt - m1) * (pred - m2)).mean()
    return ((2 * m1 * m2 + c1) * (2 * cov + c2)) / ((m1 * m1 + m2 * m2 + c1) * (v1 + v2 + c2))


def metrics(gt, pred):
    l1 = np.abs(gt - pred).mean()
    mse = ((gt - pred) ** 2).mean()
    psnr = 10 * np.log10(1.0 / (mse + 1e-12)) if mse > 0 else float("inf")
    return l1, psnr, global_ssim(gt, pred)


def center_crop(img, frac):
    h, w = img.shape
    fh, fw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    y0, x0 = (h - fh) // 2, (w - fw) // 2
    return img[y0:y0 + fh, x0:x0 + fw]


def report(nrrd_path, fractions=(1.0, 0.5, 0.25, 0.125)):
    vol = load_nrrd_f32(nrrd_path)
    gt, pred = split_gt_pred(vol)
    n, h, w = gt.shape
    print(f"== {Path(nrrd_path).name}: {n} views, gt/pred {h}x{w}")
    print(f"{'ROI (边长比例)':>16} {'窗(px)':>10} {'L1':>8} {'PSNR':>8} {'SSIM':>8}  "
          f"{'|ΔL1 vs全图':>12} {'ΔSSIM vs全图':>12}")
    base = {}
    prev = None  # (l1, ssim) 上一窗 (小窗)
    for frac in sorted(fractions, reverse=True):
        l1s, psnrs, ssims = [], [], []
        for i in range(n):
            l1, psnr, ssim = metrics(center_crop(gt[i], frac), center_crop(pred[i], frac))
            l1s.append(l1); psnrs.append(psnr); ssims.append(ssim)
        l1, psnr, ssim = np.mean(l1s), np.mean(psnrs), np.mean(ssims)
        if frac == 1.0:
            base = dict(l1=l1, psnr=psnr, ssim=ssim)
        dl = l1 - base["l1"]
        ds = ssim - base["ssim"]
        fh, fw = int(round(h * frac)), int(round(w * frac))
        print(f"{frac:>14.3f} {fh}x{fw:>8} {l1:8.4f} {psnr:8.3f} {ssim:8.4f}  {dl:+12.5f} {ds:+12.5f}")
        prev = (l1, ssim)
    # 边框环: 全图 - 最小窗
    fmin = min(fractions)
    ring_l1s, ring_psnrs, ring_ssims = [], [], []
    for i in range(n):
        inner = center_crop(gt[i], fmin)
        y0 = (h - inner.shape[0]) // 2
        x0 = (w - inner.shape[1]) // 2
        ring = np.ones_like(gt[i], dtype=bool)
        ring[y0:y0 + inner.shape[0], x0:x0 + inner.shape[1]] = False
        ring_l1, ring_psnr, ring_ssim = metrics(gt[i][ring], pred[i][ring])
        ring_l1s.append(ring_l1); ring_psnrs.append(ring_psnr); ring_ssims.append(ring_ssim)
    rl, rp, rs = np.mean(ring_l1s), np.mean(ring_psnrs), np.mean(ring_ssims)
    dl = rl - base["l1"]; ds = rs - base["ssim"]
    print(f"{'边框环(外)':>16} {('-%dx%d' % center_crop(gt[0], fmin).shape):>10} {rl:8.4f} {rp:8.3f} {rs:8.4f}  {dl:+12.5f} {ds:+12.5f}")


def heatmap(nrrd_path, grid=16, out=None):
    """粗网格局部 L1 热图 (mean over views), 找出误差集中区域."""
    vol = load_nrrd_f32(nrrd_path)
    gt, pred = split_gt_pred(vol)
    n, h, w = gt.shape
    gy, gx = np.linspace(0, h, grid + 1).astype(int), np.linspace(0, w, grid + 1).astype(int)
    heat = np.zeros((grid, grid))
    for i in range(n):
        err = np.abs(gt[i] - pred[i])
        for a in range(grid):
            for b in range(grid):
                heat[a, b] += err[gy[a]:gy[a + 1], gx[b]:gx[b + 1]].mean()
    heat /= n
    print(f"-- {grid}x{grid} 局部 L1 热图 (mean over {n} views) [{Path(nrrd_path).name}] --")
    for a in range(grid):
        print(" ".join(f"{v:.3f}" for v in heat[a]))
    if out:
        np.savetxt(out, heat, fmt="%.4f")
        print(f"saved heatmap -> {out}")
    # 统计: 热图最大值是均值的几倍
    print(f"max/mean = {heat.max() / heat.mean():.2f}x, argmax grid (row,col) = {np.unravel_index(heat.argmax(), heat.shape)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    nrrd = args[0]
    report(nrrd)
    heatmap(nrrd)
