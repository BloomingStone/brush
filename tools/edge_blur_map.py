#!/usr/bin/env python3
"""空间边缘锐度对比: pig vs dsa 重建的边缘模糊分布。

对每个数据集 (final gt_pred nrrd), 对每个视图计算:
  blur_map = pred_grad / gt_grad   (边缘像素, 粗网格聚合)
  → 模糊分布: 中心 vs 边缘, 强边缘 vs 弱边缘。

输出:
  1. 8x8 空间 blur_ratio 热图 (mean over views) — 糊在局部还是全域
  2. 按 GT 边缘强度分桶的 blur_ratio — 强/弱边缘糊的程度
  3. 边缘像素中 pred 比 GT 软超过 X% 的比例

用法:
  uv run --with numpy --with pillow python3 tools/edge_blur_map.py \
    target/exp/pig_opt40k/gt_pred_40000.nrrd target/exp/dsa_opt40k/gt_pred_40000.nrrd
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools.edge_blur_analysis import load_nrrd_f32, split_gt_pred, sobel_mag


def blur_map(gt, pred, grid=8):
    """逐视图 pred/gt 梯度比, 在 grid×grid 网格 + 边缘掩码内聚合."""
    n, h, w = gt.shape
    gy = np.linspace(0, h, grid + 1).astype(int)
    gx = np.linspace(0, w, grid + 1).astype(int)
    ratio = np.zeros((grid, grid))
    counts = np.zeros((grid, grid))
    edge_pixels = 0
    soft_frac = 0.0
    for i in range(n):
        sg, sp = sobel_mag(gt[i]), sobel_mag(pred[i])
        thr = np.percentile(sg, 90)
        edge = sg >= thr
        r = sp / (sg + 1e-9)
        for a in range(grid):
            for b in range(grid):
                m = edge[gy[a]:gy[a + 1], gx[b]:gx[b + 1]]
                c = int(m.sum())
                if c:
                    ratio[a, b] += r[gy[a]:gy[a + 1], gx[b]:gx[b + 1]][m].sum()
                    counts[a, b] += c
        edge_pixels += edge.sum()
        soft_frac += (r[edge] < 0.7).mean()
    ratio = np.divide(ratio, np.maximum(counts, 1), out=np.zeros_like(ratio))
    return ratio, edge_pixels / n, soft_frac / n


def bucket_by_strength(gt, pred, nb=5):
    """按 GT 边缘强度分桶 (等频), 每桶 pred/gt 梯度比."""
    n = gt.shape[0]
    ratios = {k: [] for k in range(nb)}
    for i in range(n):
        sg, sp = sobel_mag(gt[i]), sobel_mag(pred[i])
        thr = np.percentile(sg, 90)
        edge = sg >= thr
        sg_e, sp_e = sg[edge], sp[edge]
        r = sp_e / (sg_e + 1e-9)
        q = np.quantile(sg_e, np.linspace(0, 1, nb + 1)[1:-1])
        bins = np.searchsorted(q, sg_e)
        for k in range(nb):
            ratios[k].append(r[bins == k].mean() if (bins == k).any() else np.nan)
    return [float(np.nanmean(ratios[k])) for k in range(nb)]


def save_heatmap(ratio, tag, out):
    grid = ratio.shape[0]
    im = (ratio - 0.4) / 0.6  # 0.4→0, 1.0→1
    im = np.clip(im, 0, 1) * 255
    Image.fromarray(im.astype(np.uint8).repeat(60, 1).repeat(60, 0)).resize(
        (grid * 60, grid * 60), Image.NEAREST).save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    results = {}
    for p in args:
        gt, pred = split_gt_pred(load_nrrd_f32(p))
        ratio, edge_frac, soft_frac = blur_map(gt, pred)
        buckets = bucket_by_strength(gt, pred)
        tag = Path(p).parent.name
        results[tag] = dict(ratio=ratio, soft_frac=soft_frac)
        print(f"== {tag} ({Path(p).parent})")
        print(f"  边缘像素占比: {edge_frac:.3f} | 边缘中 pred<0.7×GT锐度比例: {soft_frac*100:.1f}%")
        print(f"  8x8 空间 blur_ratio 热图 (pred/GT 梯度比):")
        for a in range(ratio.shape[0]):
            print("    " + " ".join(f"{v:.2f}" for v in ratio[a]))
        print(f"  整体 blur_ratio: {ratio.mean():.3f} | 中心 2x2: {ratio[3:5,3:5].mean():.3f} | 周边: {np.mean(np.delete(np.delete(ratio, [3,4],0),[3,4],1)):.3f}")
        print(f"  按 GT 边缘强度分桶 (弱→强) blur_ratio: " + " ".join(f"{v:.3f}" for v in buckets))
        save_heatmap(ratio, tag, Path(p).parent / f"blur_map_{tag}.png")
    if len(results) == 2:
        tags = list(results)
        print("\n--- 对比 ---")
        r1, r2 = results[tags[0]]["ratio"], results[tags[1]]["ratio"]
        print(f"  整体 blur_ratio: {tags[0]} {r1.mean():.3f} vs {tags[1]} {r2.mean():.3f}")
        print(f"  中心 2x2: {r1[3:5,3:5].mean():.3f} vs {r2[3:5,3:5].mean():.3f}")
        print(f"  软边缘比例: {results[tags[0]]['soft_frac']*100:.1f}% vs {results[tags[1]]['soft_frac']*100:.1f}%")
