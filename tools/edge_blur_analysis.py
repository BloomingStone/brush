#!/usr/bin/env python3
"""边缘模糊验证: 用 Sobel 边缘 + 沿梯度方向剖面量化"局部边缘模糊"。

指标:
  edge_vs_bg   边缘像素 L1 vs 非边缘像素 L1 (边缘是否误差更大)
  blur_ratio   pred 梯度幅度 / gt 梯度幅度 (<1 = 预测更糊)
  hf_err       高频 (gt-平滑) 域误差: 模糊直接体现在高频成分丢失
  trans_width  沿边缘法向 10→90% 强度过渡宽度 (pred/gt, >1 = 更糊)

输出: 表格 + 每视图 montage PNG (GT | pred | 误差 | 边缘掩码)。

用法:
  python3 tools/edge_blur_analysis.py target/exp/pig_opt40k/gt_pred_40000.nrrd [iter1 iter2 ...]
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
        if ln.startswith("type:"):
            typ = ln.split(":")[1].strip()
        if ln.startswith("endian:"):
            endian = ln.split(":")[1].strip()
    arr = np.frombuffer(data[hdr_end + 2:], dtype=np.dtype(np.float32).newbyteorder(
        ">" if endian == "big" else "<"))
    return arr.reshape(list(reversed(sizes)))


def split_gt_pred(vol):
    n, h, w2 = vol.shape
    w = w2 // 2
    return vol[:, :, :w], vol[:, :, w:]


def sobel_mag(img):
    p = np.pad(img, 1, mode="reflect")
    gx = (-p[0:-2, 0:-2] + p[0:-2, 2:] - 2 * p[1:-1, 0:-2]
          + 2 * p[1:-1, 2:] - p[2:, 0:-2] + p[2:, 2:])
    gy = (-p[0:-2, 0:-2] + p[2:, 0:-2] - 2 * p[0:-2, 1:-1]
          + 2 * p[2:, 1:-1] - p[0:-2, 2:] + p[2:, 2:])
    return np.hypot(gx, gy) / 4.0  # /4 = Sobel 归一


def gauss1d(n=9, sigma=1.5):
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(img, sigma=1.5):
    k = gauss1d(n=9, sigma=sigma)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
    return out


def line_profile(img, y, x, dy, dx, half=30):
    """沿单位方向 (dx,dy) 双线性采样 img, 长度 2*half+1, 返回剖面."""
    norm = max(np.hypot(dx, dy), 1e-9)
    dx, dy = dx / norm, dy / norm
    h, w = img.shape
    prof = []
    for t in range(-half, half + 1):
        yy, xx = y + dy * t, x + dx * t
        y0, x0 = int(np.floor(yy)), int(np.floor(xx))
        fy, fx = yy - y0, xx - x0
        y0, x0 = min(max(y0, 0), h - 2), min(max(x0, 0), w - 2)
        v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
             + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
        prof.append(v)
    return np.array(prof)


def trans_width(prof, lo=0.1, hi=0.9):
    """10→90% 过渡宽度 (像素). 返回 None 若剖面单调性不足."""
    p = prof - prof.min()
    denom = p.max()
    if denom < 1e-6:
        return None
    p = p / denom
    i_lo = np.where(p >= lo)[0]
    i_hi = np.where(p >= hi)[0]
    if i_lo.size == 0 or i_hi.size == 0:
        return None
    return float(i_hi[0] - i_lo[0])


def analyze(vol, n_edge=40, outdir=None):
    gt, pred = split_gt_pred(vol)
    n, h, w = gt.shape
    rows = {"edge_l1": [], "bg_l1": [], "blur_ratio": [], "hf_err": [],
            "tw_gt": [], "tw_pred": []}
    for i in range(n):
        g, p = gt[i], pred[i]
        sg = sobel_mag(g)
        sp = sobel_mag(p)
        thr = np.percentile(sg, 90)
        edge = sg >= thr
        bg = ~edge
        rows["edge_l1"].append(np.abs(g - p)[edge].mean())
        rows["bg_l1"].append(np.abs(g - p)[bg].mean())
        rows["blur_ratio"].append((sp[edge]).mean() / (sg[edge].mean() + 1e-9))
        hf_g, hf_p = g - blur(g), p - blur(p)
        rows["hf_err"].append(np.abs(hf_g - hf_p).mean())
        # 沿最 K 强边缘法向测过渡宽度
        ys, xs = np.nonzero(sg)
        if ys.size == 0:
            continue
        mags = sg[ys, xs]
        k = ys.size
        order = np.argpartition(mags, -min(k, n_edge))[-min(k, n_edge):]
        for idx in order:
            yy, xx = ys[idx], xs[idx]
            gy, gx = sobel_mag(g), sobel_mag(g)
            # 梯度方向 (朝向强度上升方向)
            pimg = np.pad(g, 1, mode="reflect")
            gy = (pimg[2:, 1:-1] - pimg[:-2, 1:-1]) / 2.0
            gx = (pimg[1:-1, 2:] - pimg[1:-1, :-2]) / 2.0
            tw_g = trans_width(line_profile(g, yy, xx, gy[yy, xx], gx[yy, xx]))
            tw_p = trans_width(line_profile(p, yy, xx, gy[yy, xx], gx[yy, xx]))
            if tw_g is not None and tw_p is not None:
                rows["tw_gt"].append(tw_g)
                rows["tw_pred"].append(tw_p)
    m = {k: float(np.mean(v)) for k, v in rows.items() if v}
    m["edge/bg_l1"] = m["edge_l1"] / m["bg_l1"]
    m["tw_pred/gt"] = m["tw_pred"] / m["tw_gt"]
    m["edge_frac"] = float(np.mean([(sobel_mag(gt[i]) >= np.percentile(sobel_mag(gt[i]), 90)).mean() for i in range(n)]))
    if outdir and n:
        _save_montage(gt[0], pred[0], outdir, "view0")
    return m


def _save_montage(g, p, outdir, tag):
    err = np.abs(g - p)
    sg = sobel_mag(g)
    edge = sg >= np.percentile(sg, 90)
    emap = np.where(edge, 1.0, 0.2 * (sg / (sg.max() + 1e-9)))
    emap = emap / emap.max() * 255
    errv = err / (err.max() + 1e-9) * 255
    panels = [g, p, errv, emap]
    h, w = g.shape
    canvas = Image.new("L", (w * len(panels) + 3 * 5, h), 0)
    for k, arr in enumerate(panels):
        im = Image.fromarray(np.clip(arr * (255 if arr.max() > 1.5 else 1), 0, 255).astype(np.uint8))
        canvas.paste(im, (k * (w + 5), 0))
    out = Path(outdir) / f"edge_montage_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"saved montage -> {out}")


def report(paths):
    print(f"{'iter':>8} {'边缘L1':>8} {'背景L1':>8} {'边/背比':>8} "
          f"{'blur_ratio':>11} {'hf_err':>8} {'tw_gt':>7} {'tw_pred':>8} {'tw比':>6}")
    for path in paths:
        tag = Path(path).stem
        m = analyze(load_nrrd_f32(path))
        print(f"{tag:>8} {m['edge_l1']:8.4f} {m['bg_l1']:8.4f} {m['edge/bg_l1']:8.2f} "
              f"{m['blur_ratio']:11.3f} {m['hf_err']:8.4f} "
              f"{m['tw_gt']:7.2f} {m['tw_pred']:8.2f} {m['tw_pred/gt']:6.2f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    outdir = args[0] + ".edge_analysis"
    if len(args) == 1:
        vol = load_nrrd_f32(args[0])
        m = analyze(vol, outdir=outdir)
        for k, v in m.items():
            print(f"{k:>14}: {v:.4f}")
    else:
        report(args)
