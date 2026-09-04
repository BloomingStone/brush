#!/usr/bin/env python3
"""Vignetting (图像边缘暗场) 估计 + 乘性补偿预处理。

背景: 探测器/光学边缘强度衰减使图像四周 ~20-25px 内灰度越靠边越暗
(实测 RXA_chest: 最外 10px 暗 -11%, 15px -4%, 20px -1%, >25px 平坦)。
Beer-Lambert 模型把暗场误读为"额外衰减" → 边缘投影不一致。

方法:
  1. 从序列"空气帧"(帧内存在未衰减射线, 帧 p99 raw 高) 提取空气像素
     (raw 高于阈值); 空气亮度应仅随到图像边缘的距离 d 变化。
  2. 帧内归一 (每帧取 d>60px 的空气像素中位数为参考) 后, 按 d 分箱得到
     乘性暗场曲线 V(d) = raw(d)/ref (V≈1 无暗场, V<1 变暗)。
  3. 补偿: raw_corr(x,y) = raw(x,y) / V(min(d(x,y), 30px)) — 只放大边缘,
     内部 (d≥30px) 不变。发生在任何归一化之前 (raw 域)。

用法:
  python3 tools/preprocess.py <in.dcm> <out.dcm> [--air-thr=2000] \
      [--frame-p99-thr=1800] [--plot=<png>] [--dry-run]

输出: 保留全部 DICOM 头, 仅替换 PixelData (uint16, clamp 0..65535)。
"""

import argparse
import sys

import numpy as np
import pydicom


def estimate_vignetting(px, air_thr, frame_p99_thr):
    """估计边框乘性暗场 V(d), d = 到最近图像边缘的距离 (px)。

    px: [F, H, W] float array。返回 (d_centers, V) 长度 ~31 的曲线。
    """
    H, W = px.shape[1], px.shape[2]
    frame_p99 = np.percentile(px, 99, axis=(1, 2))
    air_frames = np.where(frame_p99 > frame_p99_thr)[0]
    if len(air_frames) < 5:
        raise RuntimeError(
            f"仅 {len(air_frames)} 帧含空气 (p99>{frame_p99_thr}), 无法估计暗场"
        )

    yy, xx = np.mgrid[0:H, 0:W]
    de = np.minimum.reduce([yy, xx, H - 1 - yy, W - 1 - xx])  # 到最近边缘

    # 每帧: 空气掩膜像素 → (d, raw/ref), ref = 该帧 d>60px 空气像素中位数
    pairs = []
    n_air = 0
    for fi in air_frames:
        im = px[fi]
        mask = im > air_thr
        if mask.sum() < 1000:
            continue
        inner = mask & (de > 60)
        if inner.sum() < 200:
            inner = mask & (de > 40)
        if inner.sum() < 200:
            continue
        ref = np.median(im[inner])
        if ref < air_thr * 0.5:
            continue
        ds = de[mask]
        q = im[mask] / ref
        # 下采样到 ~8000 点/帧
        if len(ds) > 8000:
            idx = np.random.RandomState(0).choice(len(ds), 8000, replace=False)
            ds, q = ds[idx], q[idx]
        pairs.append(np.stack([ds, q], 1))
        n_air += 1
    if not pairs:
        raise RuntimeError("没有可用空气帧")

    pairs = np.concatenate(pairs, 0)
    d_all, q_all = pairs[:, 0], pairs[:, 1]
    dmax = 31
    centers = np.arange(dmax)
    v = np.ones(dmax)
    for d in range(dmax):
        sel = (d_all >= d - 1) & (d_all <= d + 1) & (d_all >= 0)
        if sel.sum() >= 100:
            v[d] = np.median(q_all[sel])
    # 平滑 + 单调化尾部: d>=25 起强制回 1.0 (无暗场区), 并保证 V≤1.02
    v = np.clip(v, 0.5, 1.02)
    for d in range(dmax - 1, -1, -1):
        if d >= 25:
            v[d] = 1.0
        elif v[d] > v[min(d + 1, dmax - 1)]:
            v[d] = v[min(d + 1, dmax - 1)]
    # 前端平滑
    for _ in range(2):
        v[1:-1] = (v[:-2] + 2 * v[1:-1] + v[2:]) / 4
    print(
        f"[vignetting] 空气帧 {n_air}/{len(air_frames)}: "
        f"V(d=0..30px) 最小 {v.min():.3f} @ d={v.argmin()}px, "
        f"d=10px: {v[min(10,dmax-1)]:.3f}, d=20px: {v[min(20,dmax-1)]:.3f}"
    )
    return centers, v


def apply_compensation(px, v):
    """raw_corr = raw / V(d)。返回 uint16 数组 (clamp)。"""
    H, W = px.shape[1], px.shape[2]
    yy, xx = np.mgrid[0:H, 0:W]
    de = np.minimum.reduce([yy, xx, H - 1 - yy, W - 1 - xx])
    vmap = v[np.minimum(de, len(v) - 1)].astype(np.float32)
    out = px / vmap
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_dcm")
    ap.add_argument("out_dcm")
    ap.add_argument("--air-thr", type=float, default=2000.0)
    ap.add_argument("--frame-p99-thr", type=float, default=1800.0)
    ap.add_argument("--no-compensate", action="store_true",
                    help="只估计并打印暗场曲线, 不写输出")
    ap.add_argument("--plot", default=None, help="保存 V(d) 曲线 PNG")
    args = ap.parse_args()

    ds = pydicom.dcmread(args.in_dcm)
    if not hasattr(ds, "pixel_array"):
        sys.exit("无像素数据")
    print(f"读取 {args.in_dcm}: {ds.Columns}x{ds.Rows} x {getattr(ds,'NumberOfFrames',1)} 帧")
    px = ds.pixel_array.astype(np.float32)

    centers, v = estimate_vignetting(px, args.air_thr, args.frame_p99_thr)
    for d in range(0, 31, 5):
        print(f"  V(d={d:2d}px) = {v[d]:.3f}  (暗 {-100*(1-v[d]):5.1f}%)")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 3))
            plt.plot(centers, v, "o-")
            plt.axhline(1.0, color="k", ls="--", lw=0.6)
            plt.xlabel("距离图像边缘 d (px)")
            plt.ylabel("乘性暗场 V(d)")
            plt.title("边缘暗场 (vignetting) 曲线")
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"[plot] -> {args.plot}")
        except Exception as e:  # matplotlib 缺失时降级
            print(f"[plot] 失败: {e}", file=sys.stderr)

    if args.no_compensate:
        return

    print("补偿: raw_corr = raw / V(d), 写回 PixelData ...")
    out = apply_compensation(px, v)
    ds_out = ds.copy()
    ds_out.PixelData = out.tobytes()
    pydicom.dcmwrite(args.out_dcm, ds_out)
    print(f"[done] -> {args.out_dcm}")

    # 验证: 补偿后空气边缘 vs 内部应平坦
    frame_p99 = np.percentile(px, 99, axis=(1, 2))
    f0 = np.where(frame_p99 > args.frame_p99_thr)[0][0]
    a = px[f0]; b = out[f0].astype(np.float32)
    for de_lo, de_hi in [(0, 10), (10, 20), (20, 30), (60, 200)]:
        m0 = np.zeros_like(a, bool); m1 = np.zeros_like(a, bool)
        yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        de = np.minimum.reduce([yy, xx, a.shape[0]-1-yy, a.shape[1]-1-xx])
        air = a > args.air_thr
        m0 = air & (de >= de_lo) & (de < de_hi)
        m1 = m0
        if m0.sum() > 50:
            print(f"  验证帧{f0} 空气 d[{de_lo},{de_hi})px: "
                  f"补偿前 {np.median(a[m0]):.0f} → 补偿后 {np.median(b[m1]):.0f}")


if __name__ == "__main__":
    main()
