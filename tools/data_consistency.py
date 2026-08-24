#!/usr/bin/env python3
"""多视角一致性 / 噪声诊断 (对比 RXA_pig 与 rotate_dsa)。

假设: 真实 CTA 投影 (rotate_dsa, 合成) 自洽性好 → 重建好; 若 RXA_pig 做过
后处理 (逐帧归一/窗位) 或噪声高, 会破坏多视角一致性 → 重建边缘糊。

指标 (所有帧, 原始 16bit):
  mean/std/p50/p95   逐帧强度统计 → 角度趋势是否平滑 (物理: 旋转扫描路径长度
                      变化 → mean 应随角度平滑变化; 若被逐帧归一则近乎恒定/跳变)
  max/min            是否被裁剪 (窗位钳制痕迹)
  noise              平坦区高频残差 std (Sobel 高梯度掩码外的局部方差)
  dm_frame           |mean(f)-mean(f-1)| 逐帧跳变 (后处理闪烁 / 未对齐)
  phase_std          相位值分布

输出: 每数据集汇总 + 关键比率。
用法:
  uv run --with pydicom --with numpy python3 tools/data_consistency.py FILE1 [FILE2...]
"""
import sys
from pathlib import Path

import numpy as np
import pydicom


def load_frames(path, roi=20):
    d = pydicom.dcmread(path)
    arr = d.pixel_array.astype(np.float32)  # [F,H,W]
    if roi:
        arr = arr[:, roi:-roi, roi:-roi]
    return arr, d


def noise_est(frames, sample=8):
    """平坦区噪声: 低梯度掩码内局部梯度 (Sobel) 中位."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(frames), sample)
    grads = []
    for i in idx:
        g = frames[i]
        p = np.pad(g, 1, mode="reflect")
        gx = (-p[0:-2, 0:-2] + p[0:-2, 2:] - 2 * p[1:-1, 0:-2]
              + 2 * p[1:-1, 2:] - p[2:, 0:-2] + p[2:, 2:])
        gy = (-p[0:-2, 0:-2] + p[2:, 0:-2] - 2 * p[0:-2, 1:-1]
              + 2 * p[2:, 1:-1] - p[0:-2, 2:] + p[2:, 2:])
        gmag = np.hypot(gx, gy) / 4.0
        thr = np.percentile(gmag, 25)
        grads.append(gmag[gmag <= thr])
    return float(np.median(np.concatenate(grads)))


def analyze(path):
    frames, d = load_frames(path)
    f = frames / 65535.0  # 归一化 [0,1] 便于跨数据比较
    means = f.mean(axis=(1, 2))
    stds = f.std(axis=(1, 2))
    p5 = np.percentile(f, 5, axis=(1, 2))
    p95 = np.percentile(f, 95, axis=(1, 2))
    vmax = f.max()
    vmin = f.min()
    n = len(f)
    # 逐帧跳变
    dm = np.abs(np.diff(means))
    # 角度平滑度: mean 的一阶差分应平滑 (用中位数绝对差分 vs 波动)
    smooth = np.median(np.abs(np.diff(means, 2)))
    phase = None
    if (0x0071, 0x1010) in d:
        try:
            phase = d[(0x0071, 0x1010)].value
            if isinstance(phase, (bytes, bytearray)):
                phase = np.frombuffer(phase, dtype="<f4")
            phase = np.asarray(phase, dtype=np.float32).reshape(-1)
        except Exception as e:
            phase = None
    name = Path(path).name
    print(f"== {name}")
    print(f"  {n} frames, {frames.shape[1]}x{frames.shape[2]}, "
          f"intensity range [{vmin:.4f}, {vmax:.4f}] (归一化, 1.0=clip 痕迹)")
    print(f"  mean: {means.mean():.4f}±{means.std():.4f} 范围 [{means.min():.4f},{means.max():.4f}]"
          f"  (全帧跨度 {means.max()-means.min():.4f})")
    print(f"  p5/p50/p95: {p5.mean():.4f} / {means.mean():.4f} / {p95.mean():.4f}")
    print(f"  std(帧): 平均 {stds.mean():.4f}")
    print(f"  逐帧 mean 跳变: 中位 {np.median(dm):.5f} p99 {np.percentile(dm,99):.5f} max {dm.max():.5f}")
    print(f"  mean 二阶差分(平滑度): {smooth:.5f}")
    print(f"  平坦区噪声: {noise_est(frames):.4f} (16bit 原始域) / {noise_est(f):.5f} (归一化域)")
    if phase is not None and len(phase) >= n:
        print(f"  phase: {len(np.unique(np.round(phase[:n],3)))} 唯一值, std {phase[:n].std():.4f}, "
              f"范围 [{phase[:n].min():.3f},{phase[:n].max():.3f}]")
    else:
        print(f"  phase: 无/解析失败")
    return name, means, dm, f


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    out = {}
    for p in args:
        name, means, dm, f = analyze(p)
        out[name] = dict(means=means, dm=dm)
    # 汇总对比
    if len(out) == 2:
        names = list(out)
        a, b = names
        print("\n--- 对比 ---")
        print(f"  mean 全帧跨度: {out[a]['name'] if False else a}: "
              f"{out[a]['means'].max()-out[a]['means'].min():.4f} vs "
              f"{b}: {out[b]['means'].max()-out[b]['means'].min():.4f}")
        print(f"  逐帧跳变 p99: {a}: {np.percentile(out[a]['dm'],99):.5f} vs "
              f"{b}: {np.percentile(out[b]['dm'],99):.5f}")
