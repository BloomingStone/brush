#!/usr/bin/env python3

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "pydicom"]
# ///

"""绘制 DCM 中提取的心脏相位 (0071,1010) 与角度。

用法:
  uv run --with pydicom --with numpy --with matplotlib python3 tools/plot_phase.py \
    RXA_pig_with_phase.dcm [rotate_dsa_raw.dcm ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def load_meta(path):
    d = pydicom.dcmread(path, stop_before_pixels=True)
    n = int(d.get("NumberOfFrames", 0)) or int(d.get("Rows", 0))
    ph = d[(0x0071, 0x1010)].value
    if hasattr(ph, "value"):
        ph = ph.value
    if isinstance(ph, (bytes, bytearray)):
        ph = np.frombuffer(ph, dtype="<f4")
    ph = np.asarray(ph, dtype=np.float32).reshape(-1)[:n]

    alpha = None
    if (0x0018, 0x1520) in d:
        v = d[(0x0018, 0x1520)].value
        if hasattr(v, "__iter__") and not isinstance(v, (bytes, bytearray)):
            alpha = np.array([float(x) for x in v])
        elif isinstance(v, (bytes, bytearray)):
            alpha = np.frombuffer(v, dtype="<f4")
    if alpha is None and (0x0018, 0x1147) in d:
        start = float(d[(0x0018, 0x1147)].value)
        inc = float(d[(0x0018, 0x1520)].value)
        alpha = np.linspace(start, start + inc * n, n)
    alpha = np.asarray(alpha, dtype=np.float32)[:n] if alpha is not None else None
    return n, ph, alpha


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    n = None
    for path in sys.argv[1:]:
        cnt, ph, alpha = load_meta(path)
        n = cnt
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fr = np.arange(cnt)

        ax[0].plot(fr, ph, "-", lw=1, label="phase (0071,1010)")
        ax[0].set_ylabel("cardiac phase [0,1]")
        ax[0].set_title(f"{Path(path).name}  (n={cnt})")
        ax[0].grid(alpha=0.3)
        ax[0].legend(loc="upper left")

        if alpha is not None:
            ax2 = ax[0].twinx()
            ax2.plot(fr, alpha, "-", lw=1, color="tab:orange", alpha=0.6, label="alpha angle")
            ax2.set_ylabel("alpha (deg)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax[1].plot(fr, ph, "-", lw=0.5, color="tab:blue")
        ax[1].set_xlabel("frame index")
        ax[1].set_ylabel("phase")
        ax[1].grid(alpha=0.3)
        # 相位回绕标记
        wraps = np.where(np.diff(ph) < -0.3)[0]
        for w in wraps:
            ax[0].axvline(w, color="red", ls="--", alpha=0.5)
            ax[1].axvline(w, color="red", ls="--", alpha=0.5)
        ax[0].set_title(f"{Path(path).name} — 回绕 {len(wraps)} 次", loc="right", fontsize=10)

        out = Path(path).with_suffix(".phase.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}  (wraps={len(wraps)})")
        plt.close(fig)


if __name__ == "__main__":
    main()
