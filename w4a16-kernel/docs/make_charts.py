"""Render the README benchmark charts from the measured Tesla T4 results.

The numbers below are the real outputs of ``benchmarks/bench_gemm.py`` on a Colab
free-tier T4 (sm_75), group_size=128. Re-run this script to regenerate the PNGs in
``docs/img/`` after a new benchmark sweep.

    python docs/make_charts.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- measured Tesla T4 results (bench_gemm.py, group_size=128) --------------
DATA = [
    dict(M=1,   K=4096,  N=4096,  w4=0.436,  fp16=0.212, dq=3.168, tflops=0.1, gbps=20, sp_fp16=0.49, sp_dq=7.26),
    dict(M=1,   K=4096,  N=11008, w4=0.767,  fp16=0.543, dq=8.358, tflops=0.1, gbps=31, sp_fp16=0.71, sp_dq=10.90),
    dict(M=1,   K=11008, N=4096,  w4=0.862,  fp16=0.616, dq=8.452, tflops=0.1, gbps=28, sp_fp16=0.71, sp_dq=9.80),
    dict(M=64,  K=4096,  N=4096,  w4=1.021,  fp16=0.159, dq=3.133, tflops=2.1, gbps=10, sp_fp16=0.16, sp_dq=3.07),
    dict(M=256, K=4096,  N=4096,  w4=3.638,  fp16=0.266, dq=3.338, tflops=2.4, gbps=4,  sp_fp16=0.07, sp_dq=0.92),
    dict(M=256, K=4096,  N=11008, w4=9.622,  fp16=1.487, dq=8.856, tflops=2.4, gbps=3,  sp_fp16=0.15, sp_dq=0.92),
    dict(M=256, K=11008, N=4096,  w4=10.764, fp16=1.601, dq=8.914, tflops=2.1, gbps=3,  sp_fp16=0.15, sp_dq=0.83),
]

# Tesla T4 hardware roofs
T4_BW_PEAK = 320.0     # GB/s (GDDR6)
T4_FP16_PEAK = 65.0    # TFLOPS (fp16 tensor cores)

C_FP16 = "#2563EB"     # blue   - fp16 cuBLAS
C_DQ = "#DC2626"       # red    - dequant + matmul
C_W4 = "#16A34A"       # green  - W4A16 fused
C_ACCENT = "#7C3AED"   # purple - accents

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, "img")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.9,
    "axes.edgecolor": "#9CA3AF",
    "axes.linewidth": 0.9,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _labels():
    return [f"M={d['M']}\n{d['K']}×{d['N']}" for d in DATA]


def _annotate(ax, bars, fmt="{:.2f}", fs=8, rot=0):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=fs, rotation=rot,
                    xytext=(0, 2), textcoords="offset points", color="#374151")


def chart_latency(path):
    labels = _labels()
    x = np.arange(len(DATA))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - w, [d["fp16"] for d in DATA], w, label="fp16 cuBLAS (full-precision W)", color=C_FP16)
    b2 = ax.bar(x,      [d["dq"] for d in DATA],   w, label="dequant → fp16 matmul (naive)", color=C_DQ)
    b3 = ax.bar(x + w,  [d["w4"] for d in DATA],   w, label="W4A16 fused (this kernel)", color=C_W4)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 20)
    ax.set_ylabel("latency  (ms, log scale — lower is better)")
    ax.set_title("Latency vs baselines  ·  Tesla T4  ·  group size 128")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    for b in (b1, b2, b3):
        _annotate(ax, b, "{:.2f}", 7)
    ax.axvline(2.5, color="#9CA3AF", ls=":", lw=1.2)
    # region labels pinned to the bottom (clear of the tall bars + legend)
    ax.text(1.0, 0.118, "DECODE  (M=1)", ha="center", fontsize=9.5, color="#6B7280", fontweight="bold")
    ax.text(4.5, 0.118, "PREFILL  (M=64, 256)", ha="center", fontsize=9.5, color="#6B7280", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def chart_speedup(path):
    labels = _labels()
    x = np.arange(len(DATA))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - w / 2, [d["sp_dq"] for d in DATA], w, label="vs naive dequant+matmul", color=C_W4)
    b2 = ax.bar(x + w / 2, [d["sp_fp16"] for d in DATA], w, label="vs fp16 cuBLAS", color=C_FP16)
    ax.axhline(1.0, color="#111827", ls="--", lw=1.2)
    ax.text(len(DATA) - 0.5, 1.06, "break-even (1.0×)", ha="right", fontsize=9, color="#111827")
    ax.set_yscale("log")
    ax.set_ylabel("speedup  (×, log scale — higher is better)")
    ax.set_title("W4A16 fused speedup: wins big vs naive dequant, loses to cuBLAS")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    _annotate(ax, b1, "{:.1f}×", 8); _annotate(ax, b2, "{:.2f}×", 8)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def chart_efficiency(path):
    labels = _labels()
    x = np.arange(len(DATA))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bw = [d["gbps"] for d in DATA]
    bars = ax1.bar(x, bw, color=C_ACCENT)
    ax1.axhline(T4_BW_PEAK, color=C_DQ, ls="--", lw=1.4)
    ax1.text(len(DATA) - 0.5, T4_BW_PEAK * 0.92, f"T4 peak ≈ {T4_BW_PEAK:.0f} GB/s", ha="right", color=C_DQ, fontsize=9)
    ax1.set_ylim(0, T4_BW_PEAK * 1.08)
    ax1.set_ylabel("effective bandwidth (GB/s)")
    ax1.set_title("Memory bandwidth achieved")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8, rotation=0)
    _annotate(ax1, bars, "{:.0f}", 8)

    tf = [d["tflops"] for d in DATA]
    bars2 = ax2.bar(x, tf, color=C_W4)
    ax2.axhline(T4_FP16_PEAK, color=C_DQ, ls="--", lw=1.4)
    ax2.text(len(DATA) - 0.5, T4_FP16_PEAK * 0.92, f"T4 fp16 peak ≈ {T4_FP16_PEAK:.0f} TFLOPS", ha="right", color=C_DQ, fontsize=9)
    ax2.set_ylim(0, T4_FP16_PEAK * 1.08)
    ax2.set_ylabel("compute throughput (TFLOPS)")
    ax2.set_title("Compute throughput achieved")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8, rotation=0)
    _annotate(ax2, bars2, "{:.1f}", 8)

    fig.suptitle("Honest headroom: the kernel reaches only a small fraction of the T4 roofs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=150); plt.close(fig)


def chart_memory(path):
    # 4096 x 4096 weight, group_size 128
    fp16_mb = 33.6
    qweight_mb = 512 * 4096 * 4 / 1e6
    sz_mb = 2 * 32 * 4096 * 2 / 1e6
    packed_total = qweight_mb + sz_mb
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["fp16 weights", "W4A16 packed"], [fp16_mb, packed_total],
                  color=[C_FP16, C_W4], width=0.55)
    # stacked breakdown on the packed bar
    ax.bar(["W4A16 packed"], [sz_mb], bottom=[qweight_mb], color="#86EFAC", width=0.55,
           label="scales + zeros (fp16)")
    ax.bar(["W4A16 packed"], [qweight_mb], color=C_W4, width=0.55, label="qweight (int4 packed)")
    ax.set_ylabel("weight memory (MB)")
    ax.set_title("Weight footprint: 4096×4096, group 128")
    for b, v in zip(bars, [fp16_mb, packed_total]):
        ax.annotate(f"{v:.1f} MB", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color="#111827",
                    xytext=(0, 3), textcoords="offset points")
    ax.annotate("3.76× smaller", (1, packed_total), xytext=(1, fp16_mb * 0.6),
                ha="center", fontsize=12, fontweight="bold", color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.6))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, fp16_mb * 1.18)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    os.makedirs(_IMG, exist_ok=True)
    chart_latency(os.path.join(_IMG, "latency_comparison.png"))
    chart_speedup(os.path.join(_IMG, "speedup_vs_baselines.png"))
    chart_efficiency(os.path.join(_IMG, "roofline_efficiency.png"))
    chart_memory(os.path.join(_IMG, "memory_footprint.png"))
    print("wrote charts to", _IMG)
    for f in sorted(os.listdir(_IMG)):
        print("  ", f)


if __name__ == "__main__":
    main()
