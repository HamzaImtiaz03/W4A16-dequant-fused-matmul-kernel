"""M4 benchmarks: latency / TFLOPS / effective bandwidth vs baselines.

Compares the fused Triton W4A16 kernel against:
  (a) fp16 cuBLAS matmul with full fp16 weights  -> `fp16_matmul`
  (b) dequant-then-fp16-matmul                    -> `dequant_then_matmul`

Covers the decode regime (M=1, memory-bound) and the prefill regime (M=256,
approaching the fp16 compute bound). Timing uses CUDA events with warmup + median.

Run:  python benchmarks/bench_gemm.py            # default sweep, prints table
      python benchmarks/bench_gemm.py --json results.json   # also dump JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from w4a16.quant import quantize_weight
from w4a16.reference import dequant_then_matmul, fp16_matmul
from w4a16.triton_kernel import w4a16_matmul


# Default sweep: (M, K, N). Decode (M=1) and prefill (M in {64,256}).
DEFAULT_SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (64, 4096, 4096),
    (256, 4096, 4096),
    (256, 4096, 11008),
    (256, 11008, 4096),
]
DEFAULT_GROUP_SIZE = 128


def _time_ms(fn, warmup: int = 25, iters: int = 100) -> float:
    """Median wall-clock latency in ms via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _bytes_w4a16(M, K, N, group_size) -> int:
    ng = K // group_size
    x = M * K * 2          # fp16 activations
    qw = (K // 8) * N * 4  # int32 packed weights
    sz = 2 * ng * N * 2    # scales + zeros fp16
    y = M * N * 2          # fp16 output
    return x + qw + sz + y


def _bytes_fp16(M, K, N) -> int:
    return M * K * 2 + K * N * 2 + M * N * 2


def benchmark_shape(M: int, K: int, N: int, group_size: int = DEFAULT_GROUP_SIZE) -> dict:
    """Benchmark one (M, K, N) and return a results dict."""
    device = "cuda"
    X = torch.randn(M, K, dtype=torch.float16, device=device)
    W = torch.randn(K, N, dtype=torch.float16, device=device)  # logical [K, N]
    qweight, scales, zeros = quantize_weight(W, group_size)
    qweight, scales, zeros = qweight.to(device), scales.to(device), zeros.to(device)

    t_fp16 = _time_ms(lambda: fp16_matmul(X, W))
    t_dq = _time_ms(lambda: dequant_then_matmul(X, qweight, scales, zeros, group_size))
    t_w4 = _time_ms(lambda: w4a16_matmul(X, qweight, scales, zeros, group_size))

    flops = 2.0 * M * N * K
    res = {
        "M": M, "K": K, "N": N, "group_size": group_size,
        "fp16_ms": t_fp16,
        "dequant_matmul_ms": t_dq,
        "w4a16_ms": t_w4,
        "w4a16_tflops": flops / (t_w4 * 1e-3) / 1e12,
        "fp16_tflops": flops / (t_fp16 * 1e-3) / 1e12,
        "w4a16_gbps": _bytes_w4a16(M, K, N, group_size) / (t_w4 * 1e-3) / 1e9,
        "fp16_gbps": _bytes_fp16(M, K, N) / (t_fp16 * 1e-3) / 1e9,
        "speedup_vs_fp16": t_fp16 / t_w4,
        "speedup_vs_dequant": t_dq / t_w4,
    }
    return res


def run_sweep(shapes=DEFAULT_SHAPES, group_size: int = DEFAULT_GROUP_SIZE) -> list[dict]:
    assert torch.cuda.is_available(), "CUDA GPU required for benchmarks"
    results = []
    for (M, K, N) in shapes:
        r = benchmark_shape(M, K, N, group_size)
        results.append(r)
        print(
            f"M={M:<4} K={K:<6} N={N:<6} | "
            f"w4a16 {r['w4a16_ms']:.3f}ms ({r['w4a16_tflops']:.1f} TF, {r['w4a16_gbps']:.0f} GB/s) | "
            f"fp16 {r['fp16_ms']:.3f}ms | dq {r['dequant_matmul_ms']:.3f}ms | "
            f"speedup vs fp16 {r['speedup_vs_fp16']:.2f}x, vs dequant {r['speedup_vs_dequant']:.2f}x"
        )
    return results


def to_markdown_table(results: list[dict]) -> str:
    """Render results as a Markdown table (also used by plot_results / README)."""
    header = (
        "| M | K | N | w4a16 (ms) | fp16 (ms) | dequant+mm (ms) | "
        "w4a16 TFLOPS | w4a16 GB/s | speedup vs fp16 | speedup vs dequant |\n"
        "|---|---|---|-----------|-----------|-----------------|"
        "--------------|------------|-----------------|--------------------|\n"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['M']} | {r['K']} | {r['N']} | {r['w4a16_ms']:.3f} | {r['fp16_ms']:.3f} | "
            f"{r['dequant_matmul_ms']:.3f} | {r['w4a16_tflops']:.1f} | {r['w4a16_gbps']:.0f} | "
            f"{r['speedup_vs_fp16']:.2f}x | {r['speedup_vs_dequant']:.2f}x |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="W4A16 GEMM benchmark")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--json", type=str, default=None, help="optional path to dump results JSON")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available — cannot benchmark. Use a Colab GPU runtime.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    results = run_sweep(group_size=args.group_size)
    print("\n" + to_markdown_table(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
