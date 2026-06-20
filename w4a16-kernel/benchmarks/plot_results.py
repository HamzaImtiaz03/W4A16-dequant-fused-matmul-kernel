"""M4 plotting: bar/line charts + a Markdown results table appended to the README.

Run:  python benchmarks/plot_results.py                 # runs the sweep, saves plots
      python benchmarks/plot_results.py --from results.json   # plot a saved sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless (Colab/CI safe)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bench_gemm import run_sweep, to_markdown_table  # noqa: E402  (same dir)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUT_DIR = os.path.join(_REPO_ROOT, "benchmarks", "plots")
_README = os.path.join(_REPO_ROOT, "README.md")
_TABLE_MARKER = "<!-- RESULTS_TABLE -->"


def _label(r: dict) -> str:
    return f"M{r['M']}\n{r['K']}x{r['N']}"


def plot_latency(results: list[dict], path: str) -> None:
    labels = [_label(r) for r in results]
    x = range(len(results))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.4), 5))
    ax.bar([i - width for i in x], [r["fp16_ms"] for r in results], width, label="fp16 cuBLAS")
    ax.bar([i for i in x], [r["dequant_matmul_ms"] for r in results], width, label="dequant + fp16 mm")
    ax.bar([i + width for i in x], [r["w4a16_ms"] for r in results], width, label="W4A16 fused (Triton)")
    ax.set_ylabel("latency (ms, lower=better)")
    ax.set_title("W4A16 vs baselines — latency")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_speedup(results: list[dict], path: str) -> None:
    labels = [_label(r) for r in results]
    x = range(len(results))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.4), 5))
    ax.bar([i - width / 2 for i in x], [r["speedup_vs_fp16"] for r in results], width, label="vs fp16 cuBLAS")
    ax.bar([i + width / 2 for i in x], [r["speedup_vs_dequant"] for r in results], width, label="vs dequant+mm")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("speedup (x, higher=better)")
    ax.set_title("W4A16 fused speedup")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def append_table_to_readme(results: list[dict]) -> None:
    """Insert/replace the results table in README.md at the marker."""
    table = "## Benchmark results\n\n" + to_markdown_table(results)
    block = f"{_TABLE_MARKER}\n{table}{_TABLE_MARKER}\n"
    if not os.path.exists(_README):
        with open(_README, "w") as f:
            f.write(block)
        return
    with open(_README, "r", encoding="utf-8") as f:
        content = f.read()
    if _TABLE_MARKER in content:
        pre, _, rest = content.partition(_TABLE_MARKER)
        _, _, post = rest.partition(_TABLE_MARKER)
        content = pre + block + post
    else:
        content = content.rstrip() + "\n\n" + block
    with open(_README, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_json", default=None, help="load results from JSON instead of running")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            results = json.load(f)
    else:
        import torch

        if not torch.cuda.is_available():
            print("No CUDA GPU — cannot run sweep. Provide --from results.json instead.")
            sys.exit(1)
        results = run_sweep(group_size=args.group_size)

    os.makedirs(_OUT_DIR, exist_ok=True)
    lat = os.path.join(_OUT_DIR, "latency.png")
    spd = os.path.join(_OUT_DIR, "speedup.png")
    plot_latency(results, lat)
    plot_speedup(results, spd)
    append_table_to_readme(results)
    print(f"Saved plots:\n  {lat}\n  {spd}")
    print(f"Updated results table in {_README}")


if __name__ == "__main__":
    main()
