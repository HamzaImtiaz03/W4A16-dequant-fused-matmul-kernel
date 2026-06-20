# W4A16 dequant-fused matmul kernel

A fused **4-bit weight / fp16 activation** GEMM — the core operation behind AWQ/GPTQ
LLM inference. Weights are stored group-wise quantized to 4 bits and **dequantized
inside the matmul kernel** (no full-precision weight ever touches DRAM), so the
memory-bound decode path reads ~4× less weight data.

- **Primary backend:** a fused Triton kernel (`src/w4a16/triton_kernel.py`).
- **Stretch backend:** a CUDA SIMT kernel (`src/w4a16/cuda/`), validated against the
  same oracle.
- **Target:** Google Colab **free tier T4** (`sm_75`, Turing — fp16 tensor cores, **no
  bf16**). The code detects the actual compute capability at runtime and adapts.

> Built and run on Colab. fp16 compute, **fp32 accumulation** everywhere.

---

## Math spec

Logical layer: `Y = X @ W`

| tensor | shape    | dtype | meaning                 |
|--------|----------|-------|-------------------------|
| `X`    | `[M, K]` | fp16  | activations             |
| `W`    | `[K, N]` | fp16  | weights (K=in, N=out)   |
| `Y`    | `[M, N]` | fp16  | output (fp32 accumulate)|

**Group-wise asymmetric 4-bit quantization along K** (group size `G`, default 128;
require `G | K` and `8 | K`). Group index of K-row `k` is `gi = k // G`.

Per group `gi`, per output column `n`:

```
gmax        = max(W[group gi, n])
gmin        = min(W[group gi, n])
scale[gi,n] = (gmax - gmin) / 15.0                      # 15 = 2^4 - 1
zero[gi,n]  = clamp(round(-gmin / scale[gi,n]), 0, 15)  # integer zero-point
q[k,n]      = clamp(round(W[k,n] / scale[gi,n]) + zero[gi,n], 0, 15)   # uint4 in [0,15]
```

Dequant: `W_dq[k,n] = (q[k,n] - zero[gi,n]) * scale[gi,n]`.

---

## Storage / packing layout

```
qweight : int32 [K//8, N]   scales : fp16 [K//G, N]   zeros : fp16 [K//G, N]
```

`qweight[i, n]` packs **8 consecutive-in-K** uint4 values (rows `k = 8*i + j`,
`j = 0..7`). Nibble `j` lives in bits `[4*j : 4*j+4]`. Nibbles are **unsigned**
(mask `0xF`; no sign extension).

```
 one int32 = qweight[i, n]   (K-rows 8*i .. 8*i+7 for column n)

  bit:  31      28 27      24 ...  11       8 7        4 3        0
        +---------+---------+-----+---------+---------+---------+---------+
        | nib j=7 | nib j=6 | ... | nib j=2 | nib j=1 | nib j=0 |   ...   |
        +---------+---------+-----+---------+---------+---------+---------+
          k=8i+7    k=8i+6           k=8i+2    k=8i+1    k=8i+0

  value(k=8i+j, n) = (qweight[i,n] >> (4*j)) & 0xF        # unsigned
```

`scales`/`zeros` are indexed by `[k // G, n]`. `zeros` stores integer zero-points as
fp16 (0..15 are exact in fp16).

### Why the kernel's group indexing is safe

The Triton kernel requires `group_size % BLOCK_K == 0` (asserted for every autotune
config). With `G | K` and `8 | K`, this guarantees `BLOCK_K | K` (no K-masking) **and**
that each `BLOCK_K` slice of the contraction lies entirely inside one group — so the
group index is constant across a K-step and we load one `[BLOCK_N]` scale/zero vector
per step instead of per-row gathers. The CUDA stretch kernel dequantizes per element
and so has no such alignment requirement.

---

## Layout of the repo

```
w4a16-kernel/
├── src/w4a16/
│   ├── quant.py          # quantize / pack-int4 / unpack / dequant (pure PyTorch)
│   ├── reference.py      # dequant->matmul oracle + fp16 / dequant-then-mm baselines
│   ├── triton_kernel.py  # fused W4A16 GEMM in Triton + autotune   <- PRIMARY
│   ├── linear.py         # W4A16Linear nn.Module (drop-in for nn.Linear)
│   ├── cuda_kernel.py    # builds/loads the CUDA extension (load_inline)  <- STRETCH
│   └── cuda/{w4a16_gemm.cu, bindings.cpp}
├── tests/                # test_quant.py (CPU-ok) + test_correctness.py (GPU)
├── benchmarks/           # bench_gemm.py + plot_results.py
├── notebooks/w4a16_colab.ipynb   # top-to-bottom Colab entry point
└── scripts/setup_colab.sh
```

---

## Quickstart (Colab)

```python
# Runtime > Change runtime type > T4 GPU, then:
import sys; sys.path.insert(0, "w4a16-kernel/src")
import torch
from w4a16 import quantize_weight, reference_w4a16, w4a16_matmul, W4A16Linear

K, N, M, G = 4096, 4096, 1, 128
W = torch.randn(K, N, dtype=torch.float16, device="cuda")
X = torch.randn(M, K, dtype=torch.float16, device="cuda")

qweight, scales, zeros = quantize_weight(W, G)
y_kernel = w4a16_matmul(X, qweight, scales, zeros, G)        # fused Triton kernel
y_ref    = reference_w4a16(X, qweight, scales, zeros, G)     # fp32-accumulate oracle
print("max abs err:", (y_kernel.float() - y_ref.float()).abs().max().item())
```

Run the tests / benchmarks:

```bash
bash scripts/setup_colab.sh
python -m pytest tests/ -v                  # quant (CPU-ok) + kernel correctness (GPU)
python benchmarks/bench_gemm.py             # latency / TFLOPS / GB/s vs baselines
python benchmarks/plot_results.py           # plots + fills the table below
```

---

## Key insight

- **Decode (small M, e.g. M=1):** memory-bound. W4A16 reads packed int4 weights
  (~4× fewer bytes than fp16), so it wins on latency even though it does extra unpack
  + dequant math.
- **Prefill (large M):** compute-bound. The GEMM approaches the fp16 tensor-core bound;
  the weight-bandwidth advantage shrinks and W4A16 trends toward fp16 cuBLAS.

<!-- RESULTS_TABLE -->
*(Run `python benchmarks/plot_results.py` on the T4 to fill in this table and the plots
under `benchmarks/plots/`.)*
<!-- RESULTS_TABLE -->

---

## Milestone status

| # | milestone | backend | how to verify |
|---|-----------|---------|----------------|
| M0 | env setup + capability probe | — | `bash scripts/setup_colab.sh` |
| M1 | quant + pack/unpack | PyTorch | `pytest tests/test_quant.py` |
| M2 | reference oracle + baselines | PyTorch | imported by tests/benchmarks |
| M3 | **fused Triton kernel** | Triton | `pytest tests/test_correctness.py` |
| M4 | `W4A16Linear` + benchmarks | Triton | `python benchmarks/bench_gemm.py` |
| M5 | CUDA SIMT kernel (stretch) | CUDA | `pytest -k cuda` (auto-skips if no nvcc) |
| M6 | end-to-end Colab notebook | — | `notebooks/w4a16_colab.ipynb` |
