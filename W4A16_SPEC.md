# W4A16 — Math & Memory Layout Specification

> This is the **authoritative spec**. Every kernel (reference, Triton, CUDA) must
> reproduce these formulas and this layout exactly. When in doubt, the worked
> examples below are ground truth.

---

## 1. The operation

Logical layer:

```
Y = X @ W
  X : [M, K]  fp16   activations
  W : [K, N]  fp16   weights      (K = in_features, N = out_features)
  Y : [M, N]  fp16   output       (accumulate in fp32, cast to fp16 at the end)
```

`W` is never materialized in fp16 at inference time. It is stored quantized as
`(qweight, scales, zeros)` and **dequantized on the fly inside the GEMM K-loop**.

---

## 2. Quantization scheme

**Group-wise, asymmetric, 4-bit (uint4), along the K (contraction) dimension.**

- Group size `G` (default **128**). Require `G | K` **and** `8 | K`.
- For weight row `k`, the group index is `gi = k // G`.
- Quantization is independent **per group `gi`, per output column `n`** — i.e. each
  `(gi, n)` pair owns one `scale` and one `zero`.

### 2.1 Derivation of scale and zero-point

Asymmetric quant maps a real value `r` to an integer code `q ∈ [0, 15]` (since
`2^4 − 1 = 15`) and back via:

```
forward :  q     = round(r / scale) + zero
dequant :  r_hat = (q - zero) * scale
```

We want the code range `q ∈ {0, …, 15}` to span the group's observed range
`[gmin, gmax]`:

```
gmax = max over the G rows of group gi, at column n
gmin = min over the G rows of group gi, at column n
```

**Scale** — 16 levels (0..15) give 15 intervals across the range:

```
scale[gi, n] = (gmax - gmin) / 15
```

**Zero-point** — anchor code `q = 0` to `gmin`. From the dequant equation, at
`q = 0`: `r_hat = (0 − zero) · scale = −zero · scale`. Setting this equal to `gmin`:

```
-zero * scale = gmin   =>   zero = -gmin / scale
zero[gi, n] = clamp(round(-gmin / scale[gi, n]), 0, 15)   # integer zero-point
```

**Check the upper end:** at `q = 15`, `r_hat = (15 − zero)·scale = 15·scale − gmin`.
Since `15·scale = gmax − gmin`, this gives `r_hat = gmax`. So the endpoints map
correctly: `gmin → 0`, `gmax → 15`. ✔

### 2.2 Encode and decode

```
q[k, n]    = clamp(round(W[k, n] / scale[gi, n]) + zero[gi, n], 0, 15)   # uint4
W_dq[k, n] = (q[k, n] - zero[gi, n]) * scale[gi, n]                      # fp16
```

### 2.3 Degenerate group guard

If a group is constant (`gmax == gmin`), `scale == 0` → division by zero.
Implementation must guard this: if `scale == 0`, set `scale = 1.0` (the group then
dequantizes to ≈ its constant via `zero`). Constant fp16 groups are rare in real
weights; this guard just keeps the math defined.

### 2.4 Worked example (verify your `quantize_weight`)

One group, `G = 4`, one column. `W = [0.1, -0.3, 0.5, -0.1]`.

```
gmax  = 0.5
gmin  = -0.3
scale = (0.5 - (-0.3)) / 15 = 0.8 / 15 = 0.053333...
zero  = clamp(round(0.3 / 0.053333), 0, 15) = round(5.625) = 6
```

| W      | round(W/scale) | +zero | clamp → q | dequant (q−zero)·scale | abs err |
|--------|----------------|-------|-----------|------------------------|---------|
| 0.1    | 2              | 8     | **8**     | 0.1067                 | 0.007   |
| −0.3   | −6             | 0     | **0**     | −0.3200                | 0.020   |
| 0.5    | 9              | 15    | **15**    | 0.4800                 | 0.020   |
| −0.1   | −2             | 4     | **4**     | −0.1067                | 0.007   |

`q = [8, 0, 15, 4]`. Note `gmin → q=0` and `gmax → q=15` exactly, as derived.
Every error is below `scale/2 ≈ 0.0267` — the expected uniform-quantizer bound.
(The min/max dequantize to −0.32 / 0.48 rather than −0.3 / 0.5 because the integer
**zero-point was rounded**; that residual offset is correct and expected.)

---

## 3. Storage layout

Three tensors replace the fp16 `W`:

```
qweight : int32  [K // 8, N]    8 uint4 codes packed per int32, along K
scales  : fp16   [K // G, N]    one scale per (group, column)
zeros   : fp16   [K // G, N]    integer zero-points (0..15 are exact in fp16)
```

Memory for a 4096×11008 layer:
`W` fp16 ≈ 90 MB → `qweight` int32 ≈ 22 MB (+ tiny scales/zeros). **~4× smaller.**
This is *why* W4A16 wins at decode (M=1): the GEMM is memory-bound on weight reads,
and there are 4× fewer bytes to move.

### 3.1 Int4 packing — bit layout of one `qweight[i, n]` word

`qweight[i, n]` packs the 8 codes for rows `k = 8·i + j`, `j ∈ 0..7`.
**Code `j` lives in nibble `j`, at bits `[4·j : 4·j+4]` (little-endian within the word):**

```
            MSB                                                       LSB
 bit:        31  28 │ 27  24 │ 23  20 │ 19  16 │ 15  12 │ 11   8 │  7   4 │  3   0
            ┌───────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
 nibble #   │   7   │   6    │   5    │   4    │   3    │   2    │   1    │   0    │
 K-row      │ 8i+7  │  8i+6  │  8i+5  │  8i+4  │  8i+3  │  8i+2  │  8i+1  │  8i+0  │
            └───────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
                                                            lowest K-row → lowest nibble
```

Pack / unpack:

```
pack:    qweight[i, n] = Σ over j∈0..7 of ( q[8i+j, n] << (4*j) )
unpack:  q[8i+j, n]    = ( qweight[i, n] >> (4*j) ) & 0xF
```

### 3.2 Worked packing example (verify your `pack_int4` / `unpack_int4`)

Codes for rows `k = 0..7`, one column: `q = [8, 0, 15, 4, 1, 7, 2, 12]`.

```
 8 << 0  = 0x00000008
 0 << 4  = 0x00000000
15 << 8  = 0x00000F00
 4 << 12 = 0x00004000
 1 << 16 = 0x00010000
 7 << 20 = 0x00700000
 2 << 24 = 0x02000000
12 << 28 = 0xC0000000
 ─────────────────────  OR
 packed  = 0xC2714F08
```

Unpacking nibble 0 → `0xC2714F08 & 0xF = 0x8 = 8` ✔
nibble 2 → `(0xC2714F08 >> 8) & 0xF = 0xF = 15` ✔
nibble 7 → `(0xC2714F08 >> 28) & 0xF = 0xC = 12` ✔

> **Critical unsigned/signed note.** `0xC2714F08` has its top bit set, so as a
> *signed* int32 it is negative. The `& 0xF` mask makes the shift's fill bits
> irrelevant — even an *arithmetic* (sign-extending) right shift yields the right
> nibble after masking. **The bug is omitting the mask, or sign-extending the
> 4-bit code itself.** Codes are always unsigned in `[0, 15]`; never sign-extend.

---

## 4. Group indexing invariant (don't get this wrong in the kernel)

Because `G ∈ {64, 128}` are multiples of 8, each packed word's 8 rows
`[8i, 8i+7]` lie **entirely within one group** — a word never straddles a group
boundary. So one `qweight` row ↔ one group's scale/zero (for a given column).

In the tiled GEMM, the K-loop steps by `BLOCK_K`. To keep group lookup to a single
`scale`/`zero` per tile, **require `G % BLOCK_K == 0`** (assert it). Then a `BLOCK_K`
tile is fully inside one group and:

```
gi    = k_start // G                 # k_start is the tile's first K index
scale = scales[gi, n_offsets]        # broadcast across the BLOCK_K rows of the tile
zero  = zeros [gi, n_offsets]
```

If instead `BLOCK_K` could span a boundary, you'd have to index scale/zero
*per row* of the tile — slower and bug-prone. The assert avoids that entirely.

---

## 5. Numeric / dtype contract (T4: fp16 tensor cores, NO bf16)

Inside the K-loop of every kernel:

1. Load `X` tile as **fp16**.
2. Load packed `qweight` as **int32**; unpack nibbles with shifts + `0xF` masks (unsigned).
3. Gather `scale`, `zero` for the tile's group; dequant `W_dq = (q - zero) * scale` in **fp16**.
4. `acc += dot(X_tile, W_dq_tile)` accumulating in **fp32**.
5. After the K-loop, cast `acc` → **fp16** and store to `Y`.

Correctness target vs the pure-PyTorch reference (using the **same**
`qweight/scales/zeros`, never re-quantized):

```
torch.allclose(kernel_out, reference_out, rtol=1e-2, atol=1e-2)
```

---

## 6. Required preconditions (raise clear errors, don't silently miscompute)

- `K % 8 == 0`            (packing requirement)
- `K % G == 0`            (group requirement)
- `G % BLOCK_K == 0`      (clean group indexing in the kernel)
- All kernel-input tensors `.contiguous()` with the expected dtype/shape — assert at entry.
- Compare kernels only against the reference built from the **identical**
  `qweight/scales/zeros`. This isolates *kernel* correctness from *quantization* error.
