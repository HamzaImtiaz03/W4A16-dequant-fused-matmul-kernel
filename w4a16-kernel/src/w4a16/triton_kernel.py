"""Fused W4A16 dequant->GEMM Triton kernel (the PRIMARY backend).

Computes ``Y = X @ dequant(W)`` where the weight is stored as packed uint4 with
group-wise asymmetric scales/zeros (see :mod:`w4a16.quant`).

Tiling
------
Each program computes one ``[BLOCK_M, BLOCK_N]`` output tile, looping over K in
``BLOCK_K`` steps and accumulating in fp32. Inside the K loop it:
  1. loads the ``[BLOCK_M, BLOCK_K]`` fp16 activation tile,
  2. loads packed int32 weights and UNPACKS nibbles with shift + ``0xF`` mask
     (unsigned; mask is applied *after* the shift so signed >> is harmless),
  3. loads the group's ``scale``/``zero`` row vectors,
  4. dequantizes ``w = (q - zero) * scale`` in fp16,
  5. accumulates ``tl.dot(x, w)`` into the fp32 accumulator.

Group-boundary safety
----------------------
We require ``group_size % BLOCK_K == 0`` (asserted in the wrapper for every autotune
config). Combined with ``G | K`` and ``8 | K`` this guarantees ``BLOCK_K | K`` (so no
K-masking is needed) AND that each ``BLOCK_K`` slice lies entirely within ONE group.
Hence the group index ``gi = k0 // G`` is constant across a K-iteration and we load a
single ``[BLOCK_N]`` scale/zero vector per step instead of per-row gathers.

T4 notes
--------
fp16 inputs + fp32 accumulate (Turing has fp16 tensor cores, no bf16). All autotune
``BLOCK_K`` values divide the supported group sizes {64, 128}.
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl

__all__ = ["w4a16_matmul"]


def _autotune_configs() -> list["triton.Config"]:
    """Small (6-config) autotune space sized for a T4.

    Every BLOCK_K is a multiple of 8 and divides the supported group sizes {64, 128},
    so the single-group-per-K-block invariant holds for all of them. The first configs
    favour the memory-bound decode regime (small M); the later ones favour prefill.
    """
    return [
        # BLOCK_M, BLOCK_N, BLOCK_K            warps  stages
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ]


_CONFIGS = _autotune_configs()


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "group_size"])
@triton.jit
def _w4a16_gemm_kernel(
    x_ptr,            # *fp16  [M, K]
    qw_ptr,           # *int32 [K//8, N]
    scale_ptr,        # *fp16  [K//G, N]
    zero_ptr,         # *fp16  [K//G, N]
    y_ptr,            # *fp16  [M, N]
    M, N, K, group_size,
    stride_xm, stride_xk,
    stride_qk, stride_qn,
    stride_sk, stride_sn,
    stride_zk, stride_zn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Nibble shift per K-row within the block. k0 is a multiple of 8 (BLOCK_K % 8 == 0),
    # so (k0 + offs_k) % 8 == offs_k % 8 and the shift is loop-invariant.
    shift = (offs_k % 8) * 4                           # [BLOCK_K] int32

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k = tl.cdiv(K, BLOCK_K)
    for k_iter in range(0, num_k):
        k0 = k_iter * BLOCK_K
        gi = k0 // group_size   # constant across this block (group_size % BLOCK_K == 0)

        # --- activations: [BLOCK_M, BLOCK_K] fp16 (K never needs masking: BLOCK_K | K) ---
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

        # --- packed weights -> unpacked uint4 nibbles: [BLOCK_K, BLOCK_N] ---
        prk = (k0 + offs_k) // 8                       # packed row index per K-row
        qw_ptrs = qw_ptr + (prk[:, None] * stride_qk + offs_n[None, :] * stride_qn)
        qpacked = tl.load(qw_ptrs, mask=mask_n[None, :], other=0)
        q = (qpacked >> shift[:, None]) & 0xF          # unsigned nibble, 0..15

        # --- per-group scale/zero row vectors: [BLOCK_N] fp16 ---
        s_ptrs = scale_ptr + (gi * stride_sk + offs_n * stride_sn)
        z_ptrs = zero_ptr + (gi * stride_zk + offs_n * stride_zn)
        scale = tl.load(s_ptrs, mask=mask_n, other=0.0)
        zero = tl.load(z_ptrs, mask=mask_n, other=0.0)

        # --- dequant in fp16: w = (q - zero) * scale ---
        w = (q.to(tl.float16) - zero[None, :]) * scale[None, :]   # [BLOCK_K, BLOCK_N] fp16

        # --- fp32 accumulate ---
        acc += tl.dot(x, w)

        x_ptrs += BLOCK_K * stride_xk

    y = acc.to(tl.float16)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def w4a16_matmul(
    x: Tensor,
    qweight: Tensor,
    scales: Tensor,
    zeros: Tensor,
    group_size: int,
    out: Tensor | None = None,
) -> Tensor:
    """Fused W4A16 GEMM: ``Y = X @ dequant(qweight)``.

    Args:
        x: activations [M, K] fp16, contiguous, on CUDA.
        qweight: int32 [K//8, N], contiguous, on CUDA.
        scales: fp16 [K//G, N], contiguous, on CUDA.
        zeros: fp16 [K//G, N], contiguous, on CUDA.
        group_size: G. Must satisfy G | K, 8 | K, and (for every autotune config)
            BLOCK_K | G.
        out: optional preallocated fp16 [M, N] output.

    Returns:
        fp16 tensor [M, N].
    """
    assert x.is_cuda, "x must be on CUDA (run this in Colab with a GPU runtime)"
    assert qweight.is_cuda and scales.is_cuda and zeros.is_cuda, "all inputs must be on CUDA"
    assert x.dtype == torch.float16, f"x must be fp16, got {x.dtype}"
    assert qweight.dtype == torch.int32, f"qweight must be int32, got {qweight.dtype}"
    assert scales.dtype == torch.float16, f"scales must be fp16, got {scales.dtype}"
    assert zeros.dtype == torch.float16, f"zeros must be fp16, got {zeros.dtype}"
    assert x.dim() == 2 and qweight.dim() == 2, "x and qweight must be 2D"

    M, K = x.shape
    Kp, N = qweight.shape
    assert Kp * 8 == K, f"qweight rows (K//8)={Kp} incompatible with x K={K}"
    assert K % 8 == 0, f"K={K} must be divisible by 8"
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size
    assert scales.shape == (num_groups, N), f"scales shape {tuple(scales.shape)} != {(num_groups, N)}"
    assert zeros.shape == (num_groups, N), f"zeros shape {tuple(zeros.shape)} != {(num_groups, N)}"

    assert x.is_contiguous(), "x must be contiguous"
    assert qweight.is_contiguous(), "qweight must be contiguous"
    assert scales.is_contiguous(), "scales must be contiguous"
    assert zeros.is_contiguous(), "zeros must be contiguous"

    # Critical safety check: every config that autotune may *run and time* must keep
    # each BLOCK_K slice inside one group, else group indexing is silently wrong.
    for cfg in _CONFIGS:
        bk = cfg.kwargs["BLOCK_K"]
        assert group_size % bk == 0, (
            f"autotune config BLOCK_K={bk} does not divide group_size={group_size}; "
            f"choose group_size as a multiple of {bk} or prune the config list"
        )

    if out is None:
        y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    else:
        assert out.shape == (M, N) and out.dtype == torch.float16 and out.is_contiguous()
        y = out

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _w4a16_gemm_kernel[grid](
        x, qweight, scales, zeros, y,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        y.stride(0), y.stride(1),
    )
    return y
