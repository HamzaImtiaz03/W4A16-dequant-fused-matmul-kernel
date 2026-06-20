"""Pure-PyTorch ground truth + benchmark baselines for W4A16 GEMM.

``reference_w4a16`` is the CORRECTNESS ORACLE for every kernel: it dequantizes the
exact same (qweight, scales, zeros) the kernel consumes and does an fp32-accumulated
matmul. Kernels are always compared against this oracle using the SAME quantized
tensors, which isolates *kernel* correctness from *quantization* error.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .quant import dequantize_weight

__all__ = ["reference_w4a16", "dequant_then_matmul", "fp16_matmul"]


def reference_w4a16(
    X: Tensor, qweight: Tensor, scales: Tensor, zeros: Tensor, group_size: int
) -> Tensor:
    """Correctness oracle: dequant -> fp32-accumulated matmul -> fp16.

    Args:
        X: activations [M, K] fp16.
        qweight: int32 [K//8, N].
        scales:  fp16  [K//G, N].
        zeros:   fp16  [K//G, N].
        group_size: G.

    Returns:
        Y = X @ dequant(qweight) as fp16, accumulated in fp32.
    """
    assert X.dtype == torch.float16, f"X must be fp16, got {X.dtype}"
    assert X.dim() == 2, f"X must be 2D [M, K], got {tuple(X.shape)}"
    M, K = X.shape
    W_dq = dequantize_weight(qweight, scales, zeros, group_size)   # fp16 [K, N]
    assert W_dq.shape[0] == K, f"K mismatch: X K={K} vs weight K={W_dq.shape[0]}"
    # fp32 accumulate (cast inputs to fp32 so the matmul reduction is full precision).
    Y = torch.matmul(X.to(torch.float32), W_dq.to(torch.float32))
    return Y.to(torch.float16)


def dequant_then_matmul(
    X: Tensor, qweight: Tensor, scales: Tensor, zeros: Tensor, group_size: int
) -> Tensor:
    """Baseline: materialize full fp16 weights, then a standard cuBLAS fp16 matmul.

    This is the "naive" path the fused kernel competes against: it must read the full
    fp16 weight matrix (~4x more bytes than packed int4) from DRAM.
    """
    assert X.dtype == torch.float16
    W_dq = dequantize_weight(qweight, scales, zeros, group_size)   # fp16 [K, N]
    return torch.matmul(X, W_dq)


def fp16_matmul(X: Tensor, W: Tensor) -> Tensor:
    """Baseline: full-precision fp16 cuBLAS matmul (no quantization at all).

    Upper bound for the compute-bound regime; weights are read at full fp16 width.

    Args:
        X: [M, K] fp16.
        W: [K, N] fp16 (already in logical [K, N] layout, NOT nn.Linear's [N, K]).
    """
    assert X.dtype == torch.float16 and W.dtype == torch.float16
    assert X.shape[1] == W.shape[0], f"K mismatch: X {tuple(X.shape)} vs W {tuple(W.shape)}"
    return torch.matmul(X, W)
