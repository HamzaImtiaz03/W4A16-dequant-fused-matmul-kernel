"""Group-wise asymmetric 4-bit weight quantization + int4 packing utilities.

Logical weight layout
---------------------
    W : [K, N] fp16   (K = in_features = contraction dim, N = out_features)

Group-wise ASYMMETRIC 4-bit quantization along K (default group size G = 128).
Requires G | K and 8 | K. Group index for K-row k is ``gi = k // G``.

Math (per group gi, per output column n)
----------------------------------------
    gmax        = max(W over the rows of group gi, column n)
    gmin        = min(W over the rows of group gi, column n)
    scale[gi,n] = (gmax - gmin) / 15.0                       # 15 = 2**4 - 1
    zero[gi,n]  = clamp(round(-gmin / scale[gi,n]), 0, 15)   # integer zero-point
    q[k,n]      = clamp(round(W[k,n] / scale[gi,n]) + zero[gi,n], 0, 15)   # uint4
Dequant:
    W_dq[k,n]   = (q[k,n] - zero[gi,n]) * scale[gi,n]

Storage / packing layout
------------------------
    qweight : int32 [K//8, N]   — 8 consecutive-in-K uint4 values per int32.
              qweight[i, n] holds rows k = 8*i + j for j in 0..7, with nibble j
              at bits [4*j : 4*j+4].  UNSIGNED nibbles (mask 0xF, no sign extension).
    scales  : fp16  [K//G, N]
    zeros   : fp16  [K//G, N]   — integer zero-points (0..15 are exact in fp16).

All ops are vectorized; the only Python loops are over the 8 fixed nibble positions
(never over tensor elements).
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["quantize_weight", "dequantize_weight", "pack_int4", "unpack_int4"]

_NIBBLES_PER_INT32 = 8
_MAX_UINT4 = 15  # 2**4 - 1


# --------------------------------------------------------------------------- #
# Packing                                                                      #
# --------------------------------------------------------------------------- #
def pack_int4(q: Tensor) -> Tensor:
    """Pack a ``[K, N]`` tensor of uint4 values (0..15) into int32 ``[K//8, N]``.

    Row ``k = 8*i + j`` maps to nibble ``j`` (bits ``[4*j : 4*j+4]``) of
    ``qweight[i, n]``. Nibbles are unsigned.

    Args:
        q: integer tensor, shape [K, N], values in [0, 15], K divisible by 8.

    Returns:
        int32 tensor of shape [K//8, N].
    """
    assert q.dim() == 2, f"pack_int4 expects 2D [K, N], got {tuple(q.shape)}"
    K, N = q.shape
    assert K % _NIBBLES_PER_INT32 == 0, f"K={K} must be divisible by 8 to pack int4"
    qi = q.to(torch.int32)
    # Guard against sign / range bugs (cheap one-time check at quantization time).
    lo, hi = int(qi.min()), int(qi.max())
    assert 0 <= lo and hi <= _MAX_UINT4, f"uint4 values must be in [0,15], got [{lo},{hi}]"

    qi = qi.reshape(K // _NIBBLES_PER_INT32, _NIBBLES_PER_INT32, N)  # [K//8, 8, N]
    packed = torch.zeros((K // _NIBBLES_PER_INT32, N), dtype=torch.int32, device=q.device)
    for j in range(_NIBBLES_PER_INT32):
        # Each nibble occupies a disjoint 4-bit field -> OR == add, no carries.
        packed |= (qi[:, j, :] & 0xF) << (4 * j)
    return packed.contiguous()


def unpack_int4(qweight: Tensor, K: int | None = None) -> Tensor:
    """Inverse of :func:`pack_int4`.

    Args:
        qweight: int32 tensor [K//8, N].
        K: optional explicit K (defaults to ``qweight.shape[0] * 8``).

    Returns:
        int32 tensor [K, N] with values in [0, 15].
    """
    assert qweight.dim() == 2, f"unpack_int4 expects 2D [K//8, N], got {tuple(qweight.shape)}"
    assert qweight.dtype == torch.int32, f"qweight must be int32, got {qweight.dtype}"
    Kp, N = qweight.shape
    if K is None:
        K = Kp * _NIBBLES_PER_INT32
    assert K == Kp * _NIBBLES_PER_INT32, f"K={K} incompatible with qweight rows {Kp}"

    out = torch.empty((Kp, _NIBBLES_PER_INT32, N), dtype=torch.int32, device=qweight.device)
    for j in range(_NIBBLES_PER_INT32):
        # Arithmetic >> may sign-extend on int32, but masking AFTER the shift keeps
        # exactly the 4 target bits, so the result is the correct unsigned nibble.
        out[:, j, :] = (qweight >> (4 * j)) & 0xF
    return out.reshape(K, N)


# --------------------------------------------------------------------------- #
# Quantize / dequantize                                                        #
# --------------------------------------------------------------------------- #
def quantize_weight(W: Tensor, group_size: int = 128) -> tuple[Tensor, Tensor, Tensor]:
    """Group-wise asymmetric 4-bit quantization of ``W`` along K.

    Args:
        W: weight tensor [K, N] (fp16 or fp32). K = in_features, N = out_features.
        group_size: G, number of K-rows per quantization group. Requires G | K, 8 | K.

    Returns:
        (qweight, scales, zeros):
            qweight : int32 [K//8, N]  (packed uint4)
            scales  : fp16  [K//G, N]
            zeros   : fp16  [K//G, N]  (integer zero-points stored as fp16)
    """
    assert W.dim() == 2, f"quantize_weight expects 2D [K, N], got {tuple(W.shape)}"
    K, N = W.shape
    if K % _NIBBLES_PER_INT32 != 0:
        raise ValueError(f"K={K} must be divisible by 8 (got remainder {K % 8})")
    if K % group_size != 0:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")
    G = group_size
    num_groups = K // G

    # All quantization math in fp32 for stable scale/zero computation.
    Wf = W.detach().to(torch.float32)
    Wg = Wf.reshape(num_groups, G, N)                # [ng, G, N]
    gmax = Wg.amax(dim=1)                            # [ng, N]
    gmin = Wg.amin(dim=1)                            # [ng, N]

    scale = (gmax - gmin) / _MAX_UINT4               # [ng, N] fp32
    # Constant groups (range == 0) would divide by zero -> use scale 1.0 there.
    # (For an all-equal group, e.g. all zeros, this dequantizes back exactly.)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    zero = torch.clamp(torch.round(-gmin / scale), 0, _MAX_UINT4)  # [ng, N] fp32 int values

    scale_e = scale.reshape(num_groups, 1, N)
    zero_e = zero.reshape(num_groups, 1, N)
    q = torch.clamp(torch.round(Wg / scale_e) + zero_e, 0, _MAX_UINT4)  # [ng, G, N]
    q = q.reshape(K, N).to(torch.int32)

    qweight = pack_int4(q)
    scales = scale.to(torch.float16).contiguous()
    zeros = zero.to(torch.float16).contiguous()
    return qweight, scales, zeros


def dequantize_weight(
    qweight: Tensor, scales: Tensor, zeros: Tensor, group_size: int
) -> Tensor:
    """Dequantize packed uint4 weights back to fp16 ``[K, N]``.

    Computes ``W_dq = (q - zero) * scale`` in fp16, mirroring the fused kernel so the
    PyTorch oracle and the Triton/CUDA kernels share identical dequant arithmetic.

    Args:
        qweight: int32 [K//8, N].
        scales:  fp16  [K//G, N].
        zeros:   fp16  [K//G, N].
        group_size: G.

    Returns:
        fp16 tensor [K, N].
    """
    assert qweight.dtype == torch.int32, f"qweight must be int32, got {qweight.dtype}"
    Kp, N = qweight.shape
    K = Kp * _NIBBLES_PER_INT32
    G = group_size
    assert K % G == 0, f"K={K} must be divisible by group_size={G}"
    num_groups = K // G
    assert scales.shape == (num_groups, N), f"scales shape {tuple(scales.shape)} != {(num_groups, N)}"
    assert zeros.shape == (num_groups, N), f"zeros shape {tuple(zeros.shape)} != {(num_groups, N)}"

    q = unpack_int4(qweight, K).to(torch.float16)            # [K, N] in 0..15
    qg = q.reshape(num_groups, G, N)
    s = scales.to(torch.float16).reshape(num_groups, 1, N)
    z = zeros.to(torch.float16).reshape(num_groups, 1, N)
    w = (qg - z) * s                                         # fp16 arithmetic
    return w.reshape(K, N).to(torch.float16).contiguous()
