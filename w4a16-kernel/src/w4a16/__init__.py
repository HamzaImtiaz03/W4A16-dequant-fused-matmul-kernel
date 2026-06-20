"""W4A16 dequant-fused matmul kernel.

4-bit group-wise asymmetric weight quantization + fp16 activations, with a fused
dequant->GEMM Triton kernel (primary) and an optional CUDA SIMT kernel (stretch).

Public API:
    quantize_weight / dequantize_weight / pack_int4 / unpack_int4   (quant.py)
    reference_w4a16 / dequant_then_matmul / fp16_matmul             (reference.py)
    w4a16_matmul                                                    (triton_kernel.py)
    W4A16Linear                                                     (linear.py)

The quant/reference helpers are pure PyTorch and import without Triton/CUDA so the
quantization layer can be exercised on CPU. The Triton path is imported lazily and
will be unavailable if `triton` is not installed.
"""
from __future__ import annotations

from .quant import quantize_weight, dequantize_weight, pack_int4, unpack_int4
from .reference import reference_w4a16, dequant_then_matmul, fp16_matmul

__all__ = [
    "quantize_weight",
    "dequantize_weight",
    "pack_int4",
    "unpack_int4",
    "reference_w4a16",
    "dequant_then_matmul",
    "fp16_matmul",
]

# Triton backend is optional at import time (e.g. CPU-only machine for quant tests).
try:  # pragma: no cover - depends on environment
    from .triton_kernel import w4a16_matmul
    from .linear import W4A16Linear

    __all__ += ["w4a16_matmul", "W4A16Linear"]
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False
