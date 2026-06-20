"""Build/load the W4A16 CUDA SIMT extension via ``load_inline`` (STRETCH goal).

This compiles ``cuda/w4a16_gemm.cu`` + ``cuda/bindings.cpp`` for the *detected* GPU
arch (fallback ``sm_75`` / Turing T4). The compiled module is cached on disk by
torch and memoized in-process, so repeated Colab cells don't recompile.

Only attempt this after the Triton path passes all correctness tests.
"""
from __future__ import annotations

import os

import torch
from torch import Tensor

__all__ = ["w4a16_matmul_cuda", "load_cuda_module", "is_available"]

_MODULE = None  # memoized compiled extension
_HERE = os.path.dirname(__file__)
_CUDA_DIR = os.path.join(_HERE, "cuda")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def is_available() -> bool:
    return torch.cuda.is_available()


def load_cuda_module(verbose: bool = True):
    """Compile (once) and return the CUDA extension module.

    Sets ``TORCH_CUDA_ARCH_LIST`` to the detected capability so nvcc targets the
    actual GPU (e.g. ``7.5`` for a T4) instead of building for every arch.
    """
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to build the W4A16 CUDA extension")

    from torch.utils.cpp_extension import load_inline

    cap = torch.cuda.get_device_capability(0)
    arch = f"{cap[0]}.{cap[1]}"  # e.g. "7.5"; fallback below if detection looks wrong
    if not arch[0].isdigit():  # pragma: no cover
        arch = "7.5"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch

    cu_src = _read(os.path.join(_CUDA_DIR, "w4a16_gemm.cu"))
    cpp_src = _read(os.path.join(_CUDA_DIR, "bindings.cpp"))

    _MODULE = load_inline(
        name="w4a16_cuda_ext",
        cpp_sources=[cpp_src],   # contains PYBIND11_MODULE
        cuda_sources=[cu_src],   # contains kernel + launcher
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
    return _MODULE


def w4a16_matmul_cuda(
    x: Tensor, qweight: Tensor, scales: Tensor, zeros: Tensor, group_size: int
) -> Tensor:
    """Fused W4A16 GEMM via the CUDA SIMT kernel (stretch). Same contract as the
    Triton ``w4a16_matmul``."""
    assert x.is_cuda, "x must be on CUDA"
    assert x.dtype == torch.float16, f"x must be fp16, got {x.dtype}"
    assert qweight.dtype == torch.int32, f"qweight must be int32, got {qweight.dtype}"
    assert scales.dtype == torch.float16 and zeros.dtype == torch.float16
    assert x.dim() == 2 and qweight.dim() == 2
    M, K = x.shape
    Kp, N = qweight.shape
    assert Kp * 8 == K, f"qweight rows (K//8)={Kp} incompatible with x K={K}"
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"

    mod = load_cuda_module()
    return mod.w4a16_gemm(
        x.contiguous(),
        qweight.contiguous(),
        scales.contiguous(),
        zeros.contiguous(),
        int(group_size),
    )
