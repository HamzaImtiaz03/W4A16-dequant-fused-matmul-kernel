"""M3 tests: Triton W4A16 kernel == fp32-accumulate reference oracle.

Both sides consume the EXACT SAME (qweight, scales, zeros) so this isolates *kernel*
correctness from *quantization* error. Requires a CUDA GPU + Triton.
"""
from __future__ import annotations

import pytest
import torch

from conftest import requires_cuda

# Triton may be missing on a CPU-only machine; skip the whole module if so.
triton = pytest.importorskip("triton", reason="triton required for kernel tests")

from w4a16.quant import quantize_weight
from w4a16.reference import reference_w4a16


pytestmark = requires_cuda


def _build_case(M, K, N, group_size, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    X = torch.randn(M, K, dtype=torch.float16, device=device)
    W = torch.randn(K, N, dtype=torch.float16, device=device)
    qweight, scales, zeros = quantize_weight(W, group_size)
    qweight = qweight.to(device)
    scales = scales.to(device)
    zeros = zeros.to(device)
    return X, qweight, scales, zeros


@pytest.mark.parametrize("M", [1, 16, 64, 256])
@pytest.mark.parametrize("K,N", [(4096, 4096), (4096, 11008), (11008, 4096)])
@pytest.mark.parametrize("group_size", [64, 128])
def test_triton_matches_reference(M, K, N, group_size):
    from w4a16.triton_kernel import w4a16_matmul

    X, qweight, scales, zeros = _build_case(M, K, N, group_size)
    ref = reference_w4a16(X, qweight, scales, zeros, group_size)
    out = w4a16_matmul(X, qweight, scales, zeros, group_size)

    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@requires_cuda
def test_kernel_asserts_contiguity():
    """Non-contiguous input should trip an assertion, not produce silent garbage."""
    from w4a16.triton_kernel import w4a16_matmul

    X, qweight, scales, zeros = _build_case(8, 128, 64, 64)
    # Transposing a [K, M] tensor yields a non-contiguous [M, K] view.
    X_bad = torch.randn(128, 8, dtype=torch.float16, device="cuda").t()  # [8,128] non-contig
    assert not X_bad.is_contiguous()
    with pytest.raises(AssertionError):
        w4a16_matmul(X_bad, qweight, scales, zeros, 64)


@requires_cuda
def test_kernel_rejects_bad_group_size():
    """group_size that no config's BLOCK_K divides must raise (not silently corrupt)."""
    from w4a16.triton_kernel import w4a16_matmul

    # 8 | K but group_size=8 is not divisible by any config BLOCK_K (>=32) -> must raise.
    X, qweight, scales, zeros = _build_case(8, 64, 64, 8)
    with pytest.raises(AssertionError):
        w4a16_matmul(X, qweight, scales, zeros, 8)


@requires_cuda
def test_w4a16_linear_end_to_end():
    """W4A16Linear.from_linear matches dequant(quant(W)) @ x within tolerance."""
    from w4a16.linear import W4A16Linear
    from w4a16.reference import reference_w4a16

    torch.manual_seed(2)
    K, N, M = 4096, 4096, 16
    lin = torch.nn.Linear(K, N, bias=True).cuda().half()
    qlin = W4A16Linear.from_linear(lin, group_size=128)

    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    y = qlin(x)

    ref = reference_w4a16(x, qlin.qweight, qlin.scales, qlin.zeros, 128) + qlin.bias
    assert y.shape == (M, N)
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------- #
# M5 (stretch): CUDA SIMT kernel — same oracle, skips if the build is unavailable.
# Building the extension can take ~1 min; kept to a couple of shapes.
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("M", [1, 64])
@pytest.mark.parametrize("K,N", [(4096, 4096), (11008, 4096)])
@pytest.mark.parametrize("group_size", [64, 128])
def test_cuda_matches_reference(M, K, N, group_size):
    try:
        from w4a16.cuda_kernel import w4a16_matmul_cuda, load_cuda_module
        load_cuda_module(verbose=False)
    except Exception as e:  # nvcc missing / compile failure on this runtime
        pytest.skip(f"CUDA extension unavailable: {e}")

    X, qweight, scales, zeros = _build_case(M, K, N, group_size)
    ref = reference_w4a16(X, qweight, scales, zeros, group_size)
    out = w4a16_matmul_cuda(X, qweight, scales, zeros, group_size)

    assert out.shape == (M, N) and out.dtype == torch.float16
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
