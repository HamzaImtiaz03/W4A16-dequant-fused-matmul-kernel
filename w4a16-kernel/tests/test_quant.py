"""M1 tests: int4 pack/unpack roundtrip + quantize/dequantize error bounds.

These are pure-PyTorch and run on CPU or GPU (no Triton required).
"""
from __future__ import annotations

import pytest
import torch

from w4a16.quant import (
    quantize_weight,
    dequantize_weight,
    pack_int4,
    unpack_int4,
)


# --------------------------------------------------------------------------- #
# pack / unpack                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [8, 64, 128, 4096])
@pytest.mark.parametrize("N", [1, 16, 4096])
def test_pack_unpack_roundtrip(K, N, device):
    """unpack(pack(x)) == x for random uint4 values."""
    x = torch.randint(0, 16, (K, N), dtype=torch.int32, device=device)
    packed = pack_int4(x)
    assert packed.shape == (K // 8, N)
    assert packed.dtype == torch.int32
    restored = unpack_int4(packed, K)
    assert restored.shape == (K, N)
    assert torch.equal(restored, x), "pack/unpack roundtrip mismatch"


def test_pack_unpack_extremes(device):
    """All-zeros and all-fifteens (sign-bit edge) must roundtrip exactly."""
    for val in (0, 15):
        x = torch.full((16, 8), val, dtype=torch.int32, device=device)
        assert torch.equal(unpack_int4(pack_int4(x)), x), f"value {val} failed roundtrip"


def test_pack_nibble_position(device):
    """A single nonzero nibble lands at the documented bit position."""
    # row k = 8*0 + j -> nibble j at bits [4j:4j+4]
    x = torch.zeros((8, 1), dtype=torch.int32, device=device)
    x[3, 0] = 0xA  # j = 3 -> bits [12:16]
    packed = pack_int4(x)
    assert int(packed[0, 0]) == (0xA << 12)


def test_pack_requires_k_div_8(device):
    with pytest.raises(AssertionError):
        pack_int4(torch.zeros((7, 4), dtype=torch.int32, device=device))


# --------------------------------------------------------------------------- #
# quantize / dequantize                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("K,N", [(128, 256), (4096, 512)])
def test_quant_shapes_dtypes(K, N, group_size, device):
    W = torch.randn(K, N, dtype=torch.float16, device=device)
    qweight, scales, zeros = quantize_weight(W, group_size)
    assert qweight.shape == (K // 8, N) and qweight.dtype == torch.int32
    assert scales.shape == (K // group_size, N) and scales.dtype == torch.float16
    assert zeros.shape == (K // group_size, N) and zeros.dtype == torch.float16
    # zero-points must be integers in [0, 15]
    assert torch.all(zeros >= 0) and torch.all(zeros <= 15)
    assert torch.all(zeros == torch.round(zeros.float()))


@pytest.mark.parametrize("group_size", [64, 128])
def test_quant_dequant_roundtrip_error(group_size, device):
    """Random-weight quantize->dequantize should have bounded mean relative error.

    For 4-bit min/max quant of Gaussian weights the step is ~range/15 ~ 0.36*sigma, so
    mean|err| ~ step/4 ~ 0.09*sigma vs mean|W| ~ 0.8*sigma => ~10% mean relative error
    (verified empirically: ~9.8% at G=64, ~10.9% at G=128). This is inherent to 4 bits,
    not a bug; we bound it at 15% as a sanity check that catches gross quantizer errors.
    """
    torch.manual_seed(1)
    K, N = 4096, 1024
    W = torch.randn(K, N, dtype=torch.float16, device=device)
    qweight, scales, zeros = quantize_weight(W, group_size)
    W_dq = dequantize_weight(qweight, scales, zeros, group_size)

    assert W_dq.shape == W.shape and W_dq.dtype == torch.float16
    num = (W_dq.float() - W.float()).abs().mean()
    den = W.float().abs().mean()
    mean_rel_err = (num / den).item()
    assert mean_rel_err < 0.15, f"mean relative error too high: {mean_rel_err:.4f}"


def test_dequant_constant_group_is_exact(device):
    """A constant group (range 0) must dequantize back to that constant (no div-by-zero)."""
    K, N, G = 128, 8, 64
    W = torch.full((K, N), 0.0, dtype=torch.float16, device=device)
    W[:, 0] = 3.0  # whole column constant
    qweight, scales, zeros = quantize_weight(W, G)
    W_dq = dequantize_weight(qweight, scales, zeros, G)
    assert torch.allclose(W_dq.float(), W.float(), atol=1e-3)


def test_quant_divisibility_errors(device):
    W = torch.randn(100, 16, dtype=torch.float16, device=device)  # 100 not div by 8
    with pytest.raises(ValueError):
        quantize_weight(W, group_size=64)
    W2 = torch.randn(96, 16, dtype=torch.float16, device=device)  # 96 not div by 64
    with pytest.raises(ValueError):
        quantize_weight(W2, group_size=64)
