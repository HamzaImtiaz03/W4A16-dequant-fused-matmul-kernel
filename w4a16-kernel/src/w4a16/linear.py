"""``W4A16Linear`` — a drop-in ``nn.Linear`` replacement backed by the Triton kernel.

Note on layouts: ``nn.Linear`` stores ``weight`` as ``[out_features, in_features]`` =
``[N, K]`` and computes ``y = x @ weight.T``. Our logical weight is ``[K, N]``, so
:meth:`from_linear` transposes before quantizing. Forward then computes ``x @ W``
directly, matching ``nn.Linear``'s output.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .quant import quantize_weight
from .triton_kernel import w4a16_matmul

__all__ = ["W4A16Linear"]


class W4A16Linear(nn.Module):
    """4-bit weight / fp16 activation linear layer.

    Buffers:
        qweight : int32 [K//8, N]
        scales  : fp16  [K//G, N]
        zeros   : fp16  [K//G, N]
        bias    : fp16  [N] (optional)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % 8 != 0:
            raise ValueError(f"in_features={in_features} must be divisible by 8")
        if in_features % group_size != 0:
            raise ValueError(f"in_features={in_features} must be divisible by group_size={group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        K, N = in_features, out_features
        self.register_buffer("qweight", torch.zeros((K // 8, N), dtype=torch.int32, device=device))
        self.register_buffer("scales", torch.zeros((K // group_size, N), dtype=torch.float16, device=device))
        self.register_buffer("zeros", torch.zeros((K // group_size, N), dtype=torch.float16, device=device))
        if bias:
            self.register_buffer("bias", torch.zeros(N, dtype=torch.float16, device=device))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "W4A16Linear":
        """Quantize an existing ``nn.Linear`` into a ``W4A16Linear``."""
        N, K = linear.weight.shape  # nn.Linear weight is [out, in]
        mod = cls(
            in_features=K,
            out_features=N,
            group_size=group_size,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )
        W = linear.weight.detach().t().contiguous()  # [N, K] -> [K, N]
        mod.quantize_from_dense(W)
        if linear.bias is not None:
            mod.bias.copy_(linear.bias.detach().to(torch.float16))
        return mod

    def quantize_from_dense(self, W: Tensor) -> None:
        """Quantize a dense weight ``W`` of shape ``[K, N]`` (in_features, out_features)."""
        assert W.shape == (self.in_features, self.out_features), (
            f"expected weight [K,N]={(self.in_features, self.out_features)}, got {tuple(W.shape)}"
        )
        qweight, scales, zeros = quantize_weight(W, self.group_size)
        self.qweight.copy_(qweight.to(self.qweight.device))
        self.scales.copy_(scales.to(self.scales.device))
        self.zeros.copy_(zeros.to(self.zeros.device))

    def forward(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.float16, f"x must be fp16, got {x.dtype}"
        assert x.shape[-1] == self.in_features, (
            f"x last dim {x.shape[-1]} != in_features {self.in_features}"
        )
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features).contiguous()
        y = w4a16_matmul(x2d, self.qweight, self.scales, self.zeros, self.group_size)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )
