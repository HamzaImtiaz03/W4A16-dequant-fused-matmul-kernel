"""Shared pytest fixtures + CUDA-availability guard.

Adds ``src/`` to ``sys.path`` so ``import w4a16`` works when running pytest from the
repo root (``python -m pytest tests/``).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Make the package importable without installation.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

CUDA_AVAILABLE = torch.cuda.is_available()

# Reusable marker for tests that require a real GPU (Triton / CUDA kernels).
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")


@pytest.fixture(scope="session")
def device() -> str:
    """Device for pure-PyTorch tests (quant/pack work on CPU or GPU)."""
    return "cuda" if CUDA_AVAILABLE else "cpu"


@pytest.fixture(scope="session", autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
