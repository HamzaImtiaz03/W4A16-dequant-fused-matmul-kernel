#!/usr/bin/env bash
# M0 — Colab environment setup + capability probe.
# Run from the repo root:  bash scripts/setup_colab.sh
set -euo pipefail

echo "==================================================================="
echo " W4A16 kernel — Colab setup"
echo "==================================================================="

# Colab already ships torch + triton with a matching CUDA build.
# Install only the lightweight extras so we don't risk breaking the runtime.
echo "[1/3] Installing lightweight extras (numpy / matplotlib / pytest)..."
pip install -q numpy matplotlib pytest

echo "[2/3] nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "  !! nvidia-smi not found. On Colab: Runtime > Change runtime type > T4 GPU."
fi

echo "[3/3] Python / torch / triton / device capability:"
python - <<'PY'
import sys
print("python  :", sys.version.split()[0])

import torch
print("torch   :", torch.__version__, "| cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

try:
    import triton
    print("triton  :", triton.__version__)
except Exception as e:  # pragma: no cover
    print("triton  : NOT IMPORTABLE ->", repr(e))

if not torch.cuda.is_available():
    print("\n*** NO CUDA GPU DETECTED ***")
    print("Triton / CUDA kernels cannot run. Switch the Colab runtime to a GPU.")
    sys.exit(1)

cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
sm = f"sm_{cap[0]}{cap[1]}"
print(f"device  : {name}")
print(f"capability: {cap}  ({sm})")
bf16 = torch.cuda.is_bf16_supported()
print(f"bf16 supported: {bf16}  (T4/Turing = False; we use fp16 + fp32 accumulate)")
print("\nEnvironment OK — proceed to the milestones.")
PY
