// torch/pybind bindings for the W4A16 CUDA SIMT kernel (stretch goal).
// The implementation lives in w4a16_gemm.cu; here we just expose it to Python.

#include <torch/extension.h>

// Forward declaration of the launcher defined in w4a16_gemm.cu.
torch::Tensor w4a16_gemm_cuda(
    torch::Tensor x, torch::Tensor qweight,
    torch::Tensor scales, torch::Tensor zeros, int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm_cuda,
          "Fused W4A16 dequant->GEMM (CUDA SIMT, shared-memory tiled)");
}
