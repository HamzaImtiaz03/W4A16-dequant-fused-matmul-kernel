// W4A16 fused dequant->GEMM, SIMT (CUDA-core) shared-memory tiled kernel.
// STRETCH GOAL: correct-first reference implementation. fp16 I/O, fp32 accumulate.
//
// Y[M,N] = X[M,K] @ dequant(qweight)[K,N]
//   qweight : int32 [K/8, N]   (8 uint4 nibbles per int32 along K, nibble j at bits 4j)
//   scales  : fp16  [K/G, N]
//   zeros   : fp16  [K/G, N]   (integer zero-points)
//   W_dq[k,n] = (nibble - zero[k/G,n]) * scale[k/G,n]
//
// Dequant is done PER ELEMENT, so there is no group-boundary alignment requirement
// here (unlike the Triton kernel). Correctness first; this is the stretch baseline.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE 16

__global__ void w4a16_gemm_kernel(
    const __half* __restrict__ X,       // [M, K]
    const int*    __restrict__ qweight, // [K/8, N]
    const __half* __restrict__ scales,  // [K/G, N]
    const __half* __restrict__ zeros,   // [K/G, N]
    __half*       __restrict__ Y,       // [M, N]
    int M, int N, int K, int group_size) {

    __shared__ float Xs[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = blockIdx.y * TILE + ty;  // output row (M)
    const int col = blockIdx.x * TILE + tx;  // output col (N)

    float acc = 0.0f;
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int kx = t * TILE + tx;  // K index loaded by this thread for X
        const int kw = t * TILE + ty;  // K index loaded by this thread for W

        // Load activation tile (fp16 -> fp32).
        Xs[ty][tx] = (row < M && kx < K) ? __half2float(X[row * K + kx]) : 0.0f;

        // Load + dequantize weight tile. The dequant (q - zero)*scale is done in fp16 to
        // EXACTLY match the fp16 reference oracle (and the Triton kernel); we widen to
        // fp32 only for accumulation. Doing the dequant in fp32 here would diverge from
        // the oracle's fp16 rounding and inflate the error (~0.1) past the test tolerance.
        // (__hsub/__hmul/__int2half_rn are used because -D__CUDA_NO_HALF_OPERATORS__ is set.)
        if (col < N && kw < K) {
            const int packed = qweight[(kw >> 3) * N + col];      // kw / 8
            const int nib = (packed >> (4 * (kw & 7))) & 0xF;     // unsigned nibble 0..15
            const int gi = kw / group_size;
            const __half s = scales[gi * N + col];
            const __half z = zeros[gi * N + col];
            const __half w = __hmul(__hsub(__int2half_rn(nib), z), s);  // fp16 dequant
            Ws[ty][tx] = __half2float(w);
        } else {
            Ws[ty][tx] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += Xs[ty][k] * Ws[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc);
    }
}

torch::Tensor w4a16_gemm_cuda(
    torch::Tensor x, torch::Tensor qweight,
    torch::Tensor scales, torch::Tensor zeros, int64_t group_size) {

    TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(qweight.scalar_type() == torch::kInt32, "qweight must be int32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be fp16");
    TORCH_CHECK(zeros.scalar_type() == torch::kFloat16, "zeros must be fp16");
    TORCH_CHECK(x.is_contiguous() && qweight.is_contiguous() &&
                scales.is_contiguous() && zeros.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dim() == 2 && qweight.dim() == 2, "x and qweight must be 2D");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t Kp = qweight.size(0);
    const int64_t N = qweight.size(1);
    TORCH_CHECK(Kp * 8 == K, "qweight rows*8 must equal K");
    TORCH_CHECK(K % 8 == 0, "K must be divisible by 8");
    TORCH_CHECK(group_size > 0 && K % group_size == 0, "K must be divisible by group_size");

    auto y = torch::empty({M, N}, x.options());

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    w4a16_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        qweight.data_ptr<int>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N),
        static_cast<int>(K), static_cast<int>(group_size));

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "w4a16_gemm_kernel launch failed: ", cudaGetErrorString(err));
    return y;
}
