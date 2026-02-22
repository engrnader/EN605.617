// Module 4: CUDA Memory Types
//
// Task: 1D Weighted Moving Average (stencil convolution).
//
//   output[i] = sum_{r=-kRadius}^{kRadius}
//               d_weights[r + kRadius] * input[i + r]
//
// Memory types demonstrated:
//   Host memory     - h_input, h_output_* (malloc on CPU)
//   Global memory   - d_input, d_output (cudaMalloc on GPU)
//   Constant memory - d_weights[] (__constant__ in GPU cache)
//   Shared memory   - s_data[] tile in KernelSharedMem
//   Registers       - idx, tid, sum, r, val (kernel locals)
//
// Two kernels timed and compared:
//   KernelGlobalMem - reads input directly from global memory
//   KernelSharedMem - caches input tile in shared memory first
//
// Usage: assignment.exe [block_size] [n_log2]
//   block_size : threads per block, multiple of 32 in [32, 1024]
//                default 256
//   n_log2     : log2 of array length, default 22 (4M elements)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

static constexpr int kRadius = 2;
static constexpr int kFilterSize = 2 * kRadius + 1;  // 5 filter taps

// Constant memory: read-only, cached, broadcast-efficient.
__constant__ float d_weights[kFilterSize];

#define CUDA_CHECK(call)                               \
  do {                                                 \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
      fprintf(stderr, "CUDA error %s:%d: %s\n",       \
              __FILE__, __LINE__,                      \
              cudaGetErrorString(err));                \
      exit(EXIT_FAILURE);                              \
    }                                                  \
  } while (0)

// Applies the stencil filter using global memory on every access.
// Uses constant memory (d_weights) and registers (idx, sum, r, j, val).
//
// input:  read-only input array (device, length n).
// n:      number of elements.
// output: result array (device, length n).
__global__ void KernelGlobalMem(const float* __restrict__ input,
                                 int n, float* output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float sum = 0.0f;
    for (int r = -kRadius; r <= kRadius; ++r) {
      int j = idx + r;
      float val = (j >= 0 && j < n) ? input[j] : 0.0f;
      sum += d_weights[r + kRadius] * val;
    }
    output[idx] = sum;
  }
}

// Applies the stencil filter using a shared memory tile with halo regions.
// Each input element is loaded from global memory exactly once; the
// convolution reads from fast on-chip shared memory.
//
// Shared memory layout (block_size + 2*kRadius floats):
//   [left_halo | block_data | right_halo]
//
// input:  read-only input array (device, length n).
// n:      number of elements.
// output: result array (device, length n).
__global__ void KernelSharedMem(const float* __restrict__ input,
                                 int n, float* output) {
  extern __shared__ float s_data[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int block_start = blockIdx.x * blockDim.x;

  s_data[tid + kRadius] = (idx < n) ? input[idx] : 0.0f;

  // First kRadius threads load the left and right halo elements.
  if (tid < kRadius) {
    int left_idx = block_start - kRadius + tid;
    s_data[tid] = (left_idx >= 0) ? input[left_idx] : 0.0f;

    int right_idx = block_start + blockDim.x + tid;
    s_data[blockDim.x + kRadius + tid] =
        (right_idx < n) ? input[right_idx] : 0.0f;
  }
  __syncthreads();

  if (idx < n) {
    float sum = 0.0f;
    for (int r = 0; r < kFilterSize; ++r) {
      sum += d_weights[r] * s_data[tid + r];
    }
    output[idx] = sum;
  }
}

// Launches KernelGlobalMem and returns kernel execution time in milliseconds.
// Copies results from d_out to h_out after the kernel completes.
//
// d_in:       device input array (read-only, length n).
// n:          number of elements.
// block_size: threads per block.
// d_out:      device output array (length n).
// h_out:      host output array (length n); receives D2H copy of d_out.
static float RunGlobal(const float* d_in, int n, int block_size,
                        float* d_out, float* h_out) {
  int num_blocks = (n + block_size - 1) / block_size;
  cudaEvent_t start, stop;
  float ms = 0.0f;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  KernelGlobalMem<<<num_blocks, block_size>>>(d_in, n, d_out);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float),
                         cudaMemcpyDeviceToHost));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

// Launches KernelSharedMem and returns kernel execution time in milliseconds.
// Shared memory size is derived from block_size and kRadius.
// Copies results from d_out to h_out after the kernel completes.
//
// d_in:       device input array (read-only, length n).
// n:          number of elements.
// block_size: threads per block.
// d_out:      device output array (length n).
// h_out:      host output array (length n); receives D2H copy of d_out.
static float RunShared(const float* d_in, int n, int block_size,
                        float* d_out, float* h_out) {
  int num_blocks = (n + block_size - 1) / block_size;
  size_t shared_size = (block_size + 2 * kRadius) * sizeof(float);
  cudaEvent_t start, stop;
  float ms = 0.0f;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  KernelSharedMem<<<num_blocks, block_size, shared_size>>>(d_in, n, d_out);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float),
                         cudaMemcpyDeviceToHost));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

// Prints the launch configuration and memory type legend to stdout.
static void PrintConfig(int n, int n_log2, int block_size) {
  int num_blocks = (n + block_size - 1) / block_size;
  printf("**************************************************\n");
  printf("CUDA Memory Types Demo\n");
  printf("**************************************************\n");
  printf("Array size  : %d elements (2^%d)\n", n, n_log2);
  printf("Block size  : %d threads\n", block_size);
  printf("Grid size   : %d blocks\n", num_blocks);
  printf("Filter      : %d taps (radius=%d)\n", kFilterSize, kRadius);
  printf("--------------------------------------------------\n");
  printf("Memory types:\n");
  printf("  Host memory     - h_input, h_output_*\n");
  printf("  Global memory   - d_input, d_output\n");
  printf("  Constant memory - d_weights[%d]\n", kFilterSize);
  printf("  Shared memory   - s_data[] in KernelSharedMem\n");
  printf("  Registers       - idx, tid, sum, r, val\n");
  printf("--------------------------------------------------\n");
}

// Fills h_input with reproducible random values in [-1, 1] and populates
// h_weights with a Gaussian-like lowpass filter (weights sum to 1.0).
//
// n:         number of elements in h_input.
// h_input:   output host array to fill (length n).
// h_weights: output filter coefficient array (length kFilterSize).
static void InitializeData(int n, float* h_input, float* h_weights) {
  srand(42);
  for (int idx = 0; idx < n; ++idx) {
    h_input[idx] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
  }
  h_weights[0] = 0.1f;
  h_weights[1] = 0.2f;
  h_weights[2] = 0.4f;
  h_weights[3] = 0.2f;
  h_weights[4] = 0.1f;
}

// Compares two output arrays element-by-element. Prints the first 5
// mismatches and a PASS/FAIL summary to stdout.
//
// n:     number of elements.
// out_g: global-kernel output (host, length n).
// out_s: shared-kernel output (host, length n).
static void VerifyResults(int n, const float* out_g, const float* out_s) {
  constexpr float kEps = 1e-4f;
  int mismatches = 0;
  for (int idx = 0; idx < n; ++idx) {
    if (fabsf(out_g[idx] - out_s[idx]) > kEps) {
      if (mismatches < 5) {
        printf("  Mismatch idx=%d  g=%.6f  s=%.6f\n",
               idx, out_g[idx], out_s[idx]);
      }
      ++mismatches;
    }
  }
  printf("Verification (global vs shared): %s  (%d/%d)\n",
         mismatches == 0 ? "PASS" : "FAIL", mismatches, n);
  printf("**************************************************\n");
}

// Launches both kernels, prints timing comparison, then verifies outputs.
//
// d_in:      device input array (read-only, length n).
// n:         number of elements.
// block_size: threads per block.
// d_out:     device scratch buffer for kernel output (length n).
// h_out_g:   host array for global-kernel results (length n).
// h_out_s:   host array for shared-kernel results (length n).
static void RunKernelsAndReport(const float* d_in, int n, int block_size,
                                float* d_out, float* h_out_g,
                                float* h_out_s) {
  float ms_g = RunGlobal(d_in, n, block_size, d_out, h_out_g);
  float ms_s = RunShared(d_in, n, block_size, d_out, h_out_s);
  printf("Timing results:\n");
  printf("  KernelGlobalMem : %8.3f ms\n", ms_g);
  printf("  KernelSharedMem : %8.3f ms\n", ms_s);
  if (ms_g > 0.0f && ms_s > 0.0f) {
    printf("  Speedup         : %.2fx\n", ms_g / ms_s);
  }
  printf("--------------------------------------------------\n");
  VerifyResults(n, h_out_g, h_out_s);
}

int main(int argc, char* argv[]) {
  int block_size = (argc > 1) ? atoi(argv[1]) : 256;
  int n_log2 = (argc > 2) ? atoi(argv[2]) : 22;
  int n = 1 << n_log2;
  if (block_size < 32 || block_size > 1024 || block_size % 32 != 0) {
    fprintf(stderr,
            "block_size must be a multiple of 32 in [32, 1024]\n");
    return EXIT_FAILURE;
  }
  PrintConfig(n, n_log2, block_size);

  // Host memory.
  float* h_input = static_cast<float*>(malloc(n * sizeof(float)));
  float* h_out_g = static_cast<float*>(malloc(n * sizeof(float)));
  float* h_out_s = static_cast<float*>(malloc(n * sizeof(float)));
  if (!h_input || !h_out_g || !h_out_s) {
    fprintf(stderr, "Host malloc failed.\n");
    return EXIT_FAILURE;
  }
  float h_weights[kFilterSize];
  InitializeData(n, h_input, h_weights);

  // Constant memory.
  CUDA_CHECK(cudaMemcpyToSymbol(d_weights, h_weights,
                                kFilterSize * sizeof(float)));
  // Global memory.
  float* d_input;
  float* d_output;
  CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));
  RunKernelsAndReport(d_input, n, block_size,
                      d_output, h_out_g, h_out_s);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  free(h_input); free(h_out_g); free(h_out_s);
  return EXIT_SUCCESS;
}
