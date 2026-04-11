/**
 * @file    kernel_sweep.cu
 * @brief   Kernel 1: Parallel sweep, smoothing, and peak detection.
 *
 *          Three-stage pipeline per channel:
 *            1. Generate peak-shaped response (1 thread = 1 setpoint)
 *            2. Smooth with uniform moving average (stencil)
 *            3. Find peak via parallel reduction in shared memory
 */

#include <math.h>

#include "kernel_sweep.h"

/* ---------------------------------------------------------------
 * Device kernels
 * --------------------------------------------------------------- */

/**
 * @brief  Generate peak-shaped sensor response.
 *
 *         L(x) = A / (1 + ((x - x0) / gamma)^2) + noise
 *         One thread per sweep step.
 */
__global__ void kern_generate_response(float* response, int n, float x0,
                                       float gamma, int noise_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    float x       = (float)idx / (float)n;
    float delta   = (x - x0) / gamma;
    float signal  = PEAK_AMPLITUDE / (1.0f + delta * delta);
    float noise   = NOISE_AMPLITUDE * sinf((float)(idx * noise_seed) * 0.037f);
    response[idx] = signal + noise;
}

/**
 * @brief  Smooth response with uniform moving average.
 *
 *         Each output sample is the mean of (2*radius + 1)
 *         neighbors. Boundary samples use a reduced window.
 */
__global__ void kern_smooth_response(float* smoothed, const float* raw, int n,
                                     int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    float sum   = 0.0f;
    int   count = 0;
    for (int k = -radius; k <= radius; k++) {
        int j = idx + k;
        if (j >= 0 && j < n) {
            sum += raw[j];
            count++;
        }
    }
    smoothed[idx] = sum / (float)count;
}

/**
 * @brief  Parallel reduction to find the maximum value
 *         and its index within each block.
 *
 *         Shared memory layout: [float vals | int indices]
 */
__global__ void kern_find_peak(float* d_block_vals, int* d_block_idxs,
                               const float* data, int n) {
    extern __shared__ float s_vals[];
    int*                    s_idxs = (int*)(s_vals + blockDim.x);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    s_vals[tid] = (idx < n) ? data[idx] : -1.0f;
    s_idxs[tid] = (idx < n) ? idx : -1;
    __syncthreads();

    /* Tree-based parallel reduction: each iteration halves the stride.
       Pass 1: threads 0..N/2-1 compare against N/2..N-1
       Pass 2: threads 0..N/4-1 compare against N/4..N/2-1
       ...until thread 0 holds the block-wide maximum. */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_vals[tid + s] > s_vals[tid]) {
            s_vals[tid] = s_vals[tid + s];
            s_idxs[tid] = s_idxs[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_vals[blockIdx.x] = s_vals[0];
        d_block_idxs[blockIdx.x] = s_idxs[0];
    }
}

/* ---------------------------------------------------------------
 * Host helpers
 * --------------------------------------------------------------- */

/**
 * @brief  CPU-side final reduction over per-block peaks.
 */
static void reduce_block_peaks(const float* vals, const int* idxs,
                               int num_blocks, SweepResult* result) {
    result->peak_val = -1.0f;
    result->peak_idx = -1;
    for (int i = 0; i < num_blocks; i++) {
        if (vals[i] > result->peak_val) {
            result->peak_val = vals[i];
            result->peak_idx = idxs[i];
        }
    }
}

/* ---------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------- */

void sweep_pipeline(float* d_raw, float* d_smoothed, int n, int block_size,
                    int num_blocks, float center_frac, float width_frac,
                    int noise_seed, cudaStream_t stream, SweepResult* result) {
    size_t bv_bytes = (size_t)num_blocks * sizeof(float);
    size_t bi_bytes = (size_t)num_blocks * sizeof(int);
    size_t shared   = (size_t)block_size * (sizeof(float) + sizeof(int));

    float* d_bv;
    int*   d_bi;
    CUDA_CHECK(cudaMalloc(&d_bv, bv_bytes));
    CUDA_CHECK(cudaMalloc(&d_bi, bi_bytes));

    kern_generate_response<<<num_blocks, block_size, 0, stream>>>(
        d_raw, n, center_frac, width_frac, noise_seed);

    kern_smooth_response<<<num_blocks, block_size, 0, stream>>>(
        d_smoothed, d_raw, n, SMOOTH_RADIUS);

    kern_find_peak<<<num_blocks, block_size, shared, stream>>>(d_bv, d_bi,
                                                               d_smoothed, n);

    float* h_bv = (float*)malloc(bv_bytes);
    int*   h_bi = (int*)malloc(bi_bytes);
    CUDA_CHECK(
        cudaMemcpyAsync(h_bv, d_bv, bv_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(h_bi, d_bi, bi_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    reduce_block_peaks(h_bv, h_bi, num_blocks, result);

    free(h_bv);
    free(h_bi);
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_bi));
}
