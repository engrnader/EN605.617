/**
 * @file    kernel_gradient.cu
 * @brief   Kernel 2: Gradient analysis, margin scoring, and
 *          operating point selection.
 *
 *          Two-stage pipeline:
 *            1. Compute central-difference gradient and margin
 *               score at every sweep point (embarrassingly parallel)
 *            2. Parallel reduction to select the point with the
 *               highest margin score
 *
 *          The margin score combines:
 *            - detector signal strength (higher is better)
 *            - low gradient magnitude (flat peak = stable)
 *            - distance from sweep edges (avoid boundary)
 */

#include <math.h>

#include "kernel_gradient.h"

/* ---------------------------------------------------------------
 * Device kernels
 * --------------------------------------------------------------- */

/**
 * @brief  Compute gradient and margin score per sweep point.
 *
 *         gradient = (y[i+r] - y[i-r]) / (2*r)
 *         edge_weight = 1 if away from edges, 0 near edges
 *         margin = signal * edge_weight / (1 + |gradient|)
 *
 * @param  d_margin     Output margin scores [n]
 * @param  d_gradient   Output gradient values [n]
 * @param  d_smoothed   Input smoothed response [n]
 * @param  n            Array length
 * @param  radius       Central difference half-width
 */
__global__ void kern_compute_margin(float* d_margin, float* d_gradient,
                                    const float* d_smoothed, int n,
                                    int radius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    /* central difference gradient */
    int left  = idx - radius;
    int right = idx + radius;
    if (left < 0)
        left = 0;
    if (right >= n)
        right = n - 1;

    float grad = (d_smoothed[right] - d_smoothed[left]) / (float)(right - left);
    d_gradient[idx] = grad;

    /* edge penalty: suppress points near sweep boundaries */
    float frac        = (float)idx / (float)n;
    float edge_weight = 1.0f;
    if (frac < EDGE_MARGIN_FRAC || frac > (1.0f - EDGE_MARGIN_FRAC))
        edge_weight = 0.0f;

    /* margin: reward high signal, penalize steep gradient */
    float signal  = d_smoothed[idx];
    float margin  = signal * edge_weight / (1.0f + fabsf(grad));
    d_margin[idx] = margin;
}

/**
 * @brief  Parallel reduction: find index of max margin.
 *
 *         Same shared-memory tree reduction as Kernel 1,
 *         but operates on margin scores.
 */
__global__ void kern_find_best_margin(float* d_block_vals, int* d_block_idxs,
                                      const float* d_margin, int n) {
    extern __shared__ float s_vals[];
    int*                    s_idxs = (int*)(s_vals + blockDim.x);

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    s_vals[tid] = (idx < n) ? d_margin[idx] : -1.0f;
    s_idxs[tid] = (idx < n) ? idx : -1;
    __syncthreads();

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
 * @brief  CPU-side final reduction over per-block results.
 */
static void reduce_best_margin(const float* vals, const int* idxs,
                               int num_blocks, int* best_idx,
                               float* best_margin) {
    *best_margin = -1.0f;
    *best_idx    = -1;
    for (int i = 0; i < num_blocks; i++) {
        if (vals[i] > *best_margin) {
            *best_margin = vals[i];
            *best_idx    = idxs[i];
        }
    }
}

/**
 * @brief  Read back gradient at a single index.
 */
static float read_gradient_at(const float* d_gradient, int idx,
                              cudaStream_t stream) {
    float val;
    CUDA_CHECK(cudaMemcpyAsync(&val, d_gradient + idx, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return val;
}

/* ---------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------- */

void gradient_pipeline(const float* d_smoothed, int n, int block_size,
                       int num_blocks, cudaStream_t stream,
                       OpPointResult* result) {
    size_t float_bytes = (size_t)n * sizeof(float);
    size_t bv_bytes    = (size_t)num_blocks * sizeof(float);
    size_t bi_bytes    = (size_t)num_blocks * sizeof(int);
    size_t shared      = (size_t)block_size * (sizeof(float) + sizeof(int));

    float* d_margin;
    float* d_gradient;
    float* d_bv;
    int*   d_bi;
    CUDA_CHECK(cudaMalloc(&d_margin, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_gradient, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_bv, bv_bytes));
    CUDA_CHECK(cudaMalloc(&d_bi, bi_bytes));

    kern_compute_margin<<<num_blocks, block_size, 0, stream>>>(
        d_margin, d_gradient, d_smoothed, n, GRADIENT_RADIUS);

    kern_find_best_margin<<<num_blocks, block_size, shared, stream>>>(
        d_bv, d_bi, d_margin, n);

    float* h_bv = (float*)malloc(bv_bytes);
    int*   h_bi = (int*)malloc(bi_bytes);
    CUDA_CHECK(
        cudaMemcpyAsync(h_bv, d_bv, bv_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(h_bi, d_bi, bi_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int   best_idx;
    float best_margin;
    reduce_best_margin(h_bv, h_bi, num_blocks, &best_idx, &best_margin);

    result->best_idx = best_idx;
    result->margin   = best_margin;
    result->gradient = read_gradient_at(d_gradient, best_idx, stream);

    free(h_bv);
    free(h_bi);
    CUDA_CHECK(cudaFree(d_margin));
    CUDA_CHECK(cudaFree(d_gradient));
    CUDA_CHECK(cudaFree(d_bv));
    CUDA_CHECK(cudaFree(d_bi));
}
